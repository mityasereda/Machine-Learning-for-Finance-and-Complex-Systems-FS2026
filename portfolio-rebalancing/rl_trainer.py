import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Sigmoid()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, state):
        features = self.feature_extractor(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

class PPOTrainer:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, gamma=0.99, 
                 epsilon=0.2, epochs=10, batch_size=64, robust_params=None):
        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.robust_params = robust_params

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state)
        
        return action_probs.squeeze(0).cpu().numpy()
    
    def _get_price_indices(self, robust_params):
        """Get indices of the most recent price for each ticker in the state vector.

        State layout: for each ticker, [prices(lookback), volumes(lookback),
        returns(lookback), volatilities(lookback)] concatenated.
        """
        num_tickers = self.action_dim  # one action per ticker
        lookback = self.state_dim // (num_tickers * 4)
        # Most recent price for ticker i is at offset: i * lookback * 4 + (lookback - 1)
        return [i * lookback * 4 + (lookback - 1) for i in range(num_tickers)]

    def _build_discretized_values(self, next_states, robust_params):
        """Construct the 3-dim value vector v by evaluating the critic at
        3 discretized next-price states: [p+eps, p, p-eps].

        For multi-asset, all tickers' prices are shifted simultaneously.

        Returns:
            v: tensor of shape [batch, 3] with critic values at each price variant.
        """
        eps = robust_params["epsilon"]
        price_indices = self._get_price_indices(robust_params)

        ns_up = next_states.clone()
        ns_same = next_states.clone()
        ns_down = next_states.clone()

        for idx in price_indices:
            ns_up[:, idx] += eps
            ns_down[:, idx] -= eps

        with torch.no_grad():
            _, v_up = self.actor_critic(ns_up)        # [batch, 1]
            _, v_same = self.actor_critic(ns_same)    # [batch, 1]
            _, v_down = self.actor_critic(ns_down)    # [batch, 1]

        return torch.cat([v_up, v_same, v_down], dim=1)  # [batch, 3]

    def calculate_u_star(self, states, actions, next_states, robust_params):
        """Calculate worst-case perturbation u* per Theorem 3.5 of the paper.

        For multi-asset portfolio rebalancing, the perturbation is applied to
        the aggregate portfolio (all tickers' prices shifted together).

        Returns:
            correction: tensor of shape [batch] — the scalar v^T u* per sample,
                        to be added to advantages as gamma * correction.
        """
        robust_type = robust_params["robust_type"]
        beta = robust_params["beta"]

        # Build 3-dim value vector: V(s') at [p+eps, p, p-eps]
        v = self._build_discretized_values(next_states, robust_params)  # [batch, 3]

        if robust_type == "p1N2":
            # Theorem 3.5(b): p=1, N=2 (elliptic uncertainty set)
            focus_buy = torch.tensor(
                robust_params["focus_buy"], dtype=torch.float32, device=self.device
            )
            focus_sell = torch.tensor(
                robust_params["focus_sell"], dtype=torch.float32, device=self.device
            )

            # For multi-asset: use net portfolio action to determine buy/sell direction
            # actions shape: [batch, num_tickers], use mean action sign as aggregate
            net_action = actions.mean(dim=1, keepdim=True)  # [batch, 1]
            is_buy = (net_action > 0.5).float()  # Sigmoid actor outputs in [0,1]; >0.5 = buy
            u1 = is_buy * focus_buy.unsqueeze(0) + (1 - is_buy) * focus_sell.unsqueeze(0)
            u2 = torch.zeros_like(u1)

            # Midpoint of foci
            midpoint = (u1 + u2) / 2

            # ||u1 - u2||_1
            u_diff_l1 = torch.sum(torch.abs(u1 - u2), dim=1, keepdim=True)

            # Scaling factor
            scale = (beta - u_diff_l1) / 2

            # Compute mu* and lambda*
            v_max = v.max(dim=1, keepdim=True).values
            v_min = v.min(dim=1, keepdim=True).values
            mu_star = -(v_max + v_min) / 2
            lambda_star = -(v_max - v_min) / 4

            v_shifted = v + mu_star

            # Indicator: |v + mu*| >= 2|lambda*|
            tol = 1e-6
            indicator = (torch.abs(v_shifted) >= 2 * torch.abs(lambda_star) - tol).float()
            sign_v = torch.sign(v_shifted)

            u_star = midpoint - scale * sign_v * indicator
            correction = torch.sum(v * u_star, dim=1)
            return correction

        elif robust_type == "p1":
            # Theorem 3.5(a): N=1, p=1 (q=inf), u1=0 (ball uncertainty set)
            v_max = v.max(dim=1, keepdim=True).values
            v_min = v.min(dim=1, keepdim=True).values
            mu_star = -(v_max + v_min) / 2

            v_shifted = v + mu_star
            abs_v = torch.abs(v_shifted)
            max_idx = torch.argmax(abs_v, dim=1, keepdim=True)
            indicator = torch.zeros_like(v).scatter_(1, max_idx, 1.0)
            sign_v = torch.sign(v_shifted)

            u_star = beta * sign_v * indicator
            correction = torch.sum(v * u_star, dim=1)
            return correction

        else:
            raise ValueError(f"Invalid robust type: {robust_type}. Use 'p1N2' or 'p1'.")
    
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device) 
        
        robust_bonus = None
        if self.robust_params is not None:
            u_star = self.calculate_u_star(states, actions, next_states, self.robust_params) 
            robust_bonus = (u_star.reshape(-1) * self.gamma).detach()
            try: 
                np.save(f'u_star_{self.robust_params["robust_type"]}.pkl', u_star.cpu().numpy()) 
            except: 
                pass 

        with torch.no_grad():
            _, values = self.actor_critic(states)
            _, next_values = self.actor_critic(next_states)
            
            rewards_for_returns = rewards
            if robust_bonus is not None:
                rewards_for_returns = rewards + robust_bonus

            returns = []
            running_return = next_values[-1].squeeze() * (1 - dones[-1])
            
            for reward, done in zip(reversed(rewards_for_returns), reversed(dones)):
                running_return = reward + self.gamma * running_return * (1 - done) 
                returns.insert(0, running_return)
            
            returns = torch.stack(returns).view(-1)
            advantages = returns - values.view(-1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_action_probs, _ = self.actor_critic(states)
        old_action_probs = old_action_probs.detach()
        
        total_loss = 0
        num_batches = len(states) // self.batch_size + (1 if len(states) % self.batch_size != 0 else 0)
        
        for _ in range(self.epochs):
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                batch_indices = indices[start_idx:start_idx + min(self.batch_size, len(states) - start_idx)]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_probs = old_action_probs[batch_indices]
                
                action_probs, values = self.actor_critic(batch_states)
                
                ratio = action_probs / (batch_old_probs + 1e-8)
                
                surr1 = ratio * batch_advantages.unsqueeze(1)
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages.unsqueeze(1)
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                loss = actor_loss + 0.5 * critic_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        self.scheduler.step(-total_loss / (self.epochs * num_batches))
        
        return total_loss / (self.epochs * num_batches)
    
    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)
    
    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path)) 
