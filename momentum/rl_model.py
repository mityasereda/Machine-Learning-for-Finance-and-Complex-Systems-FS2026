import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
import pickle

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_init=-1.0):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Output actions in [-1, 1]
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, state):
        features = self.feature_extractor(state)
        action_mean = self.actor(features)
        value = self.critic(features)
        return action_mean, value

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

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        self._action_eps = 1e-6

    def _build_policy(self, action_mean):
        clamped_mean = action_mean.clamp(-1 + self._action_eps, 1 - self._action_eps)
        latent_mean = torch.atanh(clamped_mean)
        std = self.actor_critic.log_std.exp().unsqueeze(0).expand_as(latent_mean)
        base_dist = Independent(Normal(latent_mean, std), 1)
        return TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])
    
    def select_action(self, state):
        """Select deterministic action for evaluation."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_mean, _ = self.actor_critic(state)
        return action_mean.squeeze(0).cpu().numpy()

    def sample_action(self, state):
        """Sample action and log-probability for PPO rollouts."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_mean, _ = self.actor_critic(state)
            policy = self._build_policy(action_mean)
            action = policy.sample()
            log_prob = policy.log_prob(action)
        return action.squeeze(0).cpu().numpy(), float(log_prob.item())
    
    def _build_discretized_values(self, next_states, robust_params):
        """Construct the 3-dim value vector v by evaluating the critic at
        3 discretized next-price states: [p+eps, p, p-eps].

        Returns:
            v: tensor of shape [batch, 3] with critic values at each price variant.
        """
        eps = robust_params["epsilon"]
        price_idx = self.state_dim // 4 - 1  # last price index in lookback window

        ns_up = next_states.clone()
        ns_up[:, price_idx] += eps

        ns_same = next_states.clone()

        ns_down = next_states.clone()
        ns_down[:, price_idx] -= eps

        with torch.no_grad():
            _, v_up = self.actor_critic(ns_up)        # [batch, 1]
            _, v_same = self.actor_critic(ns_same)    # [batch, 1]
            _, v_down = self.actor_critic(ns_down)    # [batch, 1]

        return torch.cat([v_up, v_same, v_down], dim=1)  # [batch, 3]

    def calculate_u_star(self, states, actions, next_states, robust_params):
        """Calculate worst-case perturbation u* per Theorem 3.5 of the paper.

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
            # Foci: u1 is action-dependent (buy/sell), u2 = 0
            focus_buy = torch.tensor(
                robust_params["focus_buy"], dtype=torch.float32, device=self.device
            )
            focus_sell = torch.tensor(
                robust_params["focus_sell"], dtype=torch.float32, device=self.device
            )

            # Select focus based on action sign: actions > 0 -> buy, <= 0 -> sell
            is_buy = (actions > 0).float()  # [batch, 1]
            if is_buy.dim() == 1:
                is_buy = is_buy.unsqueeze(1)
            u1 = is_buy * focus_buy.unsqueeze(0) + (1 - is_buy) * focus_sell.unsqueeze(0)  # [batch, 3]
            u2 = torch.zeros_like(u1)  # [batch, 3]

            # Midpoint of foci
            midpoint = (u1 + u2) / 2  # [batch, 3]

            # ||u1 - u2||_1
            u_diff_l1 = torch.sum(torch.abs(u1 - u2), dim=1, keepdim=True)  # [batch, 1]

            # Scaling factor: (beta - ||u1 - u2||_1) / 2
            scale = (beta - u_diff_l1) / 2  # [batch, 1]

            # Compute mu* and lambda* from Theorem 3.5(b)
            v_max = v.max(dim=1, keepdim=True).values   # [batch, 1]
            v_min = v.min(dim=1, keepdim=True).values   # [batch, 1]
            mu_star = -(v_max + v_min) / 2              # [batch, 1]
            lambda_star = -(v_max - v_min) / 4          # [batch, 1]

            # v + mu* * 1
            v_shifted = v + mu_star  # [batch, 3]

            # Indicator: |v + mu*| >= 2|lambda*| (relaxed from exact equality)
            tol = 1e-6
            indicator = (torch.abs(v_shifted) >= 2 * torch.abs(lambda_star) - tol).float()

            # sign(v + mu*)
            sign_v = torch.sign(v_shifted)

            # u* = midpoint - scale * sign(v + mu*) * indicator
            u_star = midpoint - scale * sign_v * indicator  # [batch, 3]

            # Return scalar correction: v^T u* per sample
            correction = torch.sum(v * u_star, dim=1)  # [batch]
            return correction

        elif robust_type == "p1":
            # Theorem 3.5(a): N=1, p=1 (q=inf), u1=0 (ball uncertainty set)
            # u* concentrates all perturbation on the coordinate with max |v + mu*|
            v_max = v.max(dim=1, keepdim=True).values
            v_min = v.min(dim=1, keepdim=True).values
            mu_star = -(v_max + v_min) / 2

            v_shifted = v + mu_star  # [batch, 3]

            # For p=1 (q=inf): u* = beta * sign(v_{j*}) * e_{j*}
            # where j* = argmax_j |v_j + mu*|
            abs_v = torch.abs(v_shifted)
            max_idx = torch.argmax(abs_v, dim=1, keepdim=True)  # [batch, 1]
            indicator = torch.zeros_like(v).scatter_(1, max_idx, 1.0)
            sign_v = torch.sign(v_shifted)

            u_star = beta * sign_v * indicator  # [batch, 3]

            correction = torch.sum(v * u_star, dim=1)  # [batch]
            return correction

        else:
            raise ValueError(f"Invalid robust type: {robust_type}. Use 'p1N2' or 'p1'.")
    
    def update(self, states, actions, rewards, next_states, dones, old_log_probs=None):
        """Update policy using PPO algorithm"""
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device) 
        if old_log_probs is None:
            with torch.no_grad():
                old_action_mean, _ = self.actor_critic(states)
                old_log_probs = self._build_policy(old_action_mean).log_prob(actions)
        else:
            old_log_probs = torch.FloatTensor(old_log_probs).to(self.device).view(-1)
        
        robust_bonus = None
        if self.robust_params is not None:
            # Calculate u^*
            u_star = self.calculate_u_star(states, actions, next_states, self.robust_params) 
            robust_bonus = (u_star.reshape(-1) * self.gamma).detach()
            # save u_star to pickle
            with open(f'u_star_{self.robust_params["robust_type"]}.pkl', 'wb') as f:
                pickle.dump(u_star, f) 

        # Calculate returns and advantages
        with torch.no_grad():
            _, values = self.actor_critic(states)
            _, next_values = self.actor_critic(next_states)
            
            rewards_for_returns = rewards
            if robust_bonus is not None:
                rewards_for_returns = rewards + robust_bonus

            # Calculate returns
            returns = []
            running_return = next_values[-1].squeeze() * (1 - dones[-1])
            
            for reward, done in zip(reversed(rewards_for_returns), reversed(dones)):
                running_return = reward + self.gamma * running_return * (1 - done) 
                returns.insert(0, running_return)
            
            returns = torch.stack(returns).view(-1)
            advantages = returns - values.view(-1)
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy for K epochs
        total_loss = 0
        num_batches = len(states) // self.batch_size + (1 if len(states) % self.batch_size != 0 else 0)
        
        for _ in range(self.epochs):
            # Generate random indices for batches
            indices = torch.randperm(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                # Get batch indices
                batch_indices = indices[start_idx:start_idx + min(self.batch_size, len(states) - start_idx)]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Get current policy and values
                action_mean, values = self.actor_critic(batch_states)
                policy = self._build_policy(action_mean)
                log_probs = policy.log_prob(batch_actions)
                
                # Calculate PPO likelihood ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                
                # Calculate actor and critic losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values.view(-1), batch_returns)
                
                # Calculate total loss
                loss = actor_loss + 0.5 * critic_loss
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        # Update learning rate based on performance
        self.scheduler.step(-total_loss / (self.epochs * num_batches))
        
        return total_loss / (self.epochs * num_batches)
    
    def save(self, path):
        """Save model"""
        torch.save(self.actor_critic.state_dict(), path)
    
    def load(self, path):
        """Load model"""
        state_dict = torch.load(path, weights_only=True)
        self.actor_critic.load_state_dict(state_dict, strict=False) 
