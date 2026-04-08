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
    
    def calculate_u_star(self, states, actions, next_states, robust_params):
        robust_type = robust_params["robust_type"]
        beta = robust_params["beta"]
        p2_coef = robust_params["p2_coef"]
        epsilon = robust_params["epsilon"]

        u_dim = robust_params["u_dim"]
        p = torch.zeros(u_dim)
        p -= epsilon/u_dim
        
        if robust_type == "p1N2":
            _, values = self.actor_critic(states) 
            max_value_indices = torch.argmax(values, dim=0)
            min_value_indices = torch.argmin(values, dim=0) 
            v_max = values[max_value_indices]
            v_min = values[min_value_indices] 
            
            p2 = torch.ones(actions.shape) * p2_coef  
            p2 = torch.abs(p2 - beta / self.state_dim) / 2  
            p2 = p2.to(self.device)
            actions = actions.to(self.device)
            return p2 * actions * self.robust_params["beta"]
            
        elif robust_type == "p1N1":
            _, values = self.actor_critic(states) 
            max_value_indices = torch.argmax(values, dim=0)
            min_value_indices = torch.argmin(values, dim=0) 
            v_max = values[max_value_indices]
            v_min = values[min_value_indices] 
            mu_start = -(v_min + v_max) / 2
            lambda_start = (v_max - v_min) / 4
            
            raise NotImplementedError("p1N1 not implemented for multi-ticker")
            
        elif robust_type == "p1":
            _, values = self.actor_critic(states) 
            max_value_indices = torch.argmax(values, dim=0)
            min_value_indices = torch.argmin(values, dim=0) 
            v_max = values[max_value_indices]
            v_min = values[min_value_indices] 
            
            actions = actions.to(self.device)
            return actions * self.robust_params["beta"]
            
        else:
            raise Exception("Invalid robust type")
    
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device) 
        
        if self.robust_params is not None:
            u_star = self.calculate_u_star(states, actions, next_states, self.robust_params) 
            try: 
                np.save(f'u_star_{self.robust_params["robust_type"]}.pkl', u_star.cpu().numpy()) 
            except: 
                pass 

        with torch.no_grad():
            _, values = self.actor_critic(states)
            _, next_values = self.actor_critic(next_states)
            
            returns = []
            running_return = next_values[-1] * (1 - dones[-1])
            
            for reward, done in zip(reversed(rewards), reversed(dones)):
                running_return = reward + self.gamma * running_return * (1 - done) 
                returns.insert(0, running_return)
            
            advantages = []
            running_advantage = 0
            for ret, value in zip(reversed(returns), reversed(values)):
                running_advantage = ret - value
                advantages.insert(0, running_advantage)
            
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device) 
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if self.robust_params is not None:
            advantages = advantages + (u_star * self.gamma).mean(dim=1)  

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