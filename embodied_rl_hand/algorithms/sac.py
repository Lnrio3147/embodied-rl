import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mean)
            return action, None

        # 重参数化采样
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 带梯度
        action = torch.tanh(x_t)

        # 对数概率修正 (tanh squashing)
        log_prob = normal.log_prob(x_t).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class SACAgent:
    def __init__(self, state_dim, action_dim, device='cuda', lr=3e-4, gamma=0.99, tau=0.005, alpha_lr=3e-4):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        # Actor
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic (双Q)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)

        # 初始化目标网络
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        # 自动温度系数 alpha
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample(state, deterministic)
        return action.cpu().numpy()[0]

    def update(self, batch):
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)

        # ---------- Critic 更新 ----------
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # ---------- Actor 更新 ----------
        new_actions, log_probs = self.actor.sample(states)
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha.detach() * log_probs - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------- Alpha 更新 ----------
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---------- 软更新 Target ----------
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'q': current_q1.mean().item(),
        }

    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.log_alpha = checkpoint['log_alpha']