"""
D3 Step 2: Behavior Cloning 训练与评估
运行: cd src && python bc_train.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from embodied_rl_hand.envs.hand_env import DexterousHandEnv


class BCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh()
        )

    def forward(self, obs):
        return self.net(obs)


class ExpertDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.obs = torch.FloatTensor(data['obs'])
        self.actions = torch.FloatTensor(data['actions'])

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


def train_bc(data_path, save_path, epochs=100, batch_size=256, lr=3e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 自动推断维度
    with open(data_path, 'rb') as f:
        tmp = pickle.load(f)
    state_dim = tmp['obs'].shape[1]
    action_dim = tmp['actions'].shape[1]

    dataset = ExpertDataset(data_path)
    if len(dataset) == 0:
        raise ValueError("数据集为空，请先运行 collect_expert_data.py")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy = BCPolicy(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for obs, expert_act in loader:
            obs, expert_act = obs.to(device), expert_act.to(device)
            pred_act = policy(obs)
            loss = criterion(pred_act, expert_act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {total_loss/len(loader):.6f}")

    torch.save(policy.state_dict(), save_path)
    print(f"BC 策略已保存至 {save_path}")


def evaluate_bc(policy_path, n_episodes=20):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = DexterousHandEnv()

    # 关键修复：自动推断维度，不硬编码
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = BCPolicy(state_dim, action_dim).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

    success, total_reward = 0, 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, ep_reward = False, 0

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy(obs_tensor).cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        total_reward += ep_reward
        if info.get('distance', float('inf')) < 0.02:
            success += 1

    print(f"\nBC 评估: 成功率={success}/{n_episodes} ({success/n_episodes:.1%}), 平均奖励={total_reward/n_episodes:.2f}")


if __name__ == '__main__':
    src_dir = Path(__file__).parent
    data_path = src_dir / "expert_data.pkl"
    policy_path = src_dir / "bc_policy.pth"

    if not data_path.exists():
        raise FileNotFoundError(f"未找到 {data_path}，请先运行 collect_expert_data.py")

    print("===== 训练 BC =====")
    train_bc(str(data_path), str(policy_path))

    print("\n===== 评估 BC =====")
    evaluate_bc(str(policy_path))