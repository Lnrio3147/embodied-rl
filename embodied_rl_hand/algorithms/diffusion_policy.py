"""
D3 Step 3: 极简 Diffusion Policy 推理 Demo
运行: cd src && python diffusion_policy_mini.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from embodied_rl_hand.envs.hand_env import DexterousHandEnv


class MiniDiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128, T=10):
        super().__init__()
        self.T = T
        self.action_dim = action_dim
        self.time_embed_dim = 16
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_embed_dim), nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )

        input_dim = obs_dim + action_dim + self.time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def time_embedding(self, t):
        return self.time_mlp(t)

    def forward(self, obs, noisy_action, t):
        t_emb = self.time_embedding(t)
        x = torch.cat([obs, noisy_action, t_emb], dim=-1)
        return self.net(x)

    def sample(self, obs, device='cpu'):
        batch_size = obs.shape[0]
        action = torch.randn(batch_size, self.action_dim, device=device)

        betas = torch.linspace(0.01, 0.1, self.T, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        for i in reversed(range(self.T)):
            t = torch.full((batch_size, 1), i, device=device, dtype=torch.float32)

            with torch.no_grad():
                noise_pred = self.forward(obs, action, t)

            alpha_t = alphas[i]
            beta_t = betas[i]

            action = (action - beta_t / sqrt_one_minus_alphas_cumprod[i] * noise_pred) / torch.sqrt(alpha_t)

            if i > 0:
                noise = torch.randn_like(action)
                sigma_t = torch.sqrt(beta_t)
                action = action + sigma_t * noise

        action = torch.clamp(action, -1.0, 1.0)
        return action


def demo_diffusion_sampling():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = DexterousHandEnv()

    # 自动推断维度
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = MiniDiffusionPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128, T=10).to(device)
    policy.eval()

    print("=" * 50)
    print("Diffusion Policy 采样演示")
    print("=" * 50)

    for i in range(3):
        obs, _ = env.reset()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

        actions = []
        for _ in range(5):
            with torch.no_grad():
                a = policy.sample(obs_tensor, device=device)
            actions.append(a.cpu().numpy()[0])

        actions = np.array(actions)
        std_per_dim = actions.std(axis=0).mean()

        print(f"\n观察 {i+1}: 5 次采样动作的平均标准差 = {std_per_dim:.4f}")
        print(f"  动作 1: {actions[0][:5]}...")
        print(f"  动作 2: {actions[1][:5]}...")

    print("\n说明：标准差 > 0.1 说明有多模态特性；≈ 0 则退化为确定性策略")


def quick_overfit_demo():
    import pickle
    from torch.utils.data import TensorDataset, DataLoader

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    src_dir = Path(__file__).parent
    data_path = src_dir / "expert_data.pkl"

    if not data_path.exists():
        raise FileNotFoundError("未找到 expert_data.pkl，跳过过拟合演示")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    obs = torch.FloatTensor(data['obs'][:500])
    actions = torch.FloatTensor(data['actions'][:500])

    noise = torch.randn_like(actions)
    noisy_actions = actions + 0.1 * noise

    dataset = TensorDataset(obs, noisy_actions, noise)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    policy = MiniDiffusionPolicy(obs_dim=obs.shape[1], action_dim=actions.shape[1], hidden_dim=128, T=10).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    criterion = nn.MSELoss()

    print("\n快速过拟合 50 轮...")
    for epoch in range(50):
        total_loss = 0
        for o, na, n in loader:
            o, na, n = o.to(device), na.to(device), n.to(device)
            t = torch.zeros(len(o), 1, device=device)
            pred_noise = policy(o, na, t)
            loss = criterion(pred_noise, n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    test_obs = obs[:1].to(device)
    with torch.no_grad():
        recovered = policy.sample(test_obs, device=device)

    err = (recovered.cpu() - actions[:1]).abs().mean().item()
    print(f"\n过拟合后，从噪声恢复的动作与专家动作误差: {err:.4f}")


if __name__ == '__main__':
    demo_diffusion_sampling()

    try:
        quick_overfit_demo()
    except FileNotFoundError as e:
        print(f"\n跳过过拟合演示：{e}")