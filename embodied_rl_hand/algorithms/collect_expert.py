"""
D3 Step 1: 用 D2 的 SAC 生成专家轨迹
运行: cd src && python collect_expert_data.py
"""
import torch
import numpy as np
from embodied_rl_hand.envs.hand_env import DexterousHandEnv
from embodied_rl_hand.algorithms.sac import SACAgent
import pickle
from pathlib import Path


def collect_expert_data(agent, env, num_trajectories=50, max_steps_per_traj=100, dist_threshold=0.05):
    all_obs, all_actions = [], []
    accepted = 0

    for ep in range(num_trajectories):
        obs, _ = env.reset()
        traj_obs, traj_actions = [], []
        done, step = False, 0

        while not done and step < max_steps_per_traj:
            action = agent.select_action(obs, deterministic=True)
            traj_obs.append(obs)
            traj_actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        final_dist = info.get('distance', float('inf'))
        if final_dist < dist_threshold:
            all_obs.extend(traj_obs)
            all_actions.extend(traj_actions)
            accepted += 1
            print(f"  Traj {ep+1}: ACCEPTED (final_dist={final_dist:.4f}, steps={step})")
        else:
            print(f"  Traj {ep+1}: rejected (final_dist={final_dist:.4f})")

    data = {
        'obs': np.array(all_obs, dtype=np.float32),
        'actions': np.array(all_actions, dtype=np.float32),
    }
    print(f"\n收集完成: {accepted}/{num_trajectories} 条轨迹被接受，总样本数 {len(all_obs)}")
    return data


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = DexterousHandEnv()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(state_dim, action_dim, device=device)

    # 修复：权重在 ../checkpoints/（src 的上一级）
    ckpt_path = Path(__file__).parent.parent / "checkpoints" / "sac_hand_final.pth"
    try:
        agent.load(str(ckpt_path))
        print(f"✅ 已加载 D2 SAC 权重: {ckpt_path}")
    except Exception as e:
        print(f"⚠️ 加载权重失败 ({e})")
        print("   将用随机策略收集（数据质量差，仅作演示）")

    data = collect_expert_data(agent, env, num_trajectories=50)

    save_path = Path(__file__).parent / "expert_data.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"专家数据已保存至 {save_path}")