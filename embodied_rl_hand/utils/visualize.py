"""
可视化：观看任意策略在 MuJoCo 中的实际表现
运行: python visualize_policy.py --policy sac  # 或 bc / random
"""
import torch
import numpy as np
import argparse
import time
import mujoco
import mujoco.viewer

import sys
from pathlib import Path

from embodied_rl_hand.envs.hand_env import DexterousHandEnv
from sac import SACAgent
from bc import BCPolicy


def visualize(policy_type='sac', max_steps=200, slow_motion=False):
    env = DexterousHandEnv()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载策略
    if policy_type == 'sac':
        agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], device=device)
        agent.load('./checkpoints/sac_hand_final.pth')
        get_action = lambda obs: agent.select_action(obs, deterministic=True)
    
    elif policy_type == 'bc':
        policy = BCPolicy(51, 20).to(device)
        policy.load_state_dict(torch.load('./bc_policy.pth', map_location=device))
        policy.eval()
        get_action = lambda obs: policy(torch.FloatTensor(obs).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
    
    else:  # random
        get_action = lambda obs: env.action_space.sample()
    
    # 启动 MuJoCo Viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        obs, info = env.reset()
        
        for step in range(max_steps):
            if not viewer.is_running():
                break
            
            action = get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 在 3D 窗口中显示实时信息
            dist = info['distance']
            viewer.user_scn.ngeom = 0  # 清除旧标记
            # 添加目标点红色标记（在目标位置画小球）
            # 注：这里仅打印，MuJoCo viewer 原生不支持动态文字，但可观察关节运动
            
            print(f"\rStep {step:3d} | Reward: {reward:7.3f} | Dist: {dist:.4f} | Action mean: {action.mean():.2f}", end="")
            
            if terminated:
                print("\n✅ 到达目标！")
                time.sleep(0.5)
                obs, info = env.reset()
            
            # 控制速度（可选慢放观察）
            sleep_time = 0.05 if slow_motion else 0.02
            time.sleep(sleep_time)
    
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='sac', choices=['sac', 'bc', 'random'])
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--slow', action='store_true', help='慢放模式')
    args = parser.parse_args()
    
    visualize(args.policy, args.steps, args.slow)