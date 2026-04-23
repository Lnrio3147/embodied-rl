"""
D2 训练脚本：SAC on DexterousHandEnv
运行: python train_sac.py
"""
import os
os.makedirs('./checkpoints', exist_ok=True)

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from d1_hand_gym_env import DexterousHandEnv
from sac import SACAgent
from replay_buffer import ReplayBuffer


def evaluate(agent, env, episodes=5, max_eval_steps=1000):
    """评估当前策略，返回平均奖励和成功率"""
    total_reward = 0
    success = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        step_count = 0
        while not done and step_count < max_eval_steps:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step_count += 1
            if terminated:
                success += 1
                break
        total_reward += ep_reward
    return total_reward / episodes, success / episodes


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    env = DexterousHandEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim, action_dim, device=device)
    buffer = ReplayBuffer(state_dim, action_dim, max_size=100000)

    writer = SummaryWriter(log_dir='./runs/sac_hand_d2')

    max_steps = 50000
    start_steps = 1000
    batch_size = 256
    update_interval = 1
    eval_interval = 1000

    obs, _ = env.reset()
    episode_reward = 0
    episode_step = 0

    pbar = tqdm(range(max_steps), desc="Training")
    for step in pbar:
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.add(obs, action, reward, next_obs, float(done))

        episode_reward += reward
        episode_step += 1
        obs = next_obs

        metrics = None
        if step >= start_steps and step % update_interval == 0:
            batch = buffer.sample(batch_size)
            metrics = agent.update(batch)

            if step % 100 == 0:
                writer.add_scalar('Loss/critic', metrics['critic_loss'], step)
                writer.add_scalar('Loss/actor', metrics['actor_loss'], step)
                writer.add_scalar('Loss/alpha', metrics['alpha_loss'], step)
                writer.add_scalar('Alpha/value', metrics['alpha'], step)
                writer.add_scalar('Q/mean', metrics['q'], step)

        if done:
            writer.add_scalar('Episode/reward', episode_reward, step)
            writer.add_scalar('Episode/length', episode_step, step)
            writer.add_scalar('Episode/distance', info['distance'], step)

            alpha_str = f'{metrics["alpha"]:.3f}' if (metrics is not None) else 'rand'
            pbar.set_postfix({
                'reward': f'{episode_reward:.2f}',
                'dist': f'{info["distance"]:.3f}',
                'alpha': alpha_str
            })

            obs, _ = env.reset()
            episode_reward = 0
            episode_step = 0

        if step > 0 and step % eval_interval == 0:
            eval_reward, success_rate = evaluate(agent, env, episodes=5)
            writer.add_scalar('Eval/reward', eval_reward, step)
            writer.add_scalar('Eval/success_rate', success_rate, step)
            print(f"\n[Step {step}] Eval Reward: {eval_reward:.2f}, Success: {success_rate:.1%}")

            if success_rate >= 0.5:
                agent.save(f'./checkpoints/sac_hand_step{step}.pth')

    agent.save('./checkpoints/sac_hand_final.pth')
    print("Training complete. Run `tensorboard --logdir=./runs` to view curves.")


if __name__ == '__main__':
    main()