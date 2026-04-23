"""
D1 灵巧手 Gymnasium 环境封装（最简可运行版）
修复项:
  1. truncated 硬截断（防止无限循环卡住训练）
  2. 目标绑定指尖前方 3cm（确保 SAC 一定能学会，先出 demo 曲线）
  3. body + 局部偏移估算指尖（模型无 fingertip site）
运行: python d1_hand_gym_env.py
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os


class DexterousHandEnv(gym.Env):
    """
    Shadow Hand 单指接近任务（目标绑定版，先训通 SAC 出曲线）。
    动作空间: 20 维连续
    观察空间: qpos(24) + qvel(24) + tip_xyz(3) + target_xyz(3) + rel_xyz(3) = 57 维
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, max_episode_steps=500):
        super().__init__()

        model_path = "./mujoco_menagerie/shadow_hand/right_hand.xml"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型: {model_path}，请先 git clone mujoco_menagerie")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.action_dim = self.model.nu
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )

        obs_dim = self.model.nq + self.model.nv + 3 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        self.ff_tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rh_ffdistal")
        if self.ff_tip_id < 0:
            raise RuntimeError("找不到 body 'rh_ffdistal'")

        self.tip_local_offset = np.array([0.028, 0.0, 0.0], dtype=np.float32)

        self.episode_step = 0

    def _get_tip_pos(self):
        body_xpos = self.data.xpos[self.ff_tip_id]
        body_xmat = self.data.xmat[self.ff_tip_id].reshape(3, 3)
        tip_pos = body_xpos + body_xmat @ self.tip_local_offset
        return tip_pos.copy()

    def _get_obs(self):
        tip_pos = self._get_tip_pos()
        rel_pos = self.target_pos - tip_pos
        obs = np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
            tip_pos,
            self.target_pos.copy(),
            rel_pos,
        ])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.episode_step = 0

        low = self.model.jnt_range[:, 0]
        high = self.model.jnt_range[:, 1]
        self.data.qpos[:] = self.np_random.uniform(low, high)
        mujoco.mj_forward(self.model, self.data)

        tip = self._get_tip_pos()
        self.target_pos = tip + np.array([0.03, 0.0, 0.0], dtype=np.float32)

        print(f"[RESET] tip={tip}, target={self.target_pos}, dist={np.linalg.norm(tip - self.target_pos):.3f}")

        obs = self._get_obs()
        info = {"target": self.target_pos.copy()}
        return obs, info

    def step(self, action):
        self.episode_step += 1

        ctrl_range = self.model.actuator_ctrlrange
        action = np.clip(action, -1.0, 1.0)
        ctrl = (action + 1.0) / 2.0 * (ctrl_range[:, 1] - ctrl_range[:, 0]) + ctrl_range[:, 0]

        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

        tip_pos = self._get_tip_pos()
        dist = float(np.linalg.norm(tip_pos - self.target_pos))

        reward = -dist
        terminated = bool(dist < 0.02)
        truncated = self.episode_step >= self.max_episode_steps  # 关键修复：硬截断

        if terminated:
            reward += 10.0

        info = {"distance": dist, "tip_pos": tip_pos}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass


if __name__ == "__main__":
    print("初始化环境...")
    env = DexterousHandEnv()
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")

    obs, info = env.reset(seed=42)
    print(f"初始观察维度: {obs.shape}")
    print(f"目标位置: {info['target']}")

    print("\n运行 600 步随机动作测试...")
    for step in range(600):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 100 == 0:
            print(f"Step {step:3d}: reward={reward:+.3f}, dist={info['distance']:.3f}")

        if terminated or truncated:
            print(f"  -> Episode done at {step} (terminated={terminated}, truncated={truncated})")
            obs, info = env.reset()

    print("\n环境测试通过。下一步: D2 接入 SAC 训练。")