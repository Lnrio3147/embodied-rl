"""
D1 灵巧手 Gymnasium 环境封装
运行: python d1_hand_gym_env.py
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import os


class DexterousHandEnv(gym.Env):
    """
    Shadow Hand 单指（食指）接近目标点任务。
    动作空间: 20 维连续 (对应 model.nu，位置控制)
    观察空间: qpos(24) + qvel(24) + ff_tip_xyz(3) = 51 维
    """
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # 模型路径
        model_path = "./mujoco_menagerie/shadow_hand/right_hand.xml"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型: {model_path}，请先 git clone mujoco_menagerie")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 动作空间: [-1, 1]^20，映射到 actuator ctrl_range
        self.action_dim = self.model.nu
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.action_dim,), 
            dtype=np.float32
        )
        
        # 观察空间
        obs_dim = self.model.nq + self.model.nv + 3  # qpos + qvel + ff_tip
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.target_pos = np.array([0.05, -0.02, 0.10])  # 手掌前方、略偏左、高处
        
        # 缓存 site id 加速
        self.ff_tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rh_ffdistal")

    def _get_obs(self):
        """获取观察"""
        tip_pos = self.data.xpos[self.ff_tip_id].copy()
        obs = np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat,
            tip_pos
        ])
        return obs.astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # 随机初始化关节角（在限位内）
        low = self.model.jnt_range[:, 0]
        high = self.model.jnt_range[:, 1]
        self.data.qpos[:] = self.np_random.uniform(low, high)
        mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_obs()
        info = {"target": self.target_pos.copy()}
        return obs, info
    
    def step(self, action):
        # 1. 动作缩放: [-1,1] -> [ctrl_low, ctrl_high]
        ctrl_range = self.model.actuator_ctrlrange
        action = np.clip(action, -1.0, 1.0)
        ctrl = (action + 1.0) / 2.0 * (ctrl_range[:, 1] - ctrl_range[:, 0]) + ctrl_range[:, 0]
        
        # 2. 执行一步仿真
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        
        # 3. 计算奖励
        tip_pos = self.data.xpos[self.ff_tip_id].copy()
        dist = np.linalg.norm(tip_pos - self.target_pos)
        reward = -dist  # 距离越近奖励越高（负得越少）
        
        # 4. 终止与截断
        terminated = bool(dist < 0.02)  # 到达目标
        truncated = False  # 可由 max_episode_steps 控制
        
        if terminated:
            reward += 10.0  # 到达奖励
        
        info = {"distance": dist, "tip_pos": tip_pos}
        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        pass  # D1 暂不实现可视化，专注接口正确性


# ==================== 测试脚本 ====================
if __name__ == "__main__":
    print("初始化环境...")
    env = DexterousHandEnv()
    
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    obs, info = env.reset(seed=42)
    print(f"\n初始观察维度: {obs.shape}")
    print(f"目标位置: {info['target']}")
    
    print("\n运行 200 步随机动作测试...")
    for step in range(200):
        action = env.action_space.sample()  # 随机动作
        #action = np.zeros(env.action_dim, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 50 == 0:
            print(f"Step {step:3d}: reward={reward:.3f}, dist={info['distance']:.3f}")
        
        if terminated:
            print(f"✅ 到达目标! Step {step}")
            obs, info = env.reset()
    
    print("\n环境测试通过。下一步: D2 接入 SAC 训练。")