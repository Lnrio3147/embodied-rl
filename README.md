# embodied-rl

MuJoCo Shadow Hand 单指触碰强化学习实验。

## 结构

```
embodied_rl_hand/
├── algorithms/          # 算法实现
│   ├── sac.py           # Soft Actor-Critic
│   ├── bc.py            # Behavior Cloning
│   ├── diffusion_policy.py  # Diffusion Policy Demo
│   ├── replay_buffer.py # 经验回放池
│   └── collect_expert.py    # 专家数据采集
├── envs/                # 环境封装
│   └── hand_env.py      # MuJoCo Shadow Hand Gymnasium wrapper
└── utils/               # 工具脚本
    ├── inspect.py       # 模型结构检查
    ├── tips.py          # 关节/末端查找
    └── visualize.py     # 策略可视化
```

## 依赖

```bash
pip install torch numpy gymnasium mujoco tensorboard
```

## 运行

```bash
# 1. SAC 训练（根目录入口）
python train_sac.py

# 2. 收集专家数据
cd embodied_rl_hand/algorithms && python collect_expert.py

# 3. BC 训练
cd embodied_rl_hand/algorithms && python bc.py

# 4. Diffusion Policy 演示
cd embodied_rl_hand/algorithms && python diffusion_policy.py
```
