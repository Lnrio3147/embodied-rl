"""
D1 Shadow Hand 模型结构解析
运行: python3 d1_inspect_hand.py
"""
import mujoco
import numpy as np

MODEL_PATH = "./mujoco_menagerie/shadow_hand/right_hand.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH);
data = mujoco.MjData(model);

print("=" * 50)
print("Shadow Hand 结构报告")
print("=" * 50)

print(f"\n[自由度] nq={model.nq}, nv={model.nv}, nu={model.nu}")

print(f"\n[关节列表] 共 {model.njnt} 个关节:")
for i in range(model.njnt):
    jnt = model.joint(i)
    print(f"  {i:2d}: {jnt.name:15s} | range=[{jnt.range[0]:.2f}, {jnt.range[1]:.2f}]")

print(f"\n[执行器列表] 共 {model.nu} 个:")
for i in range(model.nu):
    act = model.actuator(i)
    ctrl_range = model.actuator_ctrlrange[i]
    print(f"  {i:2d}: {act.name:20s} | ctrl_range=[{ctrl_range[0]:.3f}, {ctrl_range[1]:.3f}]")

fingertips_body = ["rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal", "rh_thdistal"]
print(f"\n[指尖 Body]:")
for name in fingertips_body:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id >= 0:
        print(f"  {name}: body_id={body_id}")
    else:
        print(f"  {name}: NOT FOUND!")

# 零位姿 FK 验证
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
print(f"\n[零位姿 FK 验证]:")
for name in fingertips_body:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id >= 0:
        pos = data.xpos[body_id].copy()
        print(f"  {name}: xyz=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    else:
        print(f"  {name}: NOT FOUND!")

print("\n" + "=" * 50)
print("解析完成。确认 5 个指尖 body_id 都 >= 0。")
print("=" * 50)