import mujoco

MODEL_PATH = "./mujoco_menagerie/shadow_hand/right_hand.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)

print("=" * 60)
print("所有 Site 名称 (共 {} 个)".format(model.nsite))
print("=" * 60)
for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    print(f"  {i:3d}: {name}")

print("\n" + "=" * 60)
print("所有 Body 名称 (筛选含 tip/fingertip/FF/MF/RF/LF/TH 的)")
print("=" * 60)
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if any(k in name for k in ["tip", "fingertip", "FF", "MF", "RF", "LF", "TH", "distal"]):
        print(f"  {i:3d}: {name}")