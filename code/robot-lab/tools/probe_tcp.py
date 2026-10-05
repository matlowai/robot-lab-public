"""Gripper + jaw mesh extents in the gripper-link frame -> fingertip (TCP) offset for the scripted expert."""
import sys
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
sys.path.insert(0, "/mnt/work/AI/robot-lab")
import gymnasium as gym, numpy as np  # noqa: E401,E402
from pxr import Usd, UsdGeom, Gf  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import robot_lab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
T = "RobotLab-SO101-MugBowl-IK-Abs-v0"
env = gym.make(T, cfg=parse_env_cfg(T, device="cuda:0", num_envs=1)).unwrapped
env.reset()
stage = env.sim.stage
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
g = stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper")
for name in ("gripper", "moving_jaw_so101_v1"):
    p = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}")
    kids = [c.GetPath().pathString for c in Usd.PrimRange(p)][:6]
    b = cache.ComputeRelativeBound(p, g).ComputeAlignedRange()
    print(f"TCP {name}: bbox in gripper frame min={np.round(np.array(b.GetMin()),4)} max={np.round(np.array(b.GetMax()),4)}")
    print(f"TCP   prims: {kids}")
env.close(); app.close()
