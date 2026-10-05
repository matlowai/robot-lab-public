"""Where is the SO-101's real camera_mount (in the gripper-link frame)? Static geometry, read relative to the body."""
import sys
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
sys.path.insert(0, "/mnt/work/AI/robot-lab")
import gymnasium as gym, numpy as np  # noqa: E401,E402
from pxr import Usd, UsdGeom  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import robot_lab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
T = "RobotLab-SO101-MugBowl-IK-Abs-v0"
env = gym.make(T, cfg=parse_env_cfg(T, device="cuda:0", num_envs=1)).unwrapped
env.reset()
stage = env.sim.stage
body = stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper")
cache = UsdGeom.XformCache()
for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
    path = prim.GetPath().pathString
    if "camera" in path.lower() and prim.IsA(UsdGeom.Mesh):
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        rel, _ = cache.ComputeRelativeTransform(prim, body)
        P = (np.c_[np.array(pts), np.ones(len(pts))] @ np.array(rel))[:, :3]
        lo, hi = P.min(0), P.max(0)
        # principal axes of the mount plate
        c = P.mean(0); u, s, vt = np.linalg.svd(P - c, full_matrices=False)
        print(f"MOUNT {path}\n  bbox min={lo.round(4)} max={hi.round(4)} center={c.round(4)}\n  "
              f"extent={(hi-lo).round(4)}  thinnest-axis(normal)={vt[2].round(3)}")
env.close(); app.close()
