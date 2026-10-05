"""Mug geometry relative to its rigid-body root frame (visual + collision), to locate the cup body center."""
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
root = stage.GetPrimAtPath("/World/envs/env_0/Mug")
cache = UsdGeom.XformCache()
for kind in ("visual", "collision"):
    pts = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        is_col = "collision" in prim.GetPath().pathString.lower() or prim.HasAPI(__import__("pxr").UsdPhysics.CollisionAPI)
        if prim.IsA(UsdGeom.Mesh) and (is_col == (kind == "collision")):
            p = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if p:
                rel, _ = cache.ComputeRelativeTransform(prim, root)
                pts.append((np.c_[np.array(p), np.ones(len(p))] @ np.array(rel))[:, :3])
    if not pts:
        print(f"MUG {kind}: none"); continue
    P = np.concatenate(pts)
    lo, hi = P.min(0), P.max(0)
    # cup body = points away from the handle: take the densest cluster via xy median
    med = np.median(P, 0)
    print(f"MUG {kind}: bbox_min={lo.round(4)} bbox_max={hi.round(4)} bbox_center={((lo+hi)/2).round(4)} median={med.round(4)}")
    # radial profile around median xy to find the body radius
    r = np.linalg.norm(P[:, :2] - med[:2], axis=1)
    print(f"MUG {kind}: radius percentiles 50/90/99 = {np.percentile(r,[50,90,99]).round(4)}")
print("MUG root scale", UsdGeom.Xformable(root).GetLocalTransformation())
env.close(); app.close()
