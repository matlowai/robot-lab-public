"""Fingertip geometry in the gripper-link frame: static finger vs moving jaw, jaw open and closed."""
import sys
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
sys.path.insert(0, "/mnt/work/AI/robot-lab")
import gymnasium as gym, numpy as np, torch  # noqa: E401,E402
from pxr import Usd, UsdGeom  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import robot_lab.tasks  # noqa: F401,E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
T = "RobotLab-SO101-MugBowl-IK-Abs-v0"
env = gym.make(T, cfg=parse_env_cfg(T, device="cuda:0", num_envs=1)).unwrapped
env.reset()
stage = env.sim.stage
r = env.scene["robot"]
g, j = r.find_bodies("gripper")[0][0], r.find_bodies("moving_jaw_so101_v1")[0][0]


def mesh_points_local(body_path):
    """Mesh points (non-collision) in the body prim's own frame: static geometry, safe to read from USD."""
    body = stage.GetPrimAtPath(body_path)
    cache = UsdGeom.XformCache()
    pts = []
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh) and "collision" not in prim.GetPath().pathString.lower():
            p = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if p:
                rel, _ = cache.ComputeRelativeTransform(prim, body)
                pts.append((np.c_[np.array(p), np.ones(len(p))] @ np.array(rel))[:, :3])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


from isaaclab.utils.math import quat_apply  # noqa: E402

LOCAL = {"static": mesh_points_local("/World/envs/env_0/Robot/gripper"),
         "jaw": mesh_points_local("/World/envs/env_0/Robot/moving_jaw_so101_v1")}
BODY = {"static": g, "jaw": j}


def in_gripper_frame(name):
    """Body-local points -> world via the *physics* body pose -> gripper-link frame via the physics gripper pose."""
    bp = r.data.body_pose_w[0, BODY[name]]
    pl = torch.tensor(LOCAL[name], device=env.device, dtype=torch.float32)
    pw = quat_apply(bp[3:7][None].expand(len(pl), 4), pl) + bp[:3]
    gp = r.data.body_pose_w[0, g]
    return quat_apply_inverse(gp[3:7][None].expand(len(pw), 4), pw - gp[:3]).cpu().numpy()


hold = None
for label, c in (("OPEN", 0.0), ("CLOSED", 1.0)):
    root, ee = r.data.root_pose_w, r.data.body_pose_w[:, g]
    from isaaclab.utils.math import subtract_frame_transforms
    pos, quat = subtract_frame_transforms(root[:, :3], root[:, 3:7], ee[:, :3], ee[:, 3:7])
    act = torch.cat([pos, quat, torch.tensor([[c]], device=env.device)], -1)
    for _ in range(40):
        env.step(act)
    for name in ("static", "jaw"):
        if len(LOCAL[name]) == 0:
            print(f"TIP {label} {name}: no mesh points"); continue
        pl = in_gripper_frame(name)
        tip = pl[pl[:, 2] < pl[:, 2].min() + 0.015]  # lowest 1.5 cm along local z = fingertip region
        print(f"TIP {label} {name}: tip centroid (gripper frame) = {tip.mean(0).round(4)}  "
              f"x-range=({tip[:,0].min():.4f},{tip[:,0].max():.4f}) y-range=({tip[:,1].min():.4f},{tip[:,1].max():.4f}) "
              f"z-min={pl[:,2].min():.4f}  jaw_joint={float(r.data.joint_pos[0, -1]):.3f}")
env.close(); app.close()
