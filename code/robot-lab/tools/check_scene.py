"""Scene-validation gate for RobotLab-SO101-MugBowl: spawn, reset, hold, render, measure.

Writes out/check/<step>_{scene,wrist}.png and prints object sizes (live stage), EE pose, joint state.
Run from the IsaacLab checkout:
  OMNI_KIT_ACCEPT_EULA=YES uv run --extra teleop python /mnt/work/AI/robot-lab/tools/check_scene.py
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=30)
args = parser.parse_args()
app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app

sys.path.insert(0, "/mnt/work/AI/robot-lab")
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import robot_lab.tasks  # noqa: E402,F401
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robot_lab.tasks.so101_mug_bowl import SCENE_CAM_EYE, SCENE_CAM_TARGET  # noqa: E402

TASK = "RobotLab-SO101-MugBowl-IK-Abs-v0"
OUT = Path("/mnt/work/AI/robot-lab/out/check")
OUT.mkdir(parents=True, exist_ok=True)

cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args.num_envs)
# Candidate wrist mounts on the gripper link: camera looks along gripper-local -Z (fingertips),
# image-up = gripper-local +X (jaw side). ROS/world conventions differ; use "world" (fwd +X, up +Z).
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
_Q_LOOK_NEG_Z = (0.0, 0.70710678, 0.0, 0.70710678)  # xyzw, R_y(+90deg): cam +X -> gripper -Z
WRIST_CANDIDATES = {
    "wa": (-0.035, 0.0, 0.00),
    "wb": (-0.035, 0.0, 0.03),
    "wc": (-0.050, 0.0, 0.05),
    "wd": (-0.020, -0.03, 0.03),
}
for _n, _pos in WRIST_CANDIDATES.items():
    setattr(cfg.scene, f"cam_{_n}", CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/gripper/cam_{_n}", update_period=0.0, height=256, width=256,
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955,
                                                             clipping_range=(0.005, 2.0)),
        offset=CameraCfg.OffsetCfg(pos=_pos, rot=_Q_LOOK_NEG_Z, convention="world")))
env = gym.make(TASK, cfg=cfg).unwrapped
obs, _ = env.reset()
print(f"CHECK step_dt={env.step_dt:.4f}s ({1 / env.step_dt:.1f} Hz)  action_dim={env.action_manager.total_action_dim}")

scene_cam = env.scene["scene_cam"]
origins = env.scene.env_origins
eye = torch.tensor(SCENE_CAM_EYE, device=env.device) + origins
tgt = torch.tensor(SCENE_CAM_TARGET, device=env.device) + origins
scene_cam.set_world_poses_from_view(eye, tgt)

robot = env.scene["robot"]
ee_idx = robot.find_bodies("gripper")[0][0]


def hold_action():
    """IK-Abs action that holds the current gripper pose (base frame) with the jaw open."""
    root = robot.data.root_pose_w
    ee = robot.data.body_pose_w[:, ee_idx]
    pos, quat = subtract_frame_transforms(root[:, :3], root[:, 3:7], ee[:, :3], ee[:, 3:7])
    return torch.cat([pos, quat, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1)


def save(tag):
    o = env.observation_manager.compute()
    for cam in ("scene", "wrist"):
        img = o["rgb_camera"][cam][0].detach().cpu().numpy()
        img = img[..., :3].astype(np.uint8) if img.dtype != np.uint8 else img[..., :3]
        iio.imwrite(OUT / f"{tag}_{cam}.png", img)


act = hold_action()
for i in range(args.steps):
    obs, rew, term, trunc, info = env.step(act)
save("hold")
for _n in WRIST_CANDIDATES:
    _img = env.scene[f"cam_{_n}"].data.output["rgb"][0][..., :3].detach().cpu().numpy().astype(np.uint8)
    iio.imwrite(OUT / f"hold_{_n}.png", _img)

# live-stage sizes
stage = env.sim.stage
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
for name in ("Mug", "Bowl"):
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath(f"/World/envs/env_0/{name}")).ComputeAlignedRange()
    print(f"CHECK {name} world size_m = {np.round(np.array(rng.GetMax() - rng.GetMin()), 4)}  "
          f"min_z={rng.GetMin()[2]:.4f}")
p = obs["policy"]
print("CHECK joint_pos", np.round(p["joint_pos"][0].cpu().numpy(), 3))
print("CHECK eef_pos (env frame)", np.round(p["eef_pos"][0].cpu().numpy(), 3))
print("CHECK mug_pose", np.round(p["mug_pose"][0].cpu().numpy(), 3), " bowl_pose",
      np.round(p["bowl_pose"][0].cpu().numpy(), 3))
print("CHECK hold action (base frame)", np.round(act[0].cpu().numpy(), 3))
print("CHECK terminated", term.tolist(), "success", env.termination_manager.get_term("success").tolist())
env.close()
app.close()
