"""Gate 2a: spawn the pick-place task (train + held-out), print per-env object name, measured center/extents, settle pose."""
import sys
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
sys.path.insert(0, "/mnt/work/AI/robot-lab")
import gymnasium as gym, numpy as np, torch  # noqa: E401,E402
import imageio.v3 as iio  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import robot_lab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robot_lab.tasks.so101_pick_place import (object_geometry, object_names, object_center_w, set_object,  # noqa: E402
    TRAIN_OBJECTS, HELDOUT_OBJECTS)
from robot_lab.tasks.so101_mug_bowl import SCENE_CAM_EYE, SCENE_CAM_TARGET  # noqa: E402
T = "RobotLab-SO101-PickPlace-IK-Abs-v0"
rows = []
for oname, _, _ in TRAIN_OBJECTS + HELDOUT_OBJECTS:
    cfg = set_object(parse_env_cfg(T, device="cuda:0", num_envs=2), oname)
    env = gym.make(T, cfg=cfg).unwrapped
    env.reset()
    o = env.scene.env_origins
    env.scene["scene_cam"].set_world_poses_from_view(torch.tensor(SCENE_CAM_EYE, device=env.device) + o,
                                                     torch.tensor(SCENE_CAM_TARGET, device=env.device) + o)
    zero_hold = None
    for _ in range(30):
        env.sim.step(render=True)
    env.scene.update(env.physics_dt)
    c, e = object_geometry(env)
    cw = object_center_w(env) - o
    names = object_names(env)
    for i in range(env.num_envs):
        print(f"OBJ env{i} {names[i]:15s} center_local={c[i].cpu().numpy().round(4)} "
              f"extents={e[i].cpu().numpy().round(4)} center_w_z={cw[i,2].item():.3f}")
    imgs = env.scene["scene_cam"].data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8)
    rows.append(imgs[0])
    env.close()
iio.imwrite('/mnt/work/AI/robot-lab/out/check/objects.png', np.concatenate(rows, 1))
app.close()
