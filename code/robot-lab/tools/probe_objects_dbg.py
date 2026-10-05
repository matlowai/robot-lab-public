import sys
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
sys.path.insert(0, "/mnt/work/AI/robot-lab")
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import robot_lab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import robot_lab.tasks.so101_pick_place as pp  # noqa: E402
pp.PickPlaceObservationsCfg.PolicyCfg.object_pose = None
pp.PickPlaceObservationsCfg.PolicyCfg.bowl_pose = None
T = "RobotLab-SO101-PickPlace-IK-Abs-v0"
cfg = parse_env_cfg(T, device="cuda:0", num_envs=4)
cfg.observations.policy.object_pose = None
cfg.observations.policy.bowl_pose = None
cfg.terminations.success = None
env = gym.make(T, cfg=cfg).unwrapped
o = env.scene["object"].data
print("DBG root_pos_w type", type(o.root_pos_w), "shape", tuple(o.root_pos_w.shape))
print("DBG num_instances", env.scene["object"].num_instances, "env_origins", tuple(env.scene.env_origins.shape))
from pxr import Usd
st = env.sim.stage
for i in range(4):
    kids = [c.GetName() for c in st.GetPrimAtPath(f"/World/envs/env_{i}").GetChildren()]
    print(f"DBG env_{i} children:", kids)
env.close(); app.close()
