import sys
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
sys.path.insert(0, "/mnt/work/AI/robot-lab")
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402


def A(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x.cpu())

import isaaclab_tasks  # noqa: F401,E402
import robot_lab.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
T = "RobotLab-SO101-MugBowl-IK-Abs-v0"
env = gym.make(T, cfg=parse_env_cfg(T, device="cuda:0", num_envs=1)).unwrapped
env.reset()
for n in ("mug", "bowl"):
    v = env.scene[n].root_physx_view
    print(f"PHYS {n} mass={A(v.get_masses()).flatten().round(4).tolist()} material(static,dyn,rest)={A(v.get_material_properties()).reshape(-1,3)[:3].round(3).tolist()}")
r = env.scene["robot"]
print("PHYS robot jaw material", A(r.root_physx_view.get_material_properties()).reshape(-1, 3)[-3:].round(3).tolist())
print("PHYS gripper effort limit", A(r.data.joint_effort_limits)[0][r.find_joints("gripper")[0]].tolist())
env.close(); app.close()
