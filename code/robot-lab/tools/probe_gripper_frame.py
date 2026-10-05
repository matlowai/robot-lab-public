"""Print the SO-101 gripper link frame (world) and the jaw offset, to design the wrist-camera mount."""
import sys
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
sys.path.insert(0, "/mnt/work/AI/robot-lab")
import gymnasium as gym, numpy as np, torch  # noqa: E401,E402
import isaaclab_tasks  # noqa: F401,E402
import robot_lab.tasks  # noqa: F401,E402
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
T = "RobotLab-SO101-MugBowl-IK-Abs-v0"
env = gym.make(T, cfg=parse_env_cfg(T, device="cuda:0", num_envs=1)).unwrapped
env.reset()
r = env.scene["robot"]
print("PROBE bodies:", r.body_names)
g = r.find_bodies("gripper")[0][0]; j = r.find_bodies("moving_jaw_so101_v1")[0][0]
gp, gq = r.data.body_pose_w[0, g, :3], r.data.body_pose_w[0, g, 3:7]
jp = r.data.body_pose_w[0, j, :3]
R = matrix_from_quat(gq[None])[0].cpu().numpy()
print("PROBE gripper pos_w", np.round(gp.cpu().numpy(), 4), "quat", np.round(gq.cpu().numpy(), 4))
print("PROBE gripper local axes in world (columns x,y,z):\n", np.round(R, 3))
print("PROBE jaw origin in gripper-local frame:", np.round(quat_apply_inverse(gq[None], (jp - gp)[None])[0].cpu().numpy(), 4))
# fingertip estimate: farthest mesh point of the static gripper body along each local axis is not
# available here; report the jaw offset and let the render loop confirm.
env.close(); app.close()
