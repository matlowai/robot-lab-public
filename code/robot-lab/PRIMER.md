# Robot-lab primer: the mental model, bottom to top

- **Date:** 2026-09-23. Each concept below appears in the code:
  - `robot_lab/tasks/so101_mug_bowl.py`: the environment.
  - `tools/record_demos.py`: the scripted expert and recorder.
- **Companion docs:**
  - `/mnt/work/AI/flux3-action/FLUX3_ACTION_MAP.md`: the model.
  - `/mnt/work/AI/flux3-action/EMBODIED_STACK_MAP.md`: the ecosystem.
  - `/mnt/work/AI/flux3-action/CONTEMPLATING_OUR_NAVEL_ROBOT_EDITION.md`: the research and benchmark plan.
  - `course/COURSE_PLAN.md`, "Backlog from the flux3-action planning": what's still over there, code included.

## 1. The layers

```
 Policy / model     FLUX 3 Action SO-101 (+ LeRobot for training/baselines)   <- "the brain"
 Data format        LeRobot v3 dataset (frames + states + actions + timestamps)
 Robot learning     Isaac Lab 3.0: environments, tasks, rewards, recording, RL <- our code lives here
 Simulator          Isaac Sim 6.1: USD scene + PhysX physics + RTX rendering
 Scene language     OpenUSD
 Hardware/OS        2x RTX PRO 6000 (GPU1 drives the display), driver 610, Ubuntu 26.04
```

Each layer only talks to its neighbours. Most early bugs came from misunderstanding one boundary.

## 2. USD: the scene is a tree of prims

- **What's in the tree.** Every robot, object, camera and light is a prim with a transform. Assets are referenced in from files and are often **instanced**: one shared copy that is cheap but read-only.
- **Asset origins are arbitrary.** Our mug's root origin is **2.2 cm from the cup body's centre** (`MUG_BODY_CENTER_LOCAL`). That one fact caused most of the missed grasps.
- **Always measure in the live sim.** `metersPerUnit`, scale and instancing change what "size" means. Tools: `tools/probe_mug_geom.py`, `probe_fingertips.py`.

## 3. Physics state is not USD state

- During simulation, physics lives in GPU buffers (Fabric), so **USD transforms go stale**.
- Read poses from the sim API (`robot.data.body_pose_w`). Read only *static geometry* from USD. See `probe_fingertips.py` for the right way.
- Isaac Lab 3.0 data fields are Warp-backed `ProxyArray`s. Use `.torch` before passing them to TorchScript math (`quat_apply`, etc.).

## 4. An Isaac Lab environment is a set of managers

| Piece | Question | Our task |
|---|---|---|
| Scene | What exists? | SO-101, mug (0.5 scale), yellow bowl, table, scene + wrist cameras |
| Actions | What can be commanded? | IK-Abs: `[pos_xyz (base frame), quat_xyzw, jaw closedness]`, 8-D |
| Observations | What is seen? | joint pos/vel, EE pose, object poses (privileged), 256×256 RGB ×2 |
| Events | What is randomized, and when? | object poses at reset; friction at startup |
| Terminations | When does an episode end? | `mug_in_bowl` success, mug dropped, timeout |
| Timing | How fast? | `sim.dt = 1/120`, `decimation = 4`, so **30 Hz** control |

- `num_envs` runs many copies in parallel on one GPU: 48 demos took about 3 minutes.
- Don't write environments from scratch. **Inherit and override**, as we did from `IsaacContrib-Stack-Cube-SO101-IK-Abs-v0`.

## 5. Controlling an arm

**Joint control vs task-space control**
- **Joint control:** "elbow → 0.8 rad". This is what the robot and FLUX output.
- **Task-space (IK) control:** "gripper → here". A solver converts it to joint angles.
- We script with IK and **record the resulting joint targets** as labels.

**The TCP (tool centre point) is not the link origin.** The SO-101 has one fixed finger (tip at gripper-frame x ≈ −1.1 cm, z ≈ −10 cm) and one sweeping jaw (100° open). A 4.4 cm cup pressed against the fixed finger has its centre at x ≈ +1.4 cm, so `TCP_LOCAL = (0.014, 0, -0.085)`.

**Degrees of freedom limit reach.** A 5-DOF arm can't point straight down everywhere, so the approach tilt grows with reach, from 0° near the base to 60° at 0.28 m. Far spawns still fail, which caps the workspace.

## 6. Demonstrations (imitation learning)

- A **scripted expert** uses privileged state (true object poses), but we record only what a real robot has: cameras, joint states and commanded actions. The policy learns to do it *from pixels*.
- A demo is a list of (observation_t, action_t) pairs with exact timestamps. Failed demos are flagged and filtered out; **quality beats quantity**.
- **Expert phases:** pre-shape jaw → pre-grasp (backed off along the approach axis) → descend → close → lift (with a grasp check) → carry → lower → release → retreat.

**Pilot success history** (every step here came from *looking* at frames or measurements):

| Stage | Success |
|---|---|
| Top-down grasps | 0–3/24 |
| Correct fingertip geometry | still low |
| Cup-centre offset fixed | 10/24 |
| Reach-adaptive tilt | **30/48** |

## 7. The policy side

- FLUX SO-101 takes the scene and wrist images, 8 recent observations and the 6-joint state. It outputs **42 future joint targets at 30 Hz (1.4 s)**, executes 32, then replans (receding horizon).
- **LoRA fine-tuning** adds small adapters on one GPU. Upstream LeRobot `main` has FLUX 3 Action as of 2026-09-23 (PR #4739), next to ACT, SmolVLA, π0.5 and GR00T, the baselines.
- **The sim must match what the policy expects:** 30 Hz, 256×256 cameras, joint order and units, and wrist-camera placement (the real SO-101 `camera_mount`).

## 8. Methodology

- Develop on a variant; test on a sealed benchmark (`CONTEMPLATING_OUR_NAVEL_ROBOT_EDITION.md` §5.0).
- Success comes from sim ground truth, never from the model.
- Change one thing, measure, look at the frames.

## 9. Gotchas

- Quaternions are **xyzw** in Isaac Lab 3.0. Example: table `rot=[0,0,0.707,0.707]`.
- **GPU1 drives the display.** Render the viewer there: `--device cuda:0 --kit_args "--/renderer/activeGpu=1 --/renderer/multiGpu/enabled=false"`.
- `isaaclab_teleop` is needed by the stack tasks: `uv run --extra teleop`.
- `libxml2.so.2` is missing on Ubuntu 26.04. This breaks the FBX/OBJ asset converter only; the URDF/MJCF importers bundle their own converters.
- The first render after a reset is stale, so drop the first frames.
- Mass overrides are ignored on instanced assets. Use events, e.g. `randomize_rigid_body_material` for friction.
- The IK-Abs base caps grip torque at 1 N·m; we use 3 N·m for the mug.
- Keep one uv venv per stack: flux-action (torch 2.10), Isaac Lab (torch 2.11), LeRobot (next).
- Only the FLUX venv has an mp4 encoder (imageio-ffmpeg).

## 10. Learn next

1. Isaac Lab's in-repo skills: `skills/user/create-environments`, `plan-manipulation-tasks`, `train-rl-agents`, `domain-randomization-events`.
2. Learn OpenUSD: https://docs.nvidia.com/learn-openusd/latest/index.html
3. Sim-to-Real SO-101 workshop (Isaac 5.1/Lab 2.3, but the same robot): https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop
4. Read `so101_mug_bowl.py` and `tools/record_demos.py` top to bottom, about 400 lines.
