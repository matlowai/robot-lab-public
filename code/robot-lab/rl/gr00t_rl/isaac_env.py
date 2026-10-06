"""Batched SO-101 pick-place env + GR00T-in-the-loop wave runner (import AFTER the Isaac Sim app has launched).

One "wave" = all N envs reset together with one seed, settle like the demos, then run up to MAX_CHUNKS RL steps of
EXEC_H=8 control ticks each. Every RL step: ONE batched GR00T plan for the still-active envs (socket to
gr00t_server.py), the residual actor's correction, 8 env.step calls. Synchronous waves keep layouts a pure function of
(seed, env index), so the base-vs-RL eval is paired on identical layouts and identical GR00T noise.

Start state (deviation from tools/gr00t_eval.py, stated in DESIGN.md): after env.reset the arm is held for one full
chunk -- 2 ticks commanding the jaw pre-shape (exactly the eval's 2 dropped frames, which leave the jaw ~16 units), then
6 ticks holding the joints where they are -- and the strict lift baseline z0 is read at the end of that chunk, after
the object's ~3 cm reset drop has settled (the eval reads it after 2 ticks).
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import robot_lab.tasks  # noqa: F401
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg
from robot_lab.tasks.so101_mug_bowl import SCENE_CAM_EYE, SCENE_CAM_TARGET, _t
from robot_lab.tasks.so101_pick_place import (CAPTION, HELDOUT_OBJECTS, TRAIN_OBJECTS, object_center_w, object_in_bowl,
                                              set_object)

from .reward import BOWL_RADIUS, Reading, RewardTracker, RewardWeights
from .seeds import mix_seed
from .units import EXEC_H, JAW_PRESHAPE_RAD, Units

TASK_ID = "RobotLab-SO101-PickPlace-Joint-v0"
TRAIN_NAMES = [n for n, _, _ in TRAIN_OBJECTS]
HELDOUT_NAMES = [n for n, _, _ in HELDOUT_OBJECTS]
ARM_SCALE_DEG = 4.0          # max |arm residual| per joint, degrees, constant over the executed chunk
OPEN_UNITS = 50.0            # release command: jaw 50 units = 45 deg (the expert's open / pre-shape)
RELEASE_HOLD_CHUNKS = 2      # a release keeps the jaw commanded open for this chunk and the next
MAX_CHUNKS = 75              # 75 x 8 ticks = 600 ticks = 20 s, the night-1 eval budget
MIN_OBJ_BOWL_XY = 0.10       # gr00t_eval.py layout rejection
TCP_LOCAL = (0.014, 0.0, -0.085)
ACTOR_LOW_DIM = 6 + 6 + 48 + len(TRAIN_NAMES) + 1 + 5 + 1 + 1     # 72
PRIV_DIM = 17
CRITIC_LOW_DIM = ACTOR_LOW_DIM + PRIV_DIM                         # 89


class ChunkEnv:
    def __init__(self, object_name: str, num_envs: int, units_path: str, seed: int = 0, device: str = "cuda:0",
                 rest_check: bool = False):
        cfg = set_object(parse_env_cfg(TASK_ID, device=device, num_envs=num_envs), object_name)
        cfg.seed = seed
        self.rest_check = rest_check
        if rest_check:  # eval only: the success term no longer auto-resets, so we can watch the object come to rest.
            cfg.terminations.success = None  # success is computed by the same function (object_in_bowl) in step()
        cfg.episode_length_s = MAX_CHUNKS * EXEC_H / 30.0 + 10.0  # we end waves ourselves; env time-out never first
        self.env = gym.make(TASK_ID, cfg=cfg).unwrapped
        self.n, self.dev = num_envs, self.env.device
        self.object_name, self.task = object_name, CAPTION.format(object_name)
        self.units = Units(units_path)
        sc = self.env.scene
        self.robot, self.obj, self.bowl = sc["robot"], sc["object"], sc["bowl"]
        self.jid = self.robot.find_joints(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
                                           "gripper"])[0]
        self.ee_idx = self.robot.find_bodies("gripper")[0][0]
        self.tcp_local = torch.tensor(TCP_LOCAL, device=self.dev)
        self.origins = sc.env_origins
        self.onehot = torch.zeros(len(TRAIN_NAMES), device=self.dev)
        if object_name in TRAIN_NAMES:
            self.onehot[TRAIN_NAMES.index(object_name)] = 1.0
        self.obs = None

    # ---- readings ----
    def jpos(self) -> torch.Tensor:
        return _t(self.robot.data.joint_pos)[:, self.jid]

    def jvel(self) -> torch.Tensor:
        return _t(self.robot.data.joint_vel)[:, self.jid]

    def reading(self) -> Reading:
        d = self.robot.data
        ee = _t(d.body_pose_w)[:, self.ee_idx]
        tcp = ee[:, :3] + quat_apply(ee[:, 3:7], self.tcp_local.expand(self.n, 3)) - self.origins
        bq = _t(self.bowl.data.root_quat_w)
        up = quat_apply(bq, torch.tensor([0.0, 0.0, 1.0], device=self.dev).expand(self.n, 3))
        tilt = torch.rad2deg(torch.arccos(up[:, 2].clamp(-1, 1)))
        return Reading(obj=object_center_w(self.env) - self.origins,
                       bowl=_t(self.bowl.data.root_pos_w) - self.origins, bowl_tilt_deg=tilt, tcp=tcp,
                       jaw=self.jpos()[:, 5], joint_vel_max=_t(d.joint_vel).abs().max(-1).values,
                       obj_speed=torch.linalg.norm(_t(self.obj.data.root_lin_vel_w), dim=-1))

    def images(self, ids: torch.Tensor):
        s = self.obs["rgb_camera"]["scene"][ids, ..., :3].to(torch.uint8).cpu().numpy()
        w = self.obs["rgb_camera"]["wrist"][ids, ..., :3].to(torch.uint8).cpu().numpy()
        return s, w

    def set_cam(self):
        self.env.scene["scene_cam"].set_world_poses_from_view(
            torch.tensor(SCENE_CAM_EYE, device=self.dev) + self.origins,
            torch.tensor(SCENE_CAM_TARGET, device=self.dev) + self.origins)

    def step(self, q: torch.Tensor):
        self.obs, _, term, trunc, _ = self.env.step(q)
        tm = self.env.termination_manager
        succ = object_in_bowl(self.env).clone() if self.rest_check else tm.get_term("success").clone()
        return term | trunc, succ, tm.get_term("object_dropping").clone()

    def reset_settle(self, seed: int) -> Reading:
        self.obs, _ = self.env.reset(seed=seed)
        self.set_cam()
        hold = self.jpos().clone()
        hold[:, 5] = JAW_PRESHAPE_RAD
        for k in range(EXEC_H):
            if k == 2:
                hold = self.jpos().clone()  # after the eval's 2 pre-shape ticks: hold everything where it is
            done, _, _ = self.step(hold)
            if bool(done.any()):
                raise RuntimeError(f"env terminated during settle (seed {seed}): {done.nonzero().flatten().tolist()}")
        return self.reading()

    def close(self):
        self.env.close()


def _frame(cenv: ChunkEnv, i: int) -> np.ndarray:
    o = cenv.obs["rgb_camera"]
    return np.concatenate([o["scene"][i, ..., :3].to(torch.uint8).cpu().numpy(),
                           o["wrist"][i, ..., :3].to(torch.uint8).cpu().numpy()], 1)


def build_obs(cenv: ChunkEnv, base_u: np.ndarray, ids: torch.Tensor, tfrac: float, prev_u: torch.Tensor,
              prev_rel: torch.Tensor, hold: torch.Tensor, r: Reading, tr: RewardTracker):
    """Actor (deployable) and critic (privileged) low-dim observations for envs `ids`."""
    U = cenv.units
    q = cenv.jpos()[ids]
    q_units = torch.as_tensor(U.to_units(q.cpu().numpy()), device=cenv.dev)
    base = torch.as_tensor(base_u[:, :EXEC_H], device=cenv.dev)              # (B, 8, 6) units
    rel_arm = base[..., :5] - q_units[:, None, :5]                           # planned arm motion vs now
    chunk_feat = torch.cat([rel_arm, base[..., 5:6] / 100.0], -1).reshape(len(ids), -1)  # 48
    a_low = torch.cat([q_units / 100.0, cenv.jvel()[ids], chunk_feat, cenv.onehot.expand(len(ids), -1),
                       torch.full((len(ids), 1), tfrac, device=cenv.dev), prev_u[ids], prev_rel[ids, None],
                       hold[ids, None].float()], -1)
    rise = r.obj[ids, 2] - tr.z0[ids]
    d_xy = torch.linalg.norm(r.obj[ids, :2] - r.bowl[ids, :2], dim=-1)
    shift = torch.linalg.norm(r.bowl[ids, :2] - tr.bowl0[ids, :2], dim=-1)
    priv = torch.cat([r.obj[ids] - r.tcp[ids], r.obj[ids] - r.bowl[ids], rise[:, None], d_xy[:, None],
                      torch.linalg.norm(r.obj[ids] - r.tcp[ids], dim=-1, keepdim=True), r.jaw[ids, None],
                      r.bowl_tilt_deg[ids, None] / 90.0, shift[:, None],
                      torch.stack([tr.reached[ids], tr.lifted[ids], tr.over[ids], tr.released[ids],
                                   tr.bowl_bad[ids]], -1).float()], -1)
    return a_low.float(), torch.cat([a_low, priv], -1).float()


def run_wave(cenv: ChunkEnv, client, ac, seed: int, mode: str, noise_key: int, weights: RewardWeights | None = None,
             video_envs: tuple = (), actor_priv: bool = False, rest_ticks: int = 0):
    """mode: 'train' (sample), 'rl' (deterministic residual), 'base' (no residual).
    rest_ticks > 0 (needs ChunkEnv(rest_check=True)): after an env's success term fires, hold its arm still with the
    jaw open for rest_ticks more ticks, then record whether the object RESTS in the bowl (centre inside the footprint,
    < 7 cm above the bowl root, bowl upright <= 20 deg and within 3 cm of its start). Diagnostic; STRICT is unchanged.
    Returns (transitions dict of (T, N, ...) tensors or None, per-env summary dict, frames dict, timing dict)."""
    assert mode in ("train", "rl", "base"), mode
    n, dev, U = cenv.n, cenv.dev, cenv.units
    t_wave = time.perf_counter()
    r = cenv.reset_settle(seed)
    tr = RewardTracker(n, dev, weights)
    tr.start(r)
    d_xy0 = torch.linalg.norm(r.obj[:, :2] - r.bowl[:, :2], dim=-1)
    valid = (d_xy0 >= MIN_OBJ_BOWL_XY) & (r.bowl_tilt_deg <= 20.0)
    prev_u = torch.zeros(n, 5, device=dev)
    prev_rel = torch.zeros(n, device=dev)
    hold_left = torch.zeros(n, dtype=torch.long, device=dev)
    T = MAX_CHUNKS
    buf = None
    if mode == "train":
        buf = {"a_low": torch.zeros(T, n, CRITIC_LOW_DIM if actor_priv else ACTOR_LOW_DIM, device=dev),
               "c_low": torch.zeros(T, n, CRITIC_LOW_DIM, device=dev),
               "emb": torch.zeros(T, n, 2048, device=dev, dtype=torch.float16),
               "u": torch.zeros(T, n, 5, device=dev), "rel": torch.zeros(T, n, device=dev),
               "logp": torch.zeros(T, n, device=dev), "val": torch.zeros(T, n, device=dev),
               "rew": torch.zeros(T, n, device=dev), "done": torch.zeros(T, n, device=dev),
               "mask": torch.zeros(T, n, device=dev)}
    comp = {k: torch.zeros(n, device=dev) for k in RewardTracker.KEYS}
    ret = torch.zeros(n, device=dev)
    frames = {i: [] for i in video_envs}
    tm = {"plan": 0.0, "sim": 0.0, "actor": 0.0, "plans": 0, "plan_envs": 0, "ticks": 0}
    p_release, p_release_over = [], []
    n_rel = torch.zeros(n, device=dev)            # release-gate chunks per env (this episode)
    first_rel = torch.full((n,), -1, dtype=torch.long, device=dev)
    tick = 0
    rest_left = torch.full((n,), -1, dtype=torch.long, device=dev)
    rest_q = torch.zeros(n, 6, device=dev)
    rested = torch.zeros(n, dtype=torch.bool, device=dev)
    jaw_open_rad = float(U.from_units(np.array([0, 0, 0, 0, 0, OPEN_UNITS], np.float32))[5])

    def rest_tick(q_all, succ, r_now, task_before):
        """Start / continue / finish post-success rest checks (no-op when rest_ticks == 0)."""
        if rest_ticks <= 0:
            return
        new = tr.task & ~task_before
        if bool(new.any()):
            rest_left[new] = rest_ticks
            rest_q[new] = cenv.jpos()[new]
            rest_q[new, 5] = jaw_open_rad
        fin = rest_left == 0
        if bool(fin.any()):
            d_xy = torch.linalg.norm(r_now.obj[:, :2] - r_now.bowl[:, :2], dim=-1)
            low = (r_now.obj[:, 2] - r_now.bowl[:, 2]) < 0.07
            shift = torch.linalg.norm(r_now.bowl[:, :2] - tr.bowl0[:, :2], dim=-1)
            ok = (d_xy < BOWL_RADIUS) & low & (r_now.bowl_tilt_deg <= 20.0) & (shift <= 0.03)
            rested[fin] = ok[fin]
            rest_left[fin] = -1

    for c in range(T):
        active = ~tr.done
        ids = active.nonzero().flatten()
        if len(ids) == 0:
            break
        q_units = U.to_units(cenv.jpos()[ids].cpu().numpy())
        scene, wrist = cenv.images(ids)
        t0 = time.perf_counter()
        client.send({"cmd": "plan", "scene": scene, "wrist": wrist, "state": q_units,
                     "tasks": [cenv.task] * len(ids),
                     "noise_seeds": np.array([mix_seed(noise_key, seed, int(i), c) for i in ids.tolist()], np.int64)})
        rep = client.recv()
        tm["plan"] += time.perf_counter() - t0
        tm["plans"] += 1
        tm["plan_envs"] += len(ids)
        base_u = rep["actions"]                                             # (B, 16, 6) units
        emb = torch.as_tensor(rep["emb"], device=dev)
        t0 = time.perf_counter()
        a_low, c_low = build_obs(cenv, base_u, ids, c / T, prev_u, prev_rel, hold_left > 0, r, tr)
        if actor_priv:
            a_low = c_low
        if mode == "base":
            u = torch.zeros(len(ids), 5, device=dev)
            rel = torch.zeros(len(ids), device=dev)
        else:
            u, rel, logp, val, pr = ac.act(a_low, emb.float(), c_low, deterministic=(mode == "rl"))
            p_release.append(pr)
            over_now = tr.lifted[ids] & (torch.linalg.norm(r.obj[ids, :2] - r.bowl[ids, :2], dim=-1) < BOWL_RADIUS)
            p_release_over.append(pr[over_now])
        tm["actor"] += time.perf_counter() - t0
        resid_deg = ARM_SCALE_DEG * torch.tanh(u)                           # (B, 5)
        n_rel[ids] += rel
        first_rel[ids] = torch.where((first_rel[ids] < 0) & (rel > 0.5), torch.full_like(first_rel[ids], c), first_rel[ids])
        hold_left[ids] = torch.where(rel > 0.5, torch.full_like(hold_left[ids], RELEASE_HOLD_CHUNKS),
                                     hold_left[ids])
        opening = hold_left[ids] > 0
        cmd = torch.as_tensor(base_u[:, :EXEC_H], device=dev).clone()     # (B, 8, 6) units
        cmd[..., :5] += resid_deg[:, None, :]
        cmd[..., 5] = torch.where(opening[:, None], torch.clamp(cmd[..., 5], min=OPEN_UNITS), cmd[..., 5])
        cmd[..., 5] = cmd[..., 5].clamp(0.0, 100.0)
        q_cmd = torch.as_tensor(U.from_units(cmd.cpu().numpy()), device=dev)  # (B, 8, 6) rad
        chunk_rew = torch.zeros(n, device=dev)
        done_before = tr.done.clone()
        t0 = time.perf_counter()
        for k in range(EXEC_H):
            q_all = cenv.jpos().clone()          # done / inactive envs hold where they are (ignored afterwards)
            live_ids = ids[~tr.done[ids]]
            sel = ~tr.done[ids]
            q_all[live_ids] = q_cmd[sel, k]
            resting = rest_left > 0
            q_all[resting] = rest_q[resting]
            task_before = tr.task.clone()
            done_env, succ, dropped = cenv.step(q_all)
            tick += 1
            rest_left[resting] -= 1
            r = cenv.reading()
            out = tr.step(r, done_env | (succ if cenv.rest_check else torch.zeros_like(succ)), succ, dropped, tick)
            rest_tick(q_all, succ, r, task_before)
            for key, v in out.items():
                comp[key] += v
                chunk_rew += v
            for i in video_envs:
                if not bool(done_before[i]) or bool(resting[i]):
                    frames[i].append(_frame(cenv, i))
        tm["sim"] += time.perf_counter() - t0
        tm["ticks"] += EXEC_H
        if c == T - 1:
            tr.finish(tick)
        ret += chunk_rew
        hold_left[ids] = (hold_left[ids] - 1).clamp(min=0)
        prev_u[ids] = u
        prev_rel[ids] = rel
        if buf is not None:
            buf["a_low"][c, ids] = a_low
            buf["c_low"][c, ids] = c_low
            buf["emb"][c, ids] = emb
            buf["u"][c, ids] = u
            buf["rel"][c, ids] = rel
            buf["logp"][c, ids] = logp
            buf["val"][c, ids] = val
            buf["rew"][c, ids] = chunk_rew[ids]
            buf["done"][c, ids] = tr.done[ids].float()
            buf["mask"][c, ids] = 1.0
    while bool((rest_left > 0).any()):  # finish pending rest checks after the last chunk
        q_all = cenv.jpos().clone()
        resting = rest_left > 0
        q_all[resting] = rest_q[resting]
        cenv.step(q_all)
        rest_left[resting] -= 1
        r = cenv.reading()
        rest_tick(q_all, None, r, tr.task.clone())
        for i in video_envs:
            if bool(resting[i]):
                frames[i].append(_frame(cenv, i))
    summ = {k: v.detach().cpu() for k, v in tr.summary().items()}
    summ["rested"] = rested.cpu()
    summ["valid_layout"] = valid.cpu()
    summ["return"] = ret.cpu()
    summ["n_release_chunks"] = n_rel.cpu()
    summ["first_release_chunk"] = first_rel.cpu()
    summ["components"] = {k: v.cpu() for k, v in comp.items()}
    tm["wave_s"] = time.perf_counter() - t_wave
    tm["p_release_mean"] = float(torch.cat(p_release).mean()) if p_release else 0.0
    po = torch.cat(p_release_over) if p_release_over else torch.zeros(0)
    tm["p_release_over_bowl"] = float(po.mean()) if len(po) else float("nan")   # lifted object inside the footprint
    tm["n_over_bowl_decisions"] = int(len(po))
    return buf, summ, frames, tm
