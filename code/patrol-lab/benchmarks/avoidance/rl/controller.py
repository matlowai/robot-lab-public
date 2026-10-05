"""The trained PPO policy as a sim2d controller: step(t, pose, goal, scan) -> (vx, vy, wz, status).

Observations are built with the env's own functions (ranges_from_points -> sectorize, goal_features, build_obs /
build_obs_lagged), so the policy sees the same numbers it was trained on. The checkpoint's obs_version picks the
feature builder ("avoid-v1": 134, "avoid-v2": 198; rl/env.py::OBS_SPECS). Inference is on CPU, deterministic (tanh
of the mean, or its clamp for squash="clip" checkpoints, D46).

Velocity feature ("current body velocity"): by default ``velocity_source="command"`` — the controller integrates
its own commands through sim2d's acceleration limits (ACC_VX/ACC_VY/ACC_WZ, dt = 0.1 s). In sim2d that reproduces
the simulator's internal velocity exactly, including across checkpoint switches (sim2d does not apply the
command on an "arrived" step, and neither does this model). ``velocity_source="pose"`` instead differentiates
successive poses (exact in sim2d except right after an arrival step, which repeats t); that is the option to use
where the walking policy does not track commands perfectly (Isaac).

Previous scan (avoid-v1): the sectors from the last call with a different t (sim2d repeats t on an arrival step; the
repeat does not shift the history).

Lagged scans (avoid-v2): ScanHistory keeps the first scan at each distinct t and returns the one whose timestamp is
closest to t - 0.3 / t - 0.6 (ties -> the older), so jittered or irregular control ticks (Isaac) pick the right
scan instead of counting calls. While the history is shorter than a lag, the oldest scan stands in, as in the env
(on the first call every channel is scan_now). A repeated t neither shifts the history nor changes the lagged scans;
time going backwards (a new run) clears it.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from benchmarks.avoidance.rl.env import (ACC_VX, ACC_VY, ACC_WZ, ARRIVAL_M, DT, OBS_SPECS, OBS_VERSION, build_obs,
                                         build_obs_lagged, goal_features, obs_spec, ranges_from_points, scale_action,
                                         sectorize)
from benchmarks.avoidance.rl.ppo import ActorCritic

_LIM = np.array([ACC_VX * DT, ACC_VY * DT, ACC_WZ * DT])


def load_policy(ckpt_path) -> tuple[ActorCritic, dict]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    version = ck.get("obs_version")
    if version not in OBS_SPECS:
        raise ValueError(f"checkpoint obs_version {version!r} is not one of {sorted(OBS_SPECS)}")
    if ck["obs_dim"] != OBS_SPECS[version]["dim"]:
        raise ValueError(f"checkpoint obs_dim {ck['obs_dim']} != {OBS_SPECS[version]['dim']} for {version}")
    model = ActorCritic(ck["obs_dim"], ck["act_dim"], tuple(ck["hidden"]), squash=ck.get("squash", "tanh"),
                        log_std_max=ck.get("log_std_max", 1.0))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


class ScanHistory:
    """Sector scans by timestamp: one entry per distinct t (the first scan at that t), oldest first."""

    def __init__(self, keep_s: float):
        self.keep_s = keep_s
        self.ts: list[float] = []
        self.sects: list[np.ndarray] = []

    def push(self, t: float, sect: np.ndarray) -> None:
        if self.ts and t < self.ts[-1]:  # time went backwards: a new run
            self.ts, self.sects = [], []
        if self.ts and t == self.ts[-1]:
            return
        self.ts.append(t)
        self.sects.append(sect)
        while len(self.ts) > 1 and self.ts[1] <= t - self.keep_s:  # the oldest can no longer be the closest
            self.ts.pop(0)
            self.sects.pop(0)

    def at(self, target: float) -> np.ndarray:
        """The scan whose timestamp is closest to target (ties -> the older; the oldest if all are newer)."""
        best, err = 0, abs(self.ts[0] - target)
        for i in range(1, len(self.ts)):
            e = abs(self.ts[i] - target)
            if e < err:
                best, err = i, e
        return self.sects[best]


def version_for_dim(dim: int) -> str:
    return next(v for v, sp in OBS_SPECS.items() if sp["dim"] == dim)


class RLController:
    name = "rl"

    def __init__(self, ckpt_path, velocity_source: str = "command", model: ActorCritic | None = None,
                 obs_version: str | None = None):
        torch.set_num_threads(1)  # many of these run side by side in eval workers
        if model is None:
            model, ck = load_policy(ckpt_path)
            obs_version = ck["obs_version"]
        self.model = model
        # a shared model (register_rl) carries no version tag: its input width says which one it is
        self.obs_version = obs_version or version_for_dim(int(model.obs_rms.mean.shape[0]))
        spec = obs_spec(self.obs_version)
        if int(model.obs_rms.mean.shape[0]) != spec["dim"]:
            raise ValueError(f"model input {int(model.obs_rms.mean.shape[0])} != {spec['dim']} for {self.obs_version}")
        self.lags_s = [k * DT for k in spec["lag_steps"]]
        self.hist = ScanHistory(keep_s=max(self.lags_s) + 0.5) if self.obs_version != OBS_VERSION else None
        self.tracks = None  # clone probe (D47): planner-derived features appended to the observation
        if spec.get("tracks"):
            from benchmarks.avoidance.rl.track_features import TrackFeatures
            self.tracks = TrackFeatures(with_commitment=spec["tracks"] == "vel+commitment")
        if velocity_source not in ("command", "pose"):
            raise ValueError(velocity_source)
        self.velocity_source = velocity_source
        self.vel = np.zeros(3)
        self._last_t = None
        self._last_sect = None  # sectors at _last_t
        self._prev_at_last = None  # the "previous" sectors used at _last_t
        self._last_pose = None
        self.last_obs = None

    def observe(self, t, pose, goal, scan) -> tuple[np.ndarray, float]:
        """Build the observation (and update scan history) without acting. Returns (obs, goal distance)."""
        x, y, yaw = pose
        dx, dy = goal[0] - x, goal[1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        gx, gy = c * dx + s * dy, -s * dx + c * dy
        sect = sectorize(ranges_from_points(scan))
        if self.hist is not None:  # avoid-v2 (and later): lagged scans by timestamp
            self.hist.push(t, sect)
            lagged = [self.hist.at(t - lag) for lag in self.lags_s]
        if self._last_t is None:
            prev = sect
        elif t == self._last_t:
            prev = self._prev_at_last
        else:
            prev = self._last_sect
        if self.velocity_source == "pose" and self._last_pose is not None and t != self._last_t:
            dt = t - self._last_t
            px, py, pyaw = self._last_pose
            wx, wy = (x - px) / dt, (y - py) / dt
            dyaw = (yaw - pyaw + math.pi) % (2 * math.pi) - math.pi
            self.vel = np.array([c * wx + s * wy, -s * wx + c * wy, dyaw / dt])
        if t != self._last_t:
            self._last_t, self._last_sect, self._prev_at_last, self._last_pose = t, sect, prev, pose
        if self.hist is None:
            obs = build_obs(sect, prev, goal_features(gx, gy), self.vel)
        else:
            obs = build_obs_lagged(sect, lagged, goal_features(gx, gy), self.vel)
        if self.tracks is not None:
            obs = np.concatenate([obs, self.tracks.update(t, pose, goal, scan)]).astype(np.float32)
        self.last_obs = obs
        return obs, math.hypot(gx, gy)

    def step(self, t, pose, goal, scan):
        obs, dist = self.observe(t, pose, goal, scan)
        if dist <= ARRIVAL_M:
            return 0.0, 0.0, 0.0, "arrived"  # sim2d skips the motion update on arrival: so does the velocity model
        with torch.no_grad():
            a = self.model.act_deterministic(torch.from_numpy(obs)[None])[0].numpy()
        vx, vy, wz = (float(v) for v in scale_action(a))
        if self.velocity_source == "command":
            self.vel = self.vel + np.clip(np.array([vx, vy, wz]) - self.vel, -_LIM, _LIM)
        return vx, vy, wz, "moving" if vx > 0.05 else "turning"


def register_rl(ckpt_path, name: str = "rl", velocity_source: str = "command") -> None:
    """Register the checkpoint as a sim2d controller (call in every worker process)."""
    from benchmarks.avoidance import sim2d

    model, ck = load_policy(ckpt_path)  # load once per process, share between episodes
    version = ck["obs_version"]
    sim2d.register(name, lambda: RLController(ckpt_path, velocity_source, model=model, obs_version=version))
