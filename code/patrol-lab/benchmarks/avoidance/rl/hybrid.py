"""RL + safety shield: a named hybrid arm (D46). Never reported as a pure RL result.

Each step the learned policy proposes a command. The heuristic planner's obstacle tracker (scan-to-scan velocity
estimates, lidar only: no privileged state) rolls that command forward over the planner's 2 s horizon against where
each tracked obstacle will be. If the command would come closer than the planner's safety margin (0.35 m), the
planner's own command is used for that step instead and the step's status is "shield"; otherwise the policy's command
goes through unchanged. sim2d counts statuses per episode, so the intervention rate is in every episode row.
"""

from __future__ import annotations

import numpy as np

from benchmarks.avoidance import sim2d
from benchmarks.avoidance.rl.controller import _LIM, RLController, load_policy
from robots.spot.local_planner import LocalPlanner, _min_clearance, _rollout


class ShieldedRLController:
    name = "rl_shield"

    def __init__(self, ckpt_path, velocity_source: str = "command", model=None, obs_version=None):
        self.rl = RLController(ckpt_path, velocity_source, model=model, obs_version=obs_version)
        self.planner = LocalPlanner()
        self._vels = None
        orig = self.planner.tracker.update

        def update(*args, **kw):  # keep the tracker's per-point velocities for the check
            self._vels = orig(*args, **kw)
            return self._vels
        self.planner.tracker.update = update

    def step(self, t, pose, goal, scan):
        p = self.planner.step(t, pose, goal, scan)  # updates the tracker every step, like the planner itself
        pre_vel = self.rl.vel.copy()
        vx, vy, wz, status = self.rl.step(t, pose, goal, scan)
        if status == "arrived":
            return vx, vy, wz, status
        cfg = self.planner.cfg
        vels = self._vels if self._vels is not None else [(0.0, 0.0)] * len(scan)
        clear = _min_clearance(list(_rollout(vx, wz, cfg, vy)), list(scan), list(vels), cfg.robot_radius, cfg.dt)
        if clear < cfg.safety_margin and p.status != "arrived":
            cmd = np.array(p.cmd, dtype=float)
            if self.rl.velocity_source == "command":  # the policy's velocity feature follows what is really sent
                self.rl.vel = pre_vel + np.clip(cmd - pre_vel, -_LIM, _LIM)
            return float(cmd[0]), float(cmd[1]), float(cmd[2]), "shield"
        return vx, vy, wz, status


def register_shield(ckpt_path, name: str = "rl_shield", velocity_source: str = "command") -> None:
    model, ck = load_policy(ckpt_path)
    version = ck["obs_version"]
    sim2d.register(name, lambda: ShieldedRLController(ckpt_path, velocity_source, model=model, obs_version=version))
