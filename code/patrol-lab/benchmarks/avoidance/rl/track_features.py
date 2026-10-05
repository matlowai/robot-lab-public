"""Planner-derived observation features for the clone probe (D47). sim2d only: the training env cannot build them.

The plain behaviour clone of the planner (D46, arm A3) failed to avoid ordinary people even after three DAgger rounds.
The planner sees more than the avoid-v2 observation: its ObstacleTracker keeps a smoothed velocity per scan cluster,
and it carries a committed detour side. These features hand the clone that extra information, from a LocalPlanner
instance fed exactly the planner's inputs (its state never depends on its own commands, so it matches the teacher's).

    avoid-v2t   avoid-v2 + track velocity per sector: for each of the 64 sectors, the tracker's trusted velocity
                (robot-frame axes, m/s / TRACK_VEL_SCALE) of the nearest return in that sector; 0 where nothing moves
    avoid-v2ts  avoid-v2t + the planner's commitment: side (+1 left / -1 right / 0), detour active (0/1), and the
                bearing of the point it steers for in the robot frame (cos, sin)
"""

from __future__ import annotations

import math

import numpy as np

from benchmarks.avoidance.rl.env import FOV, MAX_RANGE, RAY_STEP, RAYS, SECTORS, TRACK_VEL_SCALE
from robots.spot.local_planner import LocalPlanner


def sector_velocities(scan, vels) -> np.ndarray:
    """[128]: per sector, the velocity of the nearest return in that sector: vx in [0:64), vy in [64:128)."""
    out = np.zeros((2, SECTORS), dtype=np.float64)
    if len(scan) == 0:
        return out.reshape(-1)
    p = np.asarray(scan, dtype=np.float64).reshape(-1, 2)
    v = np.asarray(vels, dtype=np.float64).reshape(-1, 2)
    r = np.hypot(p[:, 0], p[:, 1])
    j = np.rint((np.arctan2(p[:, 1], p[:, 0]) + FOV / 2) / RAY_STEP).astype(np.int64)  # as ranges_from_points
    ok = (j >= 0) & (j < RAYS) & (r <= MAX_RANGE)
    k = (j[ok] * SECTORS) // RAYS
    order = np.argsort(-r[ok], kind="stable")  # farthest first: the nearest return in a sector is written last
    out[:, k[order]] = (v[ok][order] / TRACK_VEL_SCALE).T
    return out.reshape(-1)


class TrackFeatures:
    def __init__(self, with_commitment: bool):
        self.planner = LocalPlanner()
        self.with_commitment = with_commitment

    def update(self, t, pose, goal, scan) -> np.ndarray:
        if not self.with_commitment:  # the tracker alone: the same velocities, without the detour search's cost
            return sector_velocities(scan, self.planner.tracker.update(t, pose, scan))
        target, vels = self.planner.update(t, pose, goal, scan)
        feat = sector_velocities(scan, vels)
        detour = target != goal
        cos_b = sin_b = 0.0
        if detour:
            x, y, yaw = pose
            b = math.atan2(target[1] - y, target[0] - x) - yaw
            cos_b, sin_b = math.cos(b), math.sin(b)
        return np.concatenate([feat, [float(self.planner._side), float(detour), cos_b, sin_b]])
