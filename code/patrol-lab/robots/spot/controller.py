"""Waypoint -> velocity command for a legged robot whose learned policy tracks (vx, vy, wz) in the body frame.

Pure Python so it's testable without a simulator. Turn toward the goal first, walk when roughly facing it,
slow down on approach. Limits stay well inside what Isaac's flat-terrain Spot policy was trained on.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WalkLimits:
    max_vx: float = 1.0  # m/s forward
    max_wz: float = 1.0  # rad/s
    heading_gain: float = 1.5  # wz = gain * heading error
    walk_heading_rad: float = 0.6  # only walk forward when facing within this of the goal
    slow_radius_m: float = 1.5  # ramp speed down inside this distance
    arrival_m: float = 0.4


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def command(pose: tuple[float, float, float], goal: tuple[float, float], lim: WalkLimits = WalkLimits()):
    """-> ((vx, vy, wz), arrived). Pose is (x, y, yaw) in the site frame."""
    x, y, yaw = pose
    dx, dy = goal[0] - x, goal[1] - y
    dist = math.hypot(dx, dy)
    if dist <= lim.arrival_m:
        return (0.0, 0.0, 0.0), True
    err = wrap(math.atan2(dy, dx) - yaw)
    wz = max(-lim.max_wz, min(lim.max_wz, lim.heading_gain * err))
    facing = max(0.0, math.cos(err)) if abs(err) < lim.walk_heading_rad else 0.0
    vx = lim.max_vx * facing * min(1.0, dist / lim.slow_radius_m)
    return (vx, 0.0, wz), False
