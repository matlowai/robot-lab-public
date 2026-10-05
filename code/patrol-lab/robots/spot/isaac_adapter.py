"""IsaacSpotAdapter: Isaac Sim's pretrained flat-terrain Spot policy behind our RobotAdapter. Isaac venv only.

The policy (RobotPolicyRunner + get_spot_spec, 500 Hz physics) tracks a body-frame velocity command; our
controller turns named-place goals into that command. Call on_physics_step(dt) from a POST_PHYSICS_STEP
callback; step() is a no-op because the simulator owns the clock.
"""

import math

import numpy as np
from isaacsim.robot.policy.examples.bundled.spot import get_spot_spec
from isaacsim.robot.policy.examples.runtime import RobotPolicyRunner

from robots.base import Capabilities, NavigationRefused, NavStatus
from robots.spot.controller import WalkLimits, command

FALLEN_BELOW_M = 0.30  # base height under which Spot counts as fallen


def yaw_from_wxyz(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class IsaacSpotAdapter:
    def __init__(self, robot_id: str, caps: Capabilities, prim_path: str, places: dict[str, tuple[float, float]],
                 position=(0.0, 0.0, 0.8), limits: WalkLimits = WalkLimits(), lidar=None, avoider=None,
                 plan_hz: float = 10.0):
        """lidar: scan(pose) -> robot-frame points. avoider: step(t, pose, goal, scan) -> (vx, vy, wz, status), the
        same contender API as benchmarks/avoidance/sim2d.py (heuristic planner, RL policy, ...). With both, goals
        are reached through the avoider; without, through the straight-line controller."""
        self.id = robot_id
        self._caps = caps
        self._places = dict(places)
        self._limits = limits
        self.runner = RobotPolicyRunner(get_spot_spec(), prim_path=prim_path, position=position)
        self._goal: tuple[float, float] | None = None
        self._nav = NavStatus("idle")
        self._cmd = np.zeros(3, dtype=np.float32)
        self.fallen = False
        self.lidar = lidar
        self.avoider = avoider if lidar is not None else None
        self._plan_period, self._since_plan, self._t = 1.0 / plan_hz, math.inf, 0.0
        self.last_status: str | None = None
        self.last_scan: list[tuple[float, float]] = []

    # lifecycle (the order Isaac's own Spot test uses): spawn -> play -> initialize -> reset
    def spawn(self) -> None:
        self.runner.spawn()

    def initialize(self) -> None:
        self.runner.initialize()
        self.runner.articulation.reset_to_default_state()

    def on_physics_step(self, dt: float) -> None:
        self._t += dt
        pose = self.pose()
        if pose is not None and self._base_z() < FALLEN_BELOW_M and not self.fallen:
            self.fallen = True
            if self._nav.state == "navigating":
                self._nav = NavStatus("failed", self._nav.target, "robot fell")
            self._goal = None
        if self._goal is not None and not self.fallen and self.avoider is not None:
            self._since_plan += dt
            if self._since_plan >= self._plan_period:  # re-plan at plan_hz; hold the command in between
                self._since_plan = 0.0
                self.last_scan = self.lidar.scan(pose)
                vx, vy, wz, self.last_status = self.avoider.step(self._t, pose, self._goal, self.last_scan)
                self._cmd[:] = (vx, vy, wz)
                if self.last_status == "arrived":
                    self._goal = None
                    self._cmd[:] = 0.0
                    self._nav = NavStatus("succeeded", self._nav.target)
        elif self._goal is not None and not self.fallen:
            (vx, vy, wz), arrived = command(pose, self._goal, self._limits)
            self._cmd[:] = (vx, vy, wz)
            if arrived:
                self._goal = None
                self._cmd[:] = 0.0
                self._nav = NavStatus("succeeded", self._nav.target)
        else:
            self._cmd[:] = 0.0
        self.runner.step(dt, self._cmd)

    # RobotAdapter
    def capabilities(self) -> Capabilities:
        return self._caps

    def _world(self):
        pos, quat = self.runner.articulation.get_world_poses()
        return pos.numpy()[0], quat.numpy()[0]

    def _base_z(self) -> float:
        return float(self._world()[0][2])

    def pose(self):
        p, q = self._world()
        return (float(p[0]), float(p[1]), yaw_from_wxyz(q))

    def velocity(self):
        return tuple(float(v) for v in self._cmd)  # commanded body-frame (vx, vy, wz)

    def goto_named(self, place: str) -> None:
        if self.fallen:
            raise NavigationRefused("robot has fallen")
        if place not in self._places:
            raise NavigationRefused(f"unknown place {place!r}")
        if self._nav.state == "navigating":
            raise NavigationRefused("navigation is active")
        self._goal = self._places[place]
        self._nav = NavStatus("navigating", place)
        self._since_plan = float("inf")

    def navigation_status(self) -> NavStatus:
        return self._nav

    def stop(self) -> None:
        if self._nav.state == "navigating":
            self._nav = NavStatus("canceled", self._nav.target, "stopped")
        self._goal = None
        self._cmd[:] = 0.0

    def health(self) -> dict:
        return {"battery": None, "faults": ["fallen"] if self.fallen else [], "localization_ok": True}

    def sensor_streams(self) -> list[str]:
        return []

    def step(self, dt: float) -> None:
        """No-op: Isaac owns the clock; control runs in on_physics_step."""

    def state(self) -> dict:
        x, y, yaw = self.pose()
        return {"id": self.id, "embodiment": self._caps.embodiment, "pose": [round(x, 3), round(y, 3), round(yaw, 4)],
                "velocity": [round(v, 3) for v in self.velocity()], "health": self.health(),
                "navigation": {"state": self._nav.state, "target": self._nav.target, "detail": self._nav.detail}}
