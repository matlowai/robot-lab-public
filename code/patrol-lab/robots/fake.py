"""FakeAdapter: a 2-D kinematic robot with no simulator (level 0, alongside NORI's MockRobot).

It drives straight at its goal at max speed and ignores obstacles. That is enough to exercise missions, the
scheduler and patrol coverage; M1 swaps in the Isaac adapters behind the same interface.
"""

import math

from robots.base import Capabilities, NavigationRefused, NavStatus, Pose2D

ARRIVAL_TOL_M = 0.25


class FakeAdapter:
    def __init__(
        self,
        robot_id: str,
        caps: Capabilities,
        start: tuple[float, float],
        places: dict[str, tuple[float, float]] | None = None,
        yaw: float = 0.0,
    ):
        self.id = robot_id
        self._caps = caps
        self._places = dict(places or {})
        self._x, self._y, self._yaw = float(start[0]), float(start[1]), yaw
        self._vx = self._vy = 0.0
        self._goal: tuple[float, float] | None = None
        self._nav = NavStatus("idle")
        self._battery = 1.0
        self.max_speed = float(caps.limits.get("max_speed_mps", 1.0))

    def capabilities(self) -> Capabilities:
        return self._caps

    def pose(self) -> Pose2D:
        return (self._x, self._y, self._yaw)

    def velocity(self) -> tuple[float, float, float]:
        return (self._vx, self._vy, 0.0)  # site frame

    def goto_named(self, place: str) -> None:
        if place not in self._places:
            raise NavigationRefused(f"unknown place {place!r}")
        if self._nav.state == "navigating":
            raise NavigationRefused("navigation is active")
        self._goal = self._places[place]
        self._nav = NavStatus("navigating", place)

    def navigation_status(self) -> NavStatus:
        return self._nav

    def look_at(self, x: float, y: float, dwell_s: float) -> None:
        self._yaw = math.atan2(y - self._y, x - self._x)

    def stop(self) -> None:
        if self._nav.state == "navigating":
            self._nav = NavStatus("canceled", self._nav.target, "stopped")
        self._goal = None
        self._vx = self._vy = 0.0

    def health(self) -> dict:
        return {"battery": round(self._battery, 4), "faults": [], "localization_ok": True}

    def sensor_streams(self) -> list[str]:
        return []

    def step(self, dt: float) -> None:
        if self._goal is None:
            return
        dx, dy = self._goal[0] - self._x, self._goal[1] - self._y
        dist = math.hypot(dx, dy)
        travel = min(dist, self.max_speed * dt)
        if dist > 1e-9:
            self._yaw = math.atan2(dy, dx)
            self._x += dx / dist * travel
            self._y += dy / dist * travel
            self._vx, self._vy = dx / dist * travel / dt, dy / dist * travel / dt
        if dist - travel <= ARRIVAL_TOL_M:
            self._goal = None
            self._vx = self._vy = 0.0
            self._nav = NavStatus("succeeded", self._nav.target)
        endurance_s = float(self._caps.limits.get("endurance_h", 1.0)) * 3600
        self._battery = max(0.0, self._battery - dt / endurance_s)

    def state(self) -> dict:
        x, y, yaw = self.pose()
        return {
            "id": self.id,
            "embodiment": self._caps.embodiment,
            "pose": [round(x, 3), round(y, 3), round(yaw, 4)],
            "velocity": [round(v, 3) for v in self.velocity()],
            "health": self.health(),
        }
