"""The capability-based robot interface (PLAN.md: RobotAdapter).

Capabilities are three-valued, like NORI's SDK (`info.supports(verb)` -> True / False / None):
True = supported, False = not supported, None = unknown / unverified. The scheduler only assigns a task to a
robot whose required verbs are all True; unknown is never treated as yes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

VERBS = (
    "outdoor_patrol",
    "indoor_patrol",
    "rough_terrain",
    "stairs",
    "open_door",
    "press_button",
    "manipulate",
    "thermal_inspection",
    "lidar_mapping",
)

Pose2D = tuple[float, float, float]  # x, y, yaw (rad), site frame


@dataclass(frozen=True)
class Capabilities:
    robot: str
    embodiment: str  # quadruped | humanoid | wheeled_bimanual | wheeled
    verbs: dict[str, bool | None]
    limits: dict[str, float] = field(default_factory=dict)
    provenance: str = ""

    def supports(self, verb: str) -> bool | None:
        if verb not in VERBS:
            raise KeyError(f"unknown capability verb {verb!r}")
        return self.verbs.get(verb)

    def satisfies(self, requires: list[str]) -> bool:
        return all(self.supports(v) is True for v in requires)

    @classmethod
    def load(cls, path: str | Path) -> "Capabilities":
        d = yaml.safe_load(Path(path).read_text())
        return cls(d["robot"], d["embodiment"], dict(d["verbs"]), dict(d.get("limits", {})), d.get("provenance", ""))


NAV_STATES = ("idle", "navigating", "succeeded", "failed", "canceled")


class NavigationRefused(RuntimeError):
    """The robot declined a goal (unknown place, E-stop latched, goal already active, ...)."""


@dataclass(frozen=True)
class NavStatus:
    state: str  # one of NAV_STATES
    target: str | None = None
    detail: str = ""


@runtime_checkable
class RobotAdapter(Protocol):
    """What every robot, simulated or real, exposes. Adapters own all simulator or SDK specifics.

    Navigation is by **named place** (a patrol checkpoint id), because that is what real robots expose: NORI's
    SDK navigates to saved waypoints and never takes coordinates. Progress comes from navigation_status(), not
    from comparing poses, because a robot may not report a site-frame pose at all (pose() -> None).
    """

    id: str

    def capabilities(self) -> Capabilities: ...
    def pose(self) -> Pose2D | None: ...  # None: the robot does not report a site-frame pose
    def velocity(self) -> tuple[float, float, float]: ...  # vx, vy, wz (the adapter documents its frame)
    def goto_named(self, place: str) -> None: ...  # raises NavigationRefused
    def navigation_status(self) -> NavStatus: ...
    def stop(self) -> None: ...
    def health(self) -> dict: ...  # battery (0-1 or None), faults (list), localization_ok (bool)
    def sensor_streams(self) -> list[str]: ...
    def step(self, dt: float) -> None: ...  # advance the adapter's clock (no-op for real robots)

    def state(self) -> dict: ...  # robot_state.schema.json
