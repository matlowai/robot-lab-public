"""Capability-based task assignment: missions say what they need, never which robot."""

import math
from dataclasses import dataclass

from robots.base import RobotAdapter


@dataclass(frozen=True)
class Task:
    type: str
    target: tuple[float, float]
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class Assignment:
    task: Task
    robot_id: str | None
    reason: str


def assign(task: Task, robots: list[RobotAdapter]) -> Assignment:
    """The nearest robot whose required capabilities are all verified True. Unknown (None) never qualifies."""
    eligible, blocked = [], []
    for r in robots:
        caps = r.capabilities()
        missing = [v for v in task.requires if caps.supports(v) is not True]
        (blocked if missing else eligible).append((r, missing))
    if not eligible:
        detail = "; ".join(
            f"{r.id}: " + ", ".join(f"{v}={r.capabilities().supports(v)}" for v in missing) for r, missing in blocked
        )
        return Assignment(task, None, f"no eligible robot ({detail})")

    def distance(robot: RobotAdapter) -> float:
        pose = robot.pose()  # robots that don't report a site-frame pose rank last among the eligible
        return math.dist(pose[:2], task.target) if pose is not None else math.inf

    best = min((r for r, _ in eligible), key=distance)
    return Assignment(task, best.id, f"{best.id} is the nearest robot with {list(task.requires) or 'no requirements'}")
