"""Virtual 2-D lidar from PhysX raycasts (Isaac venv). Cheap, no rendering: it hits anything with a collider.

Rays are horizontal in the world frame (yaw only) at fixed heights, so body pitch never tilts them into the
ground. Two scan heights catch both low things (a pallet is 0.25 m tall) and people. Each ray starts outside
Spot's own footprint and hits on Spot itself are ignored. Returns points in the robot frame (x forward), the
same format the local planner and its fake-world tests use.
"""

import math

import carb
from omni.physx import get_physx_scene_query_interface


class PhysxLidar:
    def __init__(self, self_prefix: str = "/World/spot", fov_deg: float = 270.0, rays: int = 181,
                 heights: tuple[float, ...] = (0.15, 0.6), max_range_m: float = 12.0, start_radius_m: float = 0.65):
        self.self_prefix, self.rays, self.heights = self_prefix, rays, heights
        self.fov, self.max_range, self.start = math.radians(fov_deg), max_range_m, start_radius_m
        self._sq = get_physx_scene_query_interface()

    def scan(self, pose: tuple[float, float, float]) -> list[tuple[float, float]]:
        x, y, yaw = pose
        points = []
        for i in range(self.rays):
            rel = -self.fov / 2 + self.fov * i / (self.rays - 1)
            a = yaw + rel
            dx, dy = math.cos(a), math.sin(a)
            best = math.inf
            for h in self.heights:
                origin = carb.Float3(x + self.start * dx, y + self.start * dy, h)
                hit = self._sq.raycast_closest(origin, carb.Float3(dx, dy, 0.0), self.max_range - self.start)
                if hit.get("hit") and not str(hit.get("collision", "")).startswith(self.self_prefix):
                    best = min(best, self.start + hit["distance"])
            if best < self.max_range:
                points.append((best * math.cos(rel), best * math.sin(rel)))
        return points
