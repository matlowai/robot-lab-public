"""Compound geometry loaded from compound_spec.yaml: perimeter, gates, zones, road segments, patrol route.

Zones are geometry, not model output: "is this track in the loading dock" is a point-in-polygon test.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

Point = tuple[float, float]
Polygon = tuple[Point, ...]

GATE_TOLERANCE_M = 0.5  # a boundary crossing within this distance of a gate opening counts as a gate passage


def _polygon(spec: dict) -> Polygon:
    if "rect" in spec:
        x0, y0, x1, y1 = spec["rect"]
        return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    return tuple((float(x), float(y)) for x, y in spec["polygon"])


def point_in_polygon(p: Point, poly: Polygon) -> bool:
    """Ray casting; points exactly on an edge count as inside for axis-aligned edges."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        if min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1):
            if (x1 - x0) * (y - y0) == (y1 - y0) * (x - x0):  # on the edge
                return True
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def segment_intersection(p1: Point, p2: Point, q1: Point, q2: Point) -> Point | None:
    """Intersection point of segments p1p2 and q1q2, or None (parallel segments never intersect here)."""
    rx, ry = p2[0] - p1[0], p2[1] - p1[1]
    sx, sy = q2[0] - q1[0], q2[1] - q1[1]
    denom = rx * sy - ry * sx
    if denom == 0:
        return None
    qpx, qpy = q1[0] - p1[0], q1[1] - p1[1]
    t = (qpx * sy - qpy * sx) / denom
    u = (qpx * ry - qpy * rx) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (p1[0] + t * rx, p1[1] + t * ry)
    return None


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = b[0] - a[0], b[1] - a[1]
    length2 = ax * ax + ay * ay
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / length2))
    dx, dy = p[0] - (a[0] + t * ax), p[1] - (a[1] + t * ay)
    return (dx * dx + dy * dy) ** 0.5


@dataclass(frozen=True)
class Zone:
    id: str
    access: str  # public | employee | restricted | high_risk
    vehicles: str  # allowed | prohibited
    priority: int
    polygon: Polygon

    @property
    def restricted(self) -> bool:
        return self.access in ("restricted", "high_risk")


@dataclass(frozen=True)
class Gate:
    id: str
    opening: tuple[Point, Point]
    normal_state: str


@dataclass(frozen=True)
class RoadSegment:
    id: str
    polygon: Polygon


@dataclass(frozen=True)
class Crossing:
    point: Point
    inbound: bool  # outside -> inside
    gate_id: str | None  # set when the crossing went through a gate opening


class Compound:
    def __init__(self, spec: dict):
        self.spec = spec
        self.id: str = spec["id"]
        self.boundary: Polygon = _polygon(spec["perimeter"])
        self.gates = tuple(
            Gate(g["id"], tuple(tuple(map(float, pt)) for pt in g["opening"]), g["normal_state"])
            for g in spec["perimeter"].get("gates", [])
        )
        self.zones = tuple(
            sorted(
                (Zone(z["id"], z["access"], z["vehicles"], int(z["priority"]), _polygon(z)) for z in spec["zones"]),
                key=lambda z: -z.priority,
            )
        )
        self.road_segments = tuple(RoadSegment(r["id"], _polygon(r)) for r in spec.get("road_segments", []))
        patrol = spec.get("patrol", {})
        self.charging_station: Point = tuple(patrol.get("charging_station", (0.0, 0.0)))
        self.route: tuple[tuple[str, Point], ...] = tuple((c["id"], tuple(c["pos"])) for c in patrol.get("route", []))
        # per-stop failure policy for missions.patrol.Stop: {"place", "required", "on_failure", "retries"}
        self.route_stops: tuple[dict, ...] = tuple(
            {"place": c["id"], **{k: c[k] for k in ("required", "on_failure", "retries") if k in c}}
            for c in patrol.get("route", [])
        )

    @classmethod
    def load(cls, path: str | Path) -> "Compound":
        return cls(yaml.safe_load(Path(path).read_text()))

    def inside_site(self, p: Point) -> bool:
        return point_in_polygon(p, self.boundary)

    def zones_at(self, p: Point) -> list[Zone]:
        """Every zone containing p, most specific first."""
        return [z for z in self.zones if point_in_polygon(p, z.polygon)]

    def zone_at(self, p: Point) -> Zone | None:
        zones = self.zones_at(p)
        return zones[0] if zones else None

    def road_segment_at(self, p: Point) -> RoadSegment | None:
        return next((r for r in self.road_segments if point_in_polygon(p, r.polygon)), None)

    def boundary_crossing(self, prev: Point, now: Point) -> Crossing | None:
        """The perimeter crossing between two consecutive positions, if the track changed sides."""
        was_in, is_in = self.inside_site(prev), self.inside_site(now)
        if was_in == is_in:
            return None
        n = len(self.boundary)
        hits = [
            pt
            for i in range(n)
            if (pt := segment_intersection(prev, now, self.boundary[i], self.boundary[(i + 1) % n])) is not None
        ]
        point = hits[0] if hits else now
        gate = next((g.id for g in self.gates if point_segment_distance(point, *g.opening) <= GATE_TOLERANCE_M), None)
        return Crossing(point=point, inbound=is_in, gate_id=gate)
