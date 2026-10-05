"""compound_spec.yaml -> USD stage: the same geometry the event rules use. Isaac venv (needs pxr).

    /mnt/weights/ai/isaac/IsaacLab/.venv/bin/python sim/compound/build_compound.py [--out data/compound/compound_v0.usda]

Collidable: ground, perimeter fence (with real gaps at the gates), buildings, gate panels (closed).
Visual only: zones as thin tinted patches, checkpoints as small posts, so the video shows what the rules mean.
The spec stays the single source of truth; this file never hard-codes a coordinate.
"""

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

from events.zones import Compound, Polygon  # noqa: E402

FENCE_H, FENCE_T = 2.2, 0.08
BUILDING_H = {"office": 7.0, "warehouse": 9.0, "generator_house": 4.0}
ZONE_COLOR = {"employee": (0.32, 0.34, 0.36), "restricted": (0.75, 0.55, 0.10), "high_risk": (0.70, 0.15, 0.12),
              "public": (0.30, 0.45, 0.30)}
ROAD_COLOR = (0.16, 0.16, 0.17)
PEDESTRIAN_COLOR = (0.20, 0.42, 0.55)


def box(stage, path, center, size, color, collide=True, yaw_deg=0.0):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    if yaw_deg:
        xf.AddRotateZOp().Set(yaw_deg)
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def rect_of(poly: Polygon):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def fence_segments(compound: Compound):
    """Perimeter edges with the gate openings cut out."""
    n = len(compound.boundary)
    for i in range(n):
        a, b = compound.boundary[i], compound.boundary[(i + 1) % n]
        length = math.dist(a, b)
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        cuts = []
        for g in compound.gates:
            ts = sorted(((p[0] - a[0]) * ux + (p[1] - a[1]) * uy) for p in g.opening)
            on_edge = all(abs((p[0] - a[0]) * uy - (p[1] - a[1]) * ux) < 1e-6 for p in g.opening)
            if on_edge and ts[1] > 0 and ts[0] < length:
                cuts.append((max(0.0, ts[0]), min(length, ts[1])))
        t = 0.0
        for c0, c1 in sorted(cuts) + [(length, length)]:
            if c0 - t > 1e-3:
                yield (a[0] + ux * t, a[1] + uy * t), (a[0] + ux * c0, a[1] + uy * c0)
            t = c1


def build(spec_path: Path, out: Path) -> Path:
    compound = Compound.load(spec_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/Compound")
    stage.SetDefaultPrim(world.GetPrim())

    x0, y0, x1, y1 = rect_of(compound.boundary)
    margin = 20.0
    box(stage, "/Compound/ground", ((x0 + x1) / 2, (y0 + y1) / 2, -0.05),
        (x1 - x0 + 2 * margin, y1 - y0 + 2 * margin, 0.1), (0.36, 0.38, 0.35))

    # zones: thin visual patches, higher priority drawn slightly higher so overlaps read correctly
    for z in sorted(compound.zones, key=lambda z: z.priority):
        if z.id == "yard":
            continue
        zx0, zy0, zx1, zy1 = rect_of(z.polygon)
        color = ROAD_COLOR if z.id.startswith("road") else PEDESTRIAN_COLOR if z.vehicles == "prohibited" and z.access == "employee" else ZONE_COLOR[z.access]
        box(stage, f"/Compound/zones/{z.id}", ((zx0 + zx1) / 2, (zy0 + zy1) / 2, 0.002 + 0.001 * z.priority),
            (zx1 - zx0, zy1 - zy0, 0.002), color, collide=False)

    for i, (a, b) in enumerate(fence_segments(compound)):
        length = math.dist(a, b)
        box(stage, f"/Compound/fence/segment_{i:02d}", ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, FENCE_H / 2),
            (length, FENCE_T, FENCE_H), (0.55, 0.57, 0.6), yaw_deg=math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))

    for g in compound.gates:  # closed gate panels: the scenario runner swings them open
        (ax, ay), (bx, by) = g.opening
        gate = UsdGeom.Xform.Define(stage, f"/Compound/gates/{g.id}")
        gate.AddTranslateOp().Set(Gf.Vec3d(ax, ay, 0))  # hinge at the first opening point
        gate.AddRotateZOp().Set(math.degrees(math.atan2(by - ay, bx - ax)))
        box(stage, f"/Compound/gates/{g.id}/panel", (math.dist((ax, ay), (bx, by)) / 2, 0, FENCE_H / 2),
            (math.dist((ax, ay), (bx, by)), FENCE_T * 1.5, FENCE_H), (0.85, 0.65, 0.1))

    for b in compound.spec.get("buildings", []):
        bx0, by0, bx1, by1 = b["rect"]
        h = BUILDING_H.get(b["id"], 6.0)
        box(stage, f"/Compound/buildings/{b['id']}", ((bx0 + bx1) / 2, (by0 + by1) / 2, h / 2), (bx1 - bx0, by1 - by0, h),
            (0.62, 0.6, 0.56))

    for cp_id, (cx, cy) in compound.route:
        post = UsdGeom.Cylinder.Define(stage, f"/Compound/checkpoints/{cp_id}")
        post.CreateRadiusAttr(0.12)
        post.CreateHeightAttr(0.6)
        post.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.8, 0.9)])
        UsdGeom.Xformable(post).AddTranslateOp().Set(Gf.Vec3d(cx, cy, 0.3))

    sun = UsdLux.DistantLight.Define(stage, "/Compound/sun")
    sun.CreateIntensityAttr(1200.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50, 0, 30))
    sky = UsdLux.DomeLight.Define(stage, "/Compound/sky")
    sky.CreateIntensityAttr(250.0)
    stage.GetRootLayer().customLayerData = {"source": str(spec_path.relative_to(REPO)), "generator": "sim/compound/build_compound.py"}
    stage.Save()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(REPO / "sim/compound/compound_spec.yaml"))
    ap.add_argument("--out", default=str(REPO / "data/compound/compound_v0.usda"))
    a = ap.parse_args()
    print(f"WROTE {build(Path(a.spec), Path(a.out))}")
