"""Mesh-point extents (m) of props whose BBoxCache came back empty."""
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR  # noqa: E402

for name, rel in (("mug", "Objects/Mug/mug.usd"), ("toy_truck", "Objects/ToyTruck/toy_truck.usd"),
                  ("box", "Objects/Box/box.usd")):
    stage = Usd.Stage.Open(f"{ISAACLAB_NUCLEUS_DIR}/{rel}", Usd.Stage.LoadAll)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    xf_cache = UsdGeom.XformCache()
    pts_all, kinds = [], {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        kinds[prim.GetTypeName()] = kinds.get(prim.GetTypeName(), 0) + 1
        if prim.IsA(UsdGeom.Mesh):
            pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if pts:
                m = np.array(xf_cache.GetLocalToWorldTransform(prim))
                p = np.c_[np.array(pts), np.ones(len(pts))] @ m
                pts_all.append(p[:, :3])
    if pts_all:
        p = np.concatenate(pts_all) * mpu
        print(f"MESH {name}: size_m = {np.round(p.max(0) - p.min(0), 3)}  min_z={p[:,2].min():.3f}")
    else:
        print(f"MESH {name}: no mesh points; prim types = {kinds}")
app.close()
