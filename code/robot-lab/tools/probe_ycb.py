"""Which YCB / prop assets exist on the Isaac 6.1 asset server, and their sizes (live-stage bbox after spawning)."""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, device="cuda:0").app
import numpy as np  # noqa: E402
import omni.client  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
for sub in ("Props/YCB/Axis_Aligned_Physics", "Props/YCB/Axis_Aligned", "Props/YCB"):
    res, entries = omni.client.list(f"{ISAAC_NUCLEUS_DIR}/{sub}")
    names = sorted(e.relative_path for e in entries) if entries else []
    print(f"LIST {sub}: {res} {names[:40]}")
    if names:
        base = sub
        break
for n in names:
    if not n.endswith(".usd"):
        continue
    st = Usd.Stage.Open(f"{ISAAC_NUCLEUS_DIR}/{base}/{n}", Usd.Stage.LoadAll)
    mpu = UsdGeom.GetStageMetersPerUnit(st)
    pts = []
    cache = UsdGeom.XformCache()
    for prim in Usd.PrimRange(st.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh):
            p = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if p:
                m = np.array(cache.GetLocalToWorldTransform(prim))
                pts.append((np.c_[np.array(p), np.ones(len(p))] @ m)[:, :3])
    if pts:
        P = np.concatenate(pts) * mpu
        print(f"YCB {n}: size_m={np.round(P.max(0)-P.min(0),3)} mpu={mpu}")
app.close()
