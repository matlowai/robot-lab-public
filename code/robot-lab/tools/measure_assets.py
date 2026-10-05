"""Bounding boxes of candidate props (world units = m) - headless Kit."""
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Usd, UsdGeom  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR  # noqa: E402

PROPS = {
    "mug": f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd",
    "toy_truck": f"{ISAACLAB_NUCLEUS_DIR}/Objects/ToyTruck/toy_truck.usd",
    "box": f"{ISAACLAB_NUCLEUS_DIR}/Objects/Box/box.usd",
    "bowl_yellow": f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bowl_yellow.usd",
    "bin_blue": f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bin_blue.usd",
    "blue_block": f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
    "so101": None,
}
for name, path in PROPS.items():
    if path is None:
        continue
    stage = Usd.Stage.Open(path)
    if stage is None:
        print(f"{name}: FAILED to open {path}")
        continue
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    size = (rng.GetMax() - rng.GetMin()) * mpu
    print(f"MEASURE {name}: size_m = ({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})  metersPerUnit={mpu}  "
          f"upAxis={UsdGeom.GetStageUpAxis(stage)}  default_prim={stage.GetDefaultPrim().GetPath()}")
app.close()
