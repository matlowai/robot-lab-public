"""M1a: does Isaac's pretrained Spot walk our patrol reliably? Isaac venv, GPU 0.

    /mnt/weights/ai/isaac/IsaacLab/.venv/bin/python tools/spot_walk_test.py [--laps 10] [--side 10] [--video]

Spot patrols the corners of a square with the same PatrolMission used everywhere else. Pass = every lap
completes, no fall, no refused or failed leg. Writes report.json (+ walk.mp4 from a lit chase camera).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--laps", type=int, default=10)
ap.add_argument("--side", type=float, default=10.0)
ap.add_argument("--video", action="store_true")
ap.add_argument("--max-sim-s", type=float, default=900.0)
ap.add_argument("--out", default=str(REPO / "data/runs"))
args = ap.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 960, "height": 540})

import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

sys.path.insert(0, str(REPO))
from missions.patrol import PatrolMission  # noqa: E402
from robots.base import Capabilities  # noqa: E402
from robots.spot.isaac_adapter import IsaacSpotAdapter  # noqa: E402

PHYSICS_DT = 1 / 500
out = Path(args.out) / f"M1a-spot-walk-{time.strftime('%Y%m%d-%H%M%S')}"
out.mkdir(parents=True, exist_ok=True)

stage_utils.create_new_stage()
stage_utils.set_stage_up_axis("Z")
stage_utils.set_stage_units(meters_per_unit=1.0)
stage_utils.define_prim("/World/PhysicsScene", "PhysicsScene")
SimulationManager.set_physics_sim_device("cpu")
SimulationManager.set_physics_dt(PHYSICS_DT)
stage_utils.add_reference_to_stage(
    usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd", path="/World/ground"
)

s = args.side
corners = {"c1": (s, 0.0), "c2": (s, s), "c3": (0.0, s), "c4": (0.0, 0.0)}
spot = IsaacSpotAdapter("spot_0001", Capabilities.load(REPO / "robots/spot/capabilities.yaml"), "/World/spot",
                        corners, position=(0.0, 0.0, 0.8))
spot.spawn()
app.update()

frames = []
cam_xf = None
if args.video:
    import omni.replicator.core as rep
    from pxr import UsdLux

    stage = omni.usd.get_context().get_stage()
    sun = UsdLux.DistantLight.Define(stage, "/World/sun")
    sun.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50, 0, 30))
    cam = UsdGeom.Camera.Define(stage, "/World/chase_cam")
    cam.CreateFocalLengthAttr(16.0)
    cam_xf = UsdGeom.Xformable(cam).AddTransformOp()
    render_product = rep.create.render_product("/World/chase_cam", (960, 540))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([render_product])


def update_chase_cam():
    """Follow 3.5 m behind and 1.8 m above Spot, looking just ahead of it."""
    import math as _m
    x, y, yaw = spot.pose()
    eye = Gf.Vec3d(x - 3.5 * _m.cos(yaw), y - 3.5 * _m.sin(yaw), 1.8)
    target = Gf.Vec3d(x + 1.0 * _m.cos(yaw), y + 1.0 * _m.sin(yaw), 0.3)
    cam_xf.Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())


timeline = omni.timeline.get_timeline_interface()
timeline.play()
app.update()
spot.initialize()
app.update()

sim_t = 0.0


def on_step(dt, _ctx):
    global sim_t
    sim_t += dt
    spot.on_physics_step(dt)


cb = SimulationManager.register_callback(on_step, IsaacEvents.POST_PHYSICS_STEP)
route = [name for _ in range(args.laps) for name in corners]
patrol = PatrolMission(spot, route)
wall0, n_updates, min_z = time.time(), 0, 10.0
while not patrol.done and not spot.fallen and sim_t < args.max_sim_s:
    patrol.tick(sim_t)
    app.update()
    n_updates += 1
    min_z = min(min_z, spot._base_z())
    if args.video:
        update_chase_cam()
    if args.video and n_updates % 6 == 0:
        data = rgb.get_data()
        if data is not None and getattr(data, "size", 0):
            frames.append(np.asarray(data)[:, :, :3].copy())
wall = time.time() - wall0
SimulationManager.deregister_callback(cb)
timeline.stop()

legs = [v for v in patrol.visits]
ok_legs = [v for v in legs if v.outcome == "succeeded"]
leg_times = [b.t - a.t for a, b in zip([None] + ok_legs[:-1], ok_legs) if a is not None]
report = {
    "laps_requested": args.laps, "side_m": s, "legs_total": len(route), "legs_succeeded": len(ok_legs),
    "fallen": spot.fallen, "min_base_z_m": round(min_z, 3), "sim_s": round(sim_t, 1), "wall_s": round(wall, 1),
    "real_time_factor": round(sim_t / wall, 2) if wall else None,
    "leg_time_s": {"median": round(float(np.median(leg_times)), 2), "max": round(float(max(leg_times)), 2)} if leg_times else None,
    "passed": (not spot.fallen) and len(ok_legs) == len(route),
}
if frames:
    video = out / "walk.mp4"
    h, w, _ = frames[0].shape
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                           "-r", "10", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE)
    for f in frames:
        ff.stdin.write(f.tobytes())
    ff.stdin.close()
    ff.wait()
    report["video"] = str(video)
    report["video_frames"] = len(frames)
(out / "report.json").write_text(json.dumps(report, indent=2))
print("REPORT " + json.dumps(report))
sys.stdout.flush()
os._exit(0 if report["passed"] else 1)
