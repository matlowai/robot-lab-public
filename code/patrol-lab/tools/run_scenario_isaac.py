"""M1: a scenario in Isaac Sim. Spot patrols the compound while staged incidents play out; the rules read facts
back from the simulated stage (oracle path) and the oracle scorer checks them. Isaac venv, GPU 0.

    /mnt/weights/ai/isaac/IsaacLab/.venv/bin/python tools/run_scenario_isaac.py scenarios/library/SCN-M0-001.yaml \
        [--video] [--max-sim-s 900]

Writes into data/runs/M1-<scenario>-<stamp>/: report.json, events.jsonl, incidents.json, timeline.txt,
trajectory.jsonl (Spot pose, command and navigation state at 10 Hz) and,
with --video, patrol.mp4 (left: whole-yard overview; right: Spot chase camera).

Truth path: scenario keyframes move actor prims -> each 0.1 s the frame is rebuilt FROM THE STAGE (prim world
positions, gate hinge angles, object visibility) -> EventEngine -> incidents -> score vs staged incidents.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("scenario")
ap.add_argument("--video", action="store_true")
ap.add_argument("--avoid", action="store_true", help="shorthand for --controller heuristic")
ap.add_argument("--record-head", default=None, metavar="DIR",
                help="record Spot's head camera (256x256, 15 Hz) + (move, strafe, turn) commands for FLUX training")
ap.add_argument("--controller", default=None,
                help="avoidance contender: control | heuristic | heuristic_v1 | rl:<checkpoint> (default: none = "
                     "straight-line walking, i.e. the control)")
ap.add_argument("--max-sim-s", type=float, default=900.0)
ap.add_argument("--compound-usd", default=str(REPO / "data/compound/compound_v0.usda"))
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
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

sys.path.insert(0, str(REPO))
from events.incidents import to_incident  # noqa: E402
from events.rules import EventEngine, _clock  # noqa: E402
from events.scorer import score  # noqa: E402
from events.zones import Compound  # noqa: E402
from missions.patrol import PatrolMission  # noqa: E402
from robots.base import Capabilities  # noqa: E402
from robots.spot.isaac_adapter import IsaacSpotAdapter  # noqa: E402
from scenarios.model import Scenario  # noqa: E402
from scenarios.oracle import _actor_at  # noqa: E402
from benchmarks.avoidance import pedestrians  # noqa: E402

PHYSICS_DT, FRAME_DT, GATE_OPEN_DEG = 1 / 500, 0.1, 85.0
PARKED = Gf.Vec3d(0, 0, -50)  # despawned actors wait underground: invisible AND out of the lidar's way
PERSON_R, SPOT_R, NEAR_MISS_M = 0.30, 0.55, 0.30
scenario = Scenario.load(args.scenario)
compound = Compound.load(scenario.compound)
out = Path(args.out) / f"M1-{scenario.id}-{(args.controller or ('heuristic' if args.avoid else 'control')).split(':')[0]}-{time.strftime('%Y%m%d-%H%M%S')}"
out.mkdir(parents=True, exist_ok=True)

stage_utils.create_new_stage()
stage_utils.set_stage_up_axis("Z")
stage_utils.set_stage_units(meters_per_unit=1.0)
stage_utils.define_prim("/World/PhysicsScene", "PhysicsScene")
SimulationManager.set_physics_sim_device("cpu")
SimulationManager.set_physics_dt(PHYSICS_DT)
stage_utils.add_reference_to_stage(usd_path=args.compound_usd, path="/World/compound")
stage = omni.usd.get_context().get_stage()

# --- actors: visual stand-ins driven by the scenario (M3 swaps in Replicator Agent characters) ---------------
ACTOR_SHAPE = {"person": ((0.5, 0.5, 1.7), (0.95, 0.25, 0.2)), "vehicle": ((4.5, 1.9, 1.5), (0.15, 0.35, 0.95)),
               "object": ((1.2, 1.0, 0.25), (0.95, 0.65, 0.1))}
actor_ops, yaw_ops, beacons = {}, {}, {}


def make_actor(aid: str, cls: str):
    (sx, sy, sz), color = ACTOR_SHAPE[cls]
    xf = UsdGeom.Xform.Define(stage, f"/World/actors/{aid}")
    t_op = xf.AddTranslateOp()
    yaw_ops[aid] = xf.AddRotateZOp()
    cube = UsdGeom.Cube.Define(stage, f"/World/actors/{aid}/body")
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(0, 0, sz / 2))
    UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
    # actors are kinematic bodies: the lidar sees them and Spot can physically bump into them
    rb = UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    t_op.Set(PARKED)
    if cls == "person":  # the book (visual only): they are reading, not looking where they walk
        book = UsdGeom.Cube.Define(stage, f"/World/actors/{aid}/book")
        book.CreateSizeAttr(1.0)
        book.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.95, 0.85)])
        UsdGeom.Xformable(book).AddTranslateOp().Set(Gf.Vec3d(0.35, 0, 1.25))
        UsdGeom.Xformable(book).AddScaleOp().Set(Gf.Vec3f(0.18, 0.26, 0.04))
    # overview beacon (visual only; truth is read from the actor Xform, never from this)
    beacon = UsdGeom.Cube.Define(stage, f"/World/actors/{aid}/beacon")
    beacon.CreateSizeAttr(1.0)
    beacon.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdGeom.Xformable(beacon).AddTranslateOp().Set(Gf.Vec3d(0, 0, sz + 2.5))
    UsdGeom.Xformable(beacon).AddScaleOp().Set(Gf.Vec3f(1.2, 1.2, 3.0))
    beacons[aid] = beacon
    if args.record_head:  # debug-only markers never reach the learner's camera (a locked hunter's beacon
        UsdGeom.Imageable(beacon).MakeInvisible()  # colour would leak its intent into FLUX training data)
    img = UsdGeom.Imageable(xf)
    img.MakeInvisible()
    actor_ops[aid] = (t_op, img, cls)


for a in scenario.actors:
    make_actor(a.id, a.cls)
# live crowd (benchmarks/avoidance/pedestrians.py): people that react to where Spot actually is. Stepped at the
# 10 Hz frame rate with Spot's pose; hunters' beacons turn magenta when locked on and yellow once committed.
crowd = pedestrians.scenario_crowd(scenario)
crowd_index = {pid: i for i, pid in enumerate(crowd.ids)} if crowd else {}
crowd_beacon_state = {}
BEACON_COLOR = {pedestrians.LOCKED: (1.0, 0.1, 0.9), pedestrians.COMMITTED: (1.0, 0.85, 0.1)}
for pid in crowd_index:
    make_actor(pid, "person")
crowd_log = {"t": [], "pos": [], "state": []}
for o in scenario.objects:
    make_actor(o.id, "object")
gate_ops = {g.id: UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/compound/gates/{g.id}")).GetOrderedXformOps()[1]
            for g in compound.gates}  # [translate, rotateZ]: the hinge
gate_base = {gid: op.Get() for gid, op in gate_ops.items()}
gate_state = {g.id: g.normal_state for g in compound.gates}


def apply_scenario(t: float) -> None:
    for a in scenario.actors:
        t_op, img, _ = actor_ops[a.id]
        at = _actor_at(a, t)
        if at is None:
            img.MakeInvisible()
            t_op.Set(PARKED)
        else:
            (x, y), _action = at
            t_op.Set(Gf.Vec3d(x, y, 0))
            img.MakeVisible()
            ahead = _actor_at(a, t + 0.5)  # face the direction of travel (the book points forward)
            if ahead is not None and math.dist(ahead[0], (x, y)) > 0.05:
                yaw_ops[a.id].Set(math.degrees(math.atan2(ahead[0][1] - y, ahead[0][0] - x)))
    for pid, i in crowd_index.items():  # positions as of the last crowd step (<= 0.1 s old)
        t_op, img, _ = actor_ops[pid]
        t_op.Set(Gf.Vec3d(float(crowd.pos[i][0]), float(crowd.pos[i][1]), 0))
        img.MakeVisible()
        yaw_ops[pid].Set(math.degrees(float(crowd.heading[i])))
        st = int(crowd.state[i])
        if not args.record_head and crowd_beacon_state.get(pid) != st:
            crowd_beacon_state[pid] = st
            beacons[pid].GetDisplayColorAttr().Set([Gf.Vec3f(*BEACON_COLOR.get(st, ACTOR_SHAPE["person"][1]))])
    for o in scenario.objects:
        t_op, img, _ = actor_ops[o.id]
        present = o.spawn_t <= t and (o.remove_t is None or t < o.remove_t)
        t_op.Set(Gf.Vec3d(o.pos[0], o.pos[1], 0) if present else PARKED)
        img.MakeVisible() if present else img.MakeInvisible()
    for c in scenario.asset_changes:
        if c.t <= t:
            gate_state[c.id] = c.state
    for gid, op in gate_ops.items():
        op.Set(gate_base[gid] + (GATE_OPEN_DEG if gate_state[gid] == "open" else 0.0))


def frame_from_stage(t: float) -> dict:
    """Rebuild the observation from what is actually in the stage, not from the scenario file."""
    cache = UsdGeom.XformCache()
    tracks = []
    kinds = {a.id: a for a in scenario.actors}
    for aid, (_, img, cls) in actor_ops.items():
        prim = img.GetPrim()
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        p = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
        track = {"id": aid, "class": cls, "position": [round(p[0], 3), round(p[1], 3), 0.0]}
        if aid in crowd_index:
            standing = crowd.state[crowd_index[aid]] == pedestrians.PAUSE
            track["action"], track["credential"] = ("standing" if standing else "walking"), "authorized"
        elif cls == "person":
            at = _actor_at(kinds[aid], t)
            if at is None:  # stage and scenario disagree about presence: a harness bug, never silently patched
                raise RuntimeError(f"{aid} visible on stage at t={t} but absent in the scenario")
            track["action"] = at[1]  # actions/credentials: scenario truth (no perception yet)
            track["credential"] = kinds[aid].credential
        tracks.append(track)
    assets = []
    for gid, op in gate_ops.items():
        opened = abs(op.Get() - gate_base[gid]) > GATE_OPEN_DEG / 2
        assets.append({"id": gid, "type": "gate", "state": "open" if opened else "closed"})
    return {"t": round(t, 3), "source": "oracle", "tracks": tracks, "assets": assets}


# --- Spot -------------------------------------------------------------------------------------------------------
places = {cp: pos for cp, pos in compound.route} | {"charging_station": compound.charging_station}
cx, cy = compound.charging_station
controller = args.controller or ("heuristic" if args.avoid else None)
lidar = avoider = None
if controller and controller != "control":
    from benchmarks.avoidance import sim2d
    from robots.spot.lidar import PhysxLidar

    if controller.startswith("rl:"):
        from benchmarks.avoidance.rl.controller import register_rl

        register_rl(controller[3:], velocity_source="pose")  # the walking policy doesn't track commands exactly
        name = "rl"
    else:
        name = controller
    avoider = sim2d.CONTROLLERS[name]()
    lidar = PhysxLidar(self_prefix="/World/spot")
args.avoid = avoider is not None
spot = IsaacSpotAdapter("spot_0001", Capabilities.load(REPO / "robots/spot/capabilities.yaml"), "/World/spot",
                        places, position=(cx, cy, 0.8), lidar=lidar, avoider=avoider)
spot.spawn()
app.update()

spot_beacon = UsdGeom.Cone.Define(stage, "/World/spot_beacon")  # visual only: where Spot is, in the overview
spot_beacon.CreateRadiusAttr(1.2)
spot_beacon.CreateHeightAttr(2.4)
spot_beacon.CreateAxisAttr("Z")
spot_beacon.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 1.0)])
spot_beacon_op = UsdGeom.Xformable(spot_beacon).AddTranslateOp()
spot_beacon_op.Set(Gf.Vec3d(cx, cy, 4.0))
if args.record_head:
    UsdGeom.Imageable(spot_beacon).MakeInvisible()

frames_rgb, cams = [], {}
if args.video:
    import omni.replicator.core as rep

    def camera(name, focal):
        cam = UsdGeom.Camera.Define(stage, f"/World/{name}")
        cam.CreateFocalLengthAttr(focal)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
        op = UsdGeom.Xformable(cam).AddTransformOp()
        rp = rep.create.render_product(f"/World/{name}", (960, 540))
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach([rp])
        return op, ann

    def look(op, eye, target):
        op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())

    cams["overview"] = camera("overview_cam", 18.0)
    w, h = compound.spec["size_m"]
    look(cams["overview"][0], (w / 2, -h * 0.55, h * 0.95), (w / 2, h * 0.45, 0))
    cams["chase"] = camera("chase_cam", 16.0)

head = None
if args.record_head:
    import omni.replicator.core as rep

    head_cam = UsdGeom.Camera.Define(stage, "/World/head_cam")
    head_cam.CreateFocalLengthAttr(12.0)  # ~ 90 deg horizontal on the default 20.955 mm aperture
    head_cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))
    head_op = UsdGeom.Xformable(head_cam).AddTransformOp()
    head_rp = rep.create.render_product("/World/head_cam", (256, 256))
    head_ann = rep.AnnotatorRegistry.get_annotator("rgb")
    head_ann.attach([head_rp])
    head = {"frames": [], "action": [], "t": [], "pose": [], "status": []}


def place_head_cam():
    """A stabilised head camera: 0.62 m up, 0.45 m ahead of the body centre, looking forward and slightly down."""
    x, y, yaw = spot.pose()
    c, s_ = math.cos(yaw), math.sin(yaw)
    eye = Gf.Vec3d(x + 0.45 * c, y + 0.45 * s_, 0.62)
    target = Gf.Vec3d(x + 5.45 * c, y + 5.45 * s_, 0.25)
    head_op.Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())


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

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

_FONT = ImageFont.load_default(size=18)
_SMALL = ImageFont.load_default(size=15)


def hud(rgb: np.ndarray, t: float) -> np.ndarray:
    """Burn the operator's view into the frame: clock, task, coverage, and events as they fire."""
    img = Image.fromarray(np.ascontiguousarray(rgb))
    d = ImageDraw.Draw(img, "RGBA")
    nav = spot.navigation_status()
    reached = sum(1 for v in patrol.visits if v.outcome == "succeeded" and v.place in patrol.route)
    d.rectangle((0, 0, 959, 58), fill=(10, 14, 18, 190))
    d.text((12, 6), f"{scenario.id}  {_clock(t)}   ORACLE PATH", font=_FONT, fill=(235, 240, 245))
    planner = f"   {controller or 'control'}: {spot.last_status}" if spot.last_status else ""
    closest = min(clearance.values(), default=None)
    d.text((12, 32), f"Spot -> {nav.target or '-'} ({nav.state})   checkpoints {reached}/{len(patrol.route)}{planner}"
           + (f"   closest person {closest:.2f} m" if closest is not None and closest < 5 else "")
           + ("   FALLEN" if spot.fallen else ""), font=_FONT, fill=(120, 230, 255))
    if crowd and crowd.locked_on():
        tags = [f"{pid} {'COMMITTED' if crowd.state[crowd_index[pid]] == pedestrians.COMMITTED else 'locked'}"
                for pid in crowd.locked_on()]
        d.rectangle((0, 58, 959, 88), fill=(90, 10, 80, 200))
        d.text((12, 63), "TARGET LOCKED:  " + ",  ".join(tags), font=_FONT, fill=(255, 190, 250))
    recent = [e for e in events if e.t_detected <= t][-5:]
    if recent:
        top = 539 - 24 * len(recent) - 12
        d.rectangle((0, top, 959, 539), fill=(10, 14, 18, 190))
        for i, e in enumerate(recent):
            d.text((12, top + 8 + 24 * i), f"{_clock(e.t_detected)}  {e.type}  {e.subject}  (onset {_clock(e.t)})",
                   font=_SMALL, fill=(255, 200, 90))
    d.rectangle((960, 0, 1919, 30), fill=(10, 14, 18, 190))
    d.text((972, 6), "CHASE CAM", font=_SMALL, fill=(235, 240, 245))
    return np.asarray(img)



def _die(exc_type, exc, tb):  # Kit swallows uncaught exceptions and exits 0; a crashed run must fail
    import traceback
    traceback.print_exception(exc_type, exc, tb)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(2)


sys.excepthook = _die
engine = EventEngine(compound, scenario.config)
patrol = PatrolMission(spot, list(compound.route_stops), return_to="charging_station")
events, trajectory, next_frame, n_updates, wall0 = [], [], 0.0, 0, time.time()
clearance: dict[str, float] = {}  # live Spot-to-person clearance (m), for the HUD
min_clearance: dict[str, float] = {}  # worst clearance per person over the run
end_t = max(scenario.duration_s, 0.0)
while sim_t < args.max_sim_s and not spot.fallen and (not patrol.done or sim_t < end_t):
    apply_scenario(sim_t)
    patrol.tick(sim_t)
    if head is not None:
        place_head_cam()
    app.update()
    n_updates += 1
    if head is not None and n_updates % 4 == 0:  # ~15 Hz of the ~60 Hz update rate
        img = head_ann.get_data()
        if img is not None and getattr(img, "size", 0):
            vx_c, vy_c, wz_c = spot.velocity()
            head["frames"].append(np.asarray(img)[:, :, :3].copy())
            head["action"].append((vx_c / 1.0, vy_c / 0.5, wz_c / 1.0))  # move, strafe, turn in [-1, 1]
            head["t"].append(round(sim_t, 3))
            head["pose"].append(spot.pose())
            head["status"].append(spot.last_status or "")
    while next_frame <= sim_t:  # rules at 10 Hz of sim time; stage set to exactly the frame's time first
        if crowd:
            crowd.step(next_frame, spot.pose()[:2], SPOT_R)
            crowd_log["t"].append(round(next_frame, 3))
            crowd_log["pos"].append(crowd.pos.astype(np.float32).copy())
            crowd_log["state"].append(crowd.state.astype(np.int8).copy())
        apply_scenario(next_frame)
        frame = frame_from_stage(next_frame)
        frame["robots"] = [spot.state()]
        trajectory.append({"t": frame["t"], **{k: frame["robots"][0][k] for k in ("pose", "velocity", "navigation")},
                           **({"planner": spot.last_status} if spot.last_status else {})})
        sx, sy = frame["robots"][0]["pose"][:2]
        clearance = {tr["id"]: math.dist((sx, sy), tr["position"][:2]) - PERSON_R - SPOT_R
                     for tr in frame["tracks"] if tr["class"] == "person"}
        for pid, c in clearance.items():
            min_clearance[pid] = min(min_clearance.get(pid, math.inf), c)
        events += engine.process(frame)
        next_frame += FRAME_DT
    apply_scenario(sim_t)  # back to "now" for rendering
    if args.video:
        x, y, yaw = spot.pose()
        spot_beacon_op.Set(Gf.Vec3d(x, y, 4.0))
        look(cams["chase"][0], (x - 3.5 * math.cos(yaw), y - 3.5 * math.sin(yaw), 1.8),
             (x + math.cos(yaw), y + math.sin(yaw), 0.3))
        if n_updates % 6 == 0:
            views = [cams[k][1].get_data() for k in ("overview", "chase")]
            if all(v is not None and getattr(v, "size", 0) for v in views):
                frames_rgb.append(hud(np.concatenate([np.asarray(v)[:, :, :3] for v in views], axis=1), sim_t))
wall = time.time() - wall0
SimulationManager.deregister_callback(cb)
timeline.stop()

report_score = score(events, list(scenario.staged))
incidents = [to_incident(e, i + 1) for i, e in enumerate(events)]
visited = [(v.place, v.t) for v in patrol.visits if v.outcome == "succeeded" and v.place in patrol.route]
staged = {(s.type, s.subject) for s in scenario.staged}
lines = [(e.t_detected, f"{_clock(e.t_detected)}  {e.type:<27} {e.subject:<16} onset {_clock(e.t)}"
          + ("" if (e.type, e.subject) in staged else "   <-- NOT STAGED")) for e in events]
lines += [(t, f"{_clock(t)}  checkpoint {cp}") for cp, t in visited]
timeline_txt = "\n".join(line for _, line in sorted(lines))
report = {
    "scenario": scenario.id, **report_score.summary(sim_t), "coverage": patrol.coverage,
    "patrol_complete": patrol.done, "fallen": spot.fallen, "sim_s": round(sim_t, 1), "wall_s": round(wall, 1),
    "real_time_factor": round(sim_t / wall, 2) if wall else None,
    "misses": [f"{m.type} {m.subject} @{m.t}" for m in report_score.misses],
    "false_alarms": [f"{e.type} {e.subject} @{e.t}" for e in report_score.false_alarms],
}
near_misses = sorted(pid for pid, c in min_clearance.items() if c < NEAR_MISS_M)
report["avoidance"] = {"enabled": args.avoid, "controller": controller or "control", "near_miss_threshold_m": NEAR_MISS_M,
                       "min_clearance_m": {pid: round(c, 3) for pid, c in sorted(min_clearance.items())},
                       "near_misses": near_misses, "contacts": sorted(p for p, c in min_clearance.items() if c < 0)}
if crowd:
    crowd.finish()
    lk = crowd.lock_summary()
    hunters_hit = {r["hunter"] for r in lk["records"] if r["hit"]}
    lk["ambient_contacts"] = sorted(p for p, c in min_clearance.items() if c < 0 and p not in hunters_hit)
    report["live_crowd"] = {"tier": scenario.raw["crowd"].get("tier"), "people": len(crowd.ids), **lk}
    np.savez_compressed(out / "crowd.npz", t=np.asarray(crowd_log["t"], np.float32), pos=np.stack(crowd_log["pos"]),
                        state=np.stack(crowd_log["state"]), ids=np.asarray(crowd.ids),
                        kind=np.asarray([pedestrians.KIND_NAMES[k] for k in crowd.kind]))
report["passed"] = (report_score.perfect and patrol.done and patrol.coverage == 1.0 and not spot.fallen
                    and not (args.avoid and near_misses))
(out / "events.jsonl").write_text("".join(json.dumps(e.to_dict()) + "\n" for e in events))
(out / "incidents.json").write_text(json.dumps(incidents, indent=2))
(out / "timeline.txt").write_text(timeline_txt + "\n")
(out / "trajectory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trajectory))
report["escalations"] = [f"{e.place} after {e.attempts} attempts ({e.last_outcome})" for e in patrol.escalations]
report["mission_status"] = patrol.status
if frames_rgb:
    hh, ww, _ = frames_rgb[0].shape
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ww}x{hh}",
                           "-r", "10", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / "patrol.mp4")], stdin=subprocess.PIPE)
    for f in frames_rgb:
        ff.stdin.write(f.tobytes())
    ff.stdin.close()
    ff.wait()
    report["video"] = str(out / "patrol.mp4")
if head is not None and head["frames"]:
    rec_dir = Path(args.record_head)
    rec_dir.mkdir(parents=True, exist_ok=True)
    rec = rec_dir / f"{scenario.id}_{controller or 'control'}.npz"
    np.savez_compressed(rec, frames=np.stack(head["frames"]).astype(np.uint8),
                        action=np.asarray(head["action"], np.float32), t=np.asarray(head["t"], np.float32),
                        pose=np.asarray(head["pose"], np.float32), status=np.asarray(head["status"]),
                        meta=json.dumps({"scenario": scenario.id, "controller": controller or "control", "fps": 15,
                                         "action_names": ["move", "strafe", "turn"],
                                         "action_scale": [1.0, 0.5, 1.0], "camera": "stabilised head 256x256"}))
    report["head_recording"] = {"file": str(rec), "frames": len(head["frames"])}
(out / "report.json").write_text(json.dumps(report, indent=2))
print(timeline_txt)
print("REPORT " + json.dumps(report))
sys.stdout.flush()
os._exit(0 if report["passed"] else 1)
