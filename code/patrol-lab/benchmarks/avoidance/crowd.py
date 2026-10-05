"""Seeded crowd scenarios for the avoidance bake-off: readers who walk straight lines on Spot's patrol legs
and never react to it. The same seed always gives the same crowd, for every contender.

Timing comes from geometry, not from any contender: each leg takes ~length / 1.0 m/s plus ~1 s per radian of
turning (a fit to the deterministic base patrol: 11.4 s per 10 m leg including turns). Walks are long, so
meetings still happen when a contender's avoidance shifts its timing. Everyone is authorized: nothing should
raise an incident, so any event is a false alarm.

    uv run python -m benchmarks.avoidance.crowd --seeds 1-5 --out data/scenarios/crowd
"""

import argparse
import math
import random
from pathlib import Path

import yaml

from events.zones import Compound

REPO = Path(__file__).resolve().parents[2]
SPEED_MPS, TURN_S_PER_RAD = 1.0, 1.0
KINDS = ("head_on", "pair_head_on", "standing", "crossing")


def legs(compound: Compound) -> list[dict]:
    """Patrol legs (charging -> checkpoints -> charging) with nominal start/end times."""
    stops = [("charging_station", compound.charging_station)] + list(compound.route) + [("charging_station", compound.charging_station)]
    out, t, heading = [], 0.0, 0.0
    for (a_id, a), (b_id, b) in zip(stops, stops[1:]):
        h = math.atan2(b[1] - a[1], b[0] - a[0])
        turn = abs((h - heading + math.pi) % (2 * math.pi) - math.pi)
        length = math.dist(a, b)
        start = t + turn * TURN_S_PER_RAD
        out.append({"from": a_id, "to": b_id, "a": a, "b": b, "length": length, "t0": start, "t1": start + length / SPEED_MPS})
        t, heading = out[-1]["t1"], h
    return out


def _blocked(compound: Compound, p) -> bool:
    if not compound.inside_site(p):
        return True
    for b in compound.spec.get("buildings", []):
        x0, y0, x1, y1 = b["rect"]
        if x0 - 0.8 <= p[0] <= x1 + 0.8 and y0 - 0.8 <= p[1] <= y1 + 0.8:
            return True
    return False


def _path_ok(compound: Compound, p0, p1, samples: int = 25) -> bool:
    return all(not _blocked(compound, (p0[0] + (p1[0] - p0[0]) * k / samples, p0[1] + (p1[1] - p0[1]) * k / samples))
               for k in range(samples + 1))


def _person(pid: str, keyframes) -> dict:
    return {"id": pid, "class": "person", "credential": "authorized",
            "keyframes": [{"t": round(t, 2), "pos": [round(x, 2), round(y, 2)], "action": act} for t, (x, y), act in keyframes]}


def make_event(kind: str, leg: dict, rng: random.Random, next_id, compound: Compound) -> list[dict] | None:
    (ax, ay), (bx, by) = leg["a"], leg["b"]
    L = leg["length"]
    if L < 12:
        return None
    ux, uy = (bx - ax) / L, (by - ay) / L
    nx, ny = -uy, ux
    at = lambda s, lat=0.0: (ax + ux * s + nx * lat, ay + uy * s + ny * lat)  # noqa: E731
    frac = rng.uniform(0.35, 0.65)
    meet_s, meet_t = frac * L, leg["t0"] + frac * L / SPEED_MPS  # where/when Spot nominally is at that point
    if kind in ("head_on", "pair_head_on"):
        v = rng.uniform(0.9, 1.4)
        lat = rng.uniform(-0.3, 0.3)
        # walk the leg backwards so the reader is at the meeting point at the meeting time
        s0, s1 = min(L - 0.5, meet_s + v * (meet_t - leg["t0"] + 4)), max(0.5, meet_s - v * 12)
        t0 = meet_t - (s0 - meet_s) / v
        offsets = (lat,) if kind == "head_on" else (lat - 0.6, lat + 0.6)
        people = []
        for off in offsets:
            p0, p1 = at(s0, off), at(s1, off)
            people.append((p0, p1, t0, t0 + math.dist(p0, p1) / v))
    elif kind == "standing":
        p = at(meet_s, rng.uniform(-0.4, 0.4))
        people = [(p, p, max(0.0, leg["t0"] - 20), leg["t1"] + 20)]
    else:  # crossing: perpendicular, timed to reach the leg as Spot does (plus jitter)
        v = rng.uniform(0.9, 1.4)
        side = rng.choice((-1, 1))
        half = rng.uniform(8, 12)
        p0, p1 = at(meet_s, side * half), at(meet_s, -side * half)
        t_mid = meet_t + rng.uniform(-2.0, 2.0)
        people = [(p0, p1, t_mid - half / v, t_mid + half / v)]
    actors = []
    for p0, p1, t0, t1 in people:
        if t0 < 0 or not _path_ok(compound, p0, p1):
            return None
        act = "standing" if p0 == p1 else "walking"
        actors.append(_person(next_id(), [(t0, p0, act), (t1, p1, act)]))
    return actors



def generate(seed: int, compound_path: Path = REPO / "sim/compound/compound_spec.yaml", n_events=(3, 6)) -> dict:
    compound = Compound.load(compound_path)
    rng = random.Random(seed)
    all_legs = legs(compound)
    counter = iter(range(1, 10_000))
    next_id = lambda: f"person_{next(counter):04d}"  # noqa: E731
    actors, kinds_used, tries = [], [], 0
    target = rng.randint(*n_events)
    used_legs = set()
    while len(kinds_used) < target and tries < 200:
        tries += 1
        i = rng.randrange(len(all_legs))
        if i in used_legs:
            continue
        kind = rng.choice(KINDS)
        event = make_event(kind, all_legs[i], rng, next_id, compound)
        if event:
            actors += event
            kinds_used.append({"kind": kind, "leg": f"{all_legs[i]['from']}->{all_legs[i]['to']}", "people": [a["id"] for a in event]})
            used_legs.add(i)
    return {
        "id": f"SCN-AV-{seed:03d}",
        "name": f"crowd_seed_{seed}",
        "compound": "sim/compound/compound_spec.yaml",
        "duration_s": round(all_legs[-1]["t1"] + 20, 1),
        "rate_hz": 10,
        "config": {"gate_open_s": 20, "fall_confirm_s": 2, "obstruction_s": 3},
        "actors": actors,
        "staged_incidents": [],
        "_crowd": {"seed": seed, "events": kinds_used},  # provenance for reports; stripped before schema checks
    }


LIVE_TIERS = {
    "base": {},
    # faster hunters that commit later: less time between "committed" and contact
    "hard": {"hunter_speed": [1.3, 1.8], "commit_m": [1.8, 3.0], "n_hunters": 18, "max_locks": 3},
}


def generate_live(seed: int, tier: str = "base") -> dict:
    """A live-crowd scenario (benchmarks/avoidance/pedestrians.py): ~50 closed-loop people, hunters posted on every
    leg. No keyframed actors; the crowd is rebuilt from (seed, tier config) by every consumer."""
    config = LIVE_TIERS[tier]
    tag = "" if tier == "base" else f"{tier.upper()}-"
    return {
        "id": f"SCN-LC-{tag}{seed:03d}",
        "name": f"live_crowd_{tier}_seed_{seed}",
        "compound": "sim/compound/compound_spec.yaml",
        "duration_s": 0.0,  # the run ends when the patrol does (no parked-robot pile-ups)
        "rate_hz": 10,
        "config": {"gate_open_s": 20, "fall_confirm_s": 2, "obstruction_s": 3},
        "actors": [],
        "staged_incidents": [],
        "crowd": {"model": "live-v1", "tier": tier, "seed": seed, "config": config},
        "_crowd": {"seed": seed, "events": [{"kind": "live-v1"}]},
    }


def to_scenario_file(doc: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    body = {k: v for k, v in doc.items() if not k.startswith("_")}
    path = out_dir / f"{doc['id']}.yaml"
    path.write_text(f"# generated by benchmarks/avoidance/crowd.py, seed {doc['_crowd']['seed']}: "
                    + ", ".join(f"{e['kind']} on {e['leg']}" if "leg" in e else e["kind"] for e in doc["_crowd"]["events"]) + "\n"
                    + yaml.safe_dump(body, sort_keys=False))
    return path


def parse_seeds(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--out", default=str(REPO / "data/scenarios/crowd"))
    ap.add_argument("--live", default=None, choices=sorted(LIVE_TIERS),
                    help="live closed-loop crowds (SCN-LC-*) of this tier instead of keyframed")
    a = ap.parse_args()
    for s in parse_seeds(a.seeds):
        print(to_scenario_file(generate_live(s, a.live) if a.live else generate(s), Path(a.out)))
