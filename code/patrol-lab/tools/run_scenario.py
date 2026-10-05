"""Run a scenario on the oracle path: frames -> rules -> incidents -> oracle scorer, with a FakeAdapter patrol.

    uv run python tools/run_scenario.py scenarios/library/SCN-M0-001.yaml [--out data/runs] [--validate]

Prints the event timeline and the score report; with --out, writes events.jsonl, incidents.json and
report.json into <out>/<scenario-id>-<timestamp>/.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from events.incidents import to_incident  # noqa: E402
from events.rules import Event, EventEngine, _clock  # noqa: E402
from events.scorer import ScoreReport, score  # noqa: E402
from events.zones import Compound  # noqa: E402
from missions.patrol import PatrolMission  # noqa: E402
from robots.base import Capabilities  # noqa: E402
from robots.fake import FakeAdapter  # noqa: E402
from scenarios.model import REPO, Scenario  # noqa: E402
from scenarios.oracle import frames  # noqa: E402
from schemas import validate  # noqa: E402


@dataclass
class RunResult:
    scenario: Scenario
    events: list[Event]
    incidents: list[dict]
    report: ScoreReport
    visited: list[tuple[str, float]] = field(default_factory=list)
    coverage: float = 1.0
    route_len: int = 0


def run(scenario_path: str | Path, *, check_schemas: bool = False) -> RunResult:
    scenario = Scenario.load(scenario_path)
    compound = Compound.load(scenario.compound)
    engine = EventEngine(compound, scenario.config)
    places = {cp_id: pos for cp_id, pos in compound.route} | {"charging_station": compound.charging_station}
    spot = FakeAdapter(
        "spot_0001", Capabilities.load(REPO / "robots/spot/capabilities.yaml"), compound.charging_station, places
    )
    patrol = PatrolMission(spot, list(compound.route_stops), return_to="charging_station")
    dt = 1.0 / scenario.rate_hz
    events: list[Event] = []
    for frame in frames(scenario, compound):
        patrol.tick(frame["t"])
        frame["robots"] = [spot.state()]
        if check_schemas:
            validate("observation", frame)
            validate("robot_state", frame["robots"][0])
        events += engine.process(frame)
        spot.step(dt)
    incidents = [to_incident(e, i + 1) for i, e in enumerate(events)]
    if check_schemas:
        for e in events:
            validate("event", e.to_dict())
        for inc in incidents:
            validate("incident", inc)
    visited = [(v.place, v.t) for v in patrol.visits if v.outcome == "succeeded" and v.place in patrol.route]
    return RunResult(
        scenario, events, incidents, score(events, list(scenario.staged)), visited, patrol.coverage, len(patrol.route)
    )


def print_result(r: RunResult) -> None:
    s = r.scenario
    print(f"{s.id} · {s.name} · {s.duration_s:.0f} s at {s.rate_hz:.0f} Hz (oracle path)\n")
    staged = {(x.type, x.subject) for x in s.staged}
    lines = [(e.t_detected, f"{_clock(e.t_detected)}  {e.type:<27} {e.subject:<16} onset {_clock(e.t)}"
              + ("" if (e.type, e.subject) in staged else "   <-- NOT STAGED")) for e in r.events]
    lines += [(t, f"{_clock(t)}  checkpoint {cp}") for cp, t in r.visited]
    for _, line in sorted(lines):
        print("  " + line)
    summary = r.report.summary(s.duration_s)
    print(f"\n  route coverage   {r.coverage:.0%} ({len(r.visited)}/{r.route_len} checkpoints)")
    for k, v in summary.items():
        print(f"  {k:<16} {v}")
    for m in r.report.misses:
        print(f"  MISSED  {m.type} {m.subject} at {_clock(m.t)}")
    for e in r.report.false_alarms:
        print(f"  FALSE   {e.type} {e.subject} at {_clock(e.t)}: {e.evidence[0]}")
    print(f"\n  verdict          {'PASS' if r.report.perfect else 'FAIL'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--out", help="directory for run artifacts (e.g. data/runs)")
    ap.add_argument("--validate", action="store_true", help="validate every frame/event/incident against schemas/")
    args = ap.parse_args()
    r = run(args.scenario, check_schemas=args.validate)
    print_result(r)
    if args.out:
        out = Path(args.out) / f"{r.scenario.id}-{time.strftime('%Y%m%d-%H%M%S')}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "events.jsonl").write_text("".join(json.dumps(e.to_dict()) + "\n" for e in r.events))
        (out / "incidents.json").write_text(json.dumps(r.incidents, indent=2))
        report = {**r.report.summary(r.scenario.duration_s), "coverage": r.coverage, "perfect": r.report.perfect}
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(f"  artifacts        {out}")
    return 0 if r.report.perfect else 1


if __name__ == "__main__":
    raise SystemExit(main())
