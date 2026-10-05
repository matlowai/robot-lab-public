"""End-to-end oracle runs: the M0 go/no-go, plus checks that the scorer is not vacuous."""

import dataclasses
from pathlib import Path

from events.scorer import StagedIncident, score
from tools.run_scenario import run

LIB = Path(__file__).resolve().parents[1] / "scenarios/library"


def test_m0_gate_three_incident_timeline_is_perfect():
    r = run(LIB / "SCN-M0-001.yaml", check_schemas=True)
    summary = r.report.summary()
    assert (summary["staged"], summary["detected"], summary["missed"], summary["false_alerts"]) == (3, 3, 0, 0)
    assert r.coverage == 1.0


def test_remaining_three_incident_types_are_perfect():
    r = run(LIB / "SCN-M0-002.yaml", check_schemas=True)
    assert r.report.perfect and len(r.report.hits) == 3
    fall = next(e for _, e in r.report.hits if e.type == "PERSON_FALL")
    assert (fall.t, fall.t_detected) == (40.0, 42.0)


def test_all_six_v1_incident_types_are_covered():
    types = {s.type for p in LIB.glob("SCN-M0-*.yaml") for s, _ in run(p).report.hits}
    assert types == {"PERIMETER_BREACH", "RESTRICTED_ZONE_ENTRY", "PERSON_FALL",
                     "VEHICLE_IN_PROHIBITED_ZONE", "GATE_LEFT_OPEN", "ROUTE_OBSTRUCTION"}


def test_scorer_is_not_vacuous():
    r = run(LIB / "SCN-M0-001.yaml")
    staged = list(r.scenario.staged)
    dropped = score(r.events, staged[1:])
    assert len(dropped.false_alarms) == 1  # an event nobody staged is a false alarm
    phantom = score(r.events, staged + [StagedIncident("PERSON_FALL", "person_0003", 50, 1)])
    assert len(phantom.misses) == 1  # a staged incident nobody detected is a miss
    shifted = score(r.events, [dataclasses.replace(s, t=s.t + 5) for s in staged])
    assert len(shifted.hits) == 0
