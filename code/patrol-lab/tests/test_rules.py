from conftest import frame, obj, person, vehicle

from events.rules import EventEngine


def run(compound, frames, **cfg):
    engine = EventEngine(compound, cfg)
    return [e for f in frames for e in engine.process(f)]


def test_restricted_entry_unknown_credential_alerts(compound):
    evs = run(compound, [frame(0, [person("person_0001", 117, 20)]), frame(1, [person("person_0001", 117, 31)])])
    assert [(e.type, e.subject, e.t, e.severity) for e in evs] == [("RESTRICTED_ZONE_ENTRY", "person_0001", 1, "high")]
    assert any("credential state: unknown" in line for line in evs[0].evidence)


def test_restricted_entry_authorized_is_silent(compound):
    fs = [frame(0, [person("person_0002", 117, 20, credential="authorized")]),
          frame(1, [person("person_0002", 117, 31, credential="authorized")])]
    assert run(compound, fs) == []


def test_high_risk_zone_is_critical(compound):
    evs = run(compound, [frame(0, [person("person_0001", 130, 68)]), frame(1, [person("person_0001", 130, 75)])])
    assert evs[0].severity == "critical"


def test_entry_fires_once_until_the_track_leaves(compound):
    p = lambda t, y: frame(t, [person("person_0001", 117, y)])  # noqa: E731
    evs = run(compound, [p(0, 20), p(1, 31), p(2, 35), p(3, 20), p(4, 31)])
    assert [e.t for e in evs] == [1, 4]


def test_perimeter_breach_over_fence(compound):
    evs = run(compound, [frame(0, [person("person_0004", 155, 60)]), frame(1, [person("person_0004", 145, 60)])])
    assert [(e.type, e.location) for e in evs] == [("PERIMETER_BREACH", (150, 60))]


def test_gate_passage_and_outbound_are_not_breaches(compound):
    through_gate = [frame(0, [person("person_0005", 74, 105)]), frame(1, [person("person_0005", 74, 95)])]
    outbound = [frame(0, [person("person_0009", 145, 60)]), frame(1, [person("person_0009", 155, 60)])]
    assert run(compound, through_gate) == [] and run(compound, outbound) == []


def test_fall_needs_confirmation(compound):
    down = lambda t, a: frame(t, [person("person_0006", 62, 20, action=a)])  # noqa: E731
    evs = run(compound, [down(0, "walking"), down(1, "fallen"), down(2, "fallen"), down(3, "fallen")], fall_confirm_s=2)
    assert [(e.type, e.t, e.t_detected) for e in evs] == [("PERSON_FALL", 1, 3)]
    brief = run(compound, [down(0, "walking"), down(1, "lying"), down(2, "walking"), down(5, "walking")], fall_confirm_s=2)
    assert brief == []


def test_vehicle_prohibited_zone(compound):
    evs = run(compound, [frame(0, [vehicle("vehicle_0002", 50, 70)]), frame(1, [vehicle("vehicle_0002", 50, 78)])])
    assert [e.type for e in evs] == ["VEHICLE_IN_PROHIBITED_ZONE"]
    road = [frame(0, [vehicle("vehicle_0003", 24, 2)]), frame(1, [vehicle("vehicle_0003", 24, 40)])]
    assert run(compound, road) == []


def test_gate_left_open_after_threshold_only(compound):
    fs = [frame(t, gates={"gate_north_01": "open"}) for t in range(0, 26)]
    evs = run(compound, fs, gate_open_s=20)
    assert [(e.type, e.subject, e.t, e.t_detected) for e in evs] == [("GATE_LEFT_OPEN", "gate_north_01", 0, 20)]
    brief = [frame(t, gates={"gate_north_01": "open" if t < 10 else "closed"}) for t in range(0, 30)]
    assert run(compound, brief, gate_open_s=20) == []


def test_route_obstruction_needs_a_stationary_object_on_a_road(compound):
    still = [frame(t, [obj("pallet_0001", 50, 42)]) for t in range(0, 5)]
    evs = run(compound, still, obstruction_s=3)
    assert [(e.type, e.subject, e.t, e.t_detected) for e in evs] == [("ROUTE_OBSTRUCTION", "pallet_0001", 0, 3)]
    off_road = [frame(t, [obj("pallet_0002", 5, 95)]) for t in range(0, 5)]
    moving = [frame(t, [obj("cart_0001", 40 + 2 * t, 42)]) for t in range(0, 5)]
    assert run(compound, off_road, obstruction_s=3) == [] and run(compound, moving, obstruction_s=3) == []


def test_every_event_explains_itself(compound):
    evs = run(compound, [frame(0, [person("person_0004", 155, 60)]), frame(1, [person("person_0004", 145, 60)])])
    assert all(e.evidence and e.provenance == "rules" for e in evs)
