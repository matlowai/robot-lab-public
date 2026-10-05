import pytest

from events.rules import Event
from events.scorer import CircularScoringError, StagedIncident, score


def ev(type_, subject, t, detected=None):
    return Event(type=type_, subject=subject, t=t, t_detected=detected if detected is not None else t,
                 location=(0, 0), severity="high", evidence=["x"], source="oracle")


def test_hit_miss_false_alarm_and_latency():
    staged = [StagedIncident("PERSON_FALL", "person_0001", 10, 1), StagedIncident("GATE_LEFT_OPEN", "gate_north_01", 50, 1)]
    predicted = [ev("PERSON_FALL", "person_0001", 10.4, 12.4), ev("ROUTE_OBSTRUCTION", "pallet_0001", 70)]
    r = score(predicted, staged)
    assert [s.subject for s, _ in r.hits] == ["person_0001"]
    assert [m.subject for m in r.misses] == ["gate_north_01"]
    assert [e.subject for e in r.false_alarms] == ["pallet_0001"]
    assert r.latencies == [pytest.approx(2.4)]
    assert not r.perfect


def test_outside_tolerance_or_wrong_subject_is_not_a_hit():
    staged = [StagedIncident("PERSON_FALL", "person_0001", 10, 1)]
    assert not score([ev("PERSON_FALL", "person_0001", 11.5)], staged).hits
    assert not score([ev("PERSON_FALL", "person_0002", 10)], staged).hits


def test_one_event_cannot_satisfy_two_staged_incidents():
    staged = [StagedIncident("PERSON_FALL", "person_0001", 10, 5), StagedIncident("PERSON_FALL", "person_0001", 12, 5)]
    r = score([ev("PERSON_FALL", "person_0001", 11)], staged)
    assert len(r.hits) == 1 and len(r.misses) == 1


def test_rule_output_is_refused_as_ground_truth():
    events = [ev("PERSON_FALL", "person_0001", 10)]
    with pytest.raises(CircularScoringError):
        score(events, events)
    with pytest.raises(CircularScoringError):
        score(events, [StagedIncident("PERSON_FALL", "person_0001", 10, provenance="rules")])
