"""Per-stop failure policy: continue, abort, retry_then_escalate; coverage counts required stops only."""

from pathlib import Path

import pytest

from missions.patrol import PatrolMission, Stop
from robots.base import Capabilities, NavigationRefused
from robots.fake import FakeAdapter

REPO = Path(__file__).resolve().parents[1]
PLACES = {"a": (3, 0), "b": (3, 3), "c": (0, 3), "home": (0, 0)}


class FlakyAdapter(FakeAdapter):
    """Refuses the first `refusals[place]` attempts at a place, then behaves."""

    def __init__(self, refusals: dict[str, int]):
        super().__init__("spot_0001", Capabilities.load(REPO / "robots/spot/capabilities.yaml"), (0, 0), PLACES)
        self.refusals = dict(refusals)

    def goto_named(self, place):
        if self.refusals.get(place, 0) > 0:
            self.refusals[place] -= 1
            raise NavigationRefused(f"{place} blocked")
        super().goto_named(place)


def run(mission: PatrolMission, robot: FakeAdapter, limit_s: float = 120.0) -> PatrolMission:
    t = 0.0
    while not mission.done and t < limit_s:
        mission.tick(t)
        robot.step(0.1)
        t += 0.1
    return mission


def outcomes(m):
    return [(v.place, v.outcome) for v in m.visits]


def test_continue_records_and_moves_on():
    robot = FlakyAdapter({"b": 99})
    m = run(PatrolMission(robot, ["a", "b", "c"], return_to="home"), robot)
    assert outcomes(m) == [("a", "succeeded"), ("b", "refused"), ("c", "succeeded"), ("home", "succeeded")]
    assert m.status == "done" and m.coverage == pytest.approx(2 / 3)


def test_abort_skips_the_rest_but_still_goes_home():
    robot = FlakyAdapter({"b": 99})
    m = run(PatrolMission(robot, ["a", Stop("b", on_failure="abort"), "c"], return_to="home"), robot)
    assert outcomes(m) == [("a", "succeeded"), ("b", "refused"), ("home", "succeeded")]
    assert m.status == "aborted" and m.aborted_at == "b"


def test_retry_succeeds_after_a_transient_failure():
    robot = FlakyAdapter({"b": 1})
    m = run(PatrolMission(robot, ["a", Stop("b", on_failure="retry_then_escalate", retries=2), "c"]), robot)
    assert outcomes(m) == [("a", "succeeded"), ("b", "refused"), ("b", "succeeded"), ("c", "succeeded")]
    assert m.escalations == [] and m.coverage == 1.0


def test_retries_exhausted_raise_an_escalation():
    robot = FlakyAdapter({"b": 99})
    m = run(PatrolMission(robot, ["a", Stop("b", on_failure="retry_then_escalate", retries=2), "c"]), robot)
    assert [v.outcome for v in m.visits if v.place == "b"] == ["refused"] * 3
    assert [(e.place, e.attempts) for e in m.escalations] == [("b", 3)]
    assert outcomes(m)[-1] == ("c", "succeeded")


def test_optional_stops_do_not_count_toward_coverage():
    robot = FlakyAdapter({"b": 99})
    m = run(PatrolMission(robot, ["a", {"place": "b", "required": False}, "c"]), robot)
    assert m.coverage == 1.0


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        Stop("a", on_failure="panic")
