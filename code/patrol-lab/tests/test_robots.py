from pathlib import Path

import pytest

from missions.patrol import PatrolMission
from robots.base import Capabilities, NavigationRefused, NavStatus, RobotAdapter
from robots.fake import FakeAdapter
from robots.scheduler import Task, assign
from schemas import validate

REPO = Path(__file__).resolve().parents[1]


def caps(name):
    return Capabilities.load(REPO / f"robots/{name}/capabilities.yaml")


def test_fake_adapter_implements_the_protocol_and_reaches_a_named_place():
    spot = FakeAdapter("spot_0001", caps("spot"), (0, 0), places={"cp_a": (16, 0)})
    assert isinstance(spot, RobotAdapter)
    spot.goto_named("cp_a")
    assert spot.navigation_status() == NavStatus("navigating", "cp_a")
    for _ in range(110):  # 11 s at 10 Hz, 1.6 m/s
        spot.step(0.1)
    assert spot.navigation_status().state == "succeeded"
    assert spot.pose()[:2] == pytest.approx((16, 0), abs=0.25)
    validate("robot_state", spot.state())


def test_fake_adapter_refuses_unknown_places_and_busy_goals():
    spot = FakeAdapter("spot_0001", caps("spot"), (0, 0), places={"cp_a": (16, 0), "cp_b": (0, 16)})
    with pytest.raises(NavigationRefused):
        spot.goto_named("nowhere")
    spot.goto_named("cp_a")
    with pytest.raises(NavigationRefused):
        spot.goto_named("cp_b")
    spot.stop()
    assert spot.navigation_status().state == "canceled"


def test_patrol_mission_counts_arrivals_not_poses():
    places = {"a": (5, 0), "b": (5, 5), "home": (0, 0)}
    spot = FakeAdapter("spot_0001", caps("spot"), (0, 0), places=places)
    patrol = PatrolMission(spot, ["a", "b", "nowhere"], return_to="home")
    t = 0.0
    while not patrol.done and t < 60:
        patrol.tick(t)
        spot.step(0.1)
        t += 0.1
    assert [(v.place, v.outcome) for v in patrol.visits] == [
        ("a", "succeeded"), ("b", "succeeded"), ("nowhere", "refused"), ("home", "succeeded")]
    assert patrol.coverage == pytest.approx(2 / 3)

