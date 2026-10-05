from pathlib import Path

import pytest

from events.zones import Compound

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def compound() -> Compound:
    return Compound.load(REPO / "sim/compound/compound_spec.yaml")


def person(tid, x, y, action="walking", credential="unknown"):
    return {"id": tid, "class": "person", "position": [x, y, 0.0], "action": action, "credential": credential}


def vehicle(tid, x, y):
    return {"id": tid, "class": "vehicle", "position": [x, y, 0.0]}


def obj(tid, x, y):
    return {"id": tid, "class": "object", "position": [x, y, 0.0]}


def frame(t, tracks=(), gates=None):
    gates = gates or {}
    return {
        "t": t, "source": "oracle", "tracks": list(tracks),
        "assets": [{"id": g, "type": "gate", "state": gates.get(g, "closed")} for g in ("gate_north_01", "gate_south_01")],
    }
