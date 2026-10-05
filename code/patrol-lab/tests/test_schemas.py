import copy
from pathlib import Path

import pytest
import yaml
from jsonschema import ValidationError

from schemas import SCHEMA_DIR, validator, validate

REPO = Path(__file__).resolve().parents[1]
GOOD_TRACK = {"id": "person_0001", "class": "person", "position": [1, 2, 0], "action": "walking", "credential": "unknown"}


@pytest.mark.parametrize("name", sorted(p.name.removesuffix(".schema.json") for p in SCHEMA_DIR.glob("*.schema.json")))
def test_every_schema_is_valid_json_schema(name):
    validator(name)


def test_compound_capabilities_and_scenarios_validate():
    validate("compound", yaml.safe_load((REPO / "sim/compound/compound_spec.yaml").read_text()))
    for caps in (REPO / "robots").glob("*/capabilities.yaml"):
        validate("capabilities", yaml.safe_load(caps.read_text()))
    for scn in (REPO / "scenarios/library").glob("*.yaml"):
        validate("scenario", yaml.safe_load(scn.read_text()))


@pytest.mark.parametrize("forbidden", ["face_embedding", "identity", "name", "age", "gender", "ethnicity",
                                       "suspicious", "threat_score", "intent", "appearance"])
def test_invariant_tracks_cannot_carry_identity_or_judgement(forbidden):
    """PLAN.md invariant 1: these concepts do not exist in the observation API."""
    frame = {"t": 0, "source": "oracle", "tracks": [{**GOOD_TRACK, forbidden: "x"}], "assets": []}
    with pytest.raises(ValidationError):
        validate("observation", frame)


def test_invariant_events_cannot_carry_extra_judgement_fields():
    event = {"id": "evt_00001", "type": "PERSON_FALL", "subject": "person_0001", "t": 1, "t_detected": 3,
             "location": [1, 2], "severity": "high", "evidence": ["x"], "source": "oracle", "provenance": "rules"}
    validate("event", event)
    bad = copy.deepcopy(event)
    bad["suspicion"] = 0.9
    with pytest.raises(ValidationError):
        validate("event", bad)
    with pytest.raises(ValidationError):
        validate("event", {**event, "type": "SUSPICIOUS_PERSON"})
