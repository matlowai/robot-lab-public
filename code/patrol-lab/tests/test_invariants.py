"""PLAN.md invariants that can be checked mechanically."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORE = ["events", "scenarios", "missions", "robots/base.py", "robots/scheduler.py", "robots/fake.py", "schemas"]


def core_files():
    for c in CORE:
        p = REPO / c
        yield from ([p] if p.is_file() else sorted(p.rglob("*.py")))


def test_core_never_imports_a_simulator():
    """Invariant 4: mission/event/memory/perception/scoring code goes through RobotAdapter only."""
    pattern = re.compile(r"^\s*(import|from)\s+(isaacsim|isaaclab|omni|pxr|carb)\b", re.M)
    offenders = [str(f.relative_to(REPO)) for f in core_files() if pattern.search(f.read_text())]
    assert offenders == []


def test_no_identity_or_judgement_vocabulary_in_code():
    """Invariant 1: the concepts must not exist, not even as variable names."""
    words = re.compile(r"\b(suspicious|suspicion|face_?(id|embedding|recogni\w*)|biometric\w*|demographic\w*|"
                       r"ethnicit\w*|gender|threat_score)\b", re.I)
    offenders = [(str(f.relative_to(REPO)), m.group(0)) for f in core_files() for m in words.finditer(f.read_text())]
    assert offenders == []


ROBOTICS = ["robots", "events", "perception", "stack", "missions", "scenarios", "schemas", "sim"]
FORBIDDEN_VERBS = re.compile(r"\bdef\s+(fire_weapon|attack_\w+|target_person|engage_target|shoot\w*)\s*\(")


def robotics_files():
    for c in ROBOTICS:
        yield from sorted((REPO / c).rglob("*.py"))


def test_robotics_never_imports_the_game_layer():
    """docs/GAME_SPEC.md §3: game combat and supernatural code live in game/ and must never leak into the
    robotics stack, above all into anything a real-hardware adapter could load."""
    pattern = re.compile(r"^\s*(import|from)\s+game(\.|\s|$)", re.M)
    offenders = [str(f.relative_to(REPO)) for f in robotics_files() if pattern.search(f.read_text())]
    assert offenders == []


def test_no_robot_api_exposes_combat_verbs():
    """GAME_SPEC §3: the robot abstraction must never expose fire_weapon/attack_*/target_person/engage_target."""
    offenders = [(str(f.relative_to(REPO)), m.group(1)) for f in robotics_files() for m in FORBIDDEN_VERBS.finditer(f.read_text())]
    assert offenders == []
