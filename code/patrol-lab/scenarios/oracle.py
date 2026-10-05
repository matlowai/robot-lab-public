"""The oracle: scenario -> observation frames with perfect facts (level 0, no physics).

M1 replaces this with frames read from Isaac Sim ground truth; M4 with frames built by perception. All three
produce observation.schema.json frames, so the event engine and scorers never change.
"""

from collections.abc import Iterator

from events.zones import Compound
from scenarios.model import Actor, Scenario


def _actor_at(actor: Actor, t: float) -> tuple[tuple[float, float], str] | None:
    """Linear interpolation between keyframes; the action is the one set at the last keyframe reached."""
    kfs = actor.keyframes
    if t < kfs[0].t or t > kfs[-1].t:
        return None
    for a, b in zip(kfs, kfs[1:]):
        if a.t <= t < b.t:  # half-open: at a keyframe's own time, that keyframe's action applies
            f = (t - a.t) / (b.t - a.t)
            return (a.pos[0] + f * (b.pos[0] - a.pos[0]), a.pos[1] + f * (b.pos[1] - a.pos[1])), a.action
    return kfs[-1].pos, kfs[-1].action


def frames(scenario: Scenario, compound: Compound) -> Iterator[dict]:
    dt = 1.0 / scenario.rate_hz
    gate_states = {g.id: g.normal_state for g in compound.gates}
    changes = sorted(scenario.asset_changes, key=lambda c: c.t)
    n = int(round(scenario.duration_s * scenario.rate_hz))
    for i in range(n + 1):
        t = round(i * dt, 6)
        while changes and changes[0].t <= t:
            c = changes.pop(0)
            gate_states[c.id] = c.state
        tracks = []
        for actor in scenario.actors:
            at = _actor_at(actor, t)
            if at is None:
                continue
            (x, y), action = at
            track = {"id": actor.id, "class": actor.cls, "position": [round(x, 3), round(y, 3), 0.0]}
            if actor.cls == "person":
                track["action"] = action
                track["credential"] = actor.credential
            tracks.append(track)
        for obj in scenario.objects:
            if obj.spawn_t <= t and (obj.remove_t is None or t < obj.remove_t):
                tracks.append({"id": obj.id, "class": "object", "position": [obj.pos[0], obj.pos[1], 0.0]})
        yield {
            "t": t,
            "source": "oracle",
            "tracks": tracks,
            "assets": [{"id": gid, "type": "gate", "state": s} for gid, s in gate_states.items()],
        }
