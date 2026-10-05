"""Deterministic, explainable event rules for the six v1 incidents.

Input: observation frames (the same schema whether they come from the simulator oracle or from perception).
Output: events that carry their own evidence. Rules report facts about anonymous tracks and assets; there is
no notion of identity, appearance or intent anywhere in this module (PLAN.md invariant 1).
"""

from dataclasses import asdict, dataclass, field

from events.world_state import ExpectedWorldState
from events.zones import Compound

EVENT_TYPES = (
    "PERIMETER_BREACH",
    "RESTRICTED_ZONE_ENTRY",
    "PERSON_FALL",
    "VEHICLE_IN_PROHIBITED_ZONE",
    "GATE_LEFT_OPEN",
    "ROUTE_OBSTRUCTION",
)
FALL_ACTIONS = ("fallen", "lying")
DEFAULT_CONFIG = {
    "fall_confirm_s": 2.0,  # prone this long before it counts as a fall
    "gate_open_s": 20.0,  # an unexpectedly open gate becomes "left open" after this long
    "obstruction_s": 3.0,  # a new object must sit still on a road cell this long
    "stationary_m": 0.5,  # an object moving more than this restarts the obstruction clock
}


@dataclass
class Event:
    type: str
    subject: str  # track id or asset id
    t: float  # onset: when the condition started
    t_detected: float  # when the rule confirmed it
    location: tuple[float, float]
    severity: str
    evidence: list[str]
    source: str  # oracle | perception
    id: str = ""
    provenance: str = field(default="rules", init=False)  # scorers refuse rule output as ground truth

    def to_dict(self) -> dict:
        d = asdict(self)
        d["location"] = [round(v, 2) for v in self.location]
        return d


def _xy(track: dict) -> tuple[float, float]:
    return float(track["position"][0]), float(track["position"][1])


def _clock(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


class EventEngine:
    """Feed frames in time order with process(); each call returns the events confirmed at that frame."""

    def __init__(self, compound: Compound, config: dict | None = None):
        self.compound = compound
        self.world = ExpectedWorldState(compound)
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self._count = 0
        self._last_pos: dict[str, tuple[float, float]] = {}
        self._in_restricted: dict[str, set[str]] = {}
        self._in_vehicle_prohibited: dict[str, set[str]] = {}
        self._fall_onset: dict[str, float] = {}
        self._gate_onset: dict[str, float] = {}
        self._obstruction: dict[str, tuple[float, tuple[float, float]]] = {}  # object -> (onset, anchor)
        self._fired: set[tuple[str, str]] = set()

    def _emit(self, events: list[Event], event: Event) -> None:
        self._count += 1
        event.id = f"evt_{self._count:05d}"
        events.append(event)

    def process(self, frame: dict) -> list[Event]:
        t, source = float(frame["t"]), frame.get("source", "oracle")
        events: list[Event] = []
        seen = set()
        for track in frame.get("tracks", []):
            tid, cls, pos = track["id"], track["class"], _xy(track)
            seen.add(tid)
            if cls == "person":
                self._perimeter(events, t, source, tid, pos)
                self._restricted(events, t, source, track, pos)
                self._fall(events, t, source, track, pos)
            elif cls == "vehicle":
                self._vehicle(events, t, source, tid, pos)
            self._last_pos[tid] = pos
        for tid in list(self._last_pos):  # forget tracks that left the frame
            if tid not in seen:
                self._forget(tid)
        self._assets(events, t, source, frame)
        return events

    def _forget(self, tid: str) -> None:
        for d in (self._last_pos, self._in_restricted, self._in_vehicle_prohibited, self._fall_onset, self._obstruction):
            d.pop(tid, None)

    # --- person rules ---------------------------------------------------------------------------------------

    def _perimeter(self, events, t, source, tid, pos):
        prev = self._last_pos.get(tid)
        if prev is None:
            return
        crossing = self.compound.boundary_crossing(prev, pos)
        if crossing is None or not crossing.inbound or crossing.gate_id is not None:
            return
        x, y = crossing.point
        self._emit(
            events,
            Event(
                type="PERIMETER_BREACH", subject=tid, t=t, t_detected=t, location=crossing.point,
                severity="high", source=source,
                evidence=[
                    f"{tid} crossed the perimeter from outside to inside at ({x:.1f}, {y:.1f}) at {_clock(t)}",
                    "the crossing point is not within a gate opening",
                ],
            ),
        )

    def _restricted(self, events, t, source, track, pos):
        tid = track["id"]
        now = {z.id: z for z in self.compound.zones_at(pos) if z.restricted}
        before = self._in_restricted.get(tid, set())
        self._in_restricted[tid] = set(now)
        credential = track.get("credential", "unknown")
        if credential == "authorized":
            return
        for zid in sorted(set(now) - before):
            zone = now[zid]
            self._emit(
                events,
                Event(
                    type="RESTRICTED_ZONE_ENTRY", subject=tid, t=t, t_detected=t, location=pos,
                    severity="critical" if zone.access == "high_risk" else "high", source=source,
                    evidence=[
                        f"{tid} entered {zid} (access: {zone.access}) at {_clock(t)}",
                        f"credential state: {credential}",
                    ],
                ),
            )

    def _fall(self, events, t, source, track, pos):
        tid = track["id"]
        if track.get("action") not in FALL_ACTIONS:
            self._fall_onset.pop(tid, None)
            self._fired.discard(("PERSON_FALL", tid))
            return
        onset = self._fall_onset.setdefault(tid, t)
        key = ("PERSON_FALL", tid)
        if t - onset >= self.cfg["fall_confirm_s"] and key not in self._fired:
            self._fired.add(key)
            self._emit(
                events,
                Event(
                    type="PERSON_FALL", subject=tid, t=onset, t_detected=t, location=pos, severity="high",
                    source=source,
                    evidence=[
                        f"{tid} action '{track['action']}' since {_clock(onset)}",
                        f"prone for {t - onset:.1f} s (threshold {self.cfg['fall_confirm_s']:.1f} s)",
                    ],
                ),
            )

    # --- vehicle rule ---------------------------------------------------------------------------------------

    def _vehicle(self, events, t, source, tid, pos):
        now = {z.id for z in self.compound.zones_at(pos) if z.vehicles == "prohibited"}
        before = self._in_vehicle_prohibited.get(tid, set())
        self._in_vehicle_prohibited[tid] = now
        for zid in sorted(now - before):
            self._emit(
                events,
                Event(
                    type="VEHICLE_IN_PROHIBITED_ZONE", subject=tid, t=t, t_detected=t, location=pos,
                    severity="high", source=source,
                    evidence=[f"{tid} entered {zid} (vehicles prohibited) at {_clock(t)}"],
                ),
            )

    # --- asset rules (Expected World State deltas) ----------------------------------------------------------

    def _assets(self, events, t, source, frame):
        deltas = self.world.compare(frame)
        open_gates = {d.asset_id: d for d in deltas if d.asset_type == "gate" and d.observed.get("state") == "open"}
        for gid in list(self._gate_onset):
            if gid not in open_gates:
                del self._gate_onset[gid]
                self._fired.discard(("GATE_LEFT_OPEN", gid))
        for gid, delta in open_gates.items():
            onset = self._gate_onset.setdefault(gid, t)
            key = ("GATE_LEFT_OPEN", gid)
            if t - onset >= self.cfg["gate_open_s"] and key not in self._fired:
                self._fired.add(key)
                gate = next(g for g in self.compound.gates if g.id == gid)
                (ax, ay), (bx, by) = gate.opening
                self._emit(
                    events,
                    Event(
                        type="GATE_LEFT_OPEN", subject=gid, t=onset, t_detected=t,
                        location=((ax + bx) / 2, (ay + by) / 2), severity="medium", source=source,
                        evidence=[
                            f"{gid} expected {delta.expected['state']}, observed {delta.observed['state']}",
                            f"open since {_clock(onset)} ({t - onset:.0f} s, threshold {self.cfg['gate_open_s']:.0f} s)",
                        ],
                    ),
                )

        on_road = {d.observed["object"]: d for d in deltas if d.change == "new_object"}
        for obj in list(self._obstruction):
            if obj not in on_road:
                del self._obstruction[obj]
                self._fired.discard(("ROUTE_OBSTRUCTION", obj))
        for obj, delta in on_road.items():
            pos = tuple(delta.observed["position"])
            onset, anchor = self._obstruction.get(obj, (t, pos))
            if ((pos[0] - anchor[0]) ** 2 + (pos[1] - anchor[1]) ** 2) ** 0.5 > self.cfg["stationary_m"]:
                onset, anchor = t, pos  # still moving: restart the clock
                self._fired.discard(("ROUTE_OBSTRUCTION", obj))
            self._obstruction[obj] = (onset, anchor)
            key = ("ROUTE_OBSTRUCTION", obj)
            if t - onset >= self.cfg["obstruction_s"] and key not in self._fired:
                self._fired.add(key)
                self._emit(
                    events,
                    Event(
                        type="ROUTE_OBSTRUCTION", subject=obj, t=onset, t_detected=t, location=anchor,
                        severity="medium", source=source,
                        evidence=[
                            f"new object {obj} on {delta.asset_id} (expected clear) since {_clock(onset)}",
                            f"stationary for {t - onset:.1f} s (threshold {self.cfg['obstruction_s']:.1f} s)",
                        ],
                    ),
                )
