"""Expected World State: what each tracked asset should be, compared against one observation frame.

We compare states, not images. On the oracle path the observed states come straight from the simulator;
on the perception path each asset type needs its own state observer (gate angle, road-cell occupancy, ...).
"""

from dataclasses import asdict, dataclass

from events.zones import Compound


@dataclass(frozen=True)
class AssetDelta:
    asset_id: str
    asset_type: str  # gate | road_segment
    change: str  # unexpected_state_transition | new_object
    expected: dict
    observed: dict

    def to_dict(self) -> dict:
        return asdict(self)


class ExpectedWorldState:
    def __init__(self, compound: Compound, known_objects: frozenset[str] = frozenset()):
        self.compound = compound
        self.expected_gates = {g.id: g.normal_state for g in compound.gates}
        self.known_objects = known_objects  # static objects that belong on the map (none in v0)

    def compare(self, frame: dict) -> list[AssetDelta]:
        deltas = []
        for asset in frame.get("assets", []):
            if asset["type"] != "gate":
                continue
            expected = self.expected_gates.get(asset["id"])
            if expected is not None and asset["state"] != expected:
                deltas.append(
                    AssetDelta(
                        asset_id=asset["id"],
                        asset_type="gate",
                        change="unexpected_state_transition",
                        expected={"state": expected},
                        observed={k: v for k, v in asset.items() if k in ("state", "angle_deg")},
                    )
                )
        for track in frame.get("tracks", []):
            if track["class"] != "object" or track["id"] in self.known_objects:
                continue
            segment = self.compound.road_segment_at(tuple(track["position"][:2]))
            if segment is not None:
                deltas.append(
                    AssetDelta(
                        asset_id=segment.id,
                        asset_type="road_segment",
                        change="new_object",
                        expected={"occupied": False},
                        observed={"occupied": True, "object": track["id"], "position": list(track["position"][:2])},
                    )
                )
        return deltas
