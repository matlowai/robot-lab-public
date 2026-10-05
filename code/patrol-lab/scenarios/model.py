"""Scenario files: actors with keyframed paths, asset state changes, spawned objects, and staged incidents."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from events.scorer import StagedIncident

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Keyframe:
    t: float
    pos: tuple[float, float]
    action: str


@dataclass(frozen=True)
class Actor:
    id: str
    cls: str  # person | vehicle
    credential: str
    keyframes: tuple[Keyframe, ...]


@dataclass(frozen=True)
class AssetChange:
    id: str
    t: float
    state: str


@dataclass(frozen=True)
class SpawnedObject:
    id: str
    spawn_t: float
    pos: tuple[float, float]
    remove_t: float | None = None


@dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    compound: Path
    duration_s: float
    rate_hz: float
    config: dict
    actors: tuple[Actor, ...]
    asset_changes: tuple[AssetChange, ...]
    objects: tuple[SpawnedObject, ...]
    staged: tuple[StagedIncident, ...]
    raw: dict

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        d = yaml.safe_load(Path(path).read_text())
        actors = tuple(
            Actor(
                a["id"], a["class"], a.get("credential", "unknown"),
                tuple(Keyframe(float(k["t"]), tuple(k["pos"]), k.get("action", "walking")) for k in a["keyframes"]),
            )
            for a in d.get("actors", [])
        )
        return cls(
            id=d["id"],
            name=d["name"],
            compound=REPO / d["compound"],
            duration_s=float(d["duration_s"]),
            rate_hz=float(d.get("rate_hz", 10)),
            config=dict(d.get("config", {})),
            actors=actors,
            asset_changes=tuple(AssetChange(c["id"], float(c["t"]), c["state"]) for c in d.get("asset_changes", [])),
            objects=tuple(
                SpawnedObject(o["id"], float(o["spawn_t"]), tuple(o["pos"]), o.get("remove_t")) for o in d.get("objects", [])
            ),
            staged=tuple(
                StagedIncident(s["type"], s["subject"], float(s["t"]), float(s.get("tolerance_s", 3.0)))
                for s in d.get("staged_incidents", [])
            ),
            raw=d,
        )
