"""Scorers. Ground truth is always what the scenario *staged*, never what the rules produced.

The oracle scorer (M0/M1) checks the rules against staged incidents. The perception scorer (M4) will compare
perception-path events against oracle-path events and attribute every miss to one stage.
"""

from dataclasses import dataclass, field
from statistics import median

from events.rules import Event


class CircularScoringError(ValueError):
    """Raised when rule output is passed in as ground truth: scoring rules against themselves proves nothing."""


@dataclass(frozen=True)
class StagedIncident:
    type: str
    subject: str
    t: float
    tolerance_s: float = 3.0
    provenance: str = "scenario"


@dataclass
class ScoreReport:
    hits: list[tuple[StagedIncident, Event]] = field(default_factory=list)
    misses: list[StagedIncident] = field(default_factory=list)
    false_alarms: list[Event] = field(default_factory=list)

    @property
    def latencies(self) -> list[float]:
        return [e.t_detected - s.t for s, e in self.hits]

    @property
    def perfect(self) -> bool:
        return not self.misses and not self.false_alarms

    def summary(self, duration_s: float | None = None) -> dict:
        out = {
            "staged": len(self.hits) + len(self.misses),
            "detected": len(self.hits),
            "missed": len(self.misses),
            "false_alerts": len(self.false_alarms),
            "latency_median_s": round(median(self.latencies), 2) if self.latencies else None,
            "latency_max_s": round(max(self.latencies), 2) if self.latencies else None,
        }
        if duration_s:
            out["false_alerts_per_hour"] = round(len(self.false_alarms) / (duration_s / 3600), 3)
        return out


def score(predicted: list[Event], staged: list[StagedIncident]) -> ScoreReport:
    """Match each staged incident to at most one predicted event of the same type and subject whose onset lies
    within the staged tolerance. Unmatched staged incidents are misses; unmatched events are false alarms."""
    for s in staged:
        if not isinstance(s, StagedIncident) or s.provenance != "scenario":
            raise CircularScoringError(f"ground truth must be staged by a scenario, got {s!r}")
    report = ScoreReport()
    unmatched = list(predicted)
    for s in sorted(staged, key=lambda s: s.t):
        candidates = [e for e in unmatched if e.type == s.type and e.subject == s.subject and abs(e.t - s.t) <= s.tolerance_s]
        if candidates:
            best = min(candidates, key=lambda e: abs(e.t - s.t))
            unmatched.remove(best)
            report.hits.append((s, best))
        else:
            report.misses.append(s)
    report.false_alarms = unmatched
    return report
