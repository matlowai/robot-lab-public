"""A patrol over named checkpoints. Coverage comes from the robot reporting arrival, never from pose checks.

Each stop carries its own failure policy (a refused, failed or canceled leg):
  continue             record the failure and move on (default; fine for optional visual checks)
  abort                end the patrol; remaining stops are skipped, the robot still heads home
  retry_then_escalate  retry up to `retries` times, then raise an escalation for the operator and move on
Coverage counts only *required* stops. The caller owns the clock: tick(t) with simulator time on the oracle
path, wall time against a real robot.
"""

from dataclasses import dataclass, field

from robots.base import NavigationRefused, RobotAdapter

POLICIES = ("continue", "abort", "retry_then_escalate")


@dataclass(frozen=True)
class Stop:
    place: str
    required: bool = True
    on_failure: str = "continue"
    retries: int = 1

    def __post_init__(self):
        if self.on_failure not in POLICIES:
            raise ValueError(f"on_failure must be one of {POLICIES}, got {self.on_failure!r}")

    @classmethod
    def of(cls, item: "str | Stop | dict") -> "Stop":
        if isinstance(item, Stop):
            return item
        if isinstance(item, str):
            return cls(item)
        return cls(item["place"], item.get("required", True), item.get("on_failure", "continue"), item.get("retries", 1))


@dataclass
class Visit:
    place: str
    t: float
    outcome: str  # succeeded | failed | canceled | refused
    detail: str = ""


@dataclass
class Escalation:
    place: str
    t: float
    attempts: int
    last_outcome: str
    detail: str = ""


@dataclass
class PatrolMission:
    robot: RobotAdapter
    route: list  # of str | Stop | dict
    return_to: str | None = None  # e.g. the charging station: visited last, never counted as coverage
    visits: list[Visit] = field(default_factory=list)
    escalations: list[Escalation] = field(default_factory=list)
    aborted_at: str | None = None
    _index: int = 0
    _issued: bool = False
    _attempts: int = 0

    def __post_init__(self):
        self.stops = [Stop.of(s) for s in self.route]
        self.route = [s.place for s in self.stops]  # plain place names, for reporting

    @property
    def done(self) -> bool:
        return self._index >= len(self._plan)

    @property
    def status(self) -> str:
        return "aborted" if self.aborted_at else "done" if self.done else "running"

    @property
    def coverage(self) -> float:
        required = {s.place for s in self.stops if s.required}
        reached = {v.place for v in self.visits if v.outcome == "succeeded"} & required
        return len(reached) / len(required) if required else 1.0

    @property
    def _plan(self) -> list[Stop]:
        home = [Stop(self.return_to, required=False)] if self.return_to else []
        return self.stops + home

    def _finish_stop(self, stop: Stop, t: float, outcome: str, detail: str) -> None:
        self.visits.append(Visit(stop.place, t, outcome, detail))
        self._issued = False
        if outcome == "succeeded":
            self._attempts = 0
            self._index += 1
            return
        self._attempts += 1
        if stop.on_failure == "retry_then_escalate" and self._attempts <= stop.retries:
            return  # same stop again on the next tick
        if stop.on_failure == "retry_then_escalate":
            self.escalations.append(Escalation(stop.place, t, self._attempts, outcome, detail))
        elif stop.on_failure == "abort":
            self.aborted_at = stop.place
            home = len(self.stops)  # skip every remaining stop, keep only the trip home
            self._index = home if self.return_to else len(self._plan)
            self._attempts = 0
            return
        self._attempts = 0
        self._index += 1

    def tick(self, t: float) -> None:
        if self.done:
            return
        stop = self._plan[self._index]
        if not self._issued:
            try:
                self.robot.goto_named(stop.place)
                self._issued = True
            except NavigationRefused as exc:
                self._finish_stop(stop, t, "refused", str(exc))
            return
        status = self.robot.navigation_status()
        if status.target == stop.place and status.state in ("succeeded", "failed", "canceled"):
            self._finish_stop(stop, t, status.state, status.detail)
