"""Live crowd (benchmarks/avoidance/pedestrians.py): determinism, lock-on rules, the commit rule that makes a lock
escapable, and people staying out of buildings."""

import math

import numpy as np

from benchmarks.avoidance import pedestrians as P
from benchmarks.avoidance.pedestrians import CrowdConfig, LiveCrowd, intercept_point
from events.zones import Compound

COMPOUND = Compound.load("sim/compound/compound_spec.yaml")


def lone_hunter(pos, heading=0.0, state=P.PAUSE, commit_m=3.0, speed=1.2) -> LiveCrowd:
    """A crowd of exactly one hunter at pos (open yard), paused for a long time so no RNG-driven wandering."""
    c = LiveCrowd.create(COMPOUND, CrowdConfig(n_wanderers=0, n_loiterers=0, n_hunters=1), seed=0)
    c.pos[0], c.post[0], c.heading[0] = np.array(pos, float), np.array(pos, float), heading
    c.state[0], c.until[0], c.commit_m[0], c.speed[0] = state, 1e9, commit_m, speed
    return c


def test_intercept_point_leads_a_moving_target():
    aim = intercept_point((0.0, 0.0), 1.5, (10.0, 0.0), (0.0, 1.0), cap_s=20.0)
    t = math.dist((0, 0), aim) / 1.5  # the hunter's travel time equals the robot's
    assert math.isclose(aim[0], 10.0) and math.isclose(aim[1], t * 1.0, rel_tol=1e-6)
    # at equal speed a robot walking straight across can't be caught: aim where it is
    assert intercept_point((0.0, 0.0), 1.0, (10.0, 0.0), (0.0, 1.0), cap_s=20.0) == (10.0, 0.0)
    assert intercept_point((0.0, 0.0), 1.0, (5.0, 0.0), (0.0, 0.0), cap_s=6.0) == (5.0, 0.0)


def test_same_seed_same_robot_path_same_crowd():
    runs = []
    for _ in range(2):
        c = LiveCrowd.create(COMPOUND, CrowdConfig(), seed=7)
        for k in range(1, 600):
            c.step(k * P.DT, (50 - 0.02 * k, 50 - 0.07 * k))
        runs.append((c.pos.copy(), c.state.copy(), [(r.hunter, r.t_lock) for r in c.locks]))
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])
    assert runs[0][2] == runs[1][2]


def test_paused_hunter_locks_on_a_visible_robot_and_commits():
    c = lone_hunter((30.0, 50.0))
    robot = [45.0, 50.0]  # 15 m away: outside trigger_m (12)
    for k in range(1, 20):
        c.step(k * P.DT, robot)
    assert not c.locks
    robot = [40.0, 50.0]
    for k in range(20, 120):
        c.step(k * P.DT, robot)
    assert len(c.locks) == 1 and c.locks[0].t_commit is not None
    assert c.locks[0].commit_dist_m <= 3.0 + 1e-6


def test_building_blocks_the_line_of_sight():
    # building [60, 30, 110, 70]: hunter west of it, robot east of it, 12 m apart would be in range without it
    c = lone_hunter((58.0, 50.0))
    for k in range(1, 50):
        c.step(k * P.DT, (111.0, 50.0))
    assert not c.locks


def test_committed_hunter_never_re_aims():
    c = lone_hunter((30.0, 50.0), commit_m=3.0)
    robot = [38.0, 50.0]
    k, heading_at_commit = 0, None
    while k < 400:
        k += 1
        if c.state[0] == P.COMMITTED and heading_at_commit is None:
            heading_at_commit = c.heading[0]
        if heading_at_commit is not None:
            robot[1] += 0.05  # the robot sidesteps after the commit ...
            assert c.heading[0] == heading_at_commit or c.state[0] != P.COMMITTED  # ... and is not followed
        c.step(k * P.DT, robot)
        if c.locks and c.locks[0].t_end is not None:
            break
    assert heading_at_commit is not None and c.locks[0].end_reason == "passed"


def test_standing_still_gets_you_hit_and_a_timely_sidestep_evades():
    def run(sidestep: bool) -> bool:
        c = lone_hunter((30.0, 50.0), commit_m=3.5, speed=1.2)
        robot = [40.0, 50.0]
        for k in range(1, 300):
            if sidestep and c.state[0] == P.COMMITTED:
                robot[1] += 0.1  # 1 m/s sideways once the hunter has committed
            c.step(k * P.DT, robot)
        c.finish()
        return c.locks[0].hit

    assert run(sidestep=False)
    assert not run(sidestep=True)


def test_nobody_walks_into_a_building_or_off_site():
    c = LiveCrowd.create(COMPOUND, CrowdConfig(), seed=3)
    w, h = COMPOUND.spec["size_m"]
    rects = [b["rect"] for b in COMPOUND.spec["buildings"]]
    route = [COMPOUND.charging_station] + [p for _, p in COMPOUND.route]
    for k in range(1, 3000):  # a robot that tours the checkpoints at ~1 m/s, so hunters engage
        leg = (k // 400) % len(route)
        a, b = route[leg], route[(leg + 1) % len(route)]
        f = (k % 400) / 400
        c.step(k * P.DT, (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
        for x, y in c.pos:
            assert 0 <= x <= w and 0 <= y <= h
            assert not any(x0 < x < x1 and y0 < y < y1 for x0, y0, x1, y1 in rects), (x, y)
    assert len(c.locks) > 0
