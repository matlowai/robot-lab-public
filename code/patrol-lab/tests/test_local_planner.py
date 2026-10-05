"""The local planner in a fake 2-D world: a unicycle robot, circular obstacles, and a ray-cast fake lidar.
Nothing here tells the planner where anything is; it only ever sees lidar points."""

import math

import pytest

from robots.spot.local_planner import LocalPlanner, ObstacleTracker, PlannerConfig, plan

CFG = PlannerConfig()


def lidar(pose, circles, walls=(), n=181, fov=math.radians(270), max_range=12.0):
    """Robot-frame hit points for n rays against circles (cx, cy, r) and wall segments."""
    x, y, yaw = pose
    pts = []
    for i in range(n):
        a = yaw - fov / 2 + fov * i / (n - 1)
        dx, dy = math.cos(a), math.sin(a)
        best = max_range
        for cx, cy, r in circles:
            fx, fy = x - cx, y - cy
            b = fx * dx + fy * dy
            c = fx * fx + fy * fy - r * r
            disc = b * b - c
            if disc >= 0:
                t = -b - math.sqrt(disc)
                if 0 < t < best:
                    best = t
        for (ax, ay), (bx, by) in walls:
            ex, ey = bx - ax, by - ay
            den = dx * ey - dy * ex
            if abs(den) > 1e-9:
                t = ((ax - x) * ey - (ay - y) * ex) / den
                u = ((ax - x) * dy - (ay - y) * dx) / den
                if 0 < t < best and 0 <= u <= 1:
                    best = t
        if best < max_range:
            rel = a - yaw
            pts.append((best * math.cos(rel), best * math.sin(rel)))
    return pts


def simulate(start, goal, obstacles, walls=(), seconds=60.0, dt=0.1, person_r=0.3):
    """obstacles: callables t -> (x, y) (people). Returns (arrived_t, min_clearance, statuses)."""
    x, y, yaw = start
    min_clear, statuses = math.inf, set()
    planner = LocalPlanner()
    for k in range(int(seconds / dt)):
        t = k * dt
        circles = [(*o(t), person_r) for o in obstacles]
        for cx, cy, r in circles:
            min_clear = min(min_clear, math.hypot(cx - x, cy - y) - r - CFG.robot_radius)
        scan = lidar((x, y, yaw), circles, walls)
        p = planner.step(t, (x, y, yaw), goal, scan)
        statuses.add(p.status)
        if p.status == "arrived":
            return t, min_clear, statuses
        vx, vy, wz = p.cmd  # body frame, including Spot's sideways walking
        yaw += wz * dt
        c, s = math.cos(yaw), math.sin(yaw)
        x, y = x + (vx * c - vy * s) * dt, y + (vx * s + vy * c) * dt
    return None, min_clear, statuses


def still(x, y):
    return lambda t: (x, y)


def walking(x0, y0, vx, vy, t0=0.0):
    return lambda t: (x0 + vx * max(0.0, t - t0), y0 + vy * max(0.0, t - t0))


def test_open_ground_goes_straight_to_the_goal():
    t, _, statuses = simulate((0, 0, 0), (10, 0), [])
    assert t is not None and t < 14 and "blocked" not in statuses


def test_goes_around_someone_reading_in_the_path():
    t, clear, _ = simulate((0, 0, 0), (12, 0), [still(6, 0)])
    assert t is not None and clear >= CFG.safety_margin * 0.8


def test_person_walking_head_on_is_avoided():
    """The book reader walks straight at the robot at 1.2 m/s and never looks up."""
    t, clear, _ = simulate((0, 0, 0), (16, 0), [walking(16, 0.0, -1.2, 0.0)])
    assert t is not None and clear >= 0.15


def test_two_readers_side_by_side_head_on():
    t, clear, _ = simulate((0, 0, 0), (18, 0), [walking(18, -0.6, -1.0, 0.0), walking(18, 0.6, -1.0, 0.0)])
    assert t is not None and clear >= 0.15


def test_fully_blocked_corridor_waits_then_proceeds_when_it_clears():
    walls = [((-1, 1.2), (20, 1.2)), ((-1, -1.2), (20, -1.2))]  # a 2.4 m corridor
    blocker = lambda t: (6.0, 0.0) if t < 15 else (6.0, 50.0)  # noqa: E731  someone stands there, then leaves
    t, clear, statuses = simulate((0, 0, 0), (12, 0), [blocker], walls=walls)
    assert "blocked" in statuses or "turning" in statuses
    assert t is not None and t > 15 and clear >= 0.1


def test_tracker_estimates_an_approaching_walker():
    tracker, pose = ObstacleTracker(), (0.0, 0.0, 0.0)
    v = None
    for k in range(6):
        t = k * 0.1
        v = tracker.update(t, pose, lidar(pose, [(8 - 1.2 * t, 0.0, 0.3)]))
    assert v and all(abs(vx + 1.2) < 0.2 and abs(vy) < 0.2 for vx, vy in v)


def test_walls_stay_static():
    tracker, pose = ObstacleTracker(), (0.0, 0.0, 0.0)
    walls = [((-5, 1.5), (20, 1.5))]
    for k in range(5):
        v = tracker.update(k * 0.1, (0.1 * k, 0.0, 0.0), lidar((0.1 * k, 0.0, 0.0), [], walls))
    assert all(math.hypot(*vv) < 1e-9 for vv in v)


def test_never_plans_a_command_into_a_close_obstacle():
    p = plan((0, 0, 0), (10, 0), lidar((0, 0, 0), [(1.1, 0.0, 0.3)]))
    vx, _, wz = p.cmd
    ahead = [(1.1 - 0.3, 0.0)]
    assert not (vx > 0 and abs(wz) < 0.2), f"drove straight at an obstacle 0.8 m ahead: {p}"


@pytest.mark.parametrize("goal_angle", [math.pi, -math.pi / 2, math.pi / 2])
def test_turns_toward_a_goal_behind_it(goal_angle):
    t, _, _ = simulate((0, 0, 0), (8 * math.cos(goal_angle), 8 * math.sin(goal_angle)), [])
    assert t is not None
