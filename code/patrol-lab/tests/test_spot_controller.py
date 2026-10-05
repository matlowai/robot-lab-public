import math

from robots.spot.controller import WalkLimits, command, wrap


def simulate(start, goal, steps=4000, dt=0.02, lim=WalkLimits()):
    """Ideal unicycle tracking the command, to check the controller converges."""
    x, y, yaw = start
    for i in range(steps):
        (vx, _, wz), arrived = command((x, y, yaw), goal, lim)
        if arrived:
            return i * dt, (x, y, yaw)
        yaw = wrap(yaw + wz * dt)
        x, y = x + vx * math.cos(yaw) * dt, y + vx * math.sin(yaw) * dt
    return None, (x, y, yaw)


def test_turns_in_place_before_walking_when_facing_away():
    (vx, vy, wz), arrived = command((0, 0, 0), (-5, 0.1))
    assert vx == 0 and vy == 0 and abs(wz) == WalkLimits().max_wz and not arrived


def test_reaches_goals_in_every_direction():
    for k in range(8):
        a = k * math.pi / 4
        t, (x, y, _) = simulate((0, 0, 0), (6 * math.cos(a), 6 * math.sin(a)))
        assert t is not None and math.dist((x, y), (6 * math.cos(a), 6 * math.sin(a))) <= WalkLimits().arrival_m


def test_respects_limits():
    lim = WalkLimits()
    for goal in [(50, 0), (0, 50), (-50, -50)]:
        (vx, vy, wz), _ = command((0, 0, 0.3), goal, lim)
        assert 0 <= vx <= lim.max_vx and abs(wz) <= lim.max_wz and vy == 0
