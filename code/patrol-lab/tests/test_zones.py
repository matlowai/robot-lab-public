from events.zones import point_in_polygon, segment_intersection


def test_point_in_polygon_inside_outside_edge():
    sq = ((0, 0), (10, 0), (10, 10), (0, 10))
    assert point_in_polygon((5, 5), sq)
    assert not point_in_polygon((15, 5), sq)
    assert point_in_polygon((10, 5), sq)  # on an edge counts as inside


def test_segment_intersection():
    assert segment_intersection((0, 0), (10, 10), (0, 10), (10, 0)) == (5, 5)
    assert segment_intersection((0, 0), (1, 0), (0, 1), (1, 1)) is None


def test_most_specific_zone_wins(compound):
    assert compound.zone_at((117, 40)).id == "loading_dock"
    assert compound.zone_at((130, 80)).id == "generator_enclosure"
    assert compound.zone_at((5, 95)).id == "yard"
    assert compound.zone_at((200, 50)) is None


def test_boundary_crossing_fence_vs_gate(compound):
    fence = compound.boundary_crossing((155, 60), (145, 60))
    assert fence.inbound and fence.gate_id is None and fence.point == (150, 60)
    gate = compound.boundary_crossing((74, 105), (74, 95))
    assert gate.inbound and gate.gate_id == "gate_north_01"
    outbound = compound.boundary_crossing((145, 60), (155, 60))
    assert not outbound.inbound
    assert compound.boundary_crossing((10, 10), (20, 20)) is None
