"""Road segmentation: roads are cut into pieces of at most segment_length_m."""

import pytest
from shapely.geometry import LineString

from src.geospatial.spatial.segment_roads import detect_intersections, split_line


@pytest.mark.parametrize("length", [250.0, 1000.0, 99.5, 100.0, 37.0])
def test_pieces_are_at_most_100m_and_cover_the_road(length):
    road = LineString([(0, 0), (length, 0)])
    pieces = split_line(road, 100)
    assert all(p.length <= 100 + 1e-6 for p in pieces)
    assert sum(p.length for p in pieces) == pytest.approx(length)


def test_short_road_stays_one_piece():
    road = LineString([(0, 0), (60, 0)])
    assert split_line(road, 100) == [road]


def test_pieces_are_contiguous_on_a_bent_road():
    road = LineString([(0, 0), (150, 0), (150, 130)])  # 280 m with a corner
    pieces = split_line(road, 100)
    assert len(pieces) == 3
    for a, b in zip(pieces, pieces[1:]):
        assert a.coords[-1] == pytest.approx(b.coords[0])


def test_intersection_needs_three_road_ends():
    import geopandas as gpd

    roads = gpd.GeoDataFrame(geometry=[
        LineString([(0, 0), (100, 0)]),
        LineString([(100, 0), (200, 0)]),
        LineString([(100, 0), (100, 100)]),   # third road ending at (100, 0) -> intersection
        LineString([(500, 0), (600, 0)]),     # isolated road, no intersection
    ], crs="EPSG:32647")
    nodes = detect_intersections(roads, "EPSG:32647")
    assert len(nodes) == 1
    assert (nodes.geometry.x.iloc[0], nodes.geometry.y.iloc[0]) == (100, 0)
