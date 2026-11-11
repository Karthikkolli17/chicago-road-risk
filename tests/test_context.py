import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, mapping

from src import config
from src.ingest import osm
from src.ingest.context import classify_poi, tidy_community_areas, tidy_pois


@pytest.mark.parametrize("tags, kind", [
    ({"amenity": "school"}, "school"),
    ({"amenity": "hospital"}, "hospital"),
    ({"healthcare": "hospital"}, "hospital"),
    ({"amenity": "clinic"}, None),
    ({}, None),
])
def test_classify_poi(tags, kind):
    assert classify_poi(pd.Series(tags, dtype=object)) == kind


def _raw_pois():
    # lon/lat around downtown Chicago; ~0.001 deg ~ 100 m
    campus = Polygon([(-87.630, 41.880), (-87.628, 41.880), (-87.628, 41.882), (-87.630, 41.882)])
    rows = [
        ("way", 1, {"amenity": "school", "name": "Campus"}, campus),
        ("node", 2, {"amenity": "school", "name": "Campus pin"}, Point(-87.629, 41.881)),     # inside campus: dup
        ("node", 3, {"amenity": "school", "name": "Other school"}, Point(-87.600, 41.850)),
        ("node", 4, {"amenity": "hospital", "name": "Hospital in campus"}, Point(-87.6291, 41.8811)),  # other kind: keep
        ("node", 5, {"amenity": "cafe", "name": "Cafe"}, Point(-87.61, 41.86)),
    ]
    idx = pd.MultiIndex.from_tuples([(e, i) for e, i, _, _ in rows], names=["element", "id"])
    return gpd.GeoDataFrame([t for _, _, t, _ in rows], index=idx,
                            geometry=[g for *_, g in rows], crs="EPSG:4326")


def test_tidy_pois_dedups_pins_inside_same_kind_campus():
    out = tidy_pois(_raw_pois())
    assert out.crs.to_epsg() == 3435
    assert sorted(out["name"]) == ["Campus", "Hospital in campus", "Other school"]
    assert set(out["osm_id"]) == {"way/1", "node/3", "node/4"}


def test_tidy_community_areas():
    sq = Polygon([(-87.63, 41.88), (-87.62, 41.88), (-87.62, 41.89), (-87.63, 41.89)])
    rows = pd.DataFrame({"the_geom": [mapping(sq), mapping(sq)], "area_numbe": ["2", "1"],
                         "community": ["WEST RIDGE", "ROGERS PARK"]})
    out = tidy_community_areas(rows)
    assert out["area_number"].tolist() == [1, 2]
    assert out["community"].tolist() == ["Rogers Park", "West Ridge"]
    assert out.crs.to_epsg() == 3435


def test_overpass_retry_backs_off_then_uses_mirror(monkeypatch):
    calls, waits = [], []
    err = osm.requests.exceptions.ConnectionError("busy")

    def download():
        calls.append(osm.ox.settings.overpass_url)
        if len(calls) <= 2:
            raise err
        return "graph"

    monkeypatch.setattr(osm, "configure_osmnx", lambda: None)
    assert osm.overpass_retry(download, attempts_per_server=2, sleep=waits.append) == "graph"
    assert calls == [osm.OVERPASS_URLS[0]] * 2 + [osm.OVERPASS_URLS[1]]
    assert waits == [30, 60]


def test_overpass_retry_gives_up(monkeypatch):
    monkeypatch.setattr(osm, "configure_osmnx", lambda: None)

    def download():
        raise osm.requests.exceptions.Timeout("slow")

    with pytest.raises(osm.requests.exceptions.Timeout):
        osm.overpass_retry(download, attempts_per_server=1, sleep=lambda s: None)


def test_clip_to_city_keeps_pois_just_outside():
    from src.ingest.context import clip_to_city
    city = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])], crs=config.CRS)
    pois = gpd.GeoDataFrame({"name": ["in", "near", "far"]},
                            geometry=[Point(500, 500), Point(1500, 500), Point(9000, 500)], crs=config.CRS)
    assert clip_to_city(pois, city, buffer_ft=1000)["name"].tolist() == ["in", "near"]


def test_read_pois_from_pbf(tmp_path):
    """Writes a tiny OSM file: a school node, a school campus way with a pin inside,
    a hospital, and a cafe; the reader must return the first three kinds with geometry."""
    import osmium
    path = tmp_path / "tiny.osm.pbf"
    w = osmium.SimpleWriter(str(path))
    ring = [(1, -87.630, 41.880), (2, -87.628, 41.880), (3, -87.628, 41.882), (4, -87.630, 41.882)]
    for nid, lon, lat in ring:
        w.add_node(osmium.osm.mutable.Node(id=nid, location=(lon, lat), tags={}))
    w.add_node(osmium.osm.mutable.Node(id=10, location=(-87.60, 41.85), tags={"amenity": "school", "name": "A"}))
    w.add_node(osmium.osm.mutable.Node(id=11, location=(-87.61, 41.86), tags={"healthcare": "hospital", "name": "H"}))
    w.add_node(osmium.osm.mutable.Node(id=12, location=(-87.62, 41.87), tags={"amenity": "cafe"}))
    w.add_way(osmium.osm.mutable.Way(id=100, nodes=[1, 2, 3, 4, 1], tags={"amenity": "school", "name": "Campus"}))
    w.close()

    from src.ingest.context import read_pois_from_pbf
    raw = read_pois_from_pbf(path)
    got = {(e, i): (row["name"], row.geometry.geom_type) for (e, i), row in raw.iterrows()}
    assert got == {("node", 10): ("A", "Point"), ("node", 11): ("H", "Point"),
                   ("way", 100): ("Campus", "MultiPolygon")}
    out = tidy_pois(raw)
    assert sorted(out["kind"]) == ["hospital", "school", "school"]
