from __future__ import annotations

import logging
import math

import duckdb
import geopandas as gpd
import mapbox_vector_tile as mvt
import numpy as np
import pandas as pd
import shapely

from src import config
from src.features import junctions as J
from src.features.segments import load_segments
from src.ingest.mapillary import DETECTIONS_PATH, ZOOM, city_tiles, tile_path
from src.ingest.osm import NODES_PATH

log = logging.getLogger(__name__)

ATTACH_M = 25  # a sign or light belongs to its nearest node if it is this close
MIDBLOCK_M = 15  # a mid-block crosswalk belongs to its nearest road within this distance
COVER_M = 25  # a place counts as photographed if a photo route passes this close
SAMPLE_M = 10  # spacing of the points used to measure a road's photo coverage

JUNCTION_FEATURES = ["mly_signal", "mly_stop", "n_crosswalks", "mly_covered"]
SEGMENT_FEATURES = ["mly_signal", "mly_stop", "midblock_crosswalks", "mly_coverage"]


def kind(value: str) -> str | None:

    if value.startswith("object--traffic-light--general"):
        return "signal"
    if value.startswith("regulatory--stop--"):
        return "stop"
    if value in ("marking--discrete--crosswalk-zebra", "construction--flat--crosswalk-plain"):
        return "crosswalk"
    return None


def _project(lon, lat) -> np.ndarray:
    g = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs(config.CRS)
    return shapely.points(g.x.to_numpy(), g.y.to_numpy())


def load_detections() -> pd.DataFrame:
    det = pd.read_parquet(DETECTIONS_PATH)
    det["kind"] = det["value"].map(kind)
    det = det[det["kind"].notna()].reset_index(drop=True)
    det["geometry"] = _project(det["lon"], det["lat"])
    return det


def load_routes() -> np.ndarray:

    n = 2 ** ZOOM
    lat = lambda t: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * t / n))))  # noqa: E731
    lines = []
    for x, y in city_tiles():
        layer = mvt.decode(tile_path("coverage", x, y).read_bytes()).get("sequence")
        if not layer:
            continue
        ext = layer.get("extent", 4096)
        for f in layer["features"]:
            g = f["geometry"]
            parts = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
            for p in parts:
                if len(p) > 1:
                    lines.append(shapely.LineString(
                        [((x + px / ext) / n * 360 - 180, lat(y + 1 - py / ext)) for px, py in p]))
    return gpd.GeoSeries(lines, crs="EPSG:4326").to_crs(config.CRS).values


# attaching
def nearest_within(pts: np.ndarray, targets: np.ndarray, max_ft: float) -> np.ndarray:

    out = np.full(len(pts), -1)
    if len(pts) == 0:
        return out
    (i, j), d = shapely.STRtree(targets).query_nearest(pts, max_distance=max_ft, return_distance=True,
                                                       all_matches=False)
    out[i] = j
    return out


def node_flags(det: pd.DataFrame, nodes: pd.DataFrame, max_ft: float) -> pd.DataFrame:

    at = nearest_within(det["geometry"].to_numpy(), shapely.points(nodes["x"], nodes["y"]), max_ft)
    hit = det.assign(node=at)[at >= 0]
    osmid = nodes["osmid"].to_numpy()
    out = pd.DataFrame({"osmid": osmid})
    for k, col in (("signal", "mly_signal"), ("stop", "mly_stop"), ("crosswalk", "n_crosswalks")):
        c = np.bincount(hit.loc[hit["kind"] == k, "node"], minlength=len(osmid))
        out[col] = c if k == "crosswalk" else c > 0
    out["attached"] = np.bincount(hit["node"], minlength=len(osmid))
    return out


def covered_points(pts: np.ndarray, routes: np.ndarray, max_ft: float) -> np.ndarray:
    hit = np.zeros(len(pts), dtype=bool)
    i, _ = shapely.STRtree(routes).query(pts, predicate="dwithin", distance=max_ft)
    hit[np.unique(i)] = True
    return hit


def length_coverage(geoms: np.ndarray, routes: np.ndarray, step_ft: float, max_ft: float) -> np.ndarray:

    lengths = shapely.length(geoms)
    n = np.maximum(np.ceil(lengths / step_ft).astype(int), 1)
    owner = np.repeat(np.arange(len(geoms)), n)
    frac = np.concatenate([(np.arange(k) + 0.5) / k for k in n])
    pts = shapely.line_interpolate_point(geoms[owner], frac, normalized=True)
    hit = covered_points(pts, routes, max_ft)
    return np.bincount(owner, weights=hit, minlength=len(geoms)) / n


# build
def build() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ft = config.FT_PER_M
    segs = load_segments()
    nodes = pd.read_parquet(NODES_PATH, columns=["osmid", "x", "y", "highway"])
    nodes = nodes[nodes["osmid"].isin(pd.concat([segs["u"], segs["v"]]))].reset_index(drop=True)
    det = load_detections()
    routes = load_routes()

    flags = node_flags(det, nodes, ATTACH_M * ft)
    # photographed: a route passes close, or something was seen there from further away
    flags["covered"] = covered_points(shapely.points(nodes["x"], nodes["y"]), routes, COVER_M * ft) \
        | (flags["attached"] > 0)

    # intersections: any node of the cluster
    jn = J.junction_nodes(segs, nodes.set_index("osmid"))
    jn["junction_id"] = J.cluster(jn["x"].to_numpy(), jn["y"].to_numpy(), jn["osmid"].to_numpy(),
                                  J.CLUSTER_M * ft)
    j = jn[["osmid", "junction_id"]].merge(flags, on="osmid").groupby("junction_id").agg(
        mly_signal=("mly_signal", "any"), mly_stop=("mly_stop", "any"),
        n_crosswalks=("n_crosswalks", "sum"), mly_covered=("covered", "any")).reset_index()

    # road stretches: signs at any node the group touches; crosswalks left over go mid-block
    groups = pd.read_parquet(config.DATA / "osm" / "segment_groups.parquet")
    s = segs[["segment_id", "u", "v", "geometry"]].merge(groups, on="segment_id")
    ends = pd.concat([s[["model_segment_id", "u"]].rename(columns={"u": "osmid"}),
                      s[["model_segment_id", "v"]].rename(columns={"v": "osmid"})]).drop_duplicates()
    g = ends.merge(flags, on="osmid").groupby("model_segment_id").agg(
        mly_signal=("mly_signal", "any"), mly_stop=("mly_stop", "any"))

    at_node = nearest_within(det["geometry"].to_numpy(), shapely.points(nodes["x"], nodes["y"]), ATTACH_M * ft)
    loose = det[(det["kind"] == "crosswalk") & (at_node < 0)]
    seg_geoms = s["geometry"].to_numpy()
    at_seg = nearest_within(loose["geometry"].to_numpy(), seg_geoms, MIDBLOCK_M * ft)
    mid = pd.Series(s["model_segment_id"].to_numpy()[at_seg[at_seg >= 0]]).value_counts()
    g["midblock_crosswalks"] = mid.reindex(g.index).fillna(0).astype(int)

    cov = length_coverage(seg_geoms, routes, SAMPLE_M * ft, COVER_M * ft)
    w = shapely.length(seg_geoms)
    g["mly_coverage"] = (pd.Series(cov * w).groupby(s["model_segment_id"].to_numpy()).sum()
                         / pd.Series(w).groupby(s["model_segment_id"].to_numpy()).sum())
    g = g.reset_index()

    stats = {"detections": len(det), "attached_to_node": int((at_node >= 0).sum()),
             "midblock_crosswalks": int(len(mid) and mid.sum()), "unattached": int((at_node < 0).sum() - mid.sum()),
             "routes": len(routes)}
    return j, g, stats


def write_tables(con: duckdb.DuckDBPyConnection, j: pd.DataFrame, g: pd.DataFrame) -> None:
    con.register("_j", j)
    con.register("_g", g)
    con.execute("CREATE OR REPLACE TABLE mly_junctions AS SELECT * FROM _j ORDER BY junction_id")
    con.execute("CREATE OR REPLACE TABLE mly_segments AS SELECT * FROM _g ORDER BY model_segment_id")
    con.unregister("_j")
    con.unregister("_g")


def checks(con: duckdb.DuckDBPyConnection) -> list[str]:
    fails = []
    q = lambda sql: con.execute(sql).fetchone()  # noqa: E731
    for t, base, key in (("mly_junctions", "model_junctions", "junction_id"),
                         ("mly_segments", "model_segments", "model_segment_id")):
        a, b, c = q(f"SELECT (SELECT count(*) FROM {t}), (SELECT count(*) FROM {base}), "
                    f"(SELECT count(*) FROM {base} JOIN {t} USING ({key}))")
        if not a == b == c:
            fails.append(f"{t}: {a} rows, {base} {b}, joined {c}")
    if q("SELECT count(*) FROM mly_segments WHERE mly_coverage NOT BETWEEN 0 AND 1 OR mly_coverage IS NULL")[0]:
        fails.append("segment coverage outside 0..1")
    for t, cov in (("mly_junctions", "NOT mly_covered"), ("mly_segments", "mly_coverage = 0")):
        n = q(f"SELECT count(*) FROM {t} WHERE {cov} AND mly_signal")[0]
        if n > 0.01 * q(f"SELECT count(*) FROM {t} WHERE mly_signal")[0]:
            fails.append(f"{t}: {n} signals seen where there are no photo routes")
    return fails


def summarise(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("""
        SELECT j.has_signal AS osm_signal, m.mly_signal, m.mly_covered,
               count(*) AS junctions, sum(j.ksi) AS ksi, round(avg(j.ksi), 3) AS ksi_per_junction
        FROM model_junctions j JOIN mly_junctions m USING (junction_id)
        GROUP BY ALL ORDER BY ALL""").df()


def run() -> None:
    j, g, stats = build()
    log.info("mapillary features: %s", stats)
    con = duckdb.connect(str(config.WAREHOUSE))
    write_tables(con, j, g)
    fails = checks(con)
    log.info("signals by source\n%s", summarise(con).to_string(index=False))
    con.close()
    if fails:
        raise RuntimeError("mapillary feature checks failed: " + "; ".join(fails))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
