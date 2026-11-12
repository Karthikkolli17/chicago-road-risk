from __future__ import annotations

import logging

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from src import config
from src.features.counts import YEARS
from src.features.segments import (_CLASS_RANK, COUNTS, NEAR_SCHOOL_M, _nearest_m, segment_attributes)
from src.features.snap import segment_name_keys
from src.ingest.context import COMMUNITY_AREAS_PATH, POIS_PATH
from src.ingest.osm import NODES_PATH, OSM_DIR

log = logging.getLogger(__name__)

JUNCTION_RADIUS_M = 20
CLUSTER_M = 25
MODEL_JUNCTIONS_PATH = OSM_DIR / "model_junctions.parquet"
MAIN_RANK = _CLASS_RANK["secondary"]

FEATURES = ["n_nodes", "n_approaches", "n_streets", "max_class_rank", "min_class_rank", "n_main_approaches",
            "max_lanes", "max_speed_mph", "n_oneway", "has_signal", "has_stop", "is_roundabout", "has_link",
            "dist_school_m", "near_school", "dist_hospital_m"]


# intersections
def junction_nodes(segs: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:

    deg = pd.concat([segs["u"], segs["v"]]).value_counts()
    ids = deg.index[deg >= 3]
    return nodes.loc[ids, ["x", "y", "highway"]].rename_axis("osmid").reset_index()


def cluster(x: np.ndarray, y: np.ndarray, ids: np.ndarray, tol: float) -> np.ndarray:

    pts = shapely.points(x, y)
    a, b = shapely.STRtree(pts).query(pts, predicate="dwithin", distance=tol)
    parent = np.arange(len(pts))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(a, b):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    root = np.array([find(i) for i in range(len(pts))])
    first = pd.Series(ids).groupby(root).transform("min").to_numpy()
    return np.array([f"J{i}" for i in first], dtype=object)


def locate_crashes(crash_xy: pd.DataFrame, jn: pd.DataFrame, seg_ends: pd.DataFrame, radius: float) -> pd.Series:
    
    of_node = dict(zip(jn["osmid"], jn["junction_id"]))
    ends = seg_ends.set_index("segment_id").reindex(crash_xy["segment_id"])
    ju = ends["u"].map(of_node).to_numpy()
    jv = ends["v"].map(of_node).to_numpy()

    pts = shapely.points(crash_xy["x"], crash_xy["y"])
    nodes = shapely.points(jn["x"], jn["y"])
    ci, ni = shapely.STRtree(nodes).query(pts, predicate="dwithin", distance=radius)
    pair_j = jn["junction_id"].to_numpy()[ni]
    own = (pair_j == ju[ci]) | (pair_j == jv[ci])
    pairs = pd.DataFrame({"crash": ci[own], "junction_id": pair_j[own],
                          "dist": shapely.distance(pts[ci[own]], nodes[ni[own]])})
    best = pairs.sort_values(["crash", "dist", "junction_id"]).drop_duplicates("crash")
    out = np.full(len(crash_xy), None, dtype=object)
    out[best["crash"].to_numpy()] = best["junction_id"].to_numpy()
    return pd.Series(out, index=crash_xy.index, dtype=object)


def midblock_length_m(segs: gpd.GeoDataFrame, jn: pd.DataFrame, radius: float) -> np.ndarray:

    of_node = dict(zip(jn["osmid"], jn["junction_id"]))
    pts_of = jn.groupby("junction_id").apply(lambda d: list(zip(d["x"], d["y"])))
    discs = []
    for u, v in zip(segs["u"], segs["v"]):
        js = {of_node.get(u), of_node.get(v)} - {None}
        pts = [p for j in js for p in pts_of[j]]
        discs.append(shapely.buffer(shapely.multipoints(pts), radius) if pts else None)
    discs = np.array(discs, dtype=object)
    geoms = segs.geometry.values
    out = shapely.length(geoms)
    has = np.array([d is not None for d in discs])
    out[has] = shapely.length(shapely.difference(geoms[has], list(discs[has])))
    return out / config.FT_PER_M


# features
def count_streets(keys) -> int:

    keys = list(keys)
    return len({ks[0] for ks in keys if ks}) + int(any(not ks for ks in keys))


def junction_features(segs: pd.DataFrame, jn: pd.DataFrame) -> pd.DataFrame:

    of_node = dict(zip(jn["osmid"], jn["junction_id"]))
    s = segs.assign(ju=segs["u"].map(of_node), jv=segs["v"].map(of_node))
    s["lanes_f"] = s["lanes_osm"].fillna(s.groupby("road_class")["lanes_osm"].transform("median")).fillna(2)
    s["keys"] = segment_name_keys(s["name"])
    internal = s["ju"].notna() & (s["ju"] == s["jv"])
    app = pd.concat([s[~internal & s["ju"].notna()].assign(junction_id=lambda d: d["ju"]),
                     s[~internal & s["jv"].notna()].assign(junction_id=lambda d: d["jv"])])
    a = app.groupby("junction_id").agg(
        n_approaches=("segment_id", "size"),
        n_streets=("keys", count_streets),
        max_class_rank=("road_class_rank", "max"),
        min_class_rank=("road_class_rank", "min"),
        n_main_approaches=("road_class_rank", lambda r: int((r >= MAIN_RANK).sum())),
        max_lanes=("lanes_f", "max"),
        max_speed_mph=("speed_mph", "max"),
        n_oneway=("oneway", "sum"),
        is_roundabout=("is_roundabout", "any"),
        has_link=("is_link", "any"))
    n = jn.groupby("junction_id").agg(
        n_nodes=("osmid", "size"), x=("x", "mean"), y=("y", "mean"),
        has_signal=("highway", lambda h: bool((h == "traffic_signals").any())),
        has_stop=("highway", lambda h: bool((h == "stop").any())))
    out = n.join(a, how="left")
    # a cluster whose every segment is internal (rare) has no approaches
    fill = {"n_approaches": 0, "n_streets": 0, "n_main_approaches": 0, "n_oneway": 0,
            "is_roundabout": False, "has_link": False}
    out = out.fillna(fill)
    for c in ["max_class_rank", "min_class_rank"]:
        out[c] = out[c].fillna(0)
    out["max_lanes"] = out["max_lanes"].fillna(2)
    out["max_speed_mph"] = out["max_speed_mph"].fillna(30)
    return out


def build(con: duckdb.DuckDBPyConnection) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    segs = segment_attributes(con)
    nodes = pd.read_parquet(NODES_PATH, columns=["osmid", "x", "y", "highway"]).set_index("osmid")
    jn = junction_nodes(segs, nodes)
    jn["junction_id"] = cluster(jn["x"].to_numpy(), jn["y"].to_numpy(), jn["osmid"].to_numpy(),
                                CLUSTER_M * config.FT_PER_M)

    # where every matched crash belongs
    c = con.execute("""SELECT s.crash_record_id, s.segment_id, c.latitude, c.longitude
                       FROM crash_segments s JOIN crashes c USING (crash_record_id)
                       WHERE s.segment_id IS NOT NULL""").df()
    xy = gpd.GeoSeries(gpd.points_from_xy(c["longitude"], c["latitude"]), crs="EPSG:4326").to_crs(config.CRS)
    c["x"], c["y"] = xy.x.to_numpy(), xy.y.to_numpy()
    c["junction_id"] = locate_crashes(c, jn, segs[["segment_id", "u", "v"]], JUNCTION_RADIUS_M * config.FT_PER_M)
    locations = c[["crash_record_id", "segment_id", "junction_id"]]

    midblock = pd.DataFrame({"segment_id": segs["segment_id"],
                             "midblock_m": midblock_length_m(segs, jn, JUNCTION_RADIUS_M * config.FT_PER_M)})

    # intersection features and context
    j = junction_features(segs, jn)
    pts = gpd.GeoSeries(gpd.points_from_xy(j["x"], j["y"]), crs=config.CRS, index=j.index)
    pois = gpd.read_parquet(POIS_PATH)
    areas = gpd.read_parquet(COMMUNITY_AREAS_PATH)
    j["dist_school_m"] = _nearest_m(pts, pois[pois["kind"] == "school"])
    j["near_school"] = j["dist_school_m"] <= NEAR_SCHOOL_M
    j["dist_hospital_m"] = _nearest_m(pts, pois[pois["kind"] == "hospital"])
    idx, _ = areas.sindex.nearest(pts, return_distance=True, return_all=False)
    j["community_area"] = 0
    j.iloc[idx[0], j.columns.get_loc("community_area")] = areas["area_number"].to_numpy()[idx[1]]
    j = gpd.GeoDataFrame(j.drop(columns=["x", "y"]), geometry=pts, crs=config.CRS)
    j.index.name = "junction_id"
    return j.reset_index(), locations, midblock


def write_tables(con: duckdb.DuckDBPyConnection, j: gpd.GeoDataFrame, locations: pd.DataFrame,
                 midblock: pd.DataFrame, years=YEARS) -> None:
    con.register("_loc", locations)
    con.register("_mid", midblock)
    con.register("_j", pd.DataFrame(j.drop(columns="geometry")))
    con.execute("CREATE OR REPLACE TABLE crash_locations AS SELECT * FROM _loc ORDER BY crash_record_id")
    con.execute("CREATE OR REPLACE TABLE segment_midblock AS SELECT * FROM _mid ORDER BY segment_id")
    counts = """count(c.crash_record_id) AS crashes, count(*) FILTER (c.is_ksi) AS ksi,
        count(*) FILTER (c.most_severe_injury = 'FATAL') AS fatal,
        count(*) FILTER (c.has_pedestrian) AS ped_crashes, count(*) FILTER (c.has_cyclist) AS bike_crashes,
        count(*) FILTER (c.has_pedestrian OR c.has_cyclist) AS vru_crashes,
        count(*) FILTER (c.is_ksi AND (c.has_pedestrian OR c.has_cyclist)) AS ksi_vru"""
    con.execute(f"""
        CREATE OR REPLACE TABLE model_junctions AS
        WITH c AS (SELECT l.junction_id, c.* FROM crash_locations l JOIN crashes c USING (crash_record_id)
                   WHERE l.junction_id IS NOT NULL AND c.crash_year BETWEEN {years[0]} AND {years[1]})
        SELECT j.*, {counts}
        FROM _j j LEFT JOIN c USING (junction_id)
        GROUP BY ALL ORDER BY junction_id
    """)
    for t in ["_loc", "_mid", "_j"]:
        con.unregister(t)
    j[["junction_id", "geometry"]].to_crs("EPSG:4326").to_parquet(MODEL_JUNCTIONS_PATH)


def checks(con: duckdb.DuckDBPyConnection, years=YEARS) -> list[str]:
    fails = []
    q = lambda sql: con.execute(sql).fetchone()
    a, b = q("SELECT (SELECT count(*) FROM crash_locations), "
             "(SELECT count(*) FROM crash_segments WHERE segment_id IS NOT NULL)")
    if a != b:
        fails.append(f"crash_locations has {a} rows, matched crashes {b}")
    for c in COUNTS:
        a, b = q(f"SELECT (SELECT sum({c}) FROM model_junctions), (SELECT count(*) FROM crash_locations l "
                 f"JOIN crashes c USING (crash_record_id) WHERE l.junction_id IS NOT NULL "
                 f"AND c.crash_year BETWEEN {years[0]} AND {years[1]} AND "
                 + {"crashes": "true", "ksi": "c.is_ksi", "fatal": "c.most_severe_injury='FATAL'",
                    "ped_crashes": "c.has_pedestrian", "bike_crashes": "c.has_cyclist",
                    "vru_crashes": "(c.has_pedestrian OR c.has_cyclist)",
                    "ksi_vru": "c.is_ksi AND (c.has_pedestrian OR c.has_cyclist)"}[c] + ")")
        if a != b:
            fails.append(f"junction {c}: {a} vs {b}")
    if q("SELECT count(*) FROM segment_midblock WHERE midblock_m < 0 OR midblock_m IS NULL")[0]:
        fails.append("negative or missing mid-block length")
    nulls = [f for f in FEATURES + ["community_area"] if q(f"SELECT count(*) FROM model_junctions WHERE {f} IS NULL")[0]]
    if nulls:
        fails.append(f"missing values in {nulls}")
    return fails


def run() -> None:
    con = duckdb.connect(str(config.WAREHOUSE))
    j, locations, midblock = build(con)
    write_tables(con, j, locations, midblock)
    fails = checks(con)
    s = con.execute(f"""
        SELECT count(*) junctions, count(*) FILTER (n_nodes > 1) multi_node, max(n_nodes) max_nodes,
               sum(crashes) crashes, sum(ksi) ksi, count(*) FILTER (ksi > 0) with_ksi,
               round(avg(ksi), 3) mean_ksi, round(var_samp(ksi), 3) var_ksi, max(ksi) max_ksi
        FROM model_junctions""").df()
    log.info("junctions\n%s", s.to_string(index=False))
    con.close()
    if fails:
        raise RuntimeError("junction checks failed: " + "; ".join(fails))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
