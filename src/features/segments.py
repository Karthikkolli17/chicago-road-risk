from __future__ import annotations

import logging
import re

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from src import config
from src.features.counts import YEARS
from src.features.snap import in_network, police_key, segment_name_keys
from src.ingest.context import COMMUNITY_AREAS_PATH, POIS_PATH
from src.ingest.osm import NODES_PATH, OSM_DIR, SEGMENTS_PATH

log = logging.getLogger(__name__)

MIN_LENGTH_M = 20
NEAR_SCHOOL_M = 150
MODEL_SEGMENTS_PATH = OSM_DIR / "model_segments.parquet"

# OSM highway tag -> road class, most important first
_CLASS_RANK = {"motorway": 6, "trunk": 5, "primary": 4, "secondary": 3, "tertiary": 2,
               "unclassified": 1, "residential": 0}
_CLASS_ALIASES = {"living_street": "residential"}

# columns the model may use (community_area is for grouping folds, not a feature)
FEATURES = ["length_m", "road_class_rank", "is_link", "lanes", "lanes_imputed", "oneway",
            "speed_mph", "sinuosity", "max_degree", "has_dead_end", "has_signal", "has_stop",
            "is_bridge", "is_tunnel", "is_roundabout", "dist_school_m", "near_school",
            "dist_hospital_m"]
COUNTS = ["crashes", "ksi", "fatal", "ped_crashes", "bike_crashes", "vru_crashes", "ksi_vru"]


# road attributes
def road_class(highway) -> tuple[str, bool]:

    best, link = "residential", False
    for tag in str(highway or "").split(";"):
        is_link = tag.endswith("_link")
        tag = _CLASS_ALIASES.get(tag.removesuffix("_link"), tag.removesuffix("_link"))
        if tag in _CLASS_RANK and _CLASS_RANK[tag] >= _CLASS_RANK[best]:
            best, link = tag, is_link
    return best, link


def max_number(value) -> float:

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value))]
    return max(nums) if nums else np.nan


def has_tag(value, tags: set[str]) -> bool:
    return value is not None and not (isinstance(value, float) and np.isnan(value)) and \
        any(t in tags for t in str(value).split(";"))


# merging
def merge_short(segs: pd.DataFrame, min_length_m: float = MIN_LENGTH_M) -> pd.Series:

    segs = segs.reset_index(drop=True)
    at_node: dict[int, list[int]] = {}
    for i, (u, v) in enumerate(zip(segs["u"], segs["v"])):
        at_node.setdefault(u, []).append(i)
        if v != u:
            at_node.setdefault(v, []).append(i)

    parent = list(range(len(segs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    name, cls, length = segs["name"].to_numpy(), segs["road_class"].to_numpy(), segs["length_m"].to_numpy()
    for i in np.flatnonzero(length < min_length_m):
        cands = {j for n in (segs.at[i, "u"], segs.at[i, "v"]) for j in at_node[n] if j != i}
        if not cands:
            continue
        same_name = lambda j: isinstance(name[i], str) and name[i] == name[j]
        j = max(cands, key=lambda j: (same_name(j), cls[j] == cls[i], length[j], segs.at[j, "segment_id"]))
        parent[find(i)] = find(j)

    root = pd.Series([find(i) for i in range(len(segs))])
    anchor = (pd.DataFrame({"root": root, "length_m": length, "segment_id": segs["segment_id"]})
              .sort_values(["length_m", "segment_id"], ascending=[False, True])
              .drop_duplicates("root").set_index("root")["segment_id"])
    return pd.Series(root.map(anchor).to_numpy(), index=segs["segment_id"], name="group_id")


# inputs
def load_segments() -> gpd.GeoDataFrame:
    s = gpd.read_parquet(SEGMENTS_PATH)
    s = s[in_network(s["highway"])].reset_index(drop=True)
    cls = s["highway"].map(road_class)
    s["road_class"] = cls.str[0]
    s["road_class_rank"] = s["road_class"].map(_CLASS_RANK)
    s["is_link"] = cls.str[1]
    return s


def street_speeds(con: duckdb.DuckDBPyConnection) -> pd.Series:

    c = con.execute("""
        SELECT street_name, street_direction, posted_speed_limit
        FROM crashes WHERE posted_speed_limit BETWEEN 10 AND 55
    """).df()
    c["key"] = [police_key(n, d) for n, d in zip(c["street_name"], c["street_direction"])]
    return c.dropna(subset=["key"]).groupby("key")["posted_speed_limit"].median()


def segment_speed(segs: pd.DataFrame, keys: list[tuple[str, ...]], by_street: pd.Series) -> pd.DataFrame:
    """OSM maxspeed, else the street's police speed, else the road-class median."""
    osm = segs["maxspeed"].map(max_number)
    police = pd.Series([max((by_street[k] for k in ks if k in by_street.index), default=np.nan)
                        for ks in keys], index=segs.index)
    speed = osm.fillna(police)
    default = speed.groupby(segs["road_class"]).transform("median").fillna(30)
    source = np.select([osm.notna(), police.notna()], ["osm", "police_street"], "class_default")
    return pd.DataFrame({"speed_mph": speed.fillna(default), "speed_source": source}, index=segs.index)


# build
def _nearest_m(geoms: gpd.GeoSeries, pois: gpd.GeoDataFrame) -> np.ndarray:
    idx, dist = pois.sindex.nearest(geoms, return_distance=True, return_all=False)
    out = np.full(len(geoms), np.nan)
    out[idx[0]] = dist
    return out / config.FT_PER_M


def _community_area(geoms: gpd.GeoSeries, areas: gpd.GeoDataFrame) -> np.ndarray:

    mid = gpd.GeoSeries(shapely.line_interpolate_point(geoms.values, 0.5, normalized=True), crs=geoms.crs)
    idx, _ = areas.sindex.nearest(mid, return_distance=True, return_all=False)
    out = np.empty(len(geoms), dtype=int)
    out[idx[0]] = areas["area_number"].to_numpy()[idx[1]]
    return out


def segment_attributes(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:

    segs = load_segments()
    segs["lanes_osm"] = segs["lanes"].map(max_number)
    segs = segs.join(segment_speed(segs, segment_name_keys(segs["name"]), street_speeds(con)))
    segs["oneway"] = segs["oneway"].astype(bool)
    segs["is_bridge"] = segs["bridge"].notna()
    segs["is_tunnel"] = segs["tunnel"].map(lambda t: has_tag(t, {"yes"}))
    segs["is_roundabout"] = segs["junction"].map(lambda t: has_tag(t, {"roundabout", "circular"}))
    return segs


def build(con: duckdb.DuckDBPyConnection) -> gpd.GeoDataFrame:
    segs = segment_attributes(con)
    nodes = pd.read_parquet(NODES_PATH, columns=["osmid", "highway", "street_count"]).set_index("osmid")
    pois = gpd.read_parquet(POIS_PATH)
    areas = gpd.read_parquet(COMMUNITY_AREAS_PATH)
    segs["group_id"] = merge_short(segs).to_numpy()

    # group geometry and length; road attributes come from the group's longest member
    anchor = segs[segs["segment_id"] == segs["group_id"]].set_index("group_id")
    geom = segs.groupby("group_id")["geometry"].agg(lambda g: shapely.line_merge(shapely.union_all(g.values)))
    g = gpd.GeoDataFrame(anchor[["road_class", "road_class_rank", "is_link", "lanes_osm", "oneway",
                                 "speed_mph", "speed_source", "is_bridge", "is_tunnel", "is_roundabout",
                                 "name"]],
                         geometry=geom.reindex(anchor.index).values, crs=segs.crs)
    g["length_m"] = segs.groupby("group_id")["length_m"].sum()
    g["n_parts"] = segs.groupby("group_id").size()

    # lanes: OSM where tagged, else the class median
    g["lanes_imputed"] = g["lanes_osm"].isna()
    g["lanes"] = g["lanes_osm"].fillna(g.groupby("road_class")["lanes_osm"].transform("median")).fillna(2)

    # shape: length over the straight line between the anchor's ends (loops capped)
    a_geom = anchor.geometry.values
    chord = shapely.distance(shapely.get_point(a_geom, 0), shapely.get_point(a_geom, -1)) / config.FT_PER_M
    g["sinuosity"] = np.clip(anchor["length_m"].to_numpy() / np.maximum(chord, 1.0), 1.0, 5.0)

    # junctions: every node the group touches
    ends = pd.concat([segs[["group_id", "u"]].rename(columns={"u": "node"}),
                      segs[["group_id", "v"]].rename(columns={"v": "node"})]).drop_duplicates()
    ends = ends.join(nodes, on="node")
    j = ends.groupby("group_id").agg(
        max_degree=("street_count", "max"),
        has_dead_end=("street_count", lambda x: bool((x == 1).any())),
        has_signal=("highway", lambda x: bool((x == "traffic_signals").any())),
        has_stop=("highway", lambda x: bool((x == "stop").any())))
    g = g.join(j)

    # context
    g["dist_school_m"] = _nearest_m(g.geometry, pois[pois["kind"] == "school"])
    g["near_school"] = g["dist_school_m"] <= NEAR_SCHOOL_M
    g["dist_hospital_m"] = _nearest_m(g.geometry, pois[pois["kind"] == "hospital"])
    g["community_area"] = _community_area(g.geometry, areas)

    g.index.name = "model_segment_id"
    segs[["segment_id", "group_id"]].rename(columns={"group_id": "model_segment_id"}) \
        .to_parquet(OSM_DIR / "segment_groups.parquet", index=False)
    return g.reset_index()


def write_tables(con: duckdb.DuckDBPyConnection, g: gpd.GeoDataFrame, years=YEARS) -> None:

    groups = pd.read_parquet(OSM_DIR / "segment_groups.parquet")
    feats = pd.DataFrame(g.drop(columns="geometry"))
    con.register("_groups", groups)
    con.register("_feats", feats)
    con.execute("CREATE OR REPLACE TABLE segment_groups AS SELECT * FROM _groups")
    con.execute(f"""
        CREATE OR REPLACE TABLE model_segments AS
        WITH c AS (
            SELECT g.model_segment_id, c.*
            FROM crash_locations l JOIN crashes c USING (crash_record_id)
            JOIN segment_groups g USING (segment_id)
            WHERE l.junction_id IS NULL AND c.crash_year BETWEEN {years[0]} AND {years[1]}),
        e AS (
            SELECT g.model_segment_id, sum(m.midblock_m) AS midblock_m
            FROM segment_groups g JOIN segment_midblock m USING (segment_id) GROUP BY 1)
        SELECT f.*, e.midblock_m,
               count(c.crash_record_id) AS crashes, count(*) FILTER (c.is_ksi) AS ksi,
               count(*) FILTER (c.most_severe_injury = 'FATAL') AS fatal,
               count(*) FILTER (c.has_pedestrian) AS ped_crashes,
               count(*) FILTER (c.has_cyclist) AS bike_crashes,
               count(*) FILTER (c.has_pedestrian OR c.has_cyclist) AS vru_crashes,
               count(*) FILTER (c.is_ksi AND (c.has_pedestrian OR c.has_cyclist)) AS ksi_vru
        FROM _feats f JOIN e USING (model_segment_id) LEFT JOIN c USING (model_segment_id)
        GROUP BY ALL ORDER BY model_segment_id
    """)
    con.unregister("_groups")
    con.unregister("_feats")
    g[["model_segment_id", "name", "geometry"]].to_crs("EPSG:4326").to_parquet(MODEL_SEGMENTS_PATH)


# checks & report
def checks(con: duckdb.DuckDBPyConnection) -> list[str]:
    fails = []
    q = lambda sql: con.execute(sql).fetchone()
    n_net, n_grouped = q("SELECT (SELECT count(*) FROM segment_counts), (SELECT count(*) FROM segment_groups)")
    if n_net != n_grouped:
        fails.append(f"segment_groups has {n_grouped} rows, network has {n_net}")
    for c in COUNTS:  # every matched crash is either mid-block or at an intersection
        a, b, j = q(f"SELECT (SELECT sum({c}) FROM segment_counts), (SELECT sum({c}) FROM model_segments), "
                    f"(SELECT sum({c}) FROM model_junctions)")
        if a != b + j:
            fails.append(f"{c}: {a} on segments vs {b} mid-block + {j} at intersections")
    if q("SELECT count(*) FROM model_segments WHERE midblock_m < 0 OR midblock_m > length_m * 1.01")[0]:
        fails.append("mid-block length outside 0..length")
    a, b = q("SELECT (SELECT sum(length_m) FROM segment_counts), (SELECT sum(length_m) FROM model_segments)")
    if abs(a - b) > 1e-6 * a:
        fails.append(f"length: {a:.0f} vs {b:.0f}")
    nulls = [f for f in FEATURES + ["community_area"]
             if q(f"SELECT count(*) FROM model_segments WHERE {f} IS NULL")[0]]
    if nulls:
        fails.append(f"missing values in {nulls}")
    if q("SELECT count(*) FROM model_segments WHERE community_area NOT BETWEEN 1 AND 77")[0]:
        fails.append("community area outside 1..77")
    return fails


def summarise(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    q = lambda sql: con.execute(sql).df()
    return {
        "merging": q(f"""SELECT (SELECT count(*) FROM segment_counts) AS network_segments,
                                count(*) AS model_segments,
                                count(*) FILTER (n_parts > 1) AS merged_groups,
                                count(*) FILTER (length_m < {MIN_LENGTH_M}) AS still_under_{MIN_LENGTH_M}m
                         FROM model_segments"""),
        "by_class": q("""SELECT road_class, count(*) segments, round(sum(length_m)/1000) km,
                                round(sum(midblock_m)/1000) midblock_km, sum(ksi) midblock_ksi,
                                round(1000 * sum(ksi) / sum(midblock_m), 2) ksi_per_midblock_km
                         FROM model_segments GROUP BY 1 ORDER BY min(-road_class_rank)"""),
        "sources": q("""SELECT speed_source, count(*) segments, round(avg(speed_mph), 1) avg_mph
                        FROM model_segments GROUP BY 1 ORDER BY 2 DESC"""),
        "near_school": q("""SELECT near_school, count(*) segments, round(sum(length_m)/1000) km, sum(ksi) ksi,
                                   sum(ksi_vru) ksi_vru, round(1000 * sum(ksi) / sum(midblock_m), 2) ksi_per_km
                            FROM model_segments GROUP BY 1 ORDER BY 1"""),
        "features": q("SELECT " + ", ".join(
            f"round(avg({f}::DOUBLE), 2) AS {f}" for f in FEATURES) + " FROM model_segments").T
            .reset_index().set_axis(["feature", "mean"], axis=1),
    }


def write_report(results: dict[str, pd.DataFrame], fails: list[str], path) -> None:
    text = [f"# Model segments and features ({YEARS[0]}–{YEARS[1]} counts)\n",
            "## Checks\n\n" + ("All passed." if not fails else "\n".join(f"- FAIL {f}" for f in fails))]
    titles = {"merging": f"Merging segments under {MIN_LENGTH_M} m", "by_class": "Mid-block serious crashes by road class",
              "sources": "Where the speed limit came from", "near_school": f"Within {NEAR_SCHOOL_M} m of a school",
              "features": "Feature means"}
    for k, t in titles.items():
        text.append(f"\n## {t}\n\n" + results[k].to_markdown(index=False))
    path.write_text("\n".join(text) + "\n")


def run() -> None:
    con = duckdb.connect(str(config.WAREHOUSE))
    g = build(con)
    write_tables(con, g)
    fails, results = checks(con), summarise(con)
    for name, df in results.items():
        log.info("%s\n%s", name, df.to_string(index=False))
    config.REPORTS.mkdir(exist_ok=True)
    write_report(results, fails, config.REPORTS / "segment_features.md")
    con.close()
    if fails:
        raise RuntimeError("model segment checks failed: " + "; ".join(fails))
    log.info("model_segments: %d rows", len(g))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
