"""Match each crash to one road segment.

Rule: among segments within 30 m, take the nearest one whose name matches the
street the police recorded. If none matches, take the nearest segment only when it
has no name (often a ramp); if it is named, it is a different street, so the crash
stays unmatched. Crashes with no segment within 30 m are unmatched too. Both kinds
of unmatched crash are counted in the report.
The street name is only used here, never as a model feature.
"""
from __future__ import annotations

import logging
import re

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from src import config
from src.ingest.osm import SEGMENTS_PATH

log = logging.getLogger(__name__)

# Not ordinary streets: buses only, emergency pull-ins, closed roads.
SNAP_EXCLUDED_HIGHWAYS = {"busway", "emergency_bay", "closed"}

_DIRECTIONS = {"N", "S", "E", "W", "NORTH", "SOUTH", "EAST", "WEST"}
_SUFFIXES = {
    "AVENUE": "AVE", "AVE": "AVE", "STREET": "ST", "ST": "ST", "ROAD": "RD", "RD": "RD",
    "DRIVE": "DR", "DR": "DR", "BOULEVARD": "BLVD", "BLVD": "BLVD", "PARKWAY": "PKWY",
    "PKWY": "PKWY", "PLACE": "PL", "PL": "PL", "COURT": "CT", "CT": "CT", "LANE": "LN",
    "LN": "LN", "TERRACE": "TER", "TER": "TER", "HIGHWAY": "HWY", "HWY": "HWY",
    "WAY": "WAY", "PLAZA": "PLZ", "PLZ": "PLZ", "CIRCLE": "CIR", "CIR": "CIR",
    "EXPRESSWAY": "EXPY", "EXPY": "EXPY", "SQUARE": "SQ", "SQ": "SQ", "ROW": "ROW",
}


# Police write lanes of divided roads as e.g. 'LAKE SHORE DR NB'.
_LANE_SUFFIXES = {"NB", "SB", "EB", "WB", "OB", "IB"}
_TRAILING_DIRECTIONS = _LANE_SUFFIXES | {"N", "S", "E", "W"}
_MAIN_ROADS = {"motorway", "trunk", "primary", "secondary"}
# Spelled-out words that the two sources write differently.
_WORDS = {
    "SAINT": "ST", "DOCTOR": "DR", "JUNIOR": "JR", "FIRST": "1ST", "SECOND": "2ND",
    "THIRD": "3RD", "FOURTH": "4TH", "FIFTH": "5TH", "SIXTH": "6TH", "SEVENTH": "7TH",
    "EIGHTH": "8TH", "NINTH": "9TH", "TENTH": "10TH",
}
# Name prefixes written either joined or apart: 'LA SALLE' / 'LaSalle'.
_JOIN_PREFIXES = {"LA", "MC", "MAC", "DE", "DU", "VAN"}
# Renamed streets: police still use the old name.
_ALIASES = {"CONGRESS": "IDA B WELLS", "JAMES M ROCHFORD": "ROCHFORD"}
# East Lake Shore Drive (by Oak St) is a short street of its own; dropping 'East' as
# a direction would merge it with the Drive. Police record it as 'LAKE SHORE DR' with
# address direction E (see police_key); 'LAKE SHORE DR E'/'W' are instead the east and
# west roadways of the Drive itself.
_RAW_ALIASES = {"EAST LAKE SHORE DRIVE": "EASTLAKESHORE"}
_POLICE_ALIASES = {("E", "LAKE SHORE DR"): "EASTLAKESHORE"}


def street_key(name) -> str | None:
    """'South Cicero Avenue', 'CICERO AVE', 'S CICERO AVE' -> 'CICERO'."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    tokens = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper()).split()
    if " ".join(tokens) in _RAW_ALIASES:
        return _RAW_ALIASES[" ".join(tokens)]
    while len(tokens) > 2 and tokens[-1] in _TRAILING_DIRECTIONS and tokens[-2] in _SUFFIXES:
        tokens = tokens[:-1]
    # suffix first, then directions, never removing the last word: so
    # 'West North Avenue' and 'NORTH AVE' both become 'NORTH'
    if len(tokens) > 1 and tokens[-1] in _SUFFIXES:
        tokens = tokens[:-1]
    while len(tokens) > 1 and tokens[0] in _DIRECTIONS:
        tokens = tokens[1:]
    tokens = [_WORDS.get(t, t) for t in tokens]
    joined = []
    for t in tokens:
        if joined and joined[-1] in _JOIN_PREFIXES:
            joined[-1] += t
        else:
            joined.append(t)
    key = " ".join(joined) or None
    return _ALIASES.get(key, key)


# Words one source adds and the other leaves out without naming a different street:
# levels of stacked roads, honorific prefixes, service drives, ramps, directions.
_QUALIFIERS = (
    {"LOWER", "UPPER", "DUSABLE", "JEAN", "BAPTISTE", "POINT", "OLD", "SUB", "SD",
     "RAMP", "OVERPASS"}
    | _DIRECTIONS | _TRAILING_DIRECTIONS | set(_SUFFIXES)
)


def _same_street(a: str, b: str) -> bool:
    """Equal, or one is the other plus qualifier words only: 'WACKER' ~ 'LOWER WACKER'
    and 'LAKE SHORE' ~ 'DUSABLE LAKE SHORE', but 'WELLS' !~ 'IDA B WELLS'."""
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    if f" {short} " not in f" {long_} ":
        return False
    extra = list(long_.split())
    for t in short.split():
        extra.remove(t)
    return all(t in _QUALIFIERS for t in extra)


def police_key(name, direction=None) -> str | None:
    """Key for the street police recorded; the address direction is needed only to
    tell East Lake Shore Drive apart from the Drive."""
    raw = None if name is None or (isinstance(name, float) and np.isnan(name)) else str(name).upper().strip()
    d = None if direction is None or (isinstance(direction, float) and np.isnan(direction)) else str(direction).upper()
    if (d, raw) in _POLICE_ALIASES:
        return _POLICE_ALIASES[(d, raw)]
    return street_key(name)


def is_lane(name) -> bool:
    """'LAKE SHORE DR SB' or 'LAKE SHORE DR E': police name one roadway of a divided road."""
    tokens = str(name or "").upper().split()
    if len(tokens) < 2:
        return False
    return tokens[-1] in _LANE_SUFFIXES or (tokens[-1] in {"N", "S", "E", "W"} and tokens[-2] in _SUFFIXES)


def names_match(crash_key: str | None, segment_keys: tuple[str, ...]) -> bool:
    if not crash_key:
        return False
    return any(_same_street(crash_key, k) for k in segment_keys)


def segment_name_keys(names: pd.Series) -> list[tuple[str, ...]]:
    """OSM names can hold several values joined by ';' (merged ways)."""
    out = []
    for n in names:
        if n is None or (isinstance(n, float) and np.isnan(n)):
            out.append(())
        else:
            out.append(tuple(k for k in (street_key(p) for p in str(n).split(";")) if k))
    return out


def snap(crashes: gpd.GeoDataFrame, segments: gpd.GeoDataFrame,
         max_dist_ft: float = config.SNAP_MAX_FT) -> pd.DataFrame:
    """crashes: crash_record_id, street_name, point geometry. segments: segment_id,
    name, line geometry. Both in the project CRS (feet). One row per crash out."""
    crashes = crashes.reset_index(drop=True)
    segments = segments.reset_index(drop=True)
    ci, si = segments.sindex.query(crashes.geometry, predicate="dwithin", distance=max_dist_ft)
    dist = shapely.distance(crashes.geometry.values[ci], segments.geometry.values[si])

    direction = crashes["street_direction"] if "street_direction" in crashes else pd.Series(None, index=crashes.index)
    crash_keys = np.array([police_key(n, d) for n, d in zip(crashes["street_name"], direction)], dtype=object)
    seg_keys = segment_name_keys(segments["name"])
    match = np.fromiter((names_match(crash_keys[c], seg_keys[s]) for c, s in zip(ci, si)),
                        dtype=bool, count=len(ci))

    unnamed = np.array([len(k) == 0 for k in seg_keys])[si]
    # 'LAKE SHORE DR SB': police name the main lanes, so among same-named roads in
    # range prefer the main carriageway over a local side road of the same name.
    lane = crashes["street_name"].map(is_lane)
    if "highway" in segments:
        main = segments["highway"].fillna("").str.split(";").map(
            lambda tags: any(t in _MAIN_ROADS for t in tags)).to_numpy()
    else:
        main = np.zeros(len(segments), dtype=bool)
    prefer_main = lane.to_numpy()[ci] & main[si] & match
    # Stacked roads (Lower/Upper Wacker, Lower Michigan...): the levels sit metres
    # apart, so distance can't pick one; follow the level the police recorded.
    crash_lower = np.array([bool(k) and "LOWER" in k.split() for k in crash_keys])
    seg_lower = np.array([any("LOWER" in k.split() for k in ks) for ks in seg_keys])
    same_level = match & (crash_lower[ci] == seg_lower[si])
    cand = pd.DataFrame({"c": ci, "s": si, "dist": dist, "match": match, "unnamed": unnamed,
                         "prefer_main": prefer_main, "same_level": same_level,
                         "segment_id": segments["segment_id"].to_numpy()[si]})
    nearest = cand.sort_values(["c", "dist", "segment_id"]).groupby("c").first()
    best = cand.sort_values(["c", "match", "prefer_main", "same_level", "dist", "segment_id"],
                            ascending=[True, False, False, False, True, True])
    best = best.groupby("c").first()
    best["n_candidates"] = cand.groupby("c").size()
    best["nearest_segment_id"] = nearest["segment_id"]

    out = pd.DataFrame({"crash_record_id": crashes["crash_record_id"]})
    out = out.join(best[["segment_id", "dist", "match", "unnamed", "n_candidates", "nearest_segment_id"]])
    no_road = out["segment_id"].isna()
    matched_name = out["match"].fillna(False).astype(bool)
    # Police street not in range: the nearest road has a different name, so it is
    # very likely the wrong road (a tollway, park drive, or a geocode >30 m off).
    # Only an unnamed road (often a ramp) can't contradict the police name.
    other_street = ~no_road & ~matched_name & ~out["unnamed"].fillna(False).astype(bool)
    out["method"] = np.select(
        [no_road, other_street, ~matched_name,
         out["segment_id"] != out["nearest_segment_id"]],
        ["no_road_in_range", "street_not_in_range", "unnamed_road", "name"],
        default="nearest",
    )
    out.loc[other_street, ["segment_id", "dist"]] = None
    out["snap_dist_m"] = out.pop("dist").astype(float) / config.FT_PER_M
    out["name_match"] = out.pop("match").astype("boolean")
    out["n_candidates"] = out["n_candidates"].fillna(0).astype(int)
    return out.drop(columns=["nearest_segment_id", "unnamed"])


def load_inputs(con) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    c = con.execute("""
        SELECT crash_record_id, street_name, street_direction, latitude, longitude
        FROM crashes WHERE coord_status = 'ok'
    """).df()
    crashes = gpd.GeoDataFrame(
        c[["crash_record_id", "street_name", "street_direction"]],
        geometry=gpd.points_from_xy(c["longitude"], c["latitude"]), crs="EPSG:4326",
    ).to_crs(config.CRS)
    segs = gpd.read_parquet(SEGMENTS_PATH, columns=["segment_id", "highway", "name", "geometry"])
    keep = ~segs["highway"].fillna("").str.split(";").map(
        lambda tags: bool(tags) and all(t in SNAP_EXCLUDED_HIGHWAYS for t in tags))
    log.info("snapping to %d of %d segments (dropped %s)", keep.sum(), len(segs),
             segs.loc[~keep, "highway"].value_counts().to_dict())
    return crashes, segs[keep]


def summarise(con) -> dict[str, pd.DataFrame]:
    overall = con.execute("""
        SELECT count(*) AS crashes_total,
               count(*) FILTER (coord_status <> 'ok') AS no_usable_location,
               count(*) FILTER (s.method = 'no_road_in_range') AS no_road_within_30m,
               count(*) FILTER (s.method = 'street_not_in_range') AS police_street_not_within_30m,
               count(*) FILTER (s.segment_id IS NOT NULL) AS matched,
               round(100.0 * count(*) FILTER (s.segment_id IS NOT NULL) / count(*), 2) AS pct_matched
        FROM crashes c LEFT JOIN crash_segments s USING (crash_record_id)
    """).df()
    by_type = con.execute("""
        SELECT CASE WHEN is_ksi THEN 'serious (KSI)' ELSE 'other' END AS crash_type,
               count(*) AS crashes,
               round(100.0 * count(*) FILTER (s.segment_id IS NULL) / count(*), 2) AS pct_unmatched
        FROM crashes c LEFT JOIN crash_segments s USING (crash_record_id)
        GROUP BY 1 ORDER BY 1 DESC
    """).df()
    by_year = con.execute("""
        SELECT crash_year, count(*) AS crashes,
               round(100.0 * count(*) FILTER (s.segment_id IS NULL) / count(*), 2) AS pct_unmatched
        FROM crashes c LEFT JOIN crash_segments s USING (crash_record_id)
        GROUP BY 1 ORDER BY 1
    """).df()
    method = con.execute("""
        SELECT method, count(*) AS crashes,
               round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct,
               round(median(snap_dist_m), 1) AS median_dist_m
        FROM crash_segments GROUP BY 1 ORDER BY 2 DESC
    """).df()
    return {"overall": overall, "by_type": by_type, "by_year": by_year, "method": method}


def write_report(results: dict[str, pd.DataFrame], path) -> None:
    text = [
        "# Snapping checks\n",
        "Each crash with a usable location is matched to a road segment within 30 m. "
        "Where several are in range, a segment whose name matches the street recorded "
        "by police wins; otherwise the nearest wins.\n",
        "## Overall\n\n" + results["overall"].to_markdown(index=False),
        "\n## Unmatched share by crash type\n\n" + results["by_type"].to_markdown(index=False),
        "\n## Unmatched share by year\n\n" + results["by_year"].to_markdown(index=False),
        "\n## How matched crashes were assigned\n\n"
        "`nearest` = the nearest segment, and its name matches the police street; "
        "`name` = a farther segment chosen because its name matches; `unnamed_road` = "
        "no name match, nearest segment has no name; `street_not_in_range` = the "
        "police street is not within 30 m and the nearest road is a different street "
        "(left unmatched); `no_road_in_range` = no segment within 30 m (unmatched).\n\n"
        + results["method"].to_markdown(index=False),
    ]
    path.write_text("\n".join(text) + "\n")


def run() -> None:
    con = duckdb.connect(str(config.WAREHOUSE))
    crashes, segments = load_inputs(con)
    result = snap(crashes, segments)
    con.register("_snap", result)
    con.execute("CREATE OR REPLACE TABLE crash_segments AS SELECT * FROM _snap")
    results = summarise(con)
    for name, df in results.items():
        log.info("%s\n%s", name, df.to_string(index=False))
    write_report(results, config.REPORTS / "snap_checks.md")
    con.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
