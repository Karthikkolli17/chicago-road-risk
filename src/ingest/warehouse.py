from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from src import config

log = logging.getLogger(__name__)


def _sql_list(values) -> str:
    return ", ".join(f"'{v}'" for v in values)


def _latest_version(con: duckdb.DuckDBPyConnection, folder: Path, key: str) -> str:
    
    files = sorted(folder.glob("month=*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {folder}")
    table = f"_files_{folder.name}"
    con.register(table, pd.DataFrame({
        "filename": [str(p) for p in files],
        "pulled_at": [p.stat().st_mtime for p in files],
    }))
    paths = ", ".join("'" + str(p).replace("'", "''") + "'" for p in files)
    return f"""
        SELECT r.* EXCLUDE (filename)
        FROM read_parquet([{paths}], filename = true, union_by_name = true) r
        JOIN {table} f USING (filename)
        QUALIFY row_number() OVER (PARTITION BY r.{key} ORDER BY f.pulled_at DESC, r.filename DESC) = 1
    """


def load_crashes(con: duckdb.DuckDBPyConnection, raw_dir: Path = config.RAW) -> None:
    b = config.CITY_BBOX
    con.execute(f"""
        CREATE OR REPLACE TABLE raw_crashes AS
        {_latest_version(con, raw_dir / "crashes", "crash_record_id")};

        CREATE OR REPLACE TABLE raw_people_vru AS
        {_latest_version(con, raw_dir / "people_vru", "person_id")};

        CREATE OR REPLACE TABLE crashes AS
        WITH flags AS (
            SELECT crash_record_id,
                   bool_or(person_type = 'PEDESTRIAN') AS has_pedestrian,
                   bool_or(person_type = 'BICYCLE')    AS has_cyclist
            FROM raw_people_vru GROUP BY 1
        )
        SELECT c.*,
               year(c.crash_date) AS crash_year,
               coalesce(c.most_severe_injury IN ({_sql_list(config.KSI_INJURIES)}), false) AS is_ksi,
               coalesce(f.has_pedestrian, false) AS has_pedestrian,
               coalesce(f.has_cyclist, false)    AS has_cyclist,
               CASE
                 WHEN c.latitude IS NULL OR c.longitude IS NULL THEN 'missing'
                 WHEN c.latitude = 0 OR c.longitude = 0         THEN 'zero'
                 WHEN c.latitude  NOT BETWEEN {b['lat_min']} AND {b['lat_max']}
                   OR c.longitude NOT BETWEEN {b['lon_min']} AND {b['lon_max']} THEN 'outside_city'
                 ELSE 'ok'
               END AS coord_status
        FROM raw_crashes c LEFT JOIN flags f USING (crash_record_id);
    """)


def load_segments(con: duckdb.DuckDBPyConnection, segments_path: Path) -> None:
    # Geometry stays as WKB bytes; geopandas reads the GeoParquet directly for spatial work.
    con.execute(f"CREATE OR REPLACE TABLE segments AS SELECT * FROM read_parquet('{segments_path}')")


def checks(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    by_year = con.execute("""
        SELECT crash_year,
               count(*)                                   AS crashes,
               count(*) FILTER (is_ksi)                   AS ksi,
               count(*) FILTER (has_pedestrian)           AS with_pedestrian,
               count(*) FILTER (has_cyclist)              AS with_cyclist,
               round(100 * avg((coord_status <> 'ok')::INT), 2) AS pct_bad_coords
        FROM crashes GROUP BY 1 ORDER BY 1
    """).df()
    coords = con.execute("""
        SELECT coord_status, count(*) AS crashes, count(*) FILTER (is_ksi) AS ksi,
               round(100 * count(*) / sum(count(*)) OVER (), 2) AS pct
        FROM crashes GROUP BY 1 ORDER BY 2 DESC
    """).df()
    orphans = con.execute("""
        SELECT count(*) AS vru_people_without_crash
        FROM raw_people_vru p ANTI JOIN raw_crashes c USING (crash_record_id)
    """).df()
    return {"by_year": by_year, "coords": coords, "orphans": orphans}


MAX_BAD_COORDS_PCT = 2.0  # about 1% today
MIN_YEAR_SHARE = 0.8  # a full year with under 80% of the median year's crashes means missing months


def gate(results: dict[str, pd.DataFrame], today: date | None = None) -> list[str]:

    fails = []
    y = results["by_year"]
    full = y[y["crash_year"] < (today or date.today()).year]
    low = full[full["crashes"] < MIN_YEAR_SHARE * full["crashes"].median()]
    if len(low):
        fails.append(f"years with too few crashes: {low['crash_year'].tolist()}")
    bad = 100 - results["coords"].set_index("coord_status")["pct"].get("ok", 0)
    if bad > MAX_BAD_COORDS_PCT:
        fails.append(f"{bad:.2f}% of crashes have no usable location")
    if results["orphans"]["vru_people_without_crash"].iloc[0]:
        fails.append(f"{results['orphans']['vru_people_without_crash'].iloc[0]} people rows without a crash")
    return fails


def write_report(results: dict[str, pd.DataFrame], extra: dict[str, object], path: Path) -> None:
    lines = [f"# Ingest checks\n\nGenerated {datetime.now():%Y-%m-%d %H:%M}.\n"]
    for k, v in extra.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("\n## Crashes per year\n\n" + results["by_year"].to_markdown(index=False))
    lines.append("\n## Coordinate quality\n\nCrashes not `ok` are kept in the warehouse "
                 "but cannot be snapped to a road.\n\n" + results["coords"].to_markdown(index=False))
    lines.append("\n## People rows\n\n" + results["orphans"].to_markdown(index=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run() -> dict[str, pd.DataFrame]:
    from src.ingest.osm import SEGMENTS_PATH

    con = duckdb.connect(str(config.WAREHOUSE))
    load_crashes(con)
    if SEGMENTS_PATH.exists():
        load_segments(con, SEGMENTS_PATH)
        n_seg = con.execute("SELECT count(*) FROM segments").fetchone()[0]
    else:
        log.warning("no OSM segments yet; run `python -m src.ingest.osm`")
        n_seg = "not loaded"
    results = checks(con)
    for name, df in results.items():
        log.info("%s\n%s", name, df.to_string(index=False))
    write_report(results, {"Road segments (motorways excluded)": n_seg}, config.REPORTS / "ingest_checks.md")
    con.close()
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
