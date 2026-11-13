from __future__ import annotations

import logging
from datetime import date

import duckdb
import geopandas as gpd
import pandas as pd

from src import config
from src.features.snap import in_network
from src.ingest.osm import SEGMENTS_PATH

log = logging.getLogger(__name__)

N_YEARS = 5


def model_years(today: date | None = None, n: int = N_YEARS) -> tuple[int, int]:
    # the last n full calendar years: in 2026 that is 2021-2025, from January 2027 2022-2026
    last = (today or date.today()).year - 1
    return last - n + 1, last


YEARS = model_years()


def build_counts(con: duckdb.DuckDBPyConnection, network: pd.DataFrame, years=YEARS) -> pd.DataFrame:

    con.register("_network", network[["segment_id", "length_m"]])
    con.execute(f"""
        CREATE OR REPLACE TABLE segment_counts AS
        WITH c AS (
            SELECT s.segment_id, c.is_ksi, c.has_pedestrian, c.has_cyclist,
                   c.most_severe_injury = 'FATAL' AS is_fatal
            FROM crash_segments s JOIN crashes c USING (crash_record_id)
            WHERE s.segment_id IS NOT NULL AND c.crash_year BETWEEN {years[0]} AND {years[1]}
        )
        SELECT n.segment_id, n.length_m,
               count(c.segment_id)                                                     AS crashes,
               count(*) FILTER (c.is_ksi)                                              AS ksi,
               count(*) FILTER (c.is_fatal)                                            AS fatal,
               count(*) FILTER (c.has_pedestrian)                                      AS ped_crashes,
               count(*) FILTER (c.has_cyclist)                                         AS bike_crashes,
               count(*) FILTER (c.has_pedestrian OR c.has_cyclist)                     AS vru_crashes,
               count(*) FILTER (c.is_ksi AND (c.has_pedestrian OR c.has_cyclist))      AS ksi_vru
        FROM _network n LEFT JOIN c USING (segment_id)
        GROUP BY n.segment_id, n.length_m
        ORDER BY n.segment_id
    """)
    con.unregister("_network")
    return con.execute("SELECT * FROM segment_counts").df()


def reconcile(con: duckdb.DuckDBPyConnection, years=YEARS) -> pd.DataFrame:

    return con.execute(f"""
        WITH src AS (
            SELECT count(*) crashes, count(*) FILTER (c.is_ksi) ksi,
                   count(*) FILTER (c.most_severe_injury = 'FATAL') fatal,
                   count(*) FILTER (c.has_pedestrian) ped_crashes,
                   count(*) FILTER (c.has_cyclist) bike_crashes,
                   count(*) FILTER (c.has_pedestrian OR c.has_cyclist) vru_crashes,
                   count(*) FILTER (c.is_ksi AND (c.has_pedestrian OR c.has_cyclist)) ksi_vru
            FROM crash_segments s JOIN crashes c USING (crash_record_id)
            JOIN segment_counts g USING (segment_id)
            WHERE c.crash_year BETWEEN {years[0]} AND {years[1]}
        ), agg AS (
            SELECT sum(crashes) crashes, sum(ksi) ksi, sum(fatal) fatal, sum(ped_crashes) ped_crashes,
                   sum(bike_crashes) bike_crashes, sum(vru_crashes) vru_crashes, sum(ksi_vru) ksi_vru
            FROM segment_counts
        )
        SELECT 'matched crashes' AS source, * FROM src
        UNION ALL SELECT 'sum over segments', * FROM agg
    """).df()


def summarise(con: duckdb.DuckDBPyConnection, years=YEARS) -> dict[str, pd.DataFrame]:
    coverage = con.execute(f"""
        SELECT count(*) AS crashes_{years[0]}_{years[1]},
               count(*) FILTER (s.segment_id IS NOT NULL) AS on_a_segment,
               round(100.0 * count(*) FILTER (s.segment_id IS NOT NULL) / count(*), 2) AS pct_on_a_segment,
               count(*) FILTER (c.is_ksi) AS ksi,
               count(*) FILTER (c.is_ksi AND s.segment_id IS NOT NULL) AS ksi_on_a_segment
        FROM crashes c LEFT JOIN crash_segments s USING (crash_record_id)
        WHERE c.crash_year BETWEEN {years[0]} AND {years[1]}
    """).df()
    dist = con.execute("""
        SELECT count(*) AS segments,
               count(*) FILTER (crashes > 0) AS with_any_crash,
               count(*) FILTER (ksi > 0) AS with_ksi,
               round(100.0 * count(*) FILTER (ksi = 0) / count(*), 1) AS pct_zero_ksi,
               round(avg(ksi), 3) AS mean_ksi,
               round(var_samp(ksi), 3) AS var_ksi,
               max(ksi) AS max_ksi
        FROM segment_counts
    """).df()
    ksi_hist = con.execute("""
        SELECT CASE WHEN ksi >= 5 THEN '5+' ELSE ksi::VARCHAR END AS ksi_per_segment,
               count(*) AS segments
        FROM segment_counts GROUP BY 1 ORDER BY min(ksi)
    """).df()
    return {"coverage": coverage, "distribution": dist, "ksi_hist": ksi_hist,
            "reconcile": reconcile(con, years)}


def write_report(results: dict[str, pd.DataFrame], path, years=YEARS) -> None:
    text = [
        f"# Crash counts per segment, {years[0]}–{years[1]}\n",
        "Every segment in the network has a row, including those with no crashes.\n",
        "## Crashes that made it onto a segment\n\n" + results["coverage"].to_markdown(index=False),
        "\n## Per-segment distribution\n\n" + results["distribution"].to_markdown(index=False),
        "\n## Serious (KSI) crashes per segment\n\n" + results["ksi_hist"].to_markdown(index=False),
        "\n## Reconciliation\n\nSums over segments must equal the matched crashes.\n\n"
        + results["reconcile"].to_markdown(index=False),
    ]
    path.write_text("\n".join(text) + "\n")


def load_network() -> pd.DataFrame:
    segs = gpd.read_parquet(SEGMENTS_PATH, columns=["segment_id", "highway", "length_m", "geometry"])
    return pd.DataFrame(segs.loc[in_network(segs["highway"]), ["segment_id", "length_m"]])


def run() -> None:
    con = duckdb.connect(str(config.WAREHOUSE))
    counts = build_counts(con, load_network())
    results = summarise(con)
    rec = results["reconcile"].drop(columns="source")
    if not (rec.iloc[0] == rec.iloc[1]).all():
        raise RuntimeError(f"segment counts do not reconcile:\n{results['reconcile']}")
    for name, df in results.items():
        log.info("%s\n%s", name, df.to_string(index=False))
    config.REPORTS.mkdir(exist_ok=True)
    write_report(results, config.REPORTS / "segment_counts.md")
    log.info("segment_counts: %d segments", len(counts))
    con.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
