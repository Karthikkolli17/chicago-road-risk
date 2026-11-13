from __future__ import annotations

import argparse
import logging
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

os.environ.setdefault("PREFECT_LOGGING_EXTRA_LOGGERS", "src")  # show the steps' own logs in Prefect
os.environ.setdefault("PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW", "ignore")  # e.g. in tests

import duckdb  # noqa: E402
from prefect import flow, serve, task  # noqa: E402

from src import config  # noqa: E402
from src.features import counts, junctions, segments, snap  # noqa: E402
from src.features import mapillary as mly_features  # noqa: E402
from src.ingest import context, crashes, osm, warehouse  # noqa: E402
from src.ingest import mapillary as mly_ingest  # noqa: E402
from src.ingest.mapillary import DETECTIONS_PATH  # noqa: E402
from src.model import predict, train  # noqa: E402

log = logging.getLogger("flows.pipeline")

REFRESH_DAYS = 30  # roads, places and street photos change slowly: refresh them monthly
WEEKLY_CRON = "0 7 * * 1"  # Mondays 07:00


class CheckFailed(RuntimeError):
    pass


def _gate(fails: list[str]) -> None:
    if fails:
        raise CheckFailed("; ".join(fails))


# weekly steps, in order; each raises if its checks fail
def step_crashes() -> None:
    failed = crashes.run()
    _gate([f"crash months failed to download: {', '.join(failed)}"] if failed else [])


def step_warehouse() -> None:
    _gate(warehouse.gate(warehouse.run()))


def step_snap() -> None:
    _gate(snap.gate(snap.run()))


WEEKLY: list[tuple[str, Callable[[], None]]] = [
    ("download crashes", step_crashes),
    ("load warehouse", step_warehouse),
    ("snap crashes to roads", step_snap),
    ("count crashes per road", counts.run),
    ("build intersections", junctions.run),
    ("build road stretches", segments.run),
    ("street-photo features", mly_features.run),
    ("train and test models", train.run),
    ("predict and publish", predict.run),
]

# slow-changing sources; a failure here is a warning, the existing files are still usable
SOURCES: list[tuple[str, Callable[[], None], object]] = [
    ("refresh community areas", context.fetch_community_areas, context.COMMUNITY_AREAS_PATH),
    ("refresh road network", osm.build_network, osm.SEGMENTS_PATH),
    ("refresh schools and hospitals", lambda: context.fetch_pois("overpass"), context.POIS_PATH),
    ("refresh street photos", lambda: mly_ingest.run(force=True), DETECTIONS_PATH),
]


def is_stale(path, days: float = REFRESH_DAYS, now: float | None = None) -> bool:
    return not path.exists() or ((now or time.time()) - path.stat().st_mtime) > days * 86400


@dataclass
class StepResult:
    name: str
    status: str  # ok | failed | skipped | warning
    seconds: float
    error: str | None = None


def run_steps(steps, sources=(), force_refresh: bool = False, wrap=lambda name, fn: fn) -> list[StepResult]:

    # Sources first (only when stale or forced), then the weekly steps. The first weekly failure
    # stops the run; later steps are marked skipped. `predictions` is only replaced by the last
    # step, after its own checks, so a failed run leaves the published map as it was.
    results = []
    for name, fn, path in sources:
        if not (force_refresh or is_stale(path)):
            continue
        t0 = time.time()
        try:
            wrap(name, fn)()
            results.append(StepResult(name, "ok", time.time() - t0))
        except Exception as exc:
            log.warning("%s failed, keeping the existing files: %s", name, exc)
            results.append(StepResult(name, "warning", time.time() - t0, str(exc)))
    failed = None
    for name, fn in steps:
        if failed:
            results.append(StepResult(name, "skipped", 0.0))
            continue
        t0 = time.time()
        try:
            wrap(name, fn)()
            results.append(StepResult(name, "ok", time.time() - t0))
        except Exception as exc:
            log.error("%s failed: %s\n%s", name, exc, traceback.format_exc())
            results.append(StepResult(name, "failed", time.time() - t0, str(exc)))
            failed = name
    return results


def record(results: list[StepResult], started: datetime, path=config.WAREHOUSE) -> str:

    status = "failed" if any(r.status == "failed" for r in results) else "ok"
    con = duckdb.connect(str(path))
    try:
        con.execute("""CREATE TABLE IF NOT EXISTS pipeline_runs (
            started_at TIMESTAMP, finished_at TIMESTAMP, status VARCHAR, failed_step VARCHAR,
            error VARCHAR, data_through TIMESTAMP, steps VARCHAR)""")
        has_crashes = con.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name = 'crashes'").fetchall()[0][0]
        through = con.execute("SELECT max(crash_date) FROM crashes").fetchall()[0][0] if has_crashes else None
        bad = next((r for r in results if r.status == "failed"), None)
        steps = "; ".join(f"{r.name}: {r.status} ({r.seconds:.0f}s)" for r in results)
        con.execute("INSERT INTO pipeline_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [started, datetime.now(), status, bad and bad.name, bad and bad.error, through, steps])
    finally:
        con.close()
    return status


def write_report(results: list[StepResult], started: datetime, status: str, path) -> None:
    rows = "\n".join(f"| {r.name} | {r.status} | {r.seconds:.0f} | {(r.error or '').replace('|', '/')[:200]} |"
                     for r in results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Pipeline run {started:%Y-%m-%d %H:%M}\n\n"
                    f"**Status: {status}.** "
                    + ("New predictions published.\n" if status == "ok" else
                       "The published predictions were not changed.\n")
                    + "\n| step | status | seconds | error |\n|---|---|---|---|\n" + rows + "\n")


@flow(name="chicago-road-risk", log_prints=True)
def weekly(force_refresh: bool = False) -> str:
    started = datetime.now()
    results = run_steps(WEEKLY, SOURCES, force_refresh,
                        wrap=lambda name, fn: task(fn, name=name, retries=0))
    status = record(results, started)
    write_report(results, started, status, config.REPORTS / "pipeline_run.md")
    for r in results:
        log.info("%-30s %-8s %5.0fs %s", r.name, r.status, r.seconds, r.error or "")
    if status != "ok":
        raise RuntimeError(f"pipeline failed at: {next(r.name for r in results if r.status == 'failed')}")
    return status


def main() -> None:
    p = argparse.ArgumentParser(description="Chicago Road Risk pipeline")
    p.add_argument("--refresh", action="store_true", help="also refresh roads, places and street photos now")
    p.add_argument("--serve", action="store_true", help=f"run on a schedule ({WEEKLY_CRON}) until stopped")
    args = p.parse_args()
    if args.serve:
        serve(weekly.to_deployment(name="weekly", cron=WEEKLY_CRON))
    else:
        weekly(force_refresh=args.refresh)


if __name__ == "__main__":
    main()
