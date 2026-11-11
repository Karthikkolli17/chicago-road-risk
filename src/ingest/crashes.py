from __future__ import annotations

import argparse
import logging
from datetime import date

import pandas as pd

from src import config
from src.ingest.soda import SodaClient

log = logging.getLogger(__name__)

REFRESH_MONTHS = 2


def month_starts(start: str, end: date) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start).replace(day=1), pd.Timestamp(end), freq="MS"))


def month_where(month: pd.Timestamp) -> str:
    nxt = month + pd.offsets.MonthBegin(1)
    return (
        f"crash_date >= '{month:%Y-%m-%dT00:00:00}' "
        f"AND crash_date < '{nxt:%Y-%m-%dT00:00:00}'"
    )


def clean_crashes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["crash_date"] = pd.to_datetime(df["crash_date"])
    for col in ["posted_speed_limit", "injuries_fatal", "injuries_incapacitating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_people(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["crash_date"] = pd.to_datetime(df["crash_date"])
    return df


def _partition_path(name: str, month: pd.Timestamp):
    return config.RAW / name / f"month={month:%Y-%m}" / "part.parquet"


def pull_month(client: SodaClient, month: pd.Timestamp) -> tuple[int, int]:
    where = month_where(month)
    crashes = client.fetch(
        config.CRASHES_ID, config.CRASH_COLUMNS, where=where, order="crash_record_id"
    )
    types = ",".join(f"'{t}'" for t in config.VRU_PERSON_TYPES)
    people = client.fetch(
        config.PEOPLE_ID,
        config.PEOPLE_COLUMNS,
        where=f"{where} AND person_type in ({types})",
        order="crash_record_id,person_id",
    )
    # Paging can silently drop rows if the server misbehaves; check against its own count.
    expected = client.count(config.CRASHES_ID, where=where)
    if len(crashes) != expected or crashes["crash_record_id"].duplicated().any():
        raise RuntimeError(
            f"{month:%Y-%m}: got {len(crashes)} crash rows "
            f"({crashes['crash_record_id'].duplicated().sum()} duplicated), server count {expected}"
        )
    for name, df, cleaner in [
        ("crashes", crashes, clean_crashes),
        ("people_vru", people, clean_people),
    ]:
        path = _partition_path(name, month)
        path.parent.mkdir(parents=True, exist_ok=True)
        # write-then-rename so a killed run never leaves a half-written month behind
        tmp = path.with_suffix(".tmp")
        cleaner(df).to_parquet(tmp, index=False)
        tmp.replace(path)
    return len(crashes), len(people)


def reconcile(client: SodaClient, start: str = config.START_DATE) -> pd.DataFrame:
    server = pd.DataFrame(client._get(config.CRASHES_ID, {
        "$select": "date_trunc_ym(crash_date) AS m, count(*) AS n",
        "$where": f"crash_date >= '{start}T00:00:00'",
        "$group": "m",
        "$limit": 10_000,
    }))
    server = pd.Series(server["n"].astype(int).values, index=pd.to_datetime(server["m"]).dt.strftime("%Y-%m"))
    local = {
        p.parent.name.removeprefix("month="): len(pd.read_parquet(p, columns=["crash_record_id"]))
        for p in (config.RAW / "crashes").glob("month=*/part.parquet")
    }
    out = pd.DataFrame({"server": server, "local": pd.Series(local)}).fillna(0).astype(int)
    out["diff"] = out["local"] - out["server"]
    return out.sort_index()


def run(start: str = config.START_DATE, end: date | None = None, force: bool = False,
        heal: bool = True) -> list[str]:
    """Pull months that are missing, recent, or (with `heal`) whose crash count no
    longer matches the server, e.g. because the city deleted or re-dated a crash.
    Returns the months that failed."""
    end = end or date.today()
    client = SodaClient(config.SODA_DOMAIN, config.SODA_APP_TOKEN)
    months = month_starts(start, end)
    refresh = set(months[-REFRESH_MONTHS:])
    stale = set()
    if heal and not force:
        try:
            r = reconcile(client, start)
            stale = {pd.Timestamp(f"{m}-01") for m in r.index[r["diff"] != 0]}
            if stale - refresh:
                log.info("re-pulling months changed on the server: %s",
                         ", ".join(sorted(f"{m:%Y-%m}" for m in stale - refresh)))
        except Exception as exc:
            log.warning("could not reconcile; pulling only missing and recent months: %s", exc)
    failed = []
    for month in months:
        done = _partition_path("crashes", month).exists() and _partition_path("people_vru", month).exists()
        if done and not force and month not in refresh and month not in stale:
            continue
        try:
            n_crash, n_people = pull_month(client, month)
        except Exception as exc:  # keep going; a rerun retries only what is missing
            log.error("%s failed: %s", f"{month:%Y-%m}", exc)
            failed.append(f"{month:%Y-%m}")
            continue
        log.info("%s: %6d crashes, %5d ped/cyclist people rows", f"{month:%Y-%m}", n_crash, n_people)
    if failed:
        log.error("%d month(s) failed: %s. Rerun to retry them.", len(failed), ", ".join(failed))
    return failed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Pull Chicago crashes + ped/cyclist people rows by month")
    p.add_argument("--start", default=config.START_DATE)
    p.add_argument("--force", action="store_true", help="re-pull months already on disk")
    p.add_argument("--reconcile", action="store_true", help="only compare local vs server counts")
    args = p.parse_args()
    if args.reconcile:
        r = reconcile(SodaClient(config.SODA_DOMAIN, config.SODA_APP_TOKEN), args.start)
        bad = r[r["diff"] != 0]
        print(f"{len(r)} months, {r['local'].sum()} local vs {r['server'].sum()} server rows")
        print("all months match" if bad.empty else bad.to_string())
        raise SystemExit(0 if bad.empty else 1)
    raise SystemExit(1 if run(args.start, force=args.force) else 0)
