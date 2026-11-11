import duckdb
import pandas as pd
import pytest
import requests

from src.ingest import crashes as ingest
from src.ingest.osm import is_excluded
from src.ingest.soda import SodaClient
from src.ingest.warehouse import checks, load_crashes


class FakeSession:

    def __init__(self, rows, fail_first=0):
        self.rows, self.calls, self.fail_first, self.headers = rows, [], fail_first, {}

    def get(self, url, params, timeout):
        self.calls.append(params)
        resp = requests.Response()
        if self.fail_first:
            self.fail_first -= 1
            resp.status_code = 503
            resp._content = b"{}"
            resp.url = url
            return resp
        start, n = params["$offset"], params["$limit"]
        resp.status_code = 200
        resp._content = pd.Series(self.rows[start:start + n], dtype=object).to_json(orient="values").encode()
        return resp


def test_paging_reads_every_row_once():
    rows = [{"id": str(i)} for i in range(25)]
    session = FakeSession(rows)
    df = SodaClient("x", page_size=10, session=session).fetch("ds", ["id"])
    assert df["id"].tolist() == [str(i) for i in range(25)]
    assert [c["$offset"] for c in session.calls] == [0, 10, 20]


def test_paging_stops_on_exact_multiple():
    session = FakeSession([{"id": str(i)} for i in range(20)])
    df = SodaClient("x", page_size=10, session=session).fetch("ds", ["id"])
    assert len(df) == 20 and len(session.calls) == 3  # last call returns empty


def test_missing_fields_become_null_columns():
    session = FakeSession([{"id": "1"}, {"id": "2", "lat": "41.9"}])
    df = SodaClient("x", session=session).fetch("ds", ["id", "lat"])
    assert list(df.columns) == ["id", "lat"] and df["lat"].isna().sum() == 1


def test_retries_server_errors(monkeypatch):
    monkeypatch.setattr(SodaClient._get.retry, "sleep", lambda s: None)
    session = FakeSession([{"id": "1"}], fail_first=2)
    df = SodaClient("x", session=session).fetch("ds", ["id"])
    assert len(df) == 1 and len(session.calls) == 3


def test_month_window_is_half_open():
    w = ingest.month_where(pd.Timestamp("2024-12-01"))
    assert "crash_date >= '2024-12-01T00:00:00'" in w
    assert "crash_date < '2025-01-01T00:00:00'" in w


@pytest.mark.parametrize(
    "highway, excluded",
    [("motorway", True), ("motorway_link", True), (["motorway", "motorway_link"], True),
     (["motorway_link", "primary"], False), ("primary", False), ("trunk", False), (None, False)],
)
def test_motorway_exclusion(highway, excluded):
    assert is_excluded(highway) is excluded


@pytest.fixture
def raw_dir(tmp_path):
    crashes = pd.DataFrame({
        "crash_record_id": ["a", "b", "c", "d", "e"],
        "crash_date": pd.to_datetime(["2021-03-01", "2021-03-02", "2022-03-03", "2022-03-04", "2022-03-05"]),
        "posted_speed_limit": [30, 30, 25, 35, 30],
        "most_severe_injury": ["FATAL", "NO INDICATION OF INJURY", "INCAPACITATING INJURY",
                               "NONINCAPACITATING INJURY", None],
        "injuries_fatal": [1, 0, 0, 0, 0],
        "injuries_incapacitating": [0, 0, 1, 0, 0],
        "latitude": [41.88, None, 0.0, 41.90, 45.0],
        "longitude": [-87.63, None, 0.0, -87.70, -87.60],
    })
    people = pd.DataFrame({
        "person_id": ["p1", "p2", "p3", "p4"],
        "crash_record_id": ["a", "a", "c", "zzz"],
        "crash_date": pd.to_datetime(["2021-03-01"] * 4),
        "person_type": ["PEDESTRIAN", "BICYCLE", "BICYCLE", "PEDESTRIAN"],
        "injury_classification": ["FATAL", None, None, None],
    })
    for name, df in [("crashes", crashes), ("people_vru", people)]:
        d = tmp_path / name / "month=2021-03"
        d.mkdir(parents=True)
        df.to_parquet(d / "part.parquet", index=False)
    # same crash re-appearing in a later month file must not be double counted
    d = tmp_path / "crashes" / "month=2022-03"
    d.mkdir(parents=True)
    crashes.iloc[[3]].to_parquet(d / "part.parquet", index=False)
    return tmp_path


def test_warehouse_flags_and_checks(raw_dir):
    con = duckdb.connect()
    load_crashes(con, raw_dir)
    c = con.execute("SELECT * FROM crashes ORDER BY crash_record_id").df().set_index("crash_record_id")
    assert len(c) == 5
    assert c["is_ksi"].tolist() == [True, False, True, False, False]
    assert c.loc["a", "has_pedestrian"] and c.loc["a", "has_cyclist"]
    assert c.loc["c", "has_cyclist"] and not c.loc["c", "has_pedestrian"]
    assert not c.loc["b", "has_pedestrian"]
    assert c["coord_status"].tolist() == ["ok", "missing", "zero", "ok", "outside_city"]

    r = checks(con)
    assert r["by_year"]["crashes"].tolist() == [2, 3]
    assert r["orphans"].iloc[0, 0] == 1


class CountingClient:

    def __init__(self, crashes, server_count, fail=False):
        self.crashes, self.server_count, self.fail = crashes, server_count, fail

    def fetch(self, dataset_id, columns, where=None, order=None):
        if self.fail:
            raise RuntimeError("boom")
        if dataset_id == ingest.config.CRASHES_ID:
            return self.crashes.reindex(columns=columns)
        return pd.DataFrame(columns=columns)

    def count(self, dataset_id, where=None):
        return self.server_count


def _crash_rows(ids):
    return pd.DataFrame({"crash_record_id": ids, "crash_date": ["2021-02-01T10:00:00.000"] * len(ids)})


@pytest.fixture
def tmp_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.config, "RAW", tmp_path)
    return tmp_path


def test_pull_month_writes_both_partitions(tmp_raw):
    n_crash, n_people = ingest.pull_month(CountingClient(_crash_rows(["a", "b"]), 2), pd.Timestamp("2021-02-01"))
    assert (n_crash, n_people) == (2, 0)
    assert len(pd.read_parquet(tmp_raw / "crashes" / "month=2021-02" / "part.parquet")) == 2
    assert (tmp_raw / "people_vru" / "month=2021-02" / "part.parquet").exists()
    assert not list(tmp_raw.rglob("*.tmp"))


@pytest.mark.parametrize("ids, server_count", [(["a", "b"], 3), (["a", "a"], 2)])
def test_pull_month_rejects_short_or_duplicated_pages(tmp_raw, ids, server_count):
    with pytest.raises(RuntimeError):
        ingest.pull_month(CountingClient(_crash_rows(ids), server_count), pd.Timestamp("2021-02-01"))
    assert not list(tmp_raw.rglob("*.parquet"))  # nothing half-written


def test_run_continues_past_failed_months(tmp_raw, monkeypatch):
    monkeypatch.setattr(ingest, "SodaClient", lambda *a, **k: CountingClient(None, 0, fail=True))
    failed = ingest.run("2021-01-01", end=pd.Timestamp("2021-03-15").date(), heal=False)
    assert failed == ["2021-01", "2021-02", "2021-03"]


def test_run_skips_done_months_but_refreshes_recent(tmp_raw, monkeypatch):
    for m in ["2021-01", "2021-02", "2021-03", "2021-04"]:
        for name in ["crashes", "people_vru"]:
            p = tmp_raw / name / f"month={m}"
            p.mkdir(parents=True)
            (p / "part.parquet").touch()
    pulled = []
    monkeypatch.setattr(ingest, "pull_month", lambda c, m: pulled.append(f"{m:%Y-%m}") or (0, 0))
    ingest.run("2021-01-01", end=pd.Timestamp("2021-05-02").date(), heal=False)
    assert pulled == ["2021-04", "2021-05"]  # last 2 months always; 2021-05 also missing


def test_count_parses_server_response():
    session = FakeSession([])
    session.get = lambda url, params, timeout: _json_response([{"n": "1234"}])
    assert SodaClient("x", session=session).count("ds", where="1=1") == 1234


def _json_response(payload):
    import json
    resp = requests.Response()
    resp.status_code = 200
    resp._content = json.dumps(payload).encode()
    return resp


def test_reconcile_compares_local_with_server(tmp_raw):
    for month, ids in [("2021-01", ["a", "b"]), ("2021-02", ["c"])]:
        d = tmp_raw / "crashes" / f"month={month}"
        d.mkdir(parents=True)
        pd.DataFrame({"crash_record_id": ids}).to_parquet(d / "part.parquet", index=False)

    class Client:
        def _get(self, dataset_id, params):
            return [{"m": "2021-01-01T00:00:00.000", "n": "2"}, {"m": "2021-02-01T00:00:00.000", "n": "3"},
                    {"m": "2021-03-01T00:00:00.000", "n": "4"}]

    r = ingest.reconcile(Client())
    assert r.loc["2021-01", "diff"] == 0
    assert r.loc["2021-02", "diff"] == -2  # two crashes missing locally
    assert r.loc["2021-03", "local"] == 0 and r.loc["2021-03", "diff"] == -4  # month never pulled


def test_moved_crash_keeps_the_most_recently_pulled_copy(raw_dir):
    import os
    # Crash 'a' was first pulled in 2021-03; the city later moved it to 2021-01 and
    # changed its severity. The 2021-01 file is newer, so its copy must win even
    # though its month sorts earlier.
    fresh = pd.read_parquet(raw_dir / "crashes" / "month=2021-03" / "part.parquet").iloc[[0]]
    fresh["most_severe_injury"] = "NO INDICATION OF INJURY"
    d = raw_dir / "crashes" / "month=2021-01"
    d.mkdir(parents=True)
    fresh.to_parquet(d / "part.parquet", index=False)
    os.utime(raw_dir / "crashes" / "month=2021-03" / "part.parquet", (1_000, 1_000))
    os.utime(raw_dir / "crashes" / "month=2022-03" / "part.parquet", (1_000, 1_000))
    os.utime(d / "part.parquet", (2_000, 2_000))

    con = duckdb.connect()
    load_crashes(con, raw_dir)
    row = con.execute("SELECT is_ksi FROM crashes WHERE crash_record_id = 'a'").fetchall()
    assert row == [(False,)]
    assert con.execute("SELECT count(*) FROM crashes").fetchone()[0] == 5


def test_run_repulls_old_months_that_changed_on_server(tmp_raw, monkeypatch):
    for m in ["2021-01", "2021-02", "2021-03", "2021-04", "2021-05"]:
        for name in ["crashes", "people_vru"]:
            p = tmp_raw / name / f"month={m}"
            p.mkdir(parents=True)
            (p / "part.parquet").touch()
    mismatch = pd.DataFrame({"server": [5, 6], "local": [5, 7], "diff": [0, 1]}, index=["2021-01", "2021-02"])
    monkeypatch.setattr(ingest, "reconcile", lambda client, start: mismatch)
    pulled = []
    monkeypatch.setattr(ingest, "pull_month", lambda c, m: pulled.append(f"{m:%Y-%m}") or (0, 0))
    ingest.run("2021-01-01", end=pd.Timestamp("2021-05-02").date())
    assert pulled == ["2021-02", "2021-04", "2021-05"]


def test_run_still_works_when_reconcile_fails(tmp_raw, monkeypatch):
    def boom(client, start):
        raise RuntimeError("portal down")
    monkeypatch.setattr(ingest, "reconcile", boom)
    pulled = []
    monkeypatch.setattr(ingest, "pull_month", lambda c, m: pulled.append(f"{m:%Y-%m}") or (0, 0))
    assert ingest.run("2021-01-01", end=pd.Timestamp("2021-02-02").date()) == []
    assert pulled == ["2021-01", "2021-02"]
