from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd
import shap

from src import config
from src.features.counts import YEARS
from src.model import train as T

log = logging.getLogger(__name__)

N_YEARS = YEARS[1] - YEARS[0] + 1
N_DRIVERS = 3
MIN_SHAP = 0.05  # a reason must raise the log-rate by at least this (about +5% risk)
NEAR_SCHOOL_M = 150

# feature -> (group, describe(value) -> text or None). Features in one group share one reason.
# None means the value is not something a reader can act on as a reason: the absence of a
# feature ("not a ramp"), a data-quality flag, segment length (exposure, not design), or
# distance to a hospital (a stand-in for busy central areas). Those still feed the model.
_CLASS_NAMES = {6: "expressway-class road", 5: "trunk road", 4: "primary road", 3: "secondary road",
                2: "tertiary road", 1: "minor road", 0: "residential street"}
_flag = lambda text: (lambda v: text if v else None)  # noqa: E731
LABELS = {
    "road_class_rank": ("road type", lambda v: _CLASS_NAMES[int(v)]),
    "max_class_rank": ("road type", lambda v: f"busiest approach: {_CLASS_NAMES[int(v)]}"),
    "min_class_rank": ("road type", lambda v: f"quietest approach: {_CLASS_NAMES[int(v)]}"),
    "n_main_approaches": ("road type", lambda v: f"{v:g} main-road approaches" if v else None),
    "lanes": ("lanes", lambda v: f"{v:g} lanes"),
    "max_lanes": ("lanes", lambda v: f"up to {v:g} lanes"),
    "lanes_imputed": ("lanes", lambda v: None),
    "length_m": ("length", lambda v: None),
    "oneway": ("one-way", lambda v: None if v else "two-way traffic"),
    "n_oneway": ("one-way", lambda v: "all approaches two-way" if v == 0 else None),
    "speed_mph": ("speed", lambda v: f"{v:g} mph limit"),
    "max_speed_mph": ("speed", lambda v: f"up to {v:g} mph"),
    "has_signal": ("signals", lambda v: "traffic signals (busy junction)" if v else "no traffic signals"),
    "has_stop": ("stop signs", _flag("stop sign")),
    "max_degree": ("junction size", lambda v: f"{v:g} roads meet at one end" if v >= 4 else None),
    "n_approaches": ("junction size", lambda v: f"{v:g} approach roads"),
    "n_streets": ("junction size", lambda v: f"{v:g} streets meet" if v >= 3 else None),
    "n_nodes": ("junction size", _flag("complex layout")),
    "has_dead_end": ("dead end", _flag("dead end")),
    "sinuosity": ("curves", lambda v: "curving road" if v > 1.1 else None),
    "dist_school_m": ("school", lambda v: f"school {v:.0f} m away" if v <= NEAR_SCHOOL_M else None),
    "near_school": ("school", _flag(f"within {NEAR_SCHOOL_M} m of a school")),
    "dist_hospital_m": ("hospital", lambda v: None),
    "is_link": ("ramp", _flag("ramp / link road")),
    "is_bridge": ("bridge", _flag("bridge")),
    "is_tunnel": ("tunnel", _flag("tunnel / underpass")),
    "is_roundabout": ("roundabout", _flag("roundabout")),
    "has_link": ("ramp", _flag("ramp joins here")),
    # street photos: signal and stop text comes from has_signal / has_stop, which show both sources
    "mly_signal": ("signals", lambda v: None),
    "mly_stop": ("stop signs", lambda v: None),
    "n_crosswalks": ("crosswalks", lambda v: f"busy crossing ({v:g} marked crosswalks)" if v >= 1 else None),
    "mly_covered": ("photo coverage", lambda v: None),
}
# a feature seen in either source: the reason text uses the combined value
_EITHER = {"has_signal": "mly_signal", "has_stop": "mly_stop"}


def readable(X: pd.DataFrame) -> pd.DataFrame:

    X = X.copy()
    for osm, mly in _EITHER.items():
        if osm in X and mly in X:
            X[osm] = X[[osm, mly]].max(axis=1)
    return X


def drivers(X: pd.DataFrame, sv: np.ndarray, n: int = N_DRIVERS, min_shap: float = MIN_SHAP) -> list[list[str]]:

    feats = list(X.columns)
    groups = sorted({LABELS[f][0] for f in feats})
    gidx = {g: [i for i, f in enumerate(feats) if LABELS[f][0] == g] for g in groups}
    gsum = np.column_stack([sv[:, gidx[g]].sum(axis=1) for g in groups])
    vals = X.to_numpy()
    out = []
    for r in range(len(X)):
        reasons = []
        for j in np.argsort(-gsum[r], kind="stable"):
            if gsum[r, j] < min_shap or len(reasons) == n:
                break
            for i in sorted(gidx[groups[j]], key=lambda i: -sv[r, i]):
                text = LABELS[feats[i]][1](vals[r, i])
                if text:
                    reasons.append(text)
                    break
        out.append(reasons + [None] * (n - len(reasons)))
    return out


def predict_unit(con: duckdb.DuckDBPyConnection, unit: T.Unit) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = T.load(con, unit)
    _, oof, _ = T.cross_validate(df, unit)
    model = T.LGBM(unit).fit(df)
    mu = model.predict(df)
    X = df[list(unit.features)]
    sv = shap.TreeExplainer(model.model.booster_).shap_values(X)
    reasons = drivers(readable(X), sv)

    out = pd.DataFrame({
        "unit_type": unit.name.rstrip("s"),
        "unit_id": df["id"],
        "community_area": df["community_area"],
        "exposure_km": df["exposure"] if unit.exposure else np.nan,
        "actual_ksi": df["y"],
        "expected_ksi": mu,
        "expected_ksi_per_year": mu / N_YEARS,
        "oof_expected_ksi": oof["lgbm"],
        "risk": T.risk_score(mu / N_YEARS, df["exposure"].to_numpy()),
    })
    out["rank"] = out["risk"].rank(ascending=False, method="first").astype(int)
    out["percentile"] = (100 * out["risk"].rank(pct=True)).round(2)
    for k in range(N_DRIVERS):
        out[f"driver_{k + 1}"] = [r[k] for r in reasons]

    imp = pd.DataFrame({"feature": unit.features, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp["group"] = imp["feature"].map(lambda f: LABELS[f][0])
    imp = (imp.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False).round(3)
           .reset_index().assign(unit=unit.name))
    return out, imp


def checks(pred: pd.DataFrame) -> list[str]:
    fails = []
    for u, d in pred.groupby("unit_type"):
        ratio = d["expected_ksi"].sum() / d["actual_ksi"].sum()
        if not 0.98 < ratio < 1.02:
            fails.append(f"{u}: expected/actual KSI {ratio:.3f}")
        if d["rank"].duplicated().any() or d["rank"].min() != 1 or d["rank"].max() != len(d):
            fails.append(f"{u}: ranks are not 1..n")
        if d[["expected_ksi", "risk", "oof_expected_ksi"]].isna().any().any() or (d["expected_ksi"] <= 0).any():
            fails.append(f"{u}: missing or non-positive predictions")
        top = d[d["percentile"] >= 99]  # the riskiest 1% must each come with at least one reason
        if top["driver_1"].isna().any():
            fails.append(f"{u}: {top['driver_1'].isna().sum()} of the riskiest 1% have no reason")
    return fails


def run() -> None:
    con = duckdb.connect(str(config.WAREHOUSE))
    preds, imps = zip(*(predict_unit(con, u) for u in T.UNITS))
    pred, imp = pd.concat(preds, ignore_index=True), pd.concat(imps, ignore_index=True)
    con.register("_p", pred)
    con.execute("CREATE OR REPLACE TABLE predictions AS SELECT * FROM _p ORDER BY unit_type, rank")
    con.unregister("_p")
    fails = checks(pred)

    names = con.execute("""
        SELECT p.unit_type, p.rank, coalesce(s.name, j.streets) AS place, p.actual_ksi,
               round(p.expected_ksi, 1) AS expected_ksi, round(p.risk, 3) AS risk_per_year,
               p.driver_1, p.driver_2, p.driver_3
        FROM predictions p
        LEFT JOIN model_segments s ON p.unit_type = 'segment' AND s.model_segment_id = p.unit_id
        LEFT JOIN (SELECT l.junction_id, string_agg(DISTINCT c.street_name, ' / ' ORDER BY c.street_name) AS streets
                   FROM crash_locations l JOIN crashes c USING (crash_record_id)
                   WHERE l.junction_id IS NOT NULL GROUP BY 1) j
               ON p.unit_type = 'junction' AND j.junction_id = p.unit_id
        WHERE p.rank <= 15 ORDER BY p.unit_type, p.rank""").df()
    con.close()
    text = ["# Predictions and drivers\n",
            "Risk = expected KSI per year (2021–2025 average level) per km of mid-block road (pieces "
            f"under {T.MIN_WINDOW_KM * 1000:.0f} m count as {T.MIN_WINDOW_KM * 1000:.0f} m), or per intersection. "
            f"Drivers are the features that raise a unit's risk most (SHAP, at least +{MIN_SHAP} on the "
            "log-rate scale); absences, data-quality flags, length and hospital distance are never shown "
            "as reasons.\n",
            "## Checks\n\n" + ("All passed." if not fails else "\n".join(f"- FAIL {f}" for f in fails)),
            "\n## What drives risk overall (mean |SHAP|, log-rate scale)\n\n" + imp.to_markdown(index=False),
            "\n## Highest-risk units\n\n" + names.to_markdown(index=False)]
    (config.REPORTS / "predictions.md").write_text("\n".join(text) + "\n")
    log.info("importance\n%s", imp.to_string(index=False))
    log.info("top\n%s", names.to_string(index=False))
    if fails:
        raise RuntimeError("prediction checks failed: " + "; ".join(fails))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
