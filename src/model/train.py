from __future__ import annotations

import logging
from dataclasses import dataclass

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold

from src import config
from src.features import junctions as J
from src.features import mapillary as MLY
from src.features import segments as S
from src.features.counts import YEARS

log = logging.getLogger(__name__)

N_FOLDS = 5
MIN_MIDBLOCK_M = 1.0
SEED = 7
LGB_PARAMS = dict(objective="poisson", learning_rate=0.03, num_leaves=15, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  n_estimators=3000, random_state=SEED, verbose=-1)
CAPTURE_AT = (0.01, 0.05, 0.10)
# a weekly run stops if a model falls well below its Day 3-4 level (D2 0.25 / 0.41, top 5% 0.21 / 0.36)
GATES = {"segments": {"d2": 0.20, "top5": 0.17}, "junctions": {"d2": 0.33, "top5": 0.30}}
MIN_WINDOW_KM = 0.1  # ranking: road pieces shorter than this count as this long


@dataclass(frozen=True)
class Unit:
    name: str
    table: str
    id: str
    features: tuple[str, ...]
    categorical: tuple[str, ...]  # one-hot in the GLM; numeric (ordinal) in LightGBM
    log_features: tuple[str, ...]  # log1p in the GLM
    exposure: str | None
    where: str
    num_leaves: int  # picked by a small grid on spatial-CV deviance (7/15/31 x 50/200/500)
    extra_table: str | None = None  # more features, one row per unit, joined on the unit id


SEGMENTS = Unit("segments", "model_segments", "model_segment_id", tuple(S.FEATURES),
                ("road_class_rank",), ("length_m", "dist_school_m", "dist_hospital_m"),
                "midblock_m", f"midblock_m >= {MIN_MIDBLOCK_M}", num_leaves=7)
# street-photo features help intersections only (Day 4 test); road stretches keep OSM alone
JUNCTIONS = Unit("junctions", "model_junctions", "junction_id", tuple(J.FEATURES + MLY.JUNCTION_FEATURES),
                 ("max_class_rank", "min_class_rank"), ("dist_school_m", "dist_hospital_m"),
                 None, "true", num_leaves=15, extra_table="mly_junctions")
UNITS = (SEGMENTS, JUNCTIONS)


# data
def load(con: duckdb.DuckDBPyConnection, unit: Unit, years=YEARS) -> pd.DataFrame:

    if unit.table == SEGMENTS.table:
        ksi = f"""SELECT g.model_segment_id AS id, count(*) AS y
                  FROM crash_locations l JOIN crashes c USING (crash_record_id)
                  JOIN segment_groups g USING (segment_id)
                  WHERE l.junction_id IS NULL AND c.is_ksi AND c.crash_year BETWEEN {years[0]} AND {years[1]}
                  GROUP BY 1"""
    else:
        ksi = f"""SELECT l.junction_id AS id, count(*) AS y
                  FROM crash_locations l JOIN crashes c USING (crash_record_id)
                  WHERE l.junction_id IS NOT NULL AND c.is_ksi AND c.crash_year BETWEEN {years[0]} AND {years[1]}
                  GROUP BY 1"""
    exposure = f"{unit.exposure} / 1000.0" if unit.exposure else "1.0"
    extra = f"LEFT JOIN {unit.extra_table} x ON x.{unit.id} = t.{unit.id}" if unit.extra_table else ""
    df = con.execute(f"""
        SELECT t.{unit.id} AS id, {", ".join(unit.features)},
               t.community_area, {exposure} AS exposure, coalesce(k.y, 0) AS y
        FROM {unit.table} t {extra} LEFT JOIN ({ksi}) k ON k.id = t.{unit.id}
        WHERE {unit.where} ORDER BY t.{unit.id}""").df()
    if unit.extra_table:
        missing = con.execute(f"""SELECT count(*) FROM {unit.table} t
                                  ANTI JOIN {unit.extra_table} x ON x.{unit.id} = t.{unit.id}""").fetchall()[0][0]
        if missing:  # a stale extra table must stop the run, not drop places from the map
            raise RuntimeError(f"{unit.extra_table} has no row for {missing} of the units in {unit.table}; "
                               "rebuild it")
    bools = [f for f in unit.features if df[f].dtype == bool]
    df[bools] = df[bools].astype(int)
    return df


def folds(df: pd.DataFrame, n: int = N_FOLDS) -> np.ndarray:

    out = np.empty(len(df), dtype=int)
    for k, (_, test) in enumerate(GroupKFold(n_splits=n).split(df, groups=df["community_area"])):
        out[test] = k
    return out


# models
class GLM:

    def __init__(self, unit: Unit):
        self.unit = unit

    def _design(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        u = self.unit
        X = df[list(u.features)].astype(float).copy()
        for f in u.log_features:
            X[f] = np.log1p(X[f])
        num = [f for f in u.features if f not in u.categorical]
        if fit:
            self.mean, self.std = X[num].mean(), X[num].std().replace(0, 1)
            self.levels = {c: sorted(df[c].unique())[1:] for c in u.categorical}  # first level is the base
        X[num] = (X[num] - self.mean) / self.std
        for c in u.categorical:
            for lv in self.levels[c]:
                X[f"{c}={lv}"] = (df[c] == lv).astype(float)
            X = X.drop(columns=c)
        return sm.add_constant(X, has_constant="add")

    def fit(self, df: pd.DataFrame) -> "GLM":
        X = self._design(df, fit=True)
        self.result = sm.GLM(df["y"], X, family=sm.families.Poisson(),
                             offset=np.log(df["exposure"])).fit(maxiter=200)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.result.predict(self._design(df, fit=False), offset=np.log(df["exposure"])).to_numpy()


class LGBM:


    def __init__(self, unit: Unit, params: dict | None = None):
        self.unit = unit
        self.params = params or {**LGB_PARAMS, "num_leaves": unit.num_leaves}

    def fit(self, df: pd.DataFrame) -> "LGBM":
        X, y, off = df[list(self.unit.features)], df["y"], np.log(df["exposure"])
        inner = folds(df, n=4)
        tr, va = inner != 0, inner == 0
        probe = lgb.LGBMRegressor(**self.params).fit(
            X[tr], y[tr], init_score=off[tr], eval_X=(X[va],), eval_y=(y[va],), eval_init_score=[off[va]],
            callbacks=[lgb.early_stopping(100, verbose=False)])
        self.n_trees = max(probe.best_iteration_, 50)
        self.model = lgb.LGBMRegressor(**{**self.params, "n_estimators": self.n_trees}).fit(X, y, init_score=off)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict(df[list(self.unit.features)], raw_score=True)
        return np.exp(raw + np.log(df["exposure"].to_numpy()))


# metrics
def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-12, None)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2 * np.sum(term - (y - mu)))


def risk_score(mu: np.ndarray, exposure: np.ndarray) -> np.ndarray:

    return mu / np.maximum(exposure, MIN_WINDOW_KM)


def capture(y: np.ndarray, mu: np.ndarray, exposure: np.ndarray, share: float) -> float:

    order = np.argsort(-risk_score(mu, exposure), kind="stable")
    cum = np.cumsum(exposure[order]) / exposure.sum()
    top = order[: np.searchsorted(cum, share) + 1]
    return float(y[top].sum() / y.sum())


def level_factor(latest_year_ksi: float, per_year: np.ndarray) -> float:
    # Yearly predictions are an average over the window; serious crashes fell 28% from 2021 to
    # 2025, so that average runs high. Scale to the latest year's city total. One factor per unit
    # type moves every place by the same share, so the ranking does not change.
    return float(latest_year_ksi / np.sum(per_year))


def evaluate(df: pd.DataFrame, preds: dict[str, np.ndarray], null: np.ndarray) -> pd.DataFrame:
    y, e = df["y"].to_numpy(), df["exposure"].to_numpy()
    d_null = poisson_deviance(y, null)
    rows = []
    for name, mu in preds.items():
        r = {"model": name, "deviance": round(poisson_deviance(y, mu), 1),
             "D2 vs null": round(1 - poisson_deviance(y, mu) / d_null, 4),
             "predicted / actual": round(mu.sum() / y.sum(), 3)}
        for s in CAPTURE_AT:
            r[f"KSI in top {s:.0%}"] = round(capture(y, mu, e, s), 3)
        rows.append(r)
    return pd.DataFrame(rows)


def calibration(df: pd.DataFrame, mu: np.ndarray, bins: int = 10) -> pd.DataFrame:
    rate = risk_score(mu, df["exposure"].to_numpy())
    q = pd.qcut(pd.Series(rate).rank(method="first"), bins, labels=False) + 1
    out = pd.DataFrame({"decile": q, "predicted": mu, "actual": df["y"].to_numpy()}).groupby("decile").sum()
    out["ratio"] = (out["actual"] / out["predicted"]).round(2)
    return out.round(1).reset_index()


# experiments
def cross_validate(df: pd.DataFrame, unit: Unit) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[int]]:

    k = folds(df)
    oof = {name: np.zeros(len(df)) for name in ("null", "glm", "lgbm")}
    trees = []
    for f in range(N_FOLDS):
        tr, te = df[k != f], df[k == f]
        oof["null"][k == f] = tr["y"].sum() / tr["exposure"].sum() * te["exposure"].to_numpy()
        oof["glm"][k == f] = GLM(unit).fit(tr).predict(te)
        m = LGBM(unit).fit(tr)
        oof["lgbm"][k == f] = m.predict(te)
        trees.append(m.n_trees)
    metrics = evaluate(df, {"null (one city-wide rate)": oof["null"], "Poisson GLM": oof["glm"],
                            "LightGBM Poisson": oof["lgbm"]}, oof["null"])
    return metrics, oof, trees


def temporal(con: duckdb.DuckDBPyConnection, unit: Unit, years=YEARS) -> pd.DataFrame:

    first, last = years
    train, test = load(con, unit, (first, last - 1)), load(con, unit, (last, last))
    recent = load(con, unit, (last - 1, last - 1))
    assert (train["id"].to_numpy() == test["id"].to_numpy()).all()
    scale = 1 / (last - first)  # training years -> one test year
    null = train["y"].sum() / train["exposure"].sum() * test["exposure"].to_numpy()
    lgbm = LGBM(unit).fit(train).predict(test) * scale
    preds = {"null (one city-wide rate)": null * scale,
             f"history (own {first}-{last - 1 - 2000:02d} KSI)": (train["y"].to_numpy() + 1e-3 * null) * scale,
             "Poisson GLM": GLM(unit).fit(train).predict(test) * scale,
             "LightGBM Poisson": lgbm,
             f"LightGBM, scaled to {last - 1} level": lgbm * level_factor(recent["y"].sum(), lgbm)}
    return evaluate(test, preds, null * scale)


def gate(results: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> list[str]:

    fails = []
    for name, (metrics, tmp) in results.items():
        m = metrics.set_index("model")
        lgbm, glm = m.loc["LightGBM Poisson"], m.loc["Poisson GLM"]
        floor = GATES[name]
        if lgbm["D2 vs null"] < floor["d2"]:
            fails.append(f"{name}: D2 {lgbm['D2 vs null']:.3f} below {floor['d2']}")
        if lgbm["KSI in top 5%"] < floor["top5"]:
            fails.append(f"{name}: top-5% capture {lgbm['KSI in top 5%']:.3f} below {floor['top5']}")
        if lgbm["KSI in top 5%"] < glm["KSI in top 5%"] - 0.02:
            fails.append(f"{name}: LightGBM top-5% capture {lgbm['KSI in top 5%']:.3f} trails the GLM's "
                         f"{glm['KSI in top 5%']:.3f}")
        t = tmp.set_index("model")
        if t.loc["LightGBM Poisson", "KSI in top 5%"] < t.loc["null (one city-wide rate)", "KSI in top 5%"] + 0.05:
            fails.append(f"{name}: next-year top-5% capture barely beats a city-wide rate")
    return fails


def run() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    
    con = duckdb.connect(str(config.WAREHOUSE), read_only=True)
    text = [f"# KSI models ({YEARS[0]}–{YEARS[1]})\n",
            f"Out-of-fold scores: {N_FOLDS}-fold GroupKFold by community area. "
            "'KSI in top x%' = share of actual serious crashes on the units the model ranks riskiest, "
            "covering x% of road km (segments) or x% of intersections.\n"]
    period = f"train {YEARS[0]}–{YEARS[1] - 1}, test {YEARS[1]}"
    results = {}
    for unit in UNITS:
        df = load(con, unit)
        metrics, oof, trees = cross_validate(df, unit)
        pearson = GLM(unit).fit(df).result
        disp = pearson.pearson_chi2 / pearson.df_resid
        log.info("%s: %d units, %d KSI, trees per fold %s\n%s", unit.name, len(df), df["y"].sum(), trees,
                 metrics.to_string(index=False))
        cal = calibration(df, oof["lgbm"])
        tmp = temporal(con, unit)
        results[unit.name] = (metrics, tmp)
        log.info("%s temporal (%s)\n%s", unit.name, period, tmp.to_string(index=False))
        text += [f"\n## {unit.name}: {len(df):,} units, {int(df['y'].sum()):,} KSI\n",
                 f"LightGBM trees per fold: {trees}. Poisson GLM dispersion (Pearson χ²/df): {disp:.2f}.\n",
                 "### Spatial cross-validation\n\n" + metrics.to_markdown(index=False),
                 "\n### LightGBM calibration by predicted-risk decile (out of fold)\n\n" + cal.to_markdown(index=False),
                 f"\n### Temporal check: {period}\n\n" + tmp.to_markdown(index=False)]
    fails = gate(results)
    text.insert(1, "## Quality gate\n\n" + ("All passed.\n" if not fails else "\n".join(f"- FAIL {f}" for f in fails) + "\n"))
    config.REPORTS.mkdir(exist_ok=True)
    (config.REPORTS / "model_eval.md").write_text("\n".join(text) + "\n")
    con.close()
    if fails:
        raise RuntimeError("model quality gate failed: " + "; ".join(fails))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
