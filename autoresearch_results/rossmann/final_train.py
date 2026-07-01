"""LightGBM baseline for Rossmann Store Sales (deliberately under-powered).

A LightGBM regressor predicting daily Sales for open stores. Runs end-to-end and
reports validation RMSPE (root mean squared percentage error, lower is better),
but is left with obvious headroom for the auto-research pipeline to improve:
  - only the FEATURES list is used (just DayOfWeek -> drops Store, Promo, date
    and store-metadata signals),
  - the ensemble is tiny (N_ESTIMATORS boosting rounds -> under-fit),
  - the trees are stumps (NUM_LEAVES=2 -> no interactions across store/date).
Each individually lowers RMSPE when fixed.

Run:
    python train.py
"""

from __future__ import annotations

# --- determinism preamble ---
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
_os.environ.setdefault("PYTHONHASHSEED", "0")
import random as _random
SEED = 42
_random.seed(SEED)

import json
from pathlib import Path

import numpy as np
np.random.seed(SEED)
import pandas as pd
import lightgbm as lgb

from data_loader import load_train
from prober import record, conclude

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
VALID_QUANTILE = 0.9   # last 10% of dates -> validation (time-ordered split)
# potential_improvement_5: VALID_QUANTILE = 0.85 — a larger validation tail (15%) gives more open-store rows in the last weeks (esp. Dec/Nov peak), which better exposes Promo2/Christmas seasonality effects the model must fit to drop RMSPE

# tunable targets (each individually lowers RMSPE):
FEATURES = ["Store","DayOfWeek","Promo","Promo2","SchoolHoliday","StateHoliday","StoreType","Assortment","CompetitionDistance","CompetitionDistance_log","Promo2SinceWeeks","year","month","day","woy"]   # potential_improvement_1: expand to ["Store","DayOfWeek","Promo","Promo2","SchoolHoliday","StateHoliday","StoreType","Assortment","CompetitionDistance","CompetitionOpenSinceMonths","Promo2SinceWeeks","year","month","day","woy"] — each adds direct signal that lowers RMSPE (applied; CompetitionOpenSinceMonths/Promo2SinceWeeks deferred to potential_improvement_6 since they require engineering not yet in _prepare)
N_ESTIMATORS = 5           # potential_improvement_2: raise to 1500-2000 (paired with early stopping below) — under-fit boosting is the dominant RMSPE gap right now
NUM_LEAVES = 127           # potential_improvement_3: raise to 63-127 so the tree can capture Store×Promo×day-of-week interactions that drive daily sales (applied; 2→8 round 2, 8→31 round 7, 31→63 round 8, 63→127 this round — 2x step inside Regime B's 2x-5x range, continues the validated successful raise direction; round 8 tail_mean dropped 0.4311→0.4225 so staying incremental per recent-progress check, 127 is the upper end of comment's 63-127 target)
LEARNING_RATE = 0.1        # potential_improvement_4: lower to 0.03-0.05 — smaller steps with more rounds give lower RMSPE than large steps with few rounds


def _rmspe(y, p):
    p = np.clip(p, 1.0, None)
    return float(np.sqrt(np.mean(((y - p) / y) ** 2)))


def _prepare() -> pd.DataFrame:
    df = load_train()
    df["Date"] = pd.to_datetime(df["Date"])
    store = pd.read_csv(BASE_DIR / "data" / "store.csv", low_memory=False)
    df = df.merge(store, on="Store", how="left")
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
    # potential_improvement_7: add df["Sales_log"] = np.log1p(df["Sales"]) and train LGBM on Sales_log (predict via expm1) — RMSPE weights relative errors on small sales heavily; training on log-space matches the metric's symmetric-relative structure and typically drops val RMSPE 0.01-0.03 (applied)
    df["Sales_log"] = np.log1p(df["Sales"])
    df["StateHoliday"] = (df["StateHoliday"].astype(str) != "0").astype(int)
    for c in ["StoreType", "Assortment"]:
        df[c] = df[c].astype("category").cat.codes
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["woy"] = df["Date"].dt.isocalendar().week.astype(int)
    # potential_improvement_6: add log1p-transformed target feature columns CompetitionDistance_log = log1p(CompetitionDistance) and Promo2SinceWeeks imputed to 0 — LGBM splits linearly on raw CompetitionDistance which compresses the long tail and misses nearby-competitor impact on sales (applied)
    df["CompetitionDistance_log"] = np.log1p(df["CompetitionDistance"].fillna(0))
    df["Promo2SinceWeeks"] = df["Promo2SinceWeek"].fillna(0)
    return df.sort_values("Date")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df = _prepare()
    cut = df["Date"].quantile(VALID_QUANTILE)
    tr = df[df["Date"] <= cut]
    va = df[df["Date"] > cut]

    y_tr = tr["Sales_log"].to_numpy(dtype=np.float64)
    y_va = va["Sales"].to_numpy(dtype=np.float64)

    model = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        num_leaves=NUM_LEAVES,
        learning_rate=LEARNING_RATE,
        # potential_improvement_8: add min_data_in_leaf=200-2000 and feature_fraction=0.7-0.9, bagging_fraction=0.8, bagging_freq=1 — leaf over-fitting on store-date rows is what stalls val RMSPE once num_leaves is raised
        random_state=SEED,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )

    def _rmspe_lgb(y_true, y_pred):
        y_pred = np.expm1(y_pred)
        y_pred = np.clip(y_pred, 1.0, None)
        return "rmspe", float(np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))), False

    eval_hist: dict = {}
    model.fit(
        tr[FEATURES], y_tr,
        eval_set=[(va[FEATURES], y_va)],
        eval_metric=_rmspe_lgb,
        # potential_improvement_9: add callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.record_evaluation(eval_hist)] and raise N_ESTIMATORS to 2000 — early stopping on val RMSPE finds the genuine optimal round count instead of under-fitting at 5
        callbacks=[lgb.record_evaluation(eval_hist)],
    )
    # potential_improvement_10: clip predictions to [0, max observed Sales in tr] via np.clip(pred, 0, tr["Sales"].max()) — RMSPE penalises overshoot on small-sales rows disproportionately; a hard upper bound at the training max cuts the worst-case RMSPE tail
    pred = np.expm1(model.predict(va[FEATURES]))
    val_rmspe = _rmspe(y_va, pred)

    for round_idx, val in enumerate(eval_hist.get("valid_0", {}).get("rmspe", []), start=1):
        record(round_idx, val)
    if not eval_hist.get("valid_0", {}).get("rmspe"):
        record(1, val_rmspe)

    with (RESULT_DIR / "validation_metrics.json").open("w", encoding="utf-8") as h:
        json.dump({"metric": "rmspe", "value": val_rmspe, "n_features": len(FEATURES),
                   "n_estimators": N_ESTIMATORS, "num_leaves": NUM_LEAVES}, h, indent=2)

    print(f"Validation RMSPE: {val_rmspe:.6f}")
    conclude()


if __name__ == "__main__":
    main()
