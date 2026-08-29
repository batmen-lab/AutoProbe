"""LightGBM baseline for Rossmann Store Sales (deliberately under-powered).

A LightGBM regressor predicting daily Sales for open stores. Runs end-to-end and
reports validation RMSPE (root mean squared percentage error, lower is better),
but is left with obvious headroom for the auto-research pipeline to improve:
  - only DayOfWeek is fed to the model (drops Store, Promo, date and
    store-metadata signals),
  - the ensemble is tiny (N_ESTIMATORS boosting rounds -> under-fit),
  - the trees are stumps (NUM_LEAVES=2 -> no interactions across store/date).

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

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
VALID_QUANTILE = 0.9   # last 10% of dates -> validation (time-ordered split)

FEATURES = ["DayOfWeek"]
N_ESTIMATORS = 5
NUM_LEAVES = 2
LEARNING_RATE = 0.1


def _rmspe(y, p):
    p = np.clip(p, 1.0, None)
    return float(np.sqrt(np.mean(((y - p) / y) ** 2)))


def _prepare() -> pd.DataFrame:
    df = load_train()
    df["Date"] = pd.to_datetime(df["Date"])
    store = pd.read_csv(BASE_DIR / "data" / "store.csv", low_memory=False)
    df = df.merge(store, on="Store", how="left")
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
    df["StateHoliday"] = (df["StateHoliday"].astype(str) != "0").astype(int)
    for c in ["StoreType", "Assortment"]:
        df[c] = df[c].astype("category").cat.codes
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["woy"] = df["Date"].dt.isocalendar().week.astype(int)
    return df.sort_values("Date")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df = _prepare()
    cut = df["Date"].quantile(VALID_QUANTILE)
    tr = df[df["Date"] <= cut]
    va = df[df["Date"] > cut]

    y_tr = tr["Sales"].to_numpy(dtype=np.float64)
    y_va = va["Sales"].to_numpy(dtype=np.float64)

    model = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        num_leaves=NUM_LEAVES,
        learning_rate=LEARNING_RATE,
        random_state=SEED,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )

    def _rmspe_lgb(y_true, y_pred):
        y_pred = np.clip(y_pred, 1.0, None)
        return "rmspe", float(np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))), False

    eval_hist: dict = {}
    model.fit(
        tr[FEATURES], y_tr,
        eval_set=[(va[FEATURES], y_va)],
        eval_metric=_rmspe_lgb,
        callbacks=[lgb.record_evaluation(eval_hist)],
    )
    pred = model.predict(va[FEATURES])
    val_rmspe = _rmspe(y_va, pred)

    with (RESULT_DIR / "validation_metrics.json").open("w", encoding="utf-8") as h:
        json.dump({"metric": "rmspe", "value": val_rmspe, "n_features": len(FEATURES),
                   "n_estimators": N_ESTIMATORS, "num_leaves": NUM_LEAVES}, h, indent=2)

    print(f"Validation RMSPE: {val_rmspe:.6f}")


if __name__ == "__main__":
    main()
