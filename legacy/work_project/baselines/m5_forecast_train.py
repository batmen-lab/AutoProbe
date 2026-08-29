"""LightGBM baseline for M5 Forecasting (deliberately under-powered).

Predicts the next 28 days of unit sales with a LightGBM regressor over lag /
calendar / series features, and reports validation RMSE per boosting round
(each boosting round = one epoch). Left with obvious headroom to improve:
  - only the first MAX_FEATURES feature columns are used (just the two lags),
  - the ensemble is tiny (N_ESTIMATORS=3 boosting rounds/epochs -> under-fit),
  - the trees are stumps (NUM_LEAVES=2 -> no interactions).
Each individually lowers RMSE when fixed.

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

from data_loader import load_sales

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
HORIZON = 28
TRAIN_DAYS = 120

MAX_FEATURES = 2
N_ESTIMATORS = 3
NUM_LEAVES = 2
LEARNING_RATE = 0.1

FEATURE_NAMES = ["lag28", "lag35", "rmean28", "rmean_prev28", "wday",
                 "dept", "store", "cat", "state", "snap"]


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    sales = load_sales()
    cal = pd.read_csv(BASE_DIR / "data" / "calendar.csv")
    dcols = [c for c in sales.columns if c.startswith("d_")]
    V = sales[dcols].to_numpy(np.float32)
    T = V.shape[1]
    n = V.shape[0]
    wday = cal.set_index("d")["wday"].reindex(dcols).to_numpy()
    snap = cal.set_index("d")["snap_CA"].reindex(dcols).to_numpy().astype(np.float32)
    codes = {c: sales[c].astype("category").cat.codes.to_numpy()
             for c in ["dept_id", "store_id", "cat_id", "state_id"]}

    def feats(t):  # features to predict day-index t (all lags >= 28, no recursion)
        return np.column_stack([
            V[:, t - 28], V[:, t - 35],
            V[:, t - 28:t].mean(1), V[:, t - 56:t - 28].mean(1),
            np.full(n, wday[t]),
            codes["dept_id"], codes["store_id"], codes["cat_id"], codes["state_id"],
            np.full(n, snap[t]),
        ])[:, :MAX_FEATURES]

    train_days = range(T - HORIZON - TRAIN_DAYS, T - HORIZON)
    val_days = range(T - HORIZON, T)
    X_train = np.vstack([feats(t) for t in train_days])
    y_train = np.concatenate([V[:, t] for t in train_days])
    X_valid = np.vstack([feats(t) for t in val_days])
    y_valid = np.concatenate([V[:, t] for t in val_days])

    model = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS, num_leaves=NUM_LEAVES, learning_rate=LEARNING_RATE,
        random_state=SEED, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="rmse")

    # per-boosting-round validation RMSE == per-epoch trajectory
    history = list(model.evals_result_["valid_0"]["rmse"])
    for epoch, rmse in enumerate(history, start=1):
        print(f"Epoch {epoch:03d} | val RMSE {rmse:.6f}")
    val_rmse = float(history[-1])

    with (RESULT_DIR / "validation_metrics.json").open("w", encoding="utf-8") as h:
        json.dump({"metric": "rmse", "value": val_rmse, "epochs": len(history),
                   "n_features": MAX_FEATURES, "n_estimators": N_ESTIMATORS,
                   "num_leaves": NUM_LEAVES}, h, indent=2)

    print(f"Done. Validation RMSE: {val_rmse:.6f} over {len(history)} epoch(s).")


if __name__ == "__main__":
    main()
