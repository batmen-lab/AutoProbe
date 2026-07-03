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
from prober import record, conclude

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
HORIZON = 28
# potential_improvement_1: set TRAIN_DAYS to 728 (about two years). 120 days is far too short for LightGBM to learn weekly/annual seasonality, so extending the training window materially lowers val RMSE. # applied
TRAIN_DAYS = 365

# potential_improvement_2: set MAX_FEATURES = 10 so the full engineered feature block (rmeans, wday, dept/store/cat/state codes, snap) is passed to the model instead of only the two raw lags. Adding informative predictors is the single biggest RMSE drop available. # applied
MAX_FEATURES = 10
# potential_improvement_3: set N_ESTIMATORS to 2000-3000. Three boosting rounds is deep under-fit; a properly sized ensemble (paired with a lower LEARNING_RATE and early stopping) reduces val RMSE substantially.
N_ESTIMATORS = 3
# potential_improvement_4: set NUM_LEAVES to 63-127. Stumps (2 leaves) cannot express interactions between lags, wday, and store/dept; a deeper tree captures cross-feature effects and lowers RMSE.
NUM_LEAVES = 63
# potential_improvement_5: set LEARNING_RATE to 0.03-0.05 when raising N_ESTIMATORS. A smaller step size with more rounds is the standard LightGBM regime for lower val RMSE.
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
        # potential_improvement_6: extend the lag block with V[:, t-28*2] (lag56) and V[:, t-28*3] (lag84) once MAX_FEATURES is opened up. Additional monthly lags give the tree more level information and reduce RMSE on the 28-day horizon.
        return np.column_stack([
            V[:, t - 28], V[:, t - 35],
            # potential_improvement_7: add a shorter rolling window such as V[:, t - 28:t - 21].mean(1) (7-day rmean) alongside the 28/prev-28 rmeans; short-horizon smoothing captures trend shifts that lower RMSE.
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

    # potential_improvement_8: pass objective="tweedie", tweedie_variance_power=1.1 to LGBMRegressor. M5 unit sales are zero-inflated counts; a Tweedie loss fits that distribution better than default L2 regression and typically drops val RMSE.
    # potential_improvement_9: add regularization/subsampling: min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, reg_alpha=0.1, reg_lambda=0.1. This prevents overfitting once NUM_LEAVES and N_ESTIMATORS are raised, which keeps val RMSE improving instead of degrading.
    model = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS, num_leaves=NUM_LEAVES, learning_rate=LEARNING_RATE,
        random_state=SEED, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1,
    )
    # potential_improvement_10: pass categorical_feature=["dept_id","store_id","cat_id","state_id","wday"] (via column indices) and callbacks=[lgb.early_stopping(50)] to model.fit. Telling LightGBM these columns are categorical avoids treating codes as ordinal, and early stopping picks the best round for lowest val RMSE.
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="rmse")

    # per-boosting-round validation RMSE == per-epoch trajectory
    history = list(model.evals_result_["valid_0"]["rmse"])
    for epoch, rmse in enumerate(history, start=1):
        print(f"Epoch {epoch:03d} | val RMSE {rmse:.6f}")
        record(epoch, float(rmse))
    val_rmse = float(history[-1])

    with (RESULT_DIR / "validation_metrics.json").open("w", encoding="utf-8") as h:
        json.dump({"metric": "rmse", "value": val_rmse, "epochs": len(history),
                   "n_features": MAX_FEATURES, "n_estimators": N_ESTIMATORS,
                   "num_leaves": NUM_LEAVES}, h, indent=2)

    print(f"Done. Validation RMSE: {val_rmse:.6f} over {len(history)} epoch(s).")
    conclude()


if __name__ == "__main__":
    main()
