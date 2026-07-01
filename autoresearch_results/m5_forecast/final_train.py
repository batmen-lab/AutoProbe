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
TRAIN_DAYS = 365   # applied: raise TRAIN_DAYS from 120 to 365 so the booster sees full yearly seasonality (comment 1)
# potential_improvement_1: raise TRAIN_DAYS to 365-500 so the booster sees full yearly seasonality; 120 days truncates the trend/seasonality signal and lifts val RMSE.

# tunable targets (each individually lowers RMSE):
# potential_improvement_2: raise MAX_FEATURES to 9 (use the full FEATURE_NAMES list incl. rmean28/rmean_prev28/wday/dept/store/cat/state); only feeding 2 lags leaves most signal on the floor.
MAX_FEATURES = 10     # applied: use full FEATURE_NAMES (lags + rollings + weekday + series ids + snap)
# potential_improvement_3: raise N_ESTIMATORS to 200-500 boosting rounds; 3 rounds is severely under-fit and dominates val RMSE upward.
N_ESTIMATORS = 200    # applied: give the booster enough rounds to converge (was 3 -> under-fit, RMSE still descending at round 3)
# potential_improvement_4: raise NUM_LEAVES to 31-63 so trees can capture weekday x dept x store interactions; stumps (==2) cannot model cross-effects.
NUM_LEAVES = 31     # applied: allow weekday x dept x store interactions
# potential_improvement_5: set LEARNING_RATE to 0.03-0.05 with the larger n_estimators above; 0.1 over 3 rounds overshoots and the per-round RMSE trajectory never converges.
LEARNING_RATE = 0.05   # applied: pair 200 rounds + early stopping with a lower LR (was 0.1 -> overshoots)

FEATURE_NAMES = ["lag28", "lag35", "rmean28", "rmean_prev28", "wday",
                 "dept", "store", "cat", "state", "snap"]
# potential_improvement_6: add a 7-day and 28-day rolling std feature (and a snap/event flag) to FEATURE_NAMES; volatility and promo/snap days are material drivers of M5 sales variance, omitting them caps achievable RMSE.


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
    # potential_improvement_7: apply np.log1p to V before building lags/rollings (and inverse on preds); M5 sales are heavily right-skewed, log-transforming the target lowers RMSE on the bulk of series.

    train_days = range(T - HORIZON - TRAIN_DAYS, T - HORIZON)
    val_days = range(T - HORIZON, T)
    # potential_improvement_8: use the last 28 days before val as an early-stopping set and rebuild train_days to end at T-2*HORIZON; training on days directly adjacent to the validation window leaks recent trend, but more importantly a dedicated early-stop set lets n_estimators auto-tune to the RMSE-minimising round instead of stopping at 3.
    X_train = np.vstack([feats(t) for t in train_days])
    y_train = np.concatenate([V[:, t] for t in train_days])
    X_valid = np.vstack([feats(t) for t in val_days])
    y_valid = np.concatenate([V[:, t] for t in val_days])

    model = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS, num_leaves=NUM_LEAVES, learning_rate=LEARNING_RATE,
        random_state=SEED, n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1,
        # potential_improvement_9: add min_data_in_leaf=100-500 and lambda_l2=1.0-5.0 to regularise the larger trees; without these, deeper trees with more rounds overfit the noisy zero-sales days and val RMSE rises.
        min_data_in_leaf=200, lambda_l2=2.0,  # applied: regularise the 31-leaf trees to combat the epoch-100+ plateau
        # potential_improvement_10: set objective="tweedie" tweedie_variance_power=1.1-1.5 instead of the default rmse/l2; M5 sales are zero-inflated count-like, and Tweedie loss consistently beats L2 on validation RMSE for this dataset.
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="rmse",
              # potential_improvement_11: add callbacks=[lgb.early_stopping(stopping_rounds=30)] so the booster stops at the RMSE-minimising round instead of running all N_ESTIMATORS; combined with a larger n_estimators this directly lowers the recorded final RMSE.
              callbacks=[lgb.early_stopping(stopping_rounds=30)]
              )

    # per-boosting-round validation RMSE == per-epoch trajectory
    history = list(model.evals_result_["valid_0"]["rmse"])
    for epoch, rmse in enumerate(history, start=1):
        print(f"Epoch {epoch:03d} | val RMSE {rmse:.6f}")
        record(epoch, rmse)
    val_rmse = float(history[-1])
    conclude()

    with (RESULT_DIR / "validation_metrics.json").open("w", encoding="utf-8") as h:
        json.dump({"metric": "rmse", "value": val_rmse, "epochs": len(history),
                   "n_features": MAX_FEATURES, "n_estimators": N_ESTIMATORS,
                   "num_leaves": NUM_LEAVES}, h, indent=2)

    print(f"Done. Validation RMSE: {val_rmse:.6f} over {len(history)} epoch(s).")


if __name__ == "__main__":
    main()
