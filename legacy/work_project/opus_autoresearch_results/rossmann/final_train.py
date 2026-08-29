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
from prober import record, conclude

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
VALID_QUANTILE = 0.9   # last 10% of dates -> validation (time-ordered split)

# potential_improvement_1: expand FEATURES to ["Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance", "Promo2", "year", "month", "day", "woy"] — restricting the model to DayOfWeek discards the Store, Promo and date signals that drive most of the Sales variance, capping how low val_rmspe can go
FEATURES = ["Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance", "Promo2", "year", "month", "day", "woy"]
# potential_improvement_2: raise N_ESTIMATORS to 3000-5000 (paired with early stopping) — only 5 boosting rounds is severely underfit and leaves huge per-row percentage error in val_rmspe
N_ESTIMATORS = 5
# potential_improvement_3: raise NUM_LEAVES to 127-255 so trees can capture store x promo x seasonality interactions; stumps with 2 leaves cannot model the multiplicative Sales structure that determines RMSPE
NUM_LEAVES = 127
# potential_improvement_4: lower LEARNING_RATE to 0.03-0.05 to pair with the larger N_ESTIMATORS above — a smaller step with more rounds converges to a lower final val_rmspe than 0.1 with few rounds
LEARNING_RATE = 0.1


def _rmspe(y, p):
    p = np.clip(p, 1.0, None)
    return float(np.sqrt(np.mean(((y - p) / y) ** 2)))


def _prepare() -> pd.DataFrame:
    df = load_train()
    df["Date"] = pd.to_datetime(df["Date"])
    store = pd.read_csv(BASE_DIR / "data" / "store.csv", low_memory=False)
    df = df.merge(store, on="Store", how="left")
    # potential_improvement_5: after the store merge, impute NaNs: fill CompetitionDistance with its median (~2325), and fill CompetitionOpenSinceYear/Month, Promo2SinceYear/Week with 0 — leaving these NaN makes the store-metadata columns unusable by the model and inflates val_rmspe
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

    # potential_improvement_6: fit on log1p(Sales) and invert with expm1(pred) before computing val_rmspe — RMSPE is a percentage error, so a log-target directly optimises the relative loss and typically cuts val_rmspe by roughly half on Rossmann  # applied
    y_tr_orig = tr["Sales"].to_numpy(dtype=np.float64)
    y_va_orig = va["Sales"].to_numpy(dtype=np.float64)
    y_tr = np.log1p(y_tr_orig)
    y_va = np.log1p(y_va_orig)

    # potential_improvement_7: pass objective='regression_l1' (or 'huber' with alpha=0.9) to LGBMRegressor — L1/Huber is more robust than the default L2 to the long-tailed Sales distribution and lowers val_rmspe by not being dominated by high-Sales outliers
    # potential_improvement_8: also pass feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0, min_child_samples=20 — stochastic regularisation prevents per-store overfitting and pushes val_rmspe down once the model is larger (see improvements 2 and 3)
    model = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        num_leaves=NUM_LEAVES,
        learning_rate=LEARNING_RATE,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )

    def _rmspe_lgb(y_true, y_pred):
        y_true = np.expm1(y_true)
        y_pred = np.clip(np.expm1(y_pred), 1.0, None)
        return "rmspe", float(np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))), False

    eval_hist: dict = {}
    # potential_improvement_9: add lgb.early_stopping(stopping_rounds=100, first_metric_only=True) to the callbacks list — once N_ESTIMATORS is large (see improvement 2) early stopping selects the round that minimises val_rmspe instead of always using the last (potentially overfit) round
    # potential_improvement_10: pass categorical_feature=["Store","StoreType","Assortment","DayOfWeek","month","StateHoliday"] to fit — LightGBM's native categorical split handles high-cardinality Store far better than integer codes and materially lowers val_rmspe
    model.fit(
        tr[FEATURES], y_tr,
        eval_set=[(va[FEATURES], y_va)],
        eval_metric=_rmspe_lgb,
        categorical_feature=["Store", "StoreType", "Assortment", "DayOfWeek", "month", "StateHoliday"],
        callbacks=[lgb.record_evaluation(eval_hist)],
    )
    pred = np.expm1(model.predict(va[FEATURES]))
    val_rmspe = _rmspe(y_va_orig, pred)

    _round_series = None
    for _ds_name, _metrics in eval_hist.items():
        if "rmspe" in _metrics:
            _round_series = _metrics["rmspe"]
            break
    if _round_series is not None:
        for _i, _v in enumerate(_round_series, start=1):
            record(_i, float(_v))
    else:
        record(1, val_rmspe)
    conclude()

    with (RESULT_DIR / "validation_metrics.json").open("w", encoding="utf-8") as h:
        json.dump({"metric": "rmspe", "value": val_rmspe, "n_features": len(FEATURES),
                   "n_estimators": N_ESTIMATORS, "num_leaves": NUM_LEAVES}, h, indent=2)

    print(f"Validation RMSPE: {val_rmspe:.6f}")


if __name__ == "__main__":
    main()
