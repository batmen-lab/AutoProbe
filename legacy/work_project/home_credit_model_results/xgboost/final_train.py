"""XGBoost baseline for Home Credit Default Risk (deliberately under-powered).

An XGBoost XGBClassifier on the numeric application features. Runs
end-to-end and reports validation AUPRC / ROC-AUC, but is left with obvious
headroom for the auto-research pipeline to improve:
  - the ensemble is tiny (N_ESTIMATORS boosting rounds -> under-fit),
  - trees are shallow stumps (MAX_DEPTH -> no feature interactions),
  - only the first MAX_FEATURE_COLUMNS columns are used (drops EXT_SOURCE_1/2/3,
    the strongest predictors).
Each individually lifts the metric when fixed.

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
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from data_loader import load_application_tables
from prober import record, conclude

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"

VALID_FRACTION = 0.15
# potential_improvement_1: lower VALID_FRACTION to 0.10 (with stratified split, see below). With ~8% positives, 0.10 still yields ~240 validation positives for a stable AUPRC estimate while giving the model more training data — both lift val_auprc.
# tunable targets (each individually lifts AUPRC):
MAX_FEATURE_COLUMNS = 120  # applied: potential_improvement_2 — use all shared numeric columns (includes EXT_SOURCE_1/2/3)
# potential_improvement_2: raise MAX_FEATURE_COLUMNS to 120 (use all shared numeric columns). The first-10 slice excludes EXT_SOURCE_1/2/3, the single strongest predictor block for Home Credit default risk — including them is the largest available AUPRC gain.
N_ESTIMATORS = 50  # applied: potential_improvement_5 — capped at 50 (epoch-rule ceiling); comment named 400 but epoch cap forbids >50
# potential_improvement_3: raise N_ESTIMATORS to 400 (paired with early_stopping_rounds=40 in fit and LEARNING_RATE=0.03). 5 boosting rounds is far too few for the model to reach its AUPRC plateau.
MAX_DEPTH = 6  # applied: potential_improvement_4 — depth-6 trees (upper end of comment's "depth 4-6" range) unlock deeper EXT_SOURCE_* × AMT_CREDIT interactions
# potential_improvement_4: raise MAX_DEPTH to 5 (with LEARNING_RATE=0.03). Depth-2 stumps cannot model interactions among EXT_SOURCE_1/2/3 and AMT_CREDIT — depth 4-6 unlocks them and lifts AUPRC.
LEARNING_RATE = 0.03  # applied: potential_improvement_5
# potential_improvement_5: lower LEARNING_RATE to 0.03 (and raise N_ESTIMATORS to 400). Smaller steps with more rounds reach a higher AUPRC optimum than 0.1 with only 5 rounds.


def _numeric_features(frame: pd.DataFrame, drop_target: bool) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).copy()
    if drop_target and "TARGET" in numeric:
        numeric = numeric.drop(columns=["TARGET"])
    return numeric
# potential_improvement_6: impute NaNs explicitly with median fill (`numeric.fillna(numeric.median())`) BEFORE the train/valid split fit. XGBoost handles NaNs natively but treating them as a separate missing branch with depth-2 stumps leaves information on the table; median fill lets the EXT_SOURCE_* gaps become usable splits at small depth, raising AUPRC when N_ESTIMATORS is small.


def _select_columns(train_features: pd.DataFrame, test_features: pd.DataFrame) -> list[str]:
    shared = [c for c in train_features.columns if c in test_features.columns]
    priority = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    shared = [c for c in priority if c in shared] + [c for c in shared if c not in priority]
    return shared[:MAX_FEATURE_COLUMNS]
# potential_improvement_7: reorder `shared` so EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3 come first (move them to the front of the list). When MAX_FEATURE_COLUMNS is small this guarantees the strongest predictors are included; combined with raising the cap it is the highest-leverage AUPRC change in the file. — applied


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train_frame, test_frame = load_application_tables()

    y = train_frame["TARGET"].astype(np.int32).to_numpy()
    train_features = _numeric_features(train_frame, drop_target=True)
    test_features = _numeric_features(test_frame, drop_target=False)

    feature_columns = _select_columns(train_features, test_features)
    if not feature_columns:
        raise RuntimeError("No overlapping numeric feature columns available")

    X = train_features[feature_columns].to_numpy(dtype=np.float32)
    X_test = test_features[feature_columns].to_numpy(dtype=np.float32)

    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=VALID_FRACTION, stratify=y, random_state=SEED,
    )
# potential_improvement_8: replace the chronological row-order split with `sklearn.model_selection.train_test_split(..., stratify=y, test_size=VALID_FRACTION, random_state=SEED)`. Home Credit TARGET has ~8% positives and application_train is not time-ordered by row index; an unstratified tail slice yields a noisy validation positive rate that depresses and destabilizes val_auprc. Stratifying keeps the validation prevalence at ~8% and aligns it with the training prevalence, which is what the probe measures. — applied

    num_neg = int((y_train == 0).sum())
    num_pos = int((y_train == 1).sum())
    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(num_neg / num_pos) if num_pos > 0 else 1.0,
        random_state=SEED,
        n_jobs=1,
        tree_method="hist",
        eval_metric="aucpr",
# potential_improvement_9: add `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, and `scale_pos_weight=(num_neg/num_pos)` computed from y_train (Home Credit ~ 11:1). At 8% prevalence, logloss under-rewards the minority class; weighting positives ~11x and subsampling rows/columns cap per-leaf overfitting to rare positives — together these typically raise validation AUPRC by 1-3 points once N_ESTIMATORS is raised. — applied
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
# potential_improvement_10: add `early_stopping_rounds=40` here. Without early stopping, raising N_ESTIMATORS to 400 risks overfitting the minority class and degrading validation AUPRC after the optimum; patience=40 lets the probe trajectory reach its peak AUPRC and stop there.
    )

    evals_result = model.evals_result()
    evals = []
    for _key, per_metric in evals_result.items():
        if "aucpr" in per_metric:
            evals = per_metric["aucpr"]
            break
    for epoch_idx, value in enumerate(evals, start=1):
        record(epoch_idx, float(value))

    valid_pred = model.predict_proba(X_valid)[:, 1]
    val_auprc = float(average_precision_score(y_valid, valid_pred))
    if not evals:
        record(1, val_auprc)
    try:
        val_auroc = float(roc_auc_score(y_valid, valid_pred))
    except ValueError:
        val_auroc = 0.5

    test_pred = model.predict_proba(X_test)[:, 1]
    pd.DataFrame({"SK_ID_CURR": test_frame["SK_ID_CURR"], "TARGET": test_pred}).to_csv(
        RESULT_DIR / "submission.csv", index=False)

    with (RESULT_DIR / "validation_metrics.json").open("w", encoding="utf-8") as h:
        json.dump({"metric": "auprc", "value": val_auprc, "val_auroc": val_auroc,
                   "n_features": int(len(feature_columns)), "n_estimators": N_ESTIMATORS,
                   "max_depth": MAX_DEPTH}, h, indent=2)

    print(f"Validation AUPRC: {val_auprc:.6f} | Validation ROC-AUC: {val_auroc:.6f}")

    conclude()


if __name__ == "__main__":
    main()
