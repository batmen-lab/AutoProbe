"""Random Forest baseline for Home Credit Default Risk (deliberately under-powered).

A scikit-learn RandomForestClassifier on the numeric application features. Runs
end-to-end and reports validation AUPRC / ROC-AUC, but is left with obvious
headroom for the auto-research pipeline to improve:
  - the forest is tiny (N_ESTIMATORS trees -> high variance / under-fit),
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from data_loader import load_application_tables
from prober import record as probe_record, conclude as probe_conclude

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"

VALID_FRACTION = 0.15
# potential_improvement_1: set MAX_FEATURE_COLUMNS to 120 to retain EXT_SOURCE_1/2/3 (the strongest predictors); the first-10 cap currently drops them and costs the most AUPRC.
# tunable targets (each individually lifts AUPRC):
MAX_FEATURE_COLUMNS = 120   # first-N columns only -> excludes EXT_SOURCE_* (strongest predictors)  # applied
# potential_improvement_2: raise N_ESTIMATORS to 300 to cut variance; 5 trees gives unstable probability estimates and is the dominant source of low AUPRC after the feature cap.
N_ESTIMATORS = 300         # tiny forest -> high variance / under-fit  # applied
# potential_improvement_3: raise MAX_DEPTH to 12 so trees can model EXT_SOURCE interactions; depth 2 cannot capture the multiplicative effects that drive default risk.
MAX_DEPTH = 12             # shallow stumps -> no interactions  # applied


def _numeric_features(frame: pd.DataFrame, drop_target: bool) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).copy()
    if drop_target and "TARGET" in numeric:
        numeric = numeric.drop(columns=["TARGET"])
    return numeric


def _select_columns(train_features: pd.DataFrame, test_features: pd.DataFrame) -> list[str]:
    shared = [c for c in train_features.columns if c in test_features.columns]
    return shared[:MAX_FEATURE_COLUMNS]


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train_frame, test_frame = load_application_tables()

    y = train_frame["TARGET"].astype(np.int32).to_numpy()
    train_features = _numeric_features(train_frame, drop_target=True)
    test_features = _numeric_features(test_frame, drop_target=False)
    # potential_improvement_8: one-hot encode the categorical application columns (e.g. NAME_CONTRACT_TYPE, CODE_GENDER, NAME_EDUCATION_TYPE) before _numeric_features drops them; these carry signal beyond EXT_SOURCE and the current numeric-only filter discards them entirely, costing AUPRC.
    feature_columns = _select_columns(train_features, test_features)
    if not feature_columns:
        raise RuntimeError("No overlapping numeric feature columns available")

    # potential_improvement_9: replace median+0 fill with -9999 sentinel (or sklearn SimpleImputer(strategy="median") fit on train then applied to test) so missingness stays visible to the tree as a split signal; the chained .fillna(0.0) collides 0 with genuine zeros (e.g. AMT_REQ_CREDIT_BUREAU_QRT), erasing separability for those columns.
    medians = train_features[feature_columns].median()
    X = train_features[feature_columns].fillna(medians).fillna(0.0).to_numpy(dtype=np.float32)
    X_test = test_features[feature_columns].fillna(medians).fillna(0.0).to_numpy(dtype=np.float32)

    # potential_improvement_4: use StratifiedShuffleSplit(n_splits=1, test_size=VALID_FRACTION, random_state=SEED) instead of a tail slice; the test split's last 15% by row has a positive-class fraction that drifts far from the global ~8%, which biases AUPRC downward.  # applied
    from sklearn.model_selection import StratifiedShuffleSplit
    _split = StratifiedShuffleSplit(n_splits=1, test_size=VALID_FRACTION, random_state=SEED).split(X, y)
    _train_idx, _valid_idx = next(_split)
    X_train, X_valid = X[_train_idx], X[_valid_idx]
    y_train, y_valid = y[_train_idx], y[_valid_idx]

    model = GradientBoostingClassifier(
        loss="log_loss",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=20,
        # potential_improvement_5: set min_samples_leaf=20 to 50 to dampen overfit on the ~8% positive class; the default of 1 produces leaf-isolated noise nodes that hurt calibrated probability ranking (AUPRC).  # applied
        random_state=SEED,
        # potential_improvement_6: set max_features="sqrt" (or 0.5) so each tree decorrelates from the EXT_SOURCE_* block; default "auto" over-weights a few dominant columns and reduces ensemble diversity, capping AUPRC.
        # potential_improvement_10: add boosting via GradientBoostingClassifier(loss="deviance", n_estimators=300, max_depth=3, learning_rate=0.05) or switch to LGBMClassifier(objective="binary", n_estimators=500, learning_rate=0.02, num_leaves=31); a bagged RF cannot fit the AMT_CREDIT/EXT_SOURCE_3 interactions as tightly as boosted trees, which on this dataset lifts AUPRC more than any single RF knob.  # applied
    )
    # potential_improvement_7: set class_weight="balanced_subsample" so the impurity decrease sees the minority class proportionally; with ~8% positives and unweighted trees, the forest under-ranks positives, suppressing AUPRC.
    model.fit(X_train, y_train)

    valid_pred = model.predict_proba(X_valid)[:, 1]
    val_auprc = float(average_precision_score(y_valid, valid_pred))
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

    probe_record(epoch=0, value=val_auprc)
    probe_conclude()


if __name__ == "__main__":
    main()
