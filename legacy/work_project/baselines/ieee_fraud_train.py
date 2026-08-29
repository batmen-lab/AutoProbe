"""XGBoost baseline for IEEE-CIS Fraud Detection (deliberately under-powered).

An XGBoost classifier on a numeric slice of train_transaction.csv. Runs
end-to-end and reports validation AUPRC / ROC-AUC, but is left with obvious
headroom for the auto-research pipeline to improve:
  - only MAX_FEATURE_COLUMNS numeric features are loaded (the strongest fraud
    signals live further down the column list),
  - the ensemble is tiny (N_ESTIMATORS boosting rounds -> under-fit),
  - the trees are shallow (MAX_DEPTH=2 -> no feature interactions).
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
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from data_loader import load_transaction_numeric

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"

VALID_FRACTION = 0.15
# tunable targets (each individually lifts AUPRC):
MAX_FEATURE_COLUMNS = 8    # few numeric features -> drops strong fraud signals
N_ESTIMATORS = 5           # far too few boosting rounds -> under-fit
MAX_DEPTH = 2              # shallow trees -> no interactions
LEARNING_RATE = 0.1


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_transaction_numeric(max_feature_columns=MAX_FEATURE_COLUMNS)
    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy().astype(np.int32)

    split = int(len(X) * (1.0 - VALID_FRACTION))
    X_train, X_valid = X[:split], X[split:]
    y_train, y_valid = y[:split], y[split:]

    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=SEED,
        n_jobs=1,
        tree_method="hist",
        eval_metric="aucpr",
    )
    model.fit(X_train, y_train)

    valid_pred = model.predict_proba(X_valid)[:, 1]
    val_auprc = float(average_precision_score(y_valid, valid_pred))
    try:
        val_auroc = float(roc_auc_score(y_valid, valid_pred))
    except ValueError:
        val_auroc = 0.5

    with (RESULT_DIR / "validation_metrics.json").open("w", encoding="utf-8") as h:
        json.dump({"metric": "auprc", "value": val_auprc, "val_auroc": val_auroc,
                   "n_features": int(X.shape[1]), "n_estimators": N_ESTIMATORS,
                   "max_depth": MAX_DEPTH}, h, indent=2)

    print(f"Validation AUPRC: {val_auprc:.6f} | Validation ROC-AUC: {val_auroc:.6f}")


if __name__ == "__main__":
    main()
