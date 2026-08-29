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
from sklearn.model_selection import train_test_split

from data_loader import load_transaction_numeric
from prober import record, conclude

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"

# potential_improvement_1: set VALID_FRACTION to 0.10 - a smaller valid split gives the model ~5% more training rows in a highly imbalanced fraud task, which typically lifts AUPRC.
VALID_FRACTION = 0.15
# tunable targets (each individually lifts AUPRC):
# potential_improvement_2: raise MAX_FEATURE_COLUMNS to 300 (or higher up to ~380 - the full "TransactionAmt/card/addr/dist/C/D" pool) so the strong C1-C14 / D1-D15 counters/deltas are included; these carry the bulk of fraud signal and AUPRC jumps substantially with them.
MAX_FEATURE_COLUMNS = 300    # few numeric features -> drops strong fraud signals
# potential_improvement_3: raise N_ESTIMATORS to 2000 (paired with early stopping and a lower LR) - 5 rounds massively under-fits; ~1500-2000 rounds is the standard AUPRC-optimal range for this dataset.
N_ESTIMATORS = 5           # far too few boosting rounds -> under-fit
# potential_improvement_4: raise MAX_DEPTH to 7 (range 6-8) so trees can capture the card-id x amount x C-counter interactions that drive fraud AUPRC; depth=2 forbids meaningful interactions.
MAX_DEPTH = 7              # shallow trees -> no interactions
# potential_improvement_5: lower LEARNING_RATE to 0.03 (range 0.02-0.05) once N_ESTIMATORS is increased; smaller steps + many rounds is a well-known AUPRC-improving recipe for XGBoost on this competition.
LEARNING_RATE = 0.1


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_transaction_numeric(max_feature_columns=MAX_FEATURE_COLUMNS)
    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy().astype(np.int32)

    # potential_improvement_6: replace this sequential head/tail split with a stratified split (e.g. sklearn.model_selection.train_test_split(..., stratify=y, test_size=VALID_FRACTION, random_state=SEED)); the current split leaves the tail 15% of rows for validation which has a different fraud rate/time distribution than train, deflating AUPRC by ~0.02-0.05. # applied
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=VALID_FRACTION, stratify=y, random_state=SEED
    )

    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=SEED,
        n_jobs=1,
        tree_method="hist",
        eval_metric="aucpr",
        # potential_improvement_7: add scale_pos_weight=28.0 (the ~1:28 non-fraud:fraud imbalance in this dataset) - re-weighting positives directly targets AUPRC in a heavily imbalanced binary task.
        # potential_improvement_8: add subsample=0.8 and colsample_bytree=0.8 (both in the 0.7-0.9 range) - these regularizers reduce variance on the high-cardinality card/addr features and consistently lift held-out AUPRC. # applied
        subsample=0.8,
        colsample_bytree=0.8,
        # potential_improvement_9: add reg_alpha=0.1, reg_lambda=1.5, min_child_weight=5, and gamma=0.1 - these regularization terms curb overfitting on rare-fraud leaves and are known to raise AUPRC on IEEE-CIS.
    )
    # potential_improvement_10: switch this call to model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=100, verbose=False) so training halts at the AUPRC-optimal round instead of blindly using all N_ESTIMATORS rounds - the single biggest AUPRC lever once N_ESTIMATORS is large.
    model.fit(X_train, y_train)

    for epoch in range(1, N_ESTIMATORS + 1):
        pred_epoch = model.predict_proba(X_valid, iteration_range=(0, epoch))[:, 1]
        record(epoch, float(average_precision_score(y_valid, pred_epoch)))

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

    conclude()


if __name__ == "__main__":
    main()
