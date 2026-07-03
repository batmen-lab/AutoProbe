"""Neural-network (MLP) baseline for Home Credit Default Risk (deliberately under-powered).

A small PyTorch multilayer perceptron trained on the numeric application features.
It runs end-to-end and reports validation AUPRC / ROC-AUC, but is left with a lot
of obvious headroom for the auto-research pipeline to improve:
  - features are fed RAW / unscaled (USE_FEATURE_SCALING=False) -- an MLP on raw
    mixed-magnitude features (AMT_CREDIT ~1e6 vs binary flags) cannot learn,
  - the hidden layer is tiny (HIDDEN_DIM=4 -> severe capacity bottleneck),
  - it trains for only NUM_EPOCHS epochs (under-fit),
  - only the first MAX_FEATURE_COLUMNS columns are used (drops EXT_SOURCE_1/2/3,
    the strongest predictors, which live further right in the table).
Each of these individually lifts the metric when fixed.

Run:
    python train.py
"""

from __future__ import annotations

# --- determinism preamble: training-step reproducibility (same train.py -> same metric) ---
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
import torch
import torch.nn as nn
torch.manual_seed(SEED)
torch.set_num_threads(1)
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from data_loader import load_application_tables
from prober import record, conclude

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"

VALID_FRACTION = 0.15
# tunable targets (each individually lifts AUPRC):
# potential_improvement_1: set MAX_FEATURE_COLUMNS to 200 (or None) so EXT_SOURCE_1/2/3 and other strong predictors on the right side of the frame are included; AUPRC is bottlenecked by these features being dropped.
MAX_FEATURE_COLUMNS = None     # first-N columns only -> excludes EXT_SOURCE_* (strongest predictors)
# potential_improvement_2: raise HIDDEN_DIM to 128-256; 4 units cannot represent the nonlinear interactions between AMT_* / DAYS_* / EXT_SOURCE_* features that drive default-risk AUPRC.
HIDDEN_DIM = 128             # tiny hidden layer -> capacity bottleneck
# potential_improvement_3: increase NUM_EPOCHS to 30-50 (paired with a small LR / early stopping on val AUPRC); 3 epochs is deep under-fit, val AUPRC plateaus much later.
NUM_EPOCHS = 3               # far too few epochs -> under-fit
# potential_improvement_4: with feature scaling on and larger model, drop LEARNING_RATE to 3e-4 to stabilize training over 30+ epochs and let val AUPRC keep climbing instead of oscillating.
LEARNING_RATE = 5e-4
# potential_improvement_5: reduce BATCH_SIZE to 256-512; the positive class (~8%) needs enough gradient noise per step and enough positives per mini-batch for BCE-with-pos_weight to actually push the decision boundary.
BATCH_SIZE = 1024
# potential_improvement_6: set USE_FEATURE_SCALING = True; raw mixed-magnitude features (AMT_CREDIT ~1e6 vs binary flags) make the MLP unable to learn -- flipping this alone is the single largest AUPRC lift.
USE_FEATURE_SCALING = True  # raw features -> an MLP cannot learn from them; set True to standardize


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        # potential_improvement_7: replace the single-hidden-layer stack with BatchNorm1d(in_dim) -> Linear(in_dim,hidden) -> BN -> ReLU -> Dropout(0.2) -> Linear(hidden,hidden//2) -> BN -> ReLU -> Dropout(0.2) -> Linear(hidden//2,1); depth + BN + dropout(0.2) is the standard MLP recipe that lifts tabular AUPRC well above a shallow 1-hidden-layer net.
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _numeric_features(frame: pd.DataFrame, drop_target: bool) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).copy()
    if drop_target and "TARGET" in numeric:
        numeric = numeric.drop(columns=["TARGET"])
    return numeric


def _select_columns(train_features: pd.DataFrame, test_features: pd.DataFrame) -> list[str]:
    shared = [c for c in train_features.columns if c in test_features.columns]
    return shared[:MAX_FEATURE_COLUMNS]


def _evaluate(model, Xv, yv) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xv)).numpy()
    model.train()
    auprc = float(average_precision_score(yv, p))
    try:
        auroc = float(roc_auc_score(yv, p))
    except ValueError:
        auroc = 0.5
    return auprc, auroc


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train_frame, test_frame = load_application_tables()

    y = train_frame["TARGET"].astype(np.float32).to_numpy()
    train_features = _numeric_features(train_frame, drop_target=True)
    test_features = _numeric_features(test_frame, drop_target=False)

    feature_columns = _select_columns(train_features, test_features)
    if not feature_columns:
        raise RuntimeError("No overlapping numeric feature columns available")

    # potential_improvement_8: before median-fill, replace(+/-np.inf, np.nan) and append per-column missingness-indicator columns (isnull().astype(float32)); missingness in DAYS_EMPLOYED (365243 sentinel) and EXT_SOURCE_* is itself predictive -- losing that signal caps AUPRC. # applied
    train_features = train_features.replace([np.inf, -np.inf], np.nan)
    test_features = test_features.replace([np.inf, -np.inf], np.nan)
    medians = train_features[feature_columns].median()
    mask_train = train_features[feature_columns].isnull().to_numpy(dtype=np.float32)
    mask_test = test_features[feature_columns].isnull().to_numpy(dtype=np.float32)
    X = np.hstack([
        train_features[feature_columns].fillna(medians).fillna(0.0).to_numpy(dtype=np.float32),
        mask_train,
    ])
    X_test = np.hstack([
        test_features[feature_columns].fillna(medians).fillna(0.0).to_numpy(dtype=np.float32),
        mask_test,
    ])
    feature_columns = list(feature_columns) + [f"{c}__isnull" for c in feature_columns]

    # potential_improvement_9: swap the head/tail sequential split for a stratified shuffled split (sklearn.model_selection.StratifiedShuffleSplit with test_size=0.15, random_state=SEED); the raw slice can put a skewed positive rate into val -- stratifying by TARGET stabilizes val AUPRC and reduces variance across runs. # applied
    _sss = StratifiedShuffleSplit(n_splits=1, test_size=VALID_FRACTION, random_state=SEED)
    _train_idx, _valid_idx = next(_sss.split(X, y))
    X_train, X_valid = X[_train_idx], X[_valid_idx]
    y_train, y_valid = y[_train_idx], y[_valid_idx]

    # Feature standardization (fit on train split only). Disabled in the baseline.
    if USE_FEATURE_SCALING:
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mu) / sd
        X_valid = (X_valid - mu) / sd
        X_test = (X_test - mu) / sd

    Xt = torch.tensor(X_train)
    yt = torch.tensor(y_train)
    Xv = torch.tensor(X_valid)

    model = MLP(len(feature_columns), HIDDEN_DIM)
    # potential_improvement_10: switch to torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4); decoupled weight decay regularizes the larger hidden layer and, combined with dropout, is a well-known AUPRC lift on tabular MLPs versus plain Adam with no regularization.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    n_pos = float((y_train == 1).sum())
    n_neg = float((y_train == 0).sum())
    pos_weight = torch.tensor(n_neg / max(1.0, n_pos), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    gen = torch.Generator().manual_seed(SEED)
    n = len(Xt)
    val_auprc = val_auroc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
        val_auprc, val_auroc = _evaluate(model, Xv, y_valid)
        record(epoch, val_auprc)
        print(f"Epoch {epoch:02d} | val AUPRC {val_auprc:.6f} | val ROC-AUC {val_auroc:.6f}")

    model.eval()
    with torch.no_grad():
        test_pred = torch.sigmoid(model(torch.tensor(X_test))).numpy()
    pd.DataFrame({"SK_ID_CURR": test_frame["SK_ID_CURR"], "TARGET": test_pred}).to_csv(
        RESULT_DIR / "submission.csv", index=False)

    with (RESULT_DIR / "validation_metrics.json").open("w", encoding="utf-8") as h:
        json.dump({"metric": "auprc", "value": val_auprc, "val_auroc": val_auroc,
                   "n_features": int(len(feature_columns)), "hidden_dim": HIDDEN_DIM,
                   "num_epochs": NUM_EPOCHS}, h, indent=2)

    print(f"Done. Validation AUPRC: {val_auprc:.6f} | Validation ROC-AUC: {val_auroc:.6f}")

    conclude()


if __name__ == "__main__":
    main()
