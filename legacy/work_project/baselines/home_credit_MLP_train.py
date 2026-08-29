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

from data_loader import load_application_tables

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"

VALID_FRACTION = 0.15
# tunable targets (each individually lifts AUPRC):
MAX_FEATURE_COLUMNS = 10     # first-N columns only -> excludes EXT_SOURCE_* (strongest predictors)
HIDDEN_DIM = 4               # tiny hidden layer -> capacity bottleneck
NUM_EPOCHS = 3               # far too few epochs -> under-fit
LEARNING_RATE = 1e-3
BATCH_SIZE = 4096
USE_FEATURE_SCALING = False  # raw features -> an MLP cannot learn from them; set True to standardize


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
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

    medians = train_features[feature_columns].median()
    X = train_features[feature_columns].fillna(medians).fillna(0.0).to_numpy(dtype=np.float32)
    X_test = test_features[feature_columns].fillna(medians).fillna(0.0).to_numpy(dtype=np.float32)

    split = int(len(X) * (1.0 - VALID_FRACTION))
    X_train, X_valid = X[:split], X[split:]
    y_train, y_valid = y[:split], y[split:]

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


if __name__ == "__main__":
    main()
