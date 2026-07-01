"""Train a logistic regression model for ICU mortality prediction (MIMIC-III).

Expects pre-computed TF-IDF feature files produced by preprocess.py:
    {split}_tfidf.npz, {split}_meta.npz

Usage:
    python train.py  [--data_dir <dir>]

Note on metrics: ICU mortality is a strongly imbalanced label (~9% positives).
AUROC is prevalence-independent and therefore optimistic here -- it can look
high while the model misses most of the rare deaths at any usable threshold.
This baseline therefore (a) trains with class-weighted BCE and (b) selects the
checkpoint on validation AUPRC (average precision), which tracks how well the
minority (death) class is actually recovered. AUROC is still reported, but as a
secondary diagnostic, not the selection objective.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import random

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import MIMICMortalityDataset
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_DIR = SCRIPT_DIR / 'checkpoint'
DEFAULT_DATA_DIR = Path('/home/xuanhe_linux_001/aim_frontend_experiment3/aim/examples/agent_example_repos/mimic/data')

ETH_NAMES = ['white', 'black', 'hispanic', 'asian', 'other']


def _seed_all(seed: int) -> None:
    """Pin every RNG we touch so back-to-back runs produce identical metrics."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LogisticRegression(nn.Module):
    """Single linear layer -> logits (sigmoid applied via BCEWithLogitsLoss)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc='  Train', leave=False):
        features = batch['features'].to(device)
        labels = batch['label'].to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in tqdm(loader, desc='  Eval ', leave=False):
        features = batch['features'].to(device)
        labels = batch['label'].to(device)

        logits = model(features)
        total_loss += criterion(logits, labels).item()

        all_probs.append(logits.sigmoid().cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (probs >= 0.5).astype(int)

    return {
        'loss': total_loss / len(loader),
        # AUPRC (average precision) is the primary metric on this imbalanced
        # label: it reflects how well the rare death class is recovered.
        'auprc': average_precision_score(labels, probs),
        'recall': recall_score(labels, preds, zero_division=0),
        'precision': precision_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        # AUROC kept as a secondary diagnostic only (prevalence-independent,
        # optimistic under heavy imbalance).
        'auroc': roc_auc_score(labels, probs),
        'acc': float((preds == labels).mean()),
    }


# ===========================================================================
# ===  USER ANALYSIS — DO NOT MODIFY / DO NOT REMOVE / DO NOT RENAME  ========
# ===========================================================================
# `user_analyze()` below and its call in main() are HUMAN-OWNED. It is an
# independent per-ethnicity fairness audit that does NOT feed the probe metric.
# The auto-research / fix-loop agent MUST leave this function, its arguments,
# its call site, and its `.agent_probe/.user_analysis` output completely
# untouched. Do not edit, wrap, reorder, gate, or delete any of it.
# ---------------------------------------------------------------------------
def user_analyze(model, val_ds, device, threshold: float = 0.5) -> None:
    """USER ANALYSIS — DO NOT MODIFY. Independent per-race-group audit.

    For each ethnicity group it records the confusion composition (TP/FP/FN/TN,
    recall) on the validation split and saves a 100%-stacked-bar chart, to
    `<workspace>/.agent_probe/.user_analysis/`. Independent of the probe metric.
    """
    model.eval()
    loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    probs_list, label_list = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['features'].to(device))
            probs_list.append(logits.sigmoid().cpu().numpy())
            label_list.append(batch['label'].numpy())
    probs = np.concatenate(probs_list)
    labels = np.concatenate(label_list).astype(int)
    eth = val_ds.eth.cpu().numpy()
    group = eth.argmax(axis=1)
    preds = (probs >= threshold).astype(int)

    out_dir = SCRIPT_DIR / '.agent_probe' / '.user_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for c, name in enumerate(ETH_NAMES):
        m = group == c
        y, p = labels[m], preds[m]
        tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
        data[name] = {
            'n': int(m.sum()), 'n_positive': int(y.sum()),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'recall': (tp / (tp + fn)) if (tp + fn) else None,
        }
    (out_dir / 'per_group.json').write_text(json.dumps(data, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.arange(len(ETH_NAMES))
    # Chart 1: full outcome composition per group (100% stacked over all members)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    segs = [('TP', '#2ca02c'), ('FN', '#d62728'), ('FP', '#ff7f0e'), ('TN', '#c7c7c7')]
    bottom = np.zeros(len(ETH_NAMES))
    for seg, col in segs:
        vals = np.array([data[g][seg] for g in ETH_NAMES], float)
        tot = np.array([max(data[g]['n'], 1) for g in ETH_NAMES], float)
        frac = vals / tot * 100.0
        ax.bar(x, frac, bottom=bottom, label=seg, color=col)
        bottom += frac
    ax.set_xticks(x); ax.set_xticklabels(ETH_NAMES); ax.set_ylim(0, 100)
    ax.set_ylabel('% of group'); ax.legend(ncol=4, fontsize=8)
    ax.set_title('Per-ethnicity prediction outcomes (100% stacked)')
    fig.tight_layout(); fig.savefig(out_dir / 'per_group_outcomes_stacked.png', dpi=130)
    plt.close(fig)

    # Chart 2: recall composition among each group's actual deaths (caught vs missed)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    caught = np.array([data[g]['TP'] for g in ETH_NAMES], float)
    missed = np.array([data[g]['FN'] for g in ETH_NAMES], float)
    pos = np.maximum(caught + missed, 1)
    ax.bar(x, caught / pos * 100, label='caught (TP)', color='#2ca02c')
    ax.bar(x, missed / pos * 100, bottom=caught / pos * 100, label='missed (FN)', color='#d62728')
    ax.set_xticks(x); ax.set_xticklabels(ETH_NAMES); ax.set_ylim(0, 100)
    ax.set_ylabel('% of group deaths'); ax.legend()
    ax.set_title('Equal-opportunity view: ICU deaths caught vs missed per ethnicity (100% stacked)')
    fig.tight_layout(); fig.savefig(out_dir / 'per_group_recall_stacked.png', dpi=130)
    plt.close(fig)

    logger.info('[user_analyze] wrote per-group audit to %s', out_dir)


# ===========================================================================
# ===  END USER ANALYSIS — DO NOT MODIFY ABOVE THIS LINE  ===================
# ===========================================================================


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(data_dir: str) -> None:
    _seed_all(SEED)
    device = select_device()

    # ---- Load pre-computed features ----
    data_path = Path(data_dir)
    train_ds = MIMICMortalityDataset(data_path, 'train')
    val_ds = MIMICMortalityDataset(data_path, 'val')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Class imbalance weight (positives are rare) ----
    n_pos = int(train_ds.labels.sum().item())
    n_neg = len(train_ds) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)
    logger.info('Class balance: %d pos / %d neg (pos_weight=%.2f)', n_pos, n_neg, pos_weight.item())

    # ---- Model, loss, optimizer ----
    input_dim = train_ds[0]['features'].shape[0]
    model = LogisticRegression(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    # ---- Checkpoint directory ----
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / 'best_model.pt'

    # ---- Training loop (select on val AUPRC, not AUROC) ----
    best_auprc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        logger.info(
            'Epoch %02d | loss %.4f | val AUPRC %.4f | recall %.3f | prec %.3f | F1 %.3f | AUROC %.4f',
            epoch, train_loss, val_metrics['auprc'], val_metrics['recall'],
            val_metrics['precision'], val_metrics['f1'], val_metrics['auroc'],
        )
        if val_metrics['auprc'] > best_auprc:
            best_auprc = val_metrics['auprc']
            torch.save(model.state_dict(), ckpt_path)
            logger.info('  -> Saved best model (AUPRC=%.4f) to %s', best_auprc, ckpt_path)

    # ---- Test with best checkpoint ----
    logger.info('Loading best checkpoint for test evaluation ...')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    # === USER ANALYSIS CALL — DO NOT MODIFY / DO NOT REMOVE / DO NOT GATE ===
    # Human-owned independent per-ethnicity fairness audit. Keep exactly as-is.
    user_analyze(model, val_ds, device)
    # === END USER ANALYSIS CALL ===

    test_ds = MIMICMortalityDataset(data_path, 'test')
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    logger.info('Test: %d samples', len(test_ds))

    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(
        'Done. Best val AUPRC: %.4f | Test AUPRC: %.4f | Test recall: %.3f | Test AUROC: %.4f',
        best_auprc, test_metrics['auprc'], test_metrics['recall'], test_metrics['auroc'],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train logistic regression for ICU mortality prediction.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help='Directory with pre-computed feature .npz files',
    )
    args = parser.parse_args()
    main(args.data_dir)
