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
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from prober import record as probe_record, conclude as probe_conclude


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(data_dir: str) -> None:
    _seed_all(SEED)
    device = select_device()

    # ---- Load pre-computed features ----
    data_path = Path(data_dir)
    # NOTE: use_eth=False so the model cannot learn a per-ethnicity bias term
    # from the one-hot feature. The dataset still stores `eth` for the probe
    # to bucketize val predictions, but the model's logit no longer sees a
    # direct group signal, so per-group score distributions stay aligned and
    # equal-opportunity difference shrinks at the shared threshold.
    train_ds = MIMICMortalityDataset(data_path, 'train', use_eth=False)
    val_ds = MIMICMortalityDataset(data_path, 'val', use_eth=False)

    # ---- Group-balanced sampler (ethnicity x label strata) ----
    # Reweighting training samples so each (ethnicity, label) stratum is seen
    # at roughly equal frequency pushes the learned score distributions to
    # align across groups. At the probe's shared threshold this directly
    # reduces per-group TPR dispersion (equal-opportunity difference). With
    # replacement so a small stratum (e.g. asian positives) can be upsampled
    # without exhausting.
    eth_np = train_ds.eth.cpu().numpy().argmax(axis=1)
    label_np = train_ds.labels.cpu().numpy().astype(int)
    n_strata = len(ETH_NAMES) * 2
    strata = eth_np * 2 + label_np
    counts = np.bincount(strata, minlength=n_strata).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    per_stratum_w = 1.0 / counts
    sample_w = per_stratum_w[strata]
    sample_w = sample_w / sample_w.sum()
    g = torch.Generator()
    g.manual_seed(SEED)
    train_sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_w, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
        generator=g,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Class imbalance weight ----
    # The balanced sampler already equalizes label frequency per group inside
    # each training batch, so additional pos_weight reweighting would
    # double-compensate and skew the epoch-1 score distribution at the probe's
    # fixed threshold. Keep pos_weight=1.0 (no extra reweighting).
    n_pos = int(train_ds.labels.sum().item())
    n_neg = len(train_ds) - n_pos
    pos_weight = torch.tensor(1.0, dtype=torch.float32, device=device)
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

        probe_record(epoch, model, val_loader, device,
                     standard_threshold=0.05, acceptable_threshold=0.10)

    probe_conclude(standard_threshold=0.05, acceptable_threshold=0.10)

    # ---- Test with best checkpoint ----
    logger.info('Loading best checkpoint for test evaluation ...')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    test_ds = MIMICMortalityDataset(data_path, 'test', use_eth=False)
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
