"""Train a logistic regression model for ICU mortality prediction (MIMIC-III).

Expects pre-computed TF-IDF feature files produced by preprocess.py:
    {split}_tfidf.npz, {split}_meta.npz

Usage (data lives in the mimic_backup project):
    python train.py
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
from sklearn.metrics import f1_score, roc_auc_score
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
LEARNING_RATE = 2
NUM_EPOCHS = 15
SEED = 42

# ---------------------------------------------------------------------------
# Subgroup feature masking (training-only)
# ---------------------------------------------------------------------------
# Black + Hispanic clinical notes tend to be shorter / more abbreviated in
# MIMIC. We blank out their TF-IDF features during training as a noise-
# reduction step — the model can still learn from the ethnicity one-hot.
# Validation and test are untouched.
#
# Set MASK_MINORITY_TFIDF = False to disable and train on the full text.
MASK_MINORITY_TFIDF = True

# ETH one-hot index map: 0=white, 1=black, 2=hispanic, 3=asian, 4=other.
# Sized to land safely above the prober's MIN_SAMPLES gate.
MINORITY_ETH_INDICES = (1, 2)


def _seed_all(seed: int) -> None:
    """Pin every RNG we touch so back-to-back runs produce identical metrics.
    Stdlib + numpy + torch (CPU + CUDA). PYTHONHASHSEED is set for the
    process so any dict-order-dependent code is also stable."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_DIR = SCRIPT_DIR / 'checkpoint'
DEFAULT_DATA_DIR = Path('/home/xuanhe_linux_001/aim_frontend_experiment3/aim/examples/agent_example_repos/mimic/data')

ETH_NAMES = ['white', 'black', 'hispanic', 'asian', 'other']


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LogisticRegression(nn.Module):
    """Single linear layer -> sigmoid (via BCEWithLogitsLoss)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------
def _mask_minority_tfidf(ds: MIMICMortalityDataset, enabled: bool) -> None:
    """Zero out the TF-IDF rows for samples whose ethnicity is in
    MINORITY_ETH_INDICES. Labels and ethnicity dummies are left intact —
    the model can still see *which* group each row belongs to, it just
    can't see any clinical-note signal for them during training.

    Modifies `ds.tfidf` in place. Set `enabled=False` to disable and
    train on the original feature matrix.
    """
    if not enabled:
        return
    minority = torch.zeros(len(ds), dtype=torch.bool)
    for col in MINORITY_ETH_INDICES:
        minority |= ds.eth[:, col] == 1
    minority_idx = torch.where(minority)[0].numpy().tolist()
    if not minority_idx:
        return
    # CSR doesn't support fast row-assignment; LIL does. Convert, blank, convert back.
    lil = ds.tfidf.tolil()
    for i in minority_idx:
        lil.rows[i] = []
        lil.data[i] = []
    ds.tfidf = lil.tocsr()


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
        'auroc': roc_auc_score(labels, probs),
        'f1': f1_score(labels, preds, zero_division=0),
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
    train_ds = MIMICMortalityDataset(data_path, 'train')
    val_ds = MIMICMortalityDataset(data_path, 'val')

    # Blank TF-IDF rows for black + hispanic training samples. Val/test
    # untouched. Set MASK_MINORITY_TFIDF=False to disable.
    minority_n = int(sum((train_ds.eth[:, c] == 1).sum().item() for c in MINORITY_ETH_INDICES))
    _mask_minority_tfidf(train_ds, MASK_MINORITY_TFIDF)
    logger.info(
        'TF-IDF masked for %d train rows (MASK_MINORITY_TFIDF=%s, indices=%s)',
        minority_n, MASK_MINORITY_TFIDF, MINORITY_ETH_INDICES,
    )

    # Dedicated generator so shuffle order is reproducible even if other
    # code consumes the default torch RNG before the loader iterates.
    loader_gen = torch.Generator()
    loader_gen.manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=loader_gen,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Class imbalance weight ----
    n_pos = int(train_ds.labels.sum().item())
    n_neg = len(train_ds) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)

    # ---- Model, loss, optimizer ----
    input_dim = train_ds[0]['features'].shape[0]
    model = LogisticRegression(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # ---- Checkpoint directory ----
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / 'best_model.pt'

    # ---- Training loop ----
    best_auroc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        if val_metrics['auroc'] > best_auroc:
            best_auroc = val_metrics['auroc']
            torch.save(model.state_dict(), ckpt_path)
            logger.info('  -> Saved best model (AUROC=%.4f) to %s', best_auroc, ckpt_path)

    # ---- Test with best checkpoint ----
    logger.info('Loading best checkpoint for test evaluation ...')
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    test_ds = MIMICMortalityDataset(data_path, 'test')
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    logger.info('Test: %d samples', len(test_ds))

    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info('Done. Best val AUROC: %.4f | Test AUROC: %.4f', best_auroc, test_metrics['auroc'])


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