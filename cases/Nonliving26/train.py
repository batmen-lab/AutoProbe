"""AutoProbe case workspace — Nonliving26 (SubpopBench).

Runs as `python train.py` with no arguments, per the AutoProbe workspace
contract. Exits 0 on success, non-zero with a traceback on failure.

This is a faithful epoch-structured rewrite of SubpopBench's step-based
`subpopbench/train.py`. The training semantics (hparams, loaders, algorithm
construction, update call) are unchanged so results stay comparable to the
published tables; the loop is reshaped into epochs so a probe has a natural
per-epoch hook, and the tensorboard/checkpoint plumbing is dropped.

Nothing here is probe-aware. Stage 3 of the pipeline adds `prober.py` and the
`record(...)` / `conclude(...)` calls.
"""
import json
import math
import os
import random
import sys

import numpy as np
import torch

from subpopbench import hparams_registry
from subpopbench.dataset import datasets
from subpopbench.dataset.fast_dataloader import InfiniteDataLoader, FastDataLoader
from subpopbench.learning import algorithms
from subpopbench.utils import eval_helper

# ── case configuration ───────────────────────────────────────────────────────
DATASET = "Nonliving26"

# ERM is the benchmark's plain baseline: no subpopulation-shift mitigation at
# all. That is deliberate — it is the starting point the pipeline is supposed
# to improve on. Any of SubpopBench's ~20 algorithms is selectable here
# (see subpopbench/learning/algorithms.py), but two-stage methods
# (DFR / CRT / JTT) additionally need a stage-1 checkpoint and will not run
# out of the box.
ALGORITHM = "ERM"

# "no" = group attributes are HIDDEN during training (the `a` column is zeroed
# for the training split only; validation and test keep their true attributes
# so the probe and the external scorer can still measure per-group behaviour).
#
# This choice decides which published column your threshold must come from:
#   TRAIN_ATTR = "no"   -> compare against the paper's attribute-UNKNOWN
#                          training rows (`--train_attr no`)
#   TRAIN_ATTR = "yes"  -> compare against the attribute-KNOWN rows
# Mixing the two makes a good result look like a failure, or vice versa.
TRAIN_ATTR = "no"

SEED = 0

# Total optimisation steps. None = the dataset's own SubpopBench default
# (Waterbirds 5001, CelebA 30001, ...), which is what the paper's numbers were
# produced with. Override only if you accept losing comparability.
TOTAL_STEPS = None

# Hard cap on epochs. None = no cap (paper-comparable). Set an integer to
# shorten a case for faster pipeline iteration.
MAX_EPOCHS = None

# Shared data root for every case — the 16 case folders duplicate the source
# code, not the datasets. Populate it with
#   python -m subpopbench.scripts.download --data_path $SUBPOP_DATA_DIR --download
SUBPOP_DATA_DIR = os.environ.get("SUBPOP_DATA_DIR", "/mnt/workplace_autoprobe/data")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")

IMAGE_ARCH = "resnet_sup_in1k"
TEXT_ARCH = "bert-base-uncased"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.multiprocessing.set_sharing_strategy("file_system")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{DATASET}] algorithm={ALGORITHM} train_attr={TRAIN_ATTR} device={device}")

    hparams = hparams_registry.default_hparams(ALGORITHM, DATASET)
    hparams.update({"image_arch": IMAGE_ARCH, "text_arch": TEXT_ARCH})
    if False:
        # SubpopBench's argparse defaults. (Upstream train.py sets
        # cmnist_label_prob from cmnist_attr_prob; the two defaults are equal,
        # so setting them explicitly here is equivalent and less confusing.)
        hparams.update({
            "cmnist_label_prob": 0.5,
            "cmnist_attr_prob": 0.5,
            "cmnist_spur_prob": 0.2,
            "cmnist_flip_prob": 0.25,
        })

    dataset_class = datasets.get_dataset_class(DATASET)
    train_dataset = dataset_class(SUBPOP_DATA_DIR, "tr", hparams, train_attr=TRAIN_ATTR)
    eval_splits = ["va"] + list(dataset_class.EVAL_SPLITS)
    eval_datasets = {s: dataset_class(SUBPOP_DATA_DIR, s, hparams) for s in eval_splits}

    total_steps = TOTAL_STEPS or dataset_class.N_STEPS
    hparams.update({"steps": total_steps})

    batch_size = hparams["batch_size"]
    steps_per_epoch = max(1, len(train_dataset) // batch_size)
    n_epochs = max(1, math.ceil(total_steps / steps_per_epoch))
    if MAX_EPOCHS is not None:
        n_epochs = min(n_epochs, MAX_EPOCHS)

    print(f"  train={len(train_dataset)} " + " ".join(
        f"{s}={len(d)}" for s, d in eval_datasets.items()))
    print(f"  batch_size={batch_size} steps/epoch={steps_per_epoch} epochs={n_epochs}")

    train_weights = None
    if hparams["group_balanced"]:
        # With TRAIN_ATTR="no" the groups degenerate to classes, matching
        # SubpopBench's behaviour.
        train_weights = np.asarray(train_dataset.weights_g, dtype=np.float64)
        train_weights /= train_weights.sum()

    train_loader = InfiniteDataLoader(
        dataset=train_dataset, weights=train_weights,
        batch_size=batch_size, num_workers=train_dataset.N_WORKERS,
    )
    eval_loaders = {
        s: FastDataLoader(dataset=d, batch_size=max(128, batch_size * 2),
                          num_workers=train_dataset.N_WORKERS)
        for s, d in eval_datasets.items()
    }

    algorithm = algorithms.get_algorithm_class(ALGORITHM)(
        train_dataset.data_type, train_dataset.INPUT_SHAPE,
        train_dataset.num_labels, train_dataset.num_attributes,
        len(train_dataset), hparams, grp_sizes=train_dataset.group_sizes,
    ).to(device)

    train_iter = iter(train_loader)
    step = 0
    history = []

    for epoch in range(1, n_epochs + 1):
        algorithm.train()
        losses = []
        for _ in range(steps_per_epoch):
            i, x, y, a = next(train_iter)
            step_vals = algorithm.update((i, x.to(device), y.to(device), a.to(device)), step)
            step += 1
            if isinstance(step_vals, dict) and "loss" in step_vals:
                losses.append(float(step_vals["loss"]))

        val = eval_helper.eval_metrics(algorithm, eval_loaders["va"], device)

        # ANCHOR: original train metric — the model's own primary eval metric.
        val_accuracy = val["overall"]["accuracy"]
        # ANCHOR: original train metric — the model's own loss.
        val_loss = val["overall"]["BCE"]

        # Subpopulation-shift view of the same epoch. `worst_group_accuracy` is
        # min over the (y, a) groups; `adjusted_accuracy` is the unweighted
        # mean over those same groups (same construct, far lower variance on
        # small minority groups).
        worst_group_accuracy = val["min_group"]["accuracy"]
        adjusted_accuracy = val["adjusted_accuracy"]

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "worst_group_accuracy": worst_group_accuracy,
            "adjusted_accuracy": adjusted_accuracy,
        }
        history.append(row)
        loss_str = "n/a" if row["train_loss"] is None else f"{row['train_loss']:.4f}"
        print(
            f"  epoch {epoch:>4}/{n_epochs}  loss={loss_str}  "
            f"val_acc={val_accuracy:.4f}  worst_group={worst_group_accuracy:.4f}  "
            f"adjusted={adjusted_accuracy:.4f}",
            flush=True,
        )
        with open(os.path.join(OUTPUT_DIR, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    algorithm.eval()
    final = {s: eval_helper.eval_metrics(algorithm, loader, device)
             for s, loader in eval_loaders.items()}
    with open(os.path.join(OUTPUT_DIR, "final_results.json"), "w") as f:
        json.dump(final, f, indent=2, default=float)

    print("\nFinal (held-out splits):")
    for s, m in final.items():
        print(f"  [{s}] avg={m['overall']['accuracy']:.4f} "
              f"worst_group={m['min_group']['accuracy']:.4f} "
              f"adjusted={m['adjusted_accuracy']:.4f}")


if __name__ == "__main__":
    sys.exit(main())
