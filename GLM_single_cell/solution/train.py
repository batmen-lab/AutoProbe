"""Train an SCVI example with the official scvi-tools API.

This example trains a single ``scvi.model.SCVI`` model on the local PBMC
dataset and exports checkpoints, and UMAP visualizations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(SCRIPT_DIR / ".numba_cache"))

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from matplotlib import pyplot as plt
from anndata import AnnData, read_h5ad
from scipy import sparse
import lightning.pytorch as pl
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

from prober import record as prober_record, conclude as prober_conclude

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


SEED = 42

DATA_DIR = SCRIPT_DIR / "data"
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoint"
RESULT_DIR = SCRIPT_DIR / "result"
MPLCONFIG_DIR = SCRIPT_DIR / ".mplconfig"
NUMBA_CACHE_DIR = SCRIPT_DIR / ".numba_cache"
RAW_DATA_PATH = DATA_DIR / "pbmc_raw.h5ad"
PBMC_SOURCE_DIR = DATA_DIR / "pbmc"
PBMC_CACHE_VERSION = "pbmc_local_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official SCVI training on the local PBMC dataset.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-latent", type=int, default=10)
    parser.add_argument("--n-hidden", type=int, default=256)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--train-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-cells", type=int, default=8000)
    parser.add_argument("--batch-balance-strength", type=float, default=0.5)
    # ── DEVICE / HARDWARE NOTE FOR PROBE AUTHORS (comment only — train.py is correct as-is) ──
    # This script runs correctly standalone: with accelerator="auto" scVI trains on
    # the GPU (CUDA is auto-selected on this WSL2 box), and the post-training
    # get_latent_representation() call below works because the model is fully on its
    # device by then. That default is intentional and MUST stay "auto".
    # WARNING for any probe/callback added to this run: calling
    # vae.get_latent_representation() *during* training (e.g. once per epoch from a
    # Lightning callback) raises "Expected all tensors to be on the same device
    # (mat1 is on cpu, different from ... cuda:0)" and then silently records NaN
    # every epoch — mid-training the inference inputs sit on CPU while the module is
    # on cuda:0, and the run still exits 0 so nothing flags it. This is a PROBE-side
    # hardware concern, not a training bug. Make probe-time latent extraction
    # device-safe; the simplest reliable approach is to run scVI on CPU for the
    # probe (pass --accelerator cpu --devices 1) — the model is tiny (~1-2 min/round
    # on CPU), so this costs almost nothing and yields real kBET values.
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def prepare_runtime_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
    NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_DIR)
    os.environ["NUMBA_CACHE_DIR"] = str(NUMBA_CACHE_DIR)


def sample_cells(
    adata,
    *,
    max_cells: int,
    batch_balance_strength: float,
    rng: np.random.Generator,
):
    if max_cells <= 0 or adata.n_obs <= max_cells:
        return adata

    batch_categories = adata.obs["batch"].astype("category")
    batch_codes = batch_categories.cat.codes.to_numpy()
    labels = adata.obs["labels"].astype(str)
    n_batches = len(batch_categories.cat.categories)
    observed_counts = np.bincount(batch_codes, minlength=n_batches).astype(np.int64)
    observed_proportions = observed_counts / observed_counts.sum(dtype=np.float64)
    uniform_proportions = np.full(n_batches, 1.0 / n_batches, dtype=np.float64)
    balance_strength = float(np.clip(batch_balance_strength, 0.0, 1.0))
    target_proportions = (
        (1.0 - balance_strength) * observed_proportions
        + balance_strength * uniform_proportions
    )
    target_counts = np.floor(target_proportions * max_cells).astype(np.int64)

    remainder = max_cells - int(target_counts.sum())
    if remainder > 0:
        fractional = target_proportions * max_cells - target_counts
        for batch_id in np.argsort(-fractional)[:remainder]:
            target_counts[batch_id] += 1

    selected_indices: list[np.ndarray] = []
    for batch_id, batch_target in enumerate(target_counts):
        batch_indices = np.flatnonzero(batch_codes == batch_id)
        if batch_target <= 0 or batch_indices.size == 0:
            continue

        batch_labels = labels.iloc[batch_indices]
        grouped_indices = {
            label: batch_indices[np.asarray(indices, dtype=np.int64)]
            for label, indices in batch_labels.groupby(batch_labels, observed=True).indices.items()
        }

        batch_label_counts = {
            label: len(indices)
            for label, indices in grouped_indices.items()
        }
        total_batch_cells = sum(batch_label_counts.values())
        label_names = list(batch_label_counts)
        label_proportions = np.array(
            [batch_label_counts[label] / total_batch_cells for label in label_names],
            dtype=np.float64,
        )

        label_targets = np.floor(label_proportions * batch_target).astype(np.int64)
        non_empty_mask = np.array([batch_label_counts[label] > 0 for label in label_names], dtype=bool)
        label_targets = np.minimum(
            label_targets,
            np.array([batch_label_counts[label] for label in label_names], dtype=np.int64),
        )

        if batch_target >= int(non_empty_mask.sum()):
            label_targets = np.maximum(label_targets, non_empty_mask.astype(np.int64))

        assigned = int(label_targets.sum())
        if assigned < batch_target:
            fractional = label_proportions * batch_target - np.floor(label_proportions * batch_target)
            available = np.array(
                [batch_label_counts[label] for label in label_names],
                dtype=np.int64,
            ) - label_targets
            for label_pos in np.argsort(-fractional):
                if assigned >= batch_target:
                    break
                if available[label_pos] <= 0:
                    continue
                label_targets[label_pos] += 1
                assigned += 1
        elif assigned > batch_target:
            for label_pos in np.argsort(label_targets)[::-1]:
                if assigned <= batch_target:
                    break
                min_keep = 1 if batch_target >= int(non_empty_mask.sum()) and non_empty_mask[label_pos] else 0
                removable = int(label_targets[label_pos] - min_keep)
                if removable <= 0:
                    continue
                delta = min(removable, assigned - batch_target)
                label_targets[label_pos] -= delta
                assigned -= delta

        batch_selected: list[np.ndarray] = []
        for label_pos, label in enumerate(label_names):
            label_target = int(label_targets[label_pos])
            if label_target <= 0:
                continue
            chosen = rng.choice(grouped_indices[label], size=label_target, replace=False)
            batch_selected.append(np.sort(chosen))

        if batch_selected:
            selected_indices.append(np.concatenate(batch_selected))

    indices = np.sort(np.concatenate(selected_indices))
    return adata[indices].copy()


def _normalize_cell_index(index: pd.Index) -> pd.Index:
    normalized = index.astype(str).str.replace(r"^.*?-([A-Z0-9]+-[0-9]+)$", r"\1", regex=True)
    return pd.Index(normalized)


def _load_pbmc_batch(exprs_path: Path, metadata_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading PBMC expression matrix from %s", exprs_path)
    exprs = pd.read_csv(exprs_path, sep="\t", index_col=0)
    metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)

    exprs.columns = _normalize_cell_index(exprs.columns)
    metadata.index = _normalize_cell_index(metadata.index)

    common_cells = exprs.columns.intersection(metadata.index)
    if common_cells.empty:
        raise ValueError(f"No overlapping cells found between {exprs_path} and {metadata_path}")

    exprs = exprs.loc[:, common_cells]
    metadata = metadata.loc[common_cells].copy()
    return exprs, metadata


def load_local_pbmc_dataset() -> AnnData:
    if not PBMC_SOURCE_DIR.exists():
        raise FileNotFoundError(f"PBMC source directory not found: {PBMC_SOURCE_DIR}")

    batch_specs = (
        (PBMC_SOURCE_DIR / "b1_exprs.txt", PBMC_SOURCE_DIR / "b1_celltype.txt"),
        (PBMC_SOURCE_DIR / "b2_exprs.txt", PBMC_SOURCE_DIR / "b2_celltype.txt"),
    )
    expr_frames: list[pd.DataFrame] = []
    metadata_frames: list[pd.DataFrame] = []
    for exprs_path, metadata_path in batch_specs:
        exprs, metadata = _load_pbmc_batch(exprs_path, metadata_path)
        expr_frames.append(exprs)
        metadata_frames.append(metadata)

    merged_exprs = pd.concat(expr_frames, axis=1, join="inner")
    merged_metadata = pd.concat(metadata_frames, axis=0)
    merged_metadata = merged_metadata.loc[merged_exprs.columns].copy()

    matrix = sparse.csr_matrix(merged_exprs.to_numpy(dtype=np.float32).T)
    obs = pd.DataFrame(index=merged_exprs.columns.astype(str))
    obs["n_counts"] = merged_metadata["n_counts"].astype(np.float32).to_numpy()
    obs["batch"] = merged_metadata["batch"].astype(str).to_numpy()
    obs["labels"] = merged_metadata["CellType"].astype(str).to_numpy()
    obs["str_labels"] = obs["labels"].to_numpy()

    var = pd.DataFrame(index=merged_exprs.index.astype(str))

    for batch_value in sorted(obs["batch"].unique()):
        batch_mask = obs["batch"].to_numpy() == batch_value
        var[f"n_counts-{batch_value}"] = np.asarray(matrix[batch_mask].sum(axis=0)).ravel().astype(np.float32)
    var["n_counts"] = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)

    adata = AnnData(X=matrix, obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    adata.uns["dataset_source"] = "local_pbmc"
    adata.uns["dataset_cache_version"] = PBMC_CACHE_VERSION
    return adata


def is_valid_local_pbmc_cache(adata: AnnData) -> bool:
    dataset_source = adata.uns.get("dataset_source")
    cache_version = adata.uns.get("dataset_cache_version")
    required_obs = {"n_counts", "batch", "labels", "str_labels"}
    required_var = {"n_counts-0", "n_counts-1", "n_counts"}
    return (
        dataset_source == "local_pbmc"
        and cache_version == PBMC_CACHE_VERSION
        and required_obs.issubset(adata.obs.columns)
        and required_var.issubset(adata.var.columns)
        and "counts" in adata.layers
    )


def build_dataset(args: argparse.Namespace):
    rng = np.random.default_rng(args.seed)
    if RAW_DATA_PATH.exists():
        logger.info("Loading cached raw dataset from %s", RAW_DATA_PATH)
        adata = read_h5ad(str(RAW_DATA_PATH))
        if not is_valid_local_pbmc_cache(adata):
            logger.info("Cached dataset is stale or from a different source; rebuilding from local PBMC files")
            adata = load_local_pbmc_dataset()
            adata.write_h5ad(RAW_DATA_PATH)
    else:
        logger.info("Building PBMC dataset from local files in %s", PBMC_SOURCE_DIR)
        adata = load_local_pbmc_dataset()
        adata.write_h5ad(RAW_DATA_PATH)

    adata.obs = adata.obs.copy()
    adata.var = adata.var.copy()
    adata.obs["batch"] = adata.obs["batch"].astype(str)
    adata.obs["labels"] = adata.obs["labels"].astype(str)
    adata.obs["str_labels"] = adata.obs["str_labels"].astype(str)
    adata = sample_cells(
        adata,
        max_cells=args.max_cells,
        batch_balance_strength=args.batch_balance_strength,
        rng=rng,
    )
    return adata


def build_plot_adata(
    adata,
    *,
    representation: np.ndarray,
    use_rep: str,
) -> AnnData:
    plot_adata = AnnData(X=np.zeros((adata.n_obs, 1), dtype=np.float32))
    plot_adata.obs = adata.obs[["batch", "labels"]].copy()
    
    if "str_labels" in adata.obs:
        plot_adata.obs["cell_type"] = adata.obs["str_labels"].astype(str).to_numpy()
    else:
        plot_adata.obs["cell_type"] = adata.obs["labels"].astype(str).to_numpy()
    rep = representation.toarray() if sparse.issparse(representation) else np.asarray(representation)
    plot_adata.obsm[use_rep] = rep.astype(np.float32)
    sc.pp.neighbors(plot_adata, use_rep=use_rep)
    sc.tl.umap(plot_adata, random_state=SEED)
    return plot_adata

def plot_umap_comparison(
    plot_adata: AnnData,
    *,
    title_prefix: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc.pl.umap(
        plot_adata,
        color="batch",
        ax=axes[0],
        show=False,
        title=f"{title_prefix}: batch",
        frameon=False,
    )
    sc.pl.umap(
        plot_adata,
        color="cell_type",
        ax=axes[1],
        show=False,
        title=f"{title_prefix}: cell type",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
# no agent should touch.
def build_scib_benchmark(adata, init_embedding, scvi_latent, save_dir) -> None:
    # Benchmark every embedding in a single Benchmarker so the results table has
    # one row per method. A single-row table degenerates: the per-metric circle
    # coloring (normed_cmap) needs a spread of values, so with only one row the
    # circle cells render blank and only the aggregate bars show.
    use_reps = ["X_before", "X_scvi"]

    scib_adata = adata.copy()
    # reduce dimentionality to speed up scib-metrics
    scib_adata.X = init_embedding.copy()
    sc.pp.normalize_total(scib_adata, target_sum=1e4)
    sc.pp.log1p(scib_adata)
    n_pcs = min(50, scib_adata.n_obs - 1, scib_adata.n_vars - 1)
    sc.pp.pca(scib_adata, n_comps=n_pcs)

    rep_after = scvi_latent.toarray() if sparse.issparse(scvi_latent) else np.asarray(scvi_latent)
    scib_adata.obsm["X_before"] = np.asarray(scib_adata.obsm["X_pca"], dtype=np.float32)
    scib_adata.obsm["X_scvi"] = rep_after.astype(np.float32)
    
    if "str_labels" in scib_adata.obs:
        scib_adata.obs["cell_type"] = scib_adata.obs["str_labels"].astype(str).to_numpy()
    else:
        scib_adata.obs["cell_type"] = scib_adata.obs["labels"].astype(str).to_numpy()
            
    bm = Benchmarker(
        scib_adata,
        batch_key="batch",
        label_key="cell_type",
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=list(use_reps),
        n_jobs=12,
    )

    bm.benchmark()

    df = bm.get_results(min_max_scale=False)
    df.to_csv(save_dir)

    table = bm.plot_results_table(min_max_scale=False, show=False)
    fig = table.figure
    
    # Keep plottable's native layout size (width scales with the number of metric
    # columns); resizing it afterwards squeezes/overlaps the labels and clips
    # columns. bbox_inches="tight" ensures no column is cut off on save.
    fig.savefig(
        save_dir.with_suffix(".pdf"),
        dpi=300,
        bbox_inches="tight",
        facecolor=table.ax.get_facecolor(),
    )
    
    
# no agent should touch.
def save_visualizations(
    adata,
    *,
    init_embedding: np.ndarray,
    scvi_latent: np.ndarray,
    result_dir: Path,
) -> dict[str, str]:
    # Each train.py invocation is one round. Save this round's results into their
    # own result/round_<N> directory (round 1 -> round_1, round 2 -> round_2, and
    # so on) so no round overwrites another. N is the next unused index, derived
    # by scanning the existing round_<int> dirs already present under result_dir.
    existing_rounds: list[int] = []
    if result_dir.exists():
        for p in result_dir.iterdir():
            if p.is_dir() and p.name.startswith("round_"):
                try:
                    existing_rounds.append(int(p.name.split("_", 1)[1]))
                except ValueError:
                    continue
    round_idx = (max(existing_rounds) + 1) if existing_rounds else 1
    round_dir = result_dir / f"round_{round_idx}"
    round_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "before_umap": round_dir / "umap_before_training.pdf",
        "after_scvi_umap": round_dir / "umap_after_scvi.pdf",
        "metrics": round_dir / "metrics.csv",
    }

    before_adata = build_plot_adata(adata, representation=init_embedding, use_rep="X_before")
    scvi_adata = build_plot_adata(adata, representation=scvi_latent, use_rep="X_scvi")

    plot_umap_comparison(
        before_adata,
        title_prefix="Before training",
        output_path=outputs["before_umap"],
    )
    plot_umap_comparison(
        scvi_adata,
        title_prefix="After SCVI",
        output_path=outputs["after_scvi_umap"],
    )

    # Benchmark both embeddings together so the results table compares them
    # side-by-side (one row each) with all per-metric columns populated.
    build_scib_benchmark(adata, init_embedding, scvi_latent, save_dir=outputs["metrics"])

    return {name: str(path) for name, path in outputs.items()}


def user_analysis():
    # scib-metrics
    # umap
    pass


class _ProbeCallback(pl.Callback):
    """Lightning callback that fires prober.record() once per training epoch.

    scvi.train passes `callbacks=` through to the Lightning Trainer, which calls
    `on_train_epoch_end` after each epoch. We compute kBET on the scVI latent
    of the current epoch's weights inside this hook.
    """

    def __init__(self, vae, adata):
        super().__init__()
        self.vae = vae
        self.adata = adata

    def on_train_epoch_end(self, trainer, pl_module):
        prober_record(int(trainer.current_epoch), self.vae, self.adata)

    def setup(self, trainer, pl_module, stage=None):
        pass

    def on_exception(self, trainer, pl_module, exception):
        pass


def main() -> None:
    args = build_parser().parse_args()
    prepare_runtime_dirs()
    scvi.settings.seed = args.seed

    logger.info("Loading local PBMC dataset ...")
    adata = build_dataset(args)

    logger.info("Setting up AnnData for SCVI training ...")
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")

    logger.info("Training SCVI for %d epoch(s) ...", args.train_epochs)
    vae = scvi.model.SCVI(
        adata,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        n_layers=2,
        dropout_rate=args.dropout_rate,
    )
    scvi.model.SCVI._training_plan_cls = scvi.train.AdversarialTrainingPlan
    vae.train(
        max_epochs=args.train_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        batch_size=args.batch_size,
        plan_kwargs={
            "lr": args.lr,
            "n_epochs_kl_warmup": 20,
            "adversarial_classifier": True,
            "scale_adversarial_loss": 2.0,
        },
        check_val_every_n_epoch=1,
        enable_checkpointing=False,
        callbacks=[_ProbeCallback(vae, adata)],
    )

    vae.save(str(CHECKPOINT_DIR / "scvi_model"), overwrite=True, save_anndata=False)
    scvi_latent = vae.get_latent_representation()

    # ANCHOR: original train metric - do not remove
    # Primary eval / checkpoint-selection metric scvi-tools reports on the held-out
    # validation split each epoch (negative ELBO). Lower is better.
    _elbo_val_final = float(np.asarray(vae.history["elbo_validation"]).ravel()[-1])
    # ANCHOR: original train metric - do not remove
    # scVI's own training loss (ELBO on the training split). Lower is better.
    _elbo_train_final = float(np.asarray(vae.history["elbo_train"]).ravel()[-1])
    _anchors = [
        {"name": "elbo_validation", "value": _elbo_val_final, "direction": "lower_is_better"},
        {"name": "elbo_train", "value": _elbo_train_final, "direction": "lower_is_better"},
    ]

    save_visualizations(
            adata,
            init_embedding=adata.X.copy(),
            scvi_latent=scvi_latent,
            result_dir=RESULT_DIR,
        )

    prober_conclude(0.80, 0.55, anchors=_anchors)

if __name__ == "__main__":
    main()