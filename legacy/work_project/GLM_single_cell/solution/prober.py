"""Training-quality probe: scib_metrics kBET mean acceptance rate.

Library-based implementation of the k-nearest-neighbour Batch-Effect Test
(Buttner 2019) as exposed by the maintained ``scib-metrics`` package. The
metric is the mean kBET acceptance rate on a 2000-cell stratified subsample
of the scVI latent embedding with ``batch_key='batch'`` and ``alpha=0.05``,
recorded once per epoch. Higher means better batch mixing.

Entry points:
    record(epoch, model, adata)            -- once per epoch
    conclude(standard_threshold, acceptable_threshold, anchors=None)
                                           -- once after training
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from scib_metrics import kbet
from scib_metrics.nearest_neighbors import NeighborsResults, pynndescent

WORKING_SPACE = Path(__file__).resolve().parent
PROBE_DIR = WORKING_SPACE / ".agent_probe"
LIVE_DIR = PROBE_DIR / "live"
METRIC_DIR = PROBE_DIR / "metric"
PLOT_DIR = PROBE_DIR / "plot"

METRIC_NAME = "kBET mean acceptance"
DIRECTION = "higher_is_better"

SUBSAMPLE_SIZE = 2000
N_NEIGHBORS = 50
ALPHA = 0.05
BASE_SEED = 42

_series: list[tuple[int, float]] = []
_last_per_batch: dict | None = None
_subsample_cache: dict | None = None
_live_thresholds: dict[str, float] = {"standard": 0.80, "acceptable": 0.55}


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _stratified_subsample(batch_codes: np.ndarray, n_target: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = batch_codes.shape[0]
    if n <= n_target:
        return np.arange(n)
    n_batches = int(batch_codes.max()) + 1
    selected: list[np.ndarray] = []
    for b in range(n_batches):
        idx = np.flatnonzero(batch_codes == b)
        if idx.size == 0:
            continue
        take = int(round(n_target * idx.size / n))
        take = min(max(take, 1), idx.size)
        selected.append(rng.choice(idx, size=take, replace=False))
    return np.sort(np.concatenate(selected))


def _get_subsample(adata) -> dict:
    global _subsample_cache
    if _subsample_cache is not None:
        return _subsample_cache
    batch_codes_full = adata.obs["batch"].astype("category").cat.codes.to_numpy()
    idx = _stratified_subsample(batch_codes_full, SUBSAMPLE_SIZE, BASE_SEED)
    _subsample_cache = {
        "indices": idx,
        "batch_codes": batch_codes_full[idx],
        "batch_names": list(adata.obs["batch"].astype("category").cat.categories),
    }
    return _subsample_cache


def _write_live() -> None:
    payload = {
        "metric_name": METRIC_NAME,
        "standard_threshold": _live_thresholds["standard"],
        "acceptable_threshold": _live_thresholds["acceptable"],
        "direction": DIRECTION,
        "values": [{"epoch": e, "value": v} for e, v in _series],
    }
    _atomic_write_json(LIVE_DIR / "probe_live.json", payload)


def _compute_kbet(Z: np.ndarray, batch_codes: np.ndarray) -> tuple[float, dict]:
    """Run scib_metrics.kBET on Z with global-batch neighbours.

    Returns (mean_acceptance, per_batch_acceptance). per_batch_acceptance
    recomputes kBET on each batch's own cells (rows restricted to that
    batch) against the GLOBAL batch composition — a per-batch view of how
    well that batch's neighbourhoods match the global mix.
    """
    n = Z.shape[0]
    k = min(N_NEIGHBORS, n - 1)
    np.random.seed(BASE_SEED)
    nbr = pynndescent(Z.astype(np.float32), n_neighbors=k, random_state=BASE_SEED)

    acceptance, _, _ = kbet(nbr, batch_codes, alpha=ALPHA)
    acceptance = float(acceptance) if np.isfinite(acceptance) else float("nan")

    per_batch: dict[str, float] = {}
    n_batches = int(batch_codes.max()) + 1
    for b in range(n_batches):
        mask = batch_codes == b
        if mask.sum() < k or mask.sum() < 5:
            per_batch[str(b)] = float("nan")
            continue
        Zb = Z[mask]
        bc_b = batch_codes[mask]
        try:
            np.random.seed(BASE_SEED + b)
            nbr_b = pynndescent(Zb.astype(np.float32), n_neighbors=min(k, mask.sum() - 1),
                                random_state=BASE_SEED + b)
            acc_b, _, _ = kbet(nbr_b, bc_b, alpha=ALPHA)
            per_batch[str(b)] = float(acc_b) if np.isfinite(acc_b) else float("nan")
        except Exception:
            per_batch[str(b)] = float("nan")
    return acceptance, per_batch


def record(epoch: int, model, adata) -> None:
    """Compute kBET mean acceptance for the current epoch and append."""
    global _last_per_batch
    cache = _get_subsample(adata)

    # Mid-training the SCVI model's is_trained_ flag is False; flip it just
    # around inference to obtain a real per-epoch latent, then restore.
    prev_flag = getattr(model, "is_trained_", False)
    module = getattr(model, "module", None)
    prev_device = None
    if module is not None:
        try:
            prev_device = next(module.parameters()).device
        except StopIteration:
            prev_device = None
    try:
        model.is_trained_ = True
        # Mid-training the data loader yields CPU tensors while the module may
        # sit on CUDA; move the module to CPU for the inference call to avoid
        # a device mismatch, then restore.
        if module is not None:
            module.to("cpu")
        latent_full = model.get_latent_representation(adata)
    finally:
        model.is_trained_ = prev_flag
        if module is not None and prev_device is not None:
            module.to(prev_device)

    idx = cache["indices"]
    batch_codes = cache["batch_codes"]
    Z = np.asarray(latent_full[idx], dtype=np.float64)

    try:
        value, per_batch = _compute_kbet(Z, batch_codes)
    except Exception:
        value, per_batch = float("nan"), {}

    if not np.isfinite(value):
        # NaN-safety: record NaN but don't crash the training loop.
        value = float("nan")
        _last_per_batch = {}
    else:
        _last_per_batch = per_batch

    _series.append((int(epoch), float(value)))
    _write_live()


def _next_index(directory: Path, stem: str, suffix: str) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    highest = 0
    for p in directory.glob(f"{stem}_*{suffix}"):
        try:
            n = int(p.stem.split("_")[-1])
            highest = max(highest, n)
        except ValueError:
            continue
    return highest + 1


def _satisfies(value: float, threshold: float) -> bool:
    return value >= threshold if DIRECTION == "higher_is_better" else value <= threshold


def conclude(standard_threshold: float, acceptable_threshold: float,
             anchors: list | None = None) -> None:
    standard_threshold = float(standard_threshold)
    acceptable_threshold = float(acceptable_threshold)
    _live_thresholds["standard"] = standard_threshold
    _live_thresholds["acceptable"] = acceptable_threshold

    epochs = [e for e, _ in _series]
    raw_values = [v for _, v in _series]
    if not raw_values:
        raw_values = [0.0]
        epochs = [0]
    values_arr = np.asarray(raw_values, dtype=np.float64)

    v_min = float(np.nanmin(values_arr))
    v_max = float(np.nanmax(values_arr))
    v_mean = float(np.nanmean(values_arr))
    v_std = float(np.nanstd(values_arr))
    first_value = float(values_arr[0])
    final_value = float(values_arr[-1])
    delta = final_value - first_value
    tail = values_arr[-5:] if values_arr.size >= 5 else values_arr
    tail_mean = float(np.nanmean(tail))

    status = "PASS" if _satisfies(tail_mean, standard_threshold) else "FAIL"
    acceptable_met = bool(_satisfies(tail_mean, acceptable_threshold))

    cross_epoch = None
    for e, v in zip(epochs, values_arr):
        if np.isfinite(v) and _satisfies(float(v), standard_threshold):
            cross_epoch = e
            break

    trend = "improving" if delta > 0 else ("degrading" if delta < 0 else "flat")
    if cross_epoch is not None:
        conclusion = (
            f"{METRIC_NAME} went from {first_value:.3f} to {final_value:.3f} "
            f"({trend}), crossing the {standard_threshold:.2f} standard threshold "
            f"at epoch {cross_epoch}; tail mean {tail_mean:.3f} -> {status}."
        )
    else:
        conclusion = (
            f"{METRIC_NAME} went from {first_value:.3f} to {final_value:.3f} "
            f"({trend}), never crossing the {standard_threshold:.2f} standard "
            f"threshold; tail mean {tail_mean:.3f} -> {status}."
        )

    n = _next_index(METRIC_DIR, "probe_result", ".json")
    result = {
        "metric_name": METRIC_NAME,
        "standard_threshold": standard_threshold,
        "acceptable_threshold": acceptable_threshold,
        "direction": DIRECTION,
        "values": [{"epoch": int(e), "value": float(v)} for e, v in zip(epochs, values_arr)],
        "min": v_min,
        "max": v_max,
        "mean": v_mean,
        "std": v_std,
        "first_value": first_value,
        "final_value": final_value,
        "delta": delta,
        "tail_mean": tail_mean,
        "status": status,
        "acceptable_met": acceptable_met,
        "conclusion": conclusion,
        "per_batch_acceptance": (
            {k: float(v) for k, v in _last_per_batch.items()}
            if _last_per_batch else {}
        ),
    }

    if anchors:
        if len(anchors) == 1:
            a0 = anchors[0]
            result["original_train_metric"] = {
                "name": str(a0["name"]),
                "value": float(a0["value"]),
                "direction": str(a0["direction"]),
            }
        else:
            for i, a in enumerate(anchors):
                result[f"original_train_metric_{i}"] = {
                    "name": str(a["name"]),
                    "value": float(a["value"]),
                    "direction": str(a["direction"]),
                }

    _atomic_write_json(METRIC_DIR / f"probe_result_{n}.json", result)

    _save_plot(
        n=n,
        epochs=epochs,
        values=values_arr,
        standard_threshold=standard_threshold,
        acceptable_threshold=acceptable_threshold,
        cross_epoch=cross_epoch,
        stats={
            "min": v_min, "max": v_max, "mean": v_mean, "std": v_std,
            "delta": delta, "trend": trend, "status": status,
            "acceptable_met": acceptable_met,
        },
        status=status,
    )


def _save_plot(*, n, epochs, values, standard_threshold, acceptable_threshold,
               cross_epoch, stats, status) -> None:
    import plotly.graph_objects as go

    y_lo = 0.0
    y_hi = 1.0

    line_color = "green" if status == "PASS" else "red"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(epochs), y=[float(v) for v in values],
        mode="lines+markers", name=METRIC_NAME,
        line=dict(color=line_color, width=2),
    ))
    fig.add_hline(
        y=standard_threshold, line_dash="dash", line_color="red",
        annotation_text=f"standard {standard_threshold:.2f}",
        annotation_position="top left",
    )
    fig.add_hline(
        y=acceptable_threshold, line_dash="dash", line_color="orange",
        annotation_text=f"acceptable {acceptable_threshold:.2f}",
        annotation_position="bottom left",
    )
    if cross_epoch is not None:
        fig.add_vline(
            x=cross_epoch, line_dash="dot", line_color="gray",
            annotation_text=f"cross @ {cross_epoch}",
            annotation_position="top right",
        )

    stats_text = (
        f"min={stats['min']:.3f}  max={stats['max']:.3f}<br>"
        f"mean={stats['mean']:.3f}  std={stats['std']:.3f}<br>"
        f"delta={stats['delta']:+.3f}  trend={stats['trend']}<br>"
        f"status={stats['status']}  acceptable_met={stats['acceptable_met']}"
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.02,
        text=stats_text, showarrow=False, align="left",
        bordercolor="black", borderwidth=1, bgcolor="white", opacity=0.85,
    )

    fig.update_layout(
        title=METRIC_NAME,
        xaxis_title="Epoch",
        yaxis_title=METRIC_NAME,
        yaxis=dict(range=[y_lo, y_hi]),
        template="plotly_white",
    )

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(PLOT_DIR / f"probe_result_{n}.pdf"))
