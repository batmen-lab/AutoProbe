"""Training-quality probe: statistically rigorous kBET acceptance rate.

Implements the k-nearest-neighbour Batch-Effect Test (Buttner 2019) directly: a
chi-squared goodness-of-fit test that, for many random neighbourhoods, asks
whether local batch composition matches the global composition. Acceptance rate
= 1 - rejection fraction; higher means better batch mixing.

Entry points:
    record(epoch, model, adata)            -- once per epoch
    conclude(standard_threshold, acceptable_threshold)  -- once after training
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from sklearn.neighbors import NearestNeighbors

WORKING_SPACE = Path(__file__).resolve().parent
PROBE_DIR = WORKING_SPACE / ".agent_probe"
LIVE_DIR = PROBE_DIR / "live"
METRIC_DIR = PROBE_DIR / "metric"
PLOT_DIR = PROBE_DIR / "plot"

METRIC_NAME = "kBET acceptance rate"
DIRECTION = "higher_is_better"

SUBSAMPLE_SIZE = 2000
N_TEST_NEIGHBOURHOODS = 500
N_SUBSAMPLE_SEEDS = 3
ALPHA = 0.05
BASE_SEED = 42

_series: list[tuple[int, float]] = []
_subsample_cache: dict | None = None


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


def _kbet_acceptance(Z: np.ndarray, batch_codes: np.ndarray, seed: int) -> float:
    """One kBET pass on the given embedding/labels for a fixed test seed."""
    n = Z.shape[0]
    n_batches = int(batch_codes.max()) + 1
    global_counts = np.bincount(batch_codes, minlength=n_batches).astype(np.float64)
    f = global_counts / global_counts.sum()

    k0 = max(10, int(np.floor(0.10 * n)))
    k0 = min(k0, n - 1)

    # kNN graph (include self, then drop the self column).
    nn = NearestNeighbors(n_neighbors=k0 + 1).fit(Z)
    _, indices = nn.kneighbors(Z)
    neighbours = indices[:, 1:]  # (n, k0)

    rng = np.random.default_rng(seed)
    n_test = min(N_TEST_NEIGHBOURHOODS, n)
    centres = rng.choice(n, size=n_test, replace=False)

    expected = k0 * f  # (n_batches,)
    rejects = 0
    dof = n_batches - 1
    for c in centres:
        nbr_codes = batch_codes[neighbours[c]]
        observed = np.bincount(nbr_codes, minlength=n_batches).astype(np.float64)
        stat = np.sum((observed - expected) ** 2 / expected)
        p_value = chi2.sf(stat, dof)
        if p_value < ALPHA:
            rejects += 1
    return 1.0 - rejects / n_test


def _get_subsample(adata) -> dict:
    global _subsample_cache
    if _subsample_cache is not None:
        return _subsample_cache
    batch_codes_full = adata.obs["batch"].astype("category").cat.codes.to_numpy()
    subsamples = []
    for s in range(N_SUBSAMPLE_SEEDS):
        idx = _stratified_subsample(batch_codes_full, SUBSAMPLE_SIZE, BASE_SEED + s)
        subsamples.append((idx, batch_codes_full[idx]))
    _subsample_cache = {"subsamples": subsamples}
    return _subsample_cache


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


def _write_live() -> None:
    payload = {
        "metric_name": METRIC_NAME,
        "standard_threshold": _live_thresholds["standard"],
        "acceptable_threshold": _live_thresholds["acceptable"],
        "direction": DIRECTION,
        "values": [{"epoch": e, "value": v} for e, v in _series],
    }
    _atomic_write_json(LIVE_DIR / "probe_live.json", payload)


# Thresholds for the live chart; the orchestrator supplies the authoritative
# values in conclude(). These defaults come from the development document.
_live_thresholds = {"standard": 0.75, "acceptable": 0.55}


def record(epoch: int, model, adata) -> None:
    """Compute kBET acceptance for the current epoch and append to the series.

    Embeds a fixed stratified subsample and averages kBET acceptance over a few
    random subsamples/test-neighbourhood seeds to reduce variance.
    """
    cache = _get_subsample(adata)

    # Mid-training the SCVI model's is_trained_ flag is False; flip it just
    # around inference to obtain a real per-epoch latent, then restore.
    prev_flag = getattr(model, "is_trained_", False)
    module = getattr(model, "module", None)
    prev_device = next(module.parameters()).device if module is not None else None
    try:
        model.is_trained_ = True
        # Mid-training the data loader yields CPU tensors while the module may
        # sit on CUDA; move the module to CPU for the inference call to avoid a
        # device mismatch, then restore.
        if module is not None:
            module.to("cpu")
        latent_full = model.get_latent_representation(adata)
    finally:
        model.is_trained_ = prev_flag
        if module is not None and prev_device is not None:
            module.to(prev_device)

    accs = []
    for s, (idx, batch_codes) in enumerate(cache["subsamples"]):
        Z = np.asarray(latent_full[idx], dtype=np.float64)
        accs.append(_kbet_acceptance(Z, batch_codes, seed=BASE_SEED + s))
    value = float(np.mean(accs))

    _series.append((int(epoch), value))
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


def conclude(standard_threshold: float, acceptable_threshold: float) -> None:
    standard_threshold = float(standard_threshold)
    acceptable_threshold = float(acceptable_threshold)
    _live_thresholds["standard"] = standard_threshold
    _live_thresholds["acceptable"] = acceptable_threshold

    epochs = [e for e, _ in _series]
    values = np.array([v for _, v in _series], dtype=np.float64)
    if values.size == 0:
        values = np.array([0.0])
        epochs = [0]

    v_min = float(values.min())
    v_max = float(values.max())
    v_mean = float(values.mean())
    v_std = float(values.std())
    first_value = float(values[0])
    final_value = float(values[-1])
    delta = final_value - first_value
    tail = values[-5:] if values.size >= 5 else values
    tail_mean = float(tail.mean())

    status = "PASS" if _satisfies(tail_mean, standard_threshold) else "FAIL"
    acceptable_met = bool(_satisfies(tail_mean, acceptable_threshold))

    # First crossing of the standard threshold.
    cross_epoch = None
    for e, v in zip(epochs, values):
        if _satisfies(float(v), standard_threshold):
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
        "values": [{"epoch": int(e), "value": float(v)} for e, v in zip(epochs, values)],
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
    }
    _atomic_write_json(METRIC_DIR / f"probe_result_{n}.json", result)

    _save_plot(
        n=n,
        epochs=epochs,
        values=values,
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

    # Fixed y-axis range for cross-iteration comparability. kBET is in [0,1];
    # include both thresholds with headroom.
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
