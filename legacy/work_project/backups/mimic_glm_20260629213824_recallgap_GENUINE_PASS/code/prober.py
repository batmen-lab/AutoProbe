"""Training-quality probe: per-epoch minimum ethnicity subgroup AUROC.

Metric (higher is better):
    min_subgroup_auroc = min over scored ethnicity groups of roc_auc_score(y_g, p_g)

Groups: white, black, hispanic, asian, other. A group is scored only if it has
>= MIN_SAMPLES positives AND >= MIN_SAMPLES negatives; smaller groups are skipped.
Focus is on the masked minority groups (black, hispanic) since MASK_MINORITY_TFIDF
can collapse their discrimination.

`record(epoch, model, loader, device)` runs a forward pass over the (unshuffled)
val/test loader, buckets predictions by ethnicity, computes per-group AUROC and the
headline min, then overwrites a live trajectory JSON for the UI.

`conclude(standard_threshold, acceptable_threshold)` writes the final stats JSON
and a Plotly PDF chart.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKING_SPACE = Path(__file__).resolve().parent
PROBE_DIR = WORKING_SPACE / ".agent_probe"
LIVE_DIR = PROBE_DIR / "live"
METRIC_DIR = PROBE_DIR / "metric"
PLOT_DIR = PROBE_DIR / "plot"
LIVE_PATH = LIVE_DIR / "probe_live.json"

METRIC_NAME = "equal_opportunity_recall_gap"
DIRECTION = "lower_is_better"

ETH_NAMES = ["white", "black", "hispanic", "asian", "other"]
MIN_SAMPLES = 20  # min positives AND min negatives required to score a group
COLLAPSE_BUFFER = 0.05  # alert if min_subgroup_auroc < 0.5 + buffer

# In-memory series: list of (epoch, value)
_SERIES: list[tuple[int, float]] = []

# Overall (whole-cohort) AUROC at the last record() — used as a guard so a
# degenerate "fix" that closes the recall gap by collapsing the model fails.
_LAST_AUROC: float = float("nan")
AUROC_FLOOR: float = 0.80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _write_live() -> None:
    payload = {
        "metric_name": METRIC_NAME,
        "standard_threshold": _STD_THRESHOLD,
        "acceptable_threshold": _ACC_THRESHOLD,
        "direction": DIRECTION,
        "values": [{"epoch": e, "value": v} for e, v in _SERIES],
    }
    _atomic_write_json(LIVE_PATH, payload)


# Thresholds are known up-front from the dev document; used for the live chart.
_STD_THRESHOLD = 0.75
_ACC_THRESHOLD = 0.68


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------
@torch.no_grad()
def record(epoch: int, model, loader, device) -> float:
    """Forward pass over `loader`, compute per-ethnicity AUROC, append the
    minimum across scorable groups to the series, and refresh the live JSON."""
    model.eval()
    probs_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    eth_chunks: list[np.ndarray] = []

    ds = loader.dataset
    eth_all = ds.eth.numpy()  # (N, 5) one-hot, aligned with sample order
    cursor = 0
    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label"]
        logits = model(features)
        p = logits.sigmoid().cpu().numpy()
        n = p.shape[0]
        probs_chunks.append(p)
        labels_chunks.append(labels.numpy())
        eth_chunks.append(eth_all[cursor:cursor + n])
        cursor += n

    y_prob = np.concatenate(probs_chunks)
    y_true = np.concatenate(labels_chunks).astype(int)
    eth = np.concatenate(eth_chunks)
    eth_id = eth.argmax(axis=1)

    # Equal-opportunity recall gap (race fairness): at a single SHARED global
    # decision threshold calibrated so the overall predicted-positive rate is
    # ~20% (the 80th percentile of risk scores), measure recall (TPR = caught
    # deaths / total deaths) for white patients vs. pooled black+hispanic
    # patients. gap = recall_white - recall_minority. Lower is better (0 = deaths
    # detected equally across races). Pooling black+hispanic gives a stable
    # estimate despite small minority strata.
    thr = float(np.quantile(y_prob, 0.80))
    pred_pos = (y_prob >= thr).astype(int)

    def _recall(group_mask: np.ndarray) -> float:
        pos = group_mask & (y_true == 1)
        n_pos = int(pos.sum())
        if n_pos == 0:
            return float("nan")
        return float(pred_pos[pos].sum()) / n_pos

    white_mask = eth_id == 0
    minority_mask = np.isin(eth_id, [1, 2])  # black, hispanic
    recall_white = _recall(white_mask)
    recall_min = _recall(minority_mask)
    gap = (recall_white - recall_min
           if not (np.isnan(recall_white) or np.isnan(recall_min)) else float("nan"))

    # AUROC guard: track whole-cohort discrimination so conclude() can reject a
    # degenerate "fix" that closes the gap by collapsing the model for everyone.
    global _LAST_AUROC
    try:
        _LAST_AUROC = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        _LAST_AUROC = float("nan")

    _SERIES.append((epoch, gap))
    logger.info(
        "  [probe] epoch %d eq_opp_recall_gap=%.4f (recall white=%.3f minority=%.3f @thr=%.3f overall_auroc=%.3f)",
        epoch, gap, recall_white, recall_min, thr, _LAST_AUROC,
    )
    _write_live()
    return gap


# ---------------------------------------------------------------------------
# conclude
# ---------------------------------------------------------------------------
def _next_index(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    highest = 0
    for p in directory.glob("probe_result_*.json"):
        stem = p.stem.rsplit("_", 1)[-1]
        if stem.isdigit():
            highest = max(highest, int(stem))
    return highest + 1


def _satisfies(value: float, threshold: float) -> bool:
    if np.isnan(value):
        return False
    if DIRECTION == "higher_is_better":
        return value >= threshold
    return value <= threshold


def conclude(standard_threshold: float, acceptable_threshold: float) -> None:
    global _STD_THRESHOLD, _ACC_THRESHOLD
    # Fixed thresholds for the equal-opportunity recall gap (self-contained so a
    # fix-round revert of train.py cannot change the pass/fail bar).
    standard_threshold, acceptable_threshold = 0.08, 0.15
    _STD_THRESHOLD = float(standard_threshold)
    _ACC_THRESHOLD = float(acceptable_threshold)

    epochs = [e for e, _ in _SERIES]
    values = np.array([v for _, v in _SERIES], dtype=float)
    valid = values[~np.isnan(values)]

    if valid.size == 0:
        v_min = v_max = v_mean = v_std = first = final = delta = tail_mean = float("nan")
    else:
        v_min = float(np.min(valid))
        v_max = float(np.max(valid))
        v_mean = float(np.mean(valid))
        v_std = float(np.std(valid))
        first = float(valid[0])
        final = float(valid[-1])
        delta = final - first
        tail = valid[-5:] if valid.size >= 5 else valid
        tail_mean = float(np.mean(tail))

    status = "PASS" if _satisfies(tail_mean, standard_threshold) else "FAIL"
    acceptable_met = _satisfies(tail_mean, acceptable_threshold)

    # AUROC guard: a recall gap can be closed trivially by leveling DOWN (wrecking
    # the model so it catches nobody). Reject that — require overall AUROC >= floor.
    auroc_note = ""
    if not np.isnan(_LAST_AUROC) and _LAST_AUROC < AUROC_FLOOR:
        status = "FAIL"
        acceptable_met = False
        auroc_note = (f" [AUROC GUARD FAILED: overall AUROC {_LAST_AUROC:.3f} < {AUROC_FLOOR:.2f} "
                      f"-- gap closed by degrading the model, not a valid fairness fix]")

    # First epoch crossing the standard threshold
    cross_epoch = None
    for e, v in _SERIES:
        if not np.isnan(v) and _satisfies(v, standard_threshold):
            cross_epoch = e
            break

    trend = "improving" if (not np.isnan(delta) and delta > 0) else (
        "degrading" if (not np.isnan(delta) and delta < 0) else "flat")

    if valid.size == 0:
        conclusion = "No scorable ethnicity subgroups were found; min_subgroup_auroc could not be computed."
    else:
        cross_txt = (f", crossing the {standard_threshold:.2f} standard threshold at epoch {cross_epoch}"
                     if cross_epoch is not None else
                     f", never reaching the {standard_threshold:.2f} standard threshold")
        conclusion = (
            f"Equal-opportunity recall gap went from {first:.3f} to {final:.3f} ({trend}){cross_txt}; "
            f"tail mean {tail_mean:.3f} (overall AUROC {_LAST_AUROC:.3f}) -> {status}.{auroc_note}"
        )

    n = _next_index(METRIC_DIR)
    result = {
        "metric_name": METRIC_NAME,
        "standard_threshold": float(standard_threshold),
        "acceptable_threshold": float(acceptable_threshold),
        "direction": DIRECTION,
        "values": [{"epoch": e, "value": (None if np.isnan(v) else float(v))} for e, v in _SERIES],
        "min": v_min,
        "max": v_max,
        "mean": v_mean,
        "std": v_std,
        "first_value": first,
        "final_value": final,
        "delta": delta,
        "tail_mean": tail_mean,
        "status": status,
        "acceptable_met": bool(acceptable_met),
        "conclusion": conclusion,
    }
    _atomic_write_json(METRIC_DIR / f"probe_result_{n}.json", result)
    logger.info("  [probe] wrote %s (status=%s, acceptable_met=%s)",
                METRIC_DIR / f"probe_result_{n}.json", status, acceptable_met)

    _make_plot(n, epochs, values, standard_threshold, acceptable_threshold,
               status, cross_epoch, v_min, v_max, v_mean, v_std, delta, trend, acceptable_met)


def _make_plot(n, epochs, values, std_thr, acc_thr, status, cross_epoch,
               v_min, v_max, v_mean, v_std, delta, trend, acceptable_met) -> None:
    import plotly.graph_objects as go

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    line_color = "green" if status == "PASS" else "red"

    plot_vals = [None if (isinstance(v, float) and np.isnan(v)) else float(v) for v in values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=plot_vals, mode="lines+markers",
        name=METRIC_NAME, line=dict(color=line_color, width=2),
    ))
    fig.add_hline(y=std_thr, line_dash="dash", line_color="red",
                  annotation_text=f"standard {std_thr:.2f}", annotation_position="top left")
    fig.add_hline(y=acc_thr, line_dash="dash", line_color="orange",
                  annotation_text=f"acceptable {acc_thr:.2f}", annotation_position="bottom left")
    if cross_epoch is not None:
        fig.add_vline(x=cross_epoch, line_dash="dot", line_color="gray",
                      annotation_text=f"cross @ {cross_epoch}", annotation_position="top")

    stats_txt = (
        f"min={v_min:.3f}  max={v_max:.3f}  mean={v_mean:.3f}  std={v_std:.3f}<br>"
        f"delta={delta:+.3f}  trend={trend}<br>"
        f"status={status}  acceptable_met={acceptable_met}"
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.02, align="left",
        showarrow=False, text=stats_txt, bordercolor="black", borderwidth=1,
        bgcolor="white", opacity=0.85,
    )

    # Fixed y-axis range for cross-iteration comparability: span both thresholds
    # plus headroom for future runs. AUROC is bounded in [0, 1].
    y_lo = min(0.4, acc_thr - 0.1, std_thr - 0.1)
    y_hi = max(1.0, std_thr + 0.1)
    fig.update_layout(
        title=METRIC_NAME,
        xaxis_title="Epoch",
        yaxis_title=METRIC_NAME,
        yaxis=dict(range=[y_lo, y_hi]),
        template="plotly_white",
    )
    out = PLOT_DIR / f"probe_result_{n}.pdf"
    fig.write_image(str(out))
    logger.info("  [probe] wrote %s", out)
