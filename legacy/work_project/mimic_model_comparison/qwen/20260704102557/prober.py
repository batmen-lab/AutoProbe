"""Training-quality probe: per-epoch ethnicity recall gap at a FIXED threshold.

Metric (lower is better):
    gap = max(TPR_g) - min(TPR_g) across the 5 ethnicity groups on the
    validation split, at a HARD-CODED sigmoid decision threshold of 0.5 that
    lives inside prober.py and is NOT exposed to the fix loop.

`record(epoch, model, loader, device)` runs a forward pass over the (unshuffled)
validation loader, buckets predictions by ethnicity group, and appends the
gap for this epoch. It also overwrites a live trajectory JSON for the UI.

`conclude(standard_threshold, acceptable_threshold)` writes the final stats
JSON and a Plotly PDF chart.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import recall_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKING_SPACE = Path(__file__).resolve().parent
PROBE_DIR = WORKING_SPACE / ".agent_probe"
LIVE_DIR = PROBE_DIR / "live"
METRIC_DIR = PROBE_DIR / "metric"
PLOT_DIR = PROBE_DIR / "plot"

METRIC_NAME = "max_per_group_recall_gap_at_fixed_0p5"
DIRECTION = "lower_is_better"

# HARD-CODED decision threshold. Frozen inside prober.py (NOT a tunable arg in
# train.py) so the fix loop can only improve the model, never move the
# operating point at which the metric is measured.
DECISION_THRESHOLD = 0.5

ETH_NAMES = ["white", "black", "hispanic", "asian", "other"]

# Per-epoch bar chart (artifact png) as described in the dev document. Per-epoch
# CSV log lives in logs/fairness_recall_gap.csv.
LOG_DIR = WORKING_SPACE / "logs"
CSV_PATH = LOG_DIR / "fairness_recall_gap.csv"
PNG_DIR = LOG_DIR / "recall_gap_per_epoch"

# In-memory series: list of (epoch, gap)
_SERIES: list[tuple[int, float]] = []
# Per-epoch per-group detail for CSV + PNG bar chart
_GROUP_HISTORY: list[dict] = []


# ---------------------------------------------------------------------------
# Eval pass
# ---------------------------------------------------------------------------
@torch.no_grad()
def _collect(model, loader, device):
    """Return (y_true, y_prob, group_idx) arrays aligned across the loader.

    The loader is assumed unshuffled so accumulated predictions line up with
    ``loader.dataset.eth`` row-for-row.
    """
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        features = batch["features"].to(device)
        logits = model(features)
        scores.append(logits.sigmoid().cpu().numpy())
        labels.append(batch["label"].cpu().numpy())

    y_prob = np.concatenate(scores)
    y_true = np.concatenate(labels).astype(int)

    eth = loader.dataset.eth.cpu().numpy()
    group_idx = eth.argmax(axis=1)[: len(y_true)]
    return y_true, y_prob, group_idx


def _group_tpr(y_true, y_prob, group_idx):
    """Compute per-group TPR (recall) at the hard-coded 0.5 decision threshold.

    Returns a dict keyed by group name with keys: support, n_pos, tpr.  Groups
    with no positive labels in the batch have tpr=None (NaN-equivalent) and are
    excluded from the gap computation with a warning.
    """
    preds = (y_prob >= DECISION_THRESHOLD).astype(int)
    out: dict[str, dict] = {}
    for gi, name in enumerate(ETH_NAMES):
        mask = group_idx == gi
        n = int(mask.sum())
        n_pos = int(y_true[mask].sum())
        if n_pos == 0:
            logger.warning(
                "[probe] group '%s' has no positive labels — TPR undefined, "
                "excluding from gap", name
            )
            out[name] = {"support": n, "n_pos": 0, "tpr": None}
            continue
        tp = int(((preds[mask] == 1) & (y_true[mask] == 1)).sum())
        tpr = tp / n_pos
        out[name] = {"support": n, "n_pos": n_pos, "tpr": float(tpr)}
    return out


def _compute_gap(groups: dict[str, dict]) -> float:
    """max(TPR_g) - min(TPR_g) over non-NaN groups. NaN if <2 groups have TPR."""
    valid = [g["tpr"] for g in groups.values() if g["tpr"] is not None]
    if len(valid) < 2:
        return float("nan")
    return float(max(valid) - min(valid))


# ---------------------------------------------------------------------------
# Live JSON
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


def _write_live(standard_threshold: float, acceptable_threshold: float) -> None:
    payload = {
        "metric_name": METRIC_NAME,
        "standard_threshold": float(standard_threshold),
        "acceptable_threshold": float(acceptable_threshold),
        "direction": DIRECTION,
        "values": [{"epoch": e, "value": v} for e, v in _SERIES],
    }
    _atomic_write_json(LIVE_DIR / "probe_live.json", payload)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def record(epoch, model, loader, device,
           standard_threshold: float = 0.10,
           acceptable_threshold: float = 0.18):
    """Compute and store the per-ethnicity recall gap for this epoch.

    Uses the hard-coded DECISION_THRESHOLD (0.5) inside prober.py — train.py
    does not see or tune this value.
    """
    y_true, y_prob, group_idx = _collect(model, loader, device)
    groups = _group_tpr(y_true, y_prob, group_idx)
    gap = _compute_gap(groups)

    global_tpr = float(recall_score(y_true, (y_prob >= DECISION_THRESHOLD).astype(int),
                                    zero_division=0))

    _SERIES.append((int(epoch), float(gap)))
    _GROUP_HISTORY.append({"epoch": int(epoch), "global_tpr": global_tpr,
                           "gap": float(gap), "groups": groups})

    # Side-channel alert per dev doc: gap > 0.20 or any minority-group TPR
    # < 0.5 * global TPR.
    minority_alert = False
    if global_tpr > 0:
        for g in ("black", "hispanic"):
            g_tpr = groups[g]["tpr"]
            if g_tpr is not None and g_tpr < 0.5 * global_tpr:
                minority_alert = True
                logger.warning(
                    "[probe] ALERT: minority group '%s' recall %.3f < 0.5 * "
                    "global recall %.3f", g, g_tpr, global_tpr,
                )
    level = logging.WARNING if (gap == gap and (gap > 0.20 or minority_alert)) else logging.INFO
    logger.log(level, "[probe] epoch %d recall_gap=%.4f (global_TPR=%.4f)",
               epoch, gap, global_tpr)

    _write_live(standard_threshold, acceptable_threshold)
    _write_csv()
    _save_epoch_bar(int(epoch), groups, gap)
    return gap


def _write_csv() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["epoch", "global_TPR"]
    for g in ETH_NAMES:
        cols += [f"{g}_TPR", f"{g}_support", f"{g}_n_pos"]
    cols += ["gap"]
    lines = [",".join(cols)]
    for row in _GROUP_HISTORY:
        vals = [row["epoch"], f"{row['global_tpr']:.6f}"]
        for g in ETH_NAMES:
            e = row["groups"][g]
            vals += [
                "" if e["tpr"] is None else f"{e['tpr']:.6f}",
                e["support"],
                e["n_pos"],
            ]
        vals.append(f"{row['gap']:.6f}")
        lines.append(",".join(str(v) for v in vals))
    CSV_PATH.write_text("\n".join(lines) + "\n")


def _save_epoch_bar(epoch: int, groups: dict[str, dict], gap: float) -> None:
    """Per-epoch 5-bar chart of per-group TPR (dev-doc artifact)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable, skipping epoch bar: %s", exc)
        return
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(ETH_NAMES))
    tprs = [groups[g]["tpr"] if groups[g]["tpr"] is not None else 0.0 for g in ETH_NAMES]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, tprs, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(ETH_NAMES, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"TPR @ {DECISION_THRESHOLD}")
    ax.set_title(f"Epoch {epoch} — per-ethnicity recall (gap={gap:.3f})")
    for i, v in enumerate(tprs):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(PNG_DIR / f"recall_gap_epoch_{epoch:03d}.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Conclude
# ---------------------------------------------------------------------------
def _next_index() -> int:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(METRIC_DIR.glob("probe_result_*.json"))
    nums = []
    for p in existing:
        stem = p.stem.replace("probe_result_", "")
        if stem.isdigit():
            nums.append(int(stem))
    return (max(nums) + 1) if nums else 1


def _satisfies(value: float, threshold: float) -> bool:
    if value != value:  # NaN
        return False
    return value <= threshold if DIRECTION == "lower_is_better" else value >= threshold


def _first_cross_epoch(values, threshold):
    for e, v in values:
        if _satisfies(v, threshold):
            return e
    return None


def conclude(standard_threshold, acceptable_threshold,
             original_train_metric=None):
    standard_threshold = float(standard_threshold)
    acceptable_threshold = float(acceptable_threshold)

    epochs = [e for e, _ in _SERIES]
    vals = np.array([v for _, v in _SERIES], dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        finite = np.array([float("nan")])

    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    vmean = float(np.nanmean(vals))
    vstd = float(np.nanstd(vals))
    first_value = float(vals[0])
    final_value = float(vals[-1])
    delta = final_value - first_value
    tail = vals[-5:] if len(vals) >= 5 else vals
    tail_mean = float(np.nanmean(tail))

    status = "PASS" if _satisfies(tail_mean, standard_threshold) else "FAIL"
    acceptable_met = _satisfies(tail_mean, acceptable_threshold)

    cross_ep = _first_cross_epoch(_SERIES, standard_threshold)
    # Recall gap: improving = delta < 0 (going down toward lower-is-better).
    trend = "improving" if delta < 0 else ("flat" if delta == 0 else "worsening")
    cross_txt = (f", first reaching the {standard_threshold} standard threshold at epoch {cross_ep}"
                 if cross_ep is not None else
                 f", never reaching the {standard_threshold} standard threshold")
    conclusion = (
        f"Max per-group recall gap at threshold 0.5 went from {first_value:.4f} "
        f"to {final_value:.4f} over {len(epochs)} epochs{cross_txt}; "
        f"tail_mean={tail_mean:.4f} -> {status} (acceptable_met={acceptable_met})."
    )

    result = {
        "metric_name": METRIC_NAME,
        "standard_threshold": standard_threshold,
        "acceptable_threshold": acceptable_threshold,
        "direction": DIRECTION,
        "values": [{"epoch": e, "value": v} for e, v in _SERIES],
        "min": vmin,
        "max": vmax,
        "mean": vmean,
        "std": vstd,
        "first_value": first_value,
        "final_value": final_value,
        "delta": delta,
        "tail_mean": tail_mean,
        "status": status,
        "acceptable_met": bool(acceptable_met),
        "conclusion": conclusion,
    }

    # Anchor(s): train.py's own final-epoch loss/eval, passed in from the caller.
    # Recorded alongside — never folded into the probe metric itself.
    if original_train_metric is None:
        pass  # no anchors supplied
    elif isinstance(original_train_metric, list):
        for i, m in enumerate(original_train_metric):
            result[f"original_train_metric_{i}"] = m
    else:
        result["original_train_metric"] = original_train_metric

    n = _next_index()
    _atomic_write_json(METRIC_DIR / f"probe_result_{n}.json", result)

    _save_plot(n, result, cross_ep)

    logger.info("[probe] conclude: %s (status=%s, n=%d)", conclusion, status, n)
    return result


# ---------------------------------------------------------------------------
# Plotly chart (mandatory artifact)
# ---------------------------------------------------------------------------
def _y_axis_range(standard_threshold, acceptable_threshold):
    """Fixed y-range covering both thresholds + headroom for comparability.

    Recall gap lives in [0, 1]; we choose a stable range that (a) includes
    both threshold values, (b) includes any observed data, (c) has a little
    headroom. Rounded up to the next 0.1 for readability.
    """
    hi_data = max(
        (v for _, v in _SERIES if v == v),
        default=0.0,
    )
    hi = max(0.3, standard_threshold, acceptable_threshold, hi_data)
    # Round up to the next 0.1 for a clean, stable axis.
    hi = float(np.ceil(hi * 10) / 10) + 0.05
    return [0.0, hi]


def _save_plot(n, result, cross_ep):
    import plotly.graph_objects as go

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    epochs = [e for e, _ in _SERIES]
    values = [v for _, v in _SERIES]
    std_thr = result["standard_threshold"]
    acc_thr = result["acceptable_threshold"]
    color = "green" if result["status"] == "PASS" else "red"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=values, mode="lines+markers",
        name=METRIC_NAME, line=dict(color=color, width=2),
    ))
    fig.add_hline(y=std_thr, line=dict(color="red", dash="dash"),
                  annotation_text=f"standard={std_thr}", annotation_position="top right")
    fig.add_hline(y=acc_thr, line=dict(color="orange", dash="dash"),
                  annotation_text=f"acceptable={acc_thr}", annotation_position="bottom right")
    if cross_ep is not None:
        fig.add_vline(x=cross_ep, line=dict(color="gray", dash="dot"),
                      annotation_text=f"crossed @ {cross_ep}", annotation_position="top left")

    textbox = (
        f"min={result['min']:.4f}  max={result['max']:.4f}<br>"
        f"mean={result['mean']:.4f}  std={result['std']:.4f}<br>"
        f"delta={result['delta']:.4f}  trend={'improving' if result['delta'] < 0 else ('flat' if result['delta'] == 0 else 'worsening')}<br>"
        f"status={result['status']}  acceptable_met={result['acceptable_met']}"
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98, align="left",
        showarrow=False, bordercolor="black", borderwidth=1,
        bgcolor="white", opacity=0.85, text=textbox,
    )

    fig.update_layout(
        title=METRIC_NAME, xaxis_title="Epoch", yaxis_title=METRIC_NAME,
        template="plotly_white",
    )
    fig.update_yaxes(range=_y_axis_range(std_thr, acc_thr))

    out = PLOT_DIR / f"probe_result_{n}.pdf"
    fig.write_image(str(out))
