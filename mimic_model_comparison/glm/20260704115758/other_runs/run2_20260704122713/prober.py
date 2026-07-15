"""Training-quality probe: equal-opportunity TPR gap at hard-coded threshold 0.5.

Metric (lower is better):
    equal_opportunity_tpr_gap_at_threshold_0p5
        = max_group_TPR - min_group_TPR
    across {white, black, hispanic, asian, other} on the validation split, where
    TPR for the death class = TP/(TP+FN) after sigmoid(logits) binarized at the
    hard-coded 0.5 threshold (which lives ONLY in prober.py, never in train.py).

`record(epoch, model, loader, device, val_auprc, val_loss)` runs a forward pass
over the (unshuffled) validation loader, buckets predictions by ethnicity, and
appends the per-epoch gap. It also overwrites a live trajectory JSON for the UI
and records the anchor metrics (val AUPRC, weighted-BCE loss) for the backend's
utility-floor guard.

`conclude(standard_threshold, acceptable_threshold)` writes the final stats JSON
(with the anchor alongside the probe fields) and a Plotly PDF chart.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — working space + output dirs
# ---------------------------------------------------------------------------
WORKING_SPACE = Path(__file__).resolve().parent
PROBE_DIR = WORKING_SPACE / ".agent_probe"
LIVE_DIR = PROBE_DIR / "live"
METRIC_DIR = PROBE_DIR / "metric"
PLOT_DIR = PROBE_DIR / "plot"

# Repo-side artifacts requested by the dev doc (CSV time series + PNG plot)
LOG_DIR = WORKING_SPACE / "logs"
CSV_PATH = LOG_DIR / "per_group_tpr.csv"
PNG_PATH = LOG_DIR / "tpr_gap_over_epochs.png"

METRIC_NAME = "equal_opportunity_tpr_gap_at_threshold_0p5"
DIRECTION = "lower_is_better"

ETH_NAMES = ["white", "black", "hispanic", "asian", "other"]

# FROZEN OPERATING POINT — hard-coded literal 0.5. This is the metric's decision
# threshold and lives ONLY in prober.py (the fix-loop cannot touch it). It is
# NOT exposed as a tunable constant or argument in train.py.
THRESHOLD = 0.5
assert THRESHOLD == 0.5 and type(THRESHOLD) is float, (
    "THRESHOLD must be a literal 0.5 (frozen operating point, not configurable)"
)

# In-memory series
_SERIES: list[tuple[int, float]] = []
# Per-epoch per-group detail for CSV / matplotlib summary
_GROUP_HISTORY: list[dict] = []
# Anchor metrics (train.py's own loss/eval) recorded per-epoch — separate,
# independent record that lives alongside the probe metric. The backend's
# utility-floor guard reverts any edit that degrades these by >20% from round-1.
_ANCHOR_HISTORY: list[dict] = []


# ---------------------------------------------------------------------------
# Eval pass
# ---------------------------------------------------------------------------
@torch.no_grad()
def _collect(model, loader, device):
    """Return (y_true, y_score, group_idx) arrays aligned across the loader.

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

    y_score = np.concatenate(scores)
    y_true = np.concatenate(labels).astype(int)

    eth = loader.dataset.eth.cpu().numpy()
    group_idx = eth.argmax(axis=1)[: len(y_true)]
    return y_true, y_score, group_idx


def _group_tpr(y_true, y_score, group_idx, threshold: float):
    """Per-group TPR for the death class at the given threshold.

    TPR = TP/(TP+FN) over rows where group==g AND label==1 (death-positive).
    Returns dict keyed by group name with tpr, n_pos, n_pred_pos.
    """
    preds = (y_score >= threshold).astype(int)
    out: dict[str, dict] = {}
    for gi, name in enumerate(ETH_NAMES):
        gmask = group_idx == gi
        pos_mask = gmask & (y_true == 1)
        n_pos = int(pos_mask.sum())
        n_pred_pos = int(preds[pos_mask].sum()) if n_pos > 0 else 0
        tpr = float(preds[pos_mask].mean()) if n_pos > 0 else float("nan")
        out[name] = {
            "tpr": tpr,
            "n_pos": n_pos,
            "n_pred_pos": n_pred_pos,
            "support": int(gmask.sum()),
        }
    return out


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
# CSV
# ---------------------------------------------------------------------------
def _write_csv() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    header = "epoch,group,tpr,n_pos,n_pred_pos"
    lines = [header]
    for row in _GROUP_HISTORY:
        ep = row["epoch"]
        for g in ETH_NAMES:
            e = row["groups"][g]
            tpr = "" if e["tpr"] != e["tpr"] else f"{e['tpr']:.6f}"
            lines.append(f"{ep},{g},{tpr},{e['n_pos']},{e['n_pred_pos']}")
    CSV_PATH.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def record(epoch, model, loader, device,
           val_auprc=None, val_loss=None,
           standard_threshold: float = 0.08,
           acceptable_threshold: float = 0.15):
    """Compute and store the equal-opportunity TPR gap for this epoch.

    Args:
        epoch: 1-indexed epoch number.
        model: trained model (will be put in eval mode).
        loader: unshuffled validation DataLoader.
        device: torch device.
        val_auprc: train.py's own validation AUPRC (anchor, higher_is_better).
        val_loss: train.py's own validation loss (anchor, lower_is_better).
        standard_threshold / acceptable_threshold: forwarded to the live JSON.

    Returns:
        The per-epoch gap (max_tpr - min_tpr across groups with finite TPR).
    """
    y_true, y_score, group_idx = _collect(model, loader, device)
    groups = _group_tpr(y_true, y_score, group_idx, THRESHOLD)

    finite_tprs = [groups[g]["tpr"] for g in ETH_NAMES
                   if groups[g]["tpr"] == groups[g]["tpr"]
                   and groups[g]["n_pos"] > 0]
    if len(finite_tprs) >= 2:
        gap = float(max(finite_tprs) - min(finite_tprs))
    else:
        gap = float("nan")

    _SERIES.append((int(epoch), float(gap)))
    _GROUP_HISTORY.append({"epoch": int(epoch), "gap": float(gap),
                           "groups": groups})

    anchor_entry = {"epoch": int(epoch),
                    "val_auprc": float(val_auprc) if val_auprc is not None else None,
                    "val_loss": float(val_loss) if val_loss is not None else None}
    _ANCHOR_HISTORY.append(anchor_entry)

    level = logging.WARNING if (gap == gap and gap > acceptable_threshold) else logging.INFO
    logger.log(level, "[probe] epoch %d tpr_gap_0p5=%.4f (val_auprc=%.4f, val_loss=%.4f)",
               epoch, gap,
               val_auprc if val_auprc is not None else float("nan"),
               val_loss if val_loss is not None else float("nan"))

    _write_live(standard_threshold, acceptable_threshold)
    _write_csv()
    return gap


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


def _anchor_payload() -> dict:
    """Build the original_train_metric{_0,_1} anchor payload from the last epoch."""
    if not _ANCHOR_HISTORY:
        return {}
    last = _ANCHOR_HISTORY[-1]
    auprc = last.get("val_auprc")
    loss = last.get("val_loss")
    anchors: list[dict] = []
    if auprc is not None:
        anchors.append({"name": "val_auprc", "value": float(auprc),
                        "direction": "higher_is_better"})
    if loss is not None:
        anchors.append({"name": "val_loss", "value": float(loss),
                        "direction": "lower_is_better"})
    if not anchors:
        return {}
    if len(anchors) == 1:
        return {"original_train_metric": anchors[0]}
    return {f"original_train_metric_{i}": a for i, a in enumerate(anchors)}


def conclude(standard_threshold, acceptable_threshold):
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
    trend = "improving" if delta < 0 else ("flat" if delta == 0 else "worsening")
    cross_txt = (f", first reaching the {standard_threshold} standard threshold at epoch {cross_ep}"
                 if cross_ep is not None else
                 f", never reaching the {standard_threshold} standard threshold")
    conclusion = (
        f"Equal-opportunity TPR gap (lower is better) went from {first_value:.4f} "
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
    result.update(_anchor_payload())

    n = _next_index()
    _atomic_write_json(METRIC_DIR / f"probe_result_{n}.json", result)

    _save_plot(n, result, cross_ep)
    _save_matplotlib_summary(standard_threshold, acceptable_threshold)

    logger.info("[probe] conclude: %s (status=%s, n=%d)", conclusion, status, n)
    return result


# ---------------------------------------------------------------------------
# Plotly chart (mandatory artifact)
# ---------------------------------------------------------------------------
def _y_axis_range(standard_threshold, acceptable_threshold):
    """Fixed y-range covering both thresholds + headroom for comparability.

    Used across all per-iteration charts so the y-axis stays comparable. Must
    include both threshold values and leave headroom for future iterations.
    """
    candidates = [0.0, standard_threshold, acceptable_threshold]
    for _, v in _SERIES:
        if v == v:
            candidates.append(v)
    lo = min(candidates)
    hi = max(candidates)
    span = max(hi - lo, 0.1)
    return [max(0.0, lo - 0.02), hi + 0.5 * span + 0.02]


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
                  annotation_text=f"standard={std_thr}",
                  annotation_position="top right")
    fig.add_hline(y=acc_thr, line=dict(color="orange", dash="dash"),
                  annotation_text=f"acceptable={acc_thr}",
                  annotation_position="bottom right")
    if cross_ep is not None:
        fig.add_vline(x=cross_ep, line=dict(color="gray", dash="dot"),
                      annotation_text=f"crossed @ {cross_ep}",
                      annotation_position="top left")

    trend_txt = ("improving" if result["delta"] < 0
                 else ("flat" if result["delta"] == 0 else "worsening"))
    textbox = (
        f"min={result['min']:.4f}  max={result['max']:.4f}<br>"
        f"mean={result['mean']:.4f}  std={result['std']:.4f}<br>"
        f"delta={result['delta']:.4f}  trend={trend_txt}<br>"
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


def _save_matplotlib_summary(standard_threshold, acceptable_threshold):
    """Two-panel matplotlib PNG: per-group TPR lines + gap series with thresholds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable, skipping summary png: %s", exc)
        return
    if not _GROUP_HISTORY:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in _GROUP_HISTORY]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax = axes[0]
    for g in ETH_NAMES:
        ys = [r["groups"][g]["tpr"] for r in _GROUP_HISTORY]
        ys = [y if y == y else np.nan for y in ys]
        ax.plot(epochs, ys, marker="o", label=g)
    ax.set_ylabel("per-group TPR (death class)")
    ax.set_title("Per-ethnicity TPR at threshold 0.5")
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(-0.02, 1.02)

    ax = axes[1]
    gaps = [r["gap"] for r in _GROUP_HISTORY]
    ax.plot(epochs, gaps, marker="s", color="black", label="TPR gap")
    ax.axhline(standard_threshold, color="red", linestyle="--",
               label=f"standard={standard_threshold}")
    ax.axhline(acceptable_threshold, color="orange", linestyle="--",
               label=f"acceptable={acceptable_threshold}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("max_tpr - min_tpr")
    ax.set_title("Equal-opportunity TPR gap (lower is better)")
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)
