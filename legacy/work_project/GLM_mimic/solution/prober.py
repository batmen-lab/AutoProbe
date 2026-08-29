"""Training-quality probe: per-epoch equal-opportunity ethnicity recall gap.

Metric (lower is better):
    gap = max_g Recall_g - min_g Recall_g
where Recall_g = TP_g / (TP_g + FN_g) over validation patients in ethnicity
group g, with y_pred = (sigmoid(logits) >= 0.5). Groups with empty
denominator (TP+FN == 0) are dropped from the gap extrema but still logged.

`record(epoch, model, loader, device, ...)` runs a forward pass over the
unshuffled validation loader, buckets predictions by ethnicity, appends the
gap for this epoch, and overwrites a live trajectory JSON for the UI. It also
accepts `original_train_metrics` (a list of anchor dicts) which it carries
through to conclude().

`conclude(standard_threshold, acceptable_threshold)` writes the final stats
JSON (including the train.py anchor metrics as `original_train_metric_{i}`)
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

METRIC_NAME = "ethnicity_equal_opportunity_recall_gap"
DIRECTION = "lower_is_better"

ETH_NAMES = ["white", "black", "hispanic", "asian", "other"]

# CSV time series (per the dev document) — also exposes per-group AUROC so a
# controller can detect a collapsed-AUROC gaming pattern (recall gap small but
# per-group AUROC all ~0.5).
LOG_DIR = WORKING_SPACE / "result" / "mimic"
CSV_PATH = LOG_DIR / "fairness_recall_gap.csv"
PNG_PATH = LOG_DIR / "fairness_recall_gap.png"
FIXED_Y_RANGE: list[float] | None = None

# In-memory series
_SERIES: list[tuple[int, float]] = []
_GROUP_HISTORY: list[dict] = []
_LAST_ANCHORS: list[dict] = []


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


def _group_metrics(y_true, y_score, group_idx):
    """Per-group recall + AUROC + support. Returns dict keyed by group name."""
    out: dict[str, dict] = {}
    preds = (y_score >= 0.5).astype(int)
    for gi, name in enumerate(ETH_NAMES):
        mask = group_idx == gi
        n = int(mask.sum())
        n_pos = int(y_true[mask].sum())
        n_neg = n - n_pos
        tp = int(((preds[mask] == 1) & (y_true[mask] == 1)).sum())
        fn = int(((preds[mask] == 0) & (y_true[mask] == 1)).sum())
        fp = int(((preds[mask] == 1) & (y_true[mask] == 0)).sum())
        tn = int(((preds[mask] == 0) & (y_true[mask] == 0)).sum())
        denom = tp + fn
        recall = float(tp / denom) if denom > 0 else None
        auroc = None
        if n_pos > 0 and n_neg > 0:
            try:
                auroc = float(roc_auc_score(y_true[mask], y_score[mask]))
            except Exception:
                auroc = None
        out[name] = {
            "support": n, "n_pos": n_pos, "n_neg": n_neg,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "recall": recall, "auroc": auroc,
        }
    return out


def _gap_from_groups(groups):
    """max recall - min recall over groups with non-None recall (denom>0)."""
    recalls = [g["recall"] for g in groups.values() if g["recall"] is not None]
    if len(recalls) < 2:
        return float("nan")
    return float(max(recalls) - min(recalls))


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
    cols = ["epoch", "gap"]
    for g in ETH_NAMES:
        cols += [f"R_{g}", f"{g}_auroc", f"n_{g}"]
    lines = [",".join(cols)]
    for row in _GROUP_HISTORY:
        vals = [row["epoch"], f"{row['gap']:.6f}"]
        for g in ETH_NAMES:
            e = row["groups"][g]
            vals += [
                "" if e["recall"] is None else f"{e['recall']:.6f}",
                "" if e["auroc"] is None else f"{e['auroc']:.6f}",
                e["support"],
            ]
        lines.append(",".join(str(v) for v in vals))
    CSV_PATH.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def record(epoch, model, loader, device,
           standard_threshold: float = 0.05,
           acceptable_threshold: float = 0.10,
           original_train_metrics: list[dict] | None = None):
    """Compute and store the equal-opportunity recall gap for this epoch.

    Args:
        original_train_metrics: optional list of anchor dicts
            [{"name": str, "value": float, "direction": str}, ...] from
            train.py's own loss/eval. Carried through to conclude().
    """
    y_true, y_score, group_idx = _collect(model, loader, device)
    groups = _group_metrics(y_true, y_score, group_idx)
    gap = _gap_from_groups(groups)

    _SERIES.append((int(epoch), float(gap)))
    _GROUP_HISTORY.append({"epoch": int(epoch), "gap": float(gap),
                           "groups": groups})
    if original_train_metrics is not None:
        # carry through the latest snapshot of train.py's anchors
        _LAST_ANCHORS[:] = [
            {"name": a["name"], "value": float(a["value"]),
             "direction": a["direction"]}
            for a in original_train_metrics
        ]

    level = logging.WARNING if (gap == gap and gap > acceptable_threshold) else logging.INFO
    recalls_str = " ".join(
        f"{g}={groups[g]['recall']:.3f}" if groups[g]["recall"] is not None
        else f"{g}=NA" for g in ETH_NAMES
    )
    logger.log(level, "[probe] epoch %d recall_gap=%.4f (%s)",
               epoch, gap, recalls_str)

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


def conclude(standard_threshold, acceptable_threshold):
    standard_threshold = float(standard_threshold)
    acceptable_threshold = float(acceptable_threshold)

    epochs = [e for e, _ in _SERIES]
    vals = np.array([v for _, v in _SERIES], dtype=float)
    if vals.size == 0:
        vals = np.array([float("nan")])
    finite = vals[np.isfinite(vals)] if np.isfinite(vals).any() else vals

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
        f"Equal-opportunity recall gap (lower is better) went from {first_value:.4f} "
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
    # Anchor: train.py's own primary eval/selection metric(s) + loss. Stored
    # as independent records alongside the probe metric — do NOT fold into
    # the gap computation. Use _0, _1, ... when more than one anchor exists.
    if len(_LAST_ANCHORS) == 1:
        result["original_train_metric"] = _LAST_ANCHORS[0]
    else:
        for i, a in enumerate(_LAST_ANCHORS):
            result[f"original_train_metric_{i}"] = a

    n = _next_index()
    _atomic_write_json(METRIC_DIR / f"probe_result_{n}.json", result)

    _save_plot(n, result, cross_ep)
    _save_matplotlib_summary()

    logger.info("[probe] conclude: %s (status=%s, n=%d)", conclusion, status, n)
    return result


# ---------------------------------------------------------------------------
# Plotly chart (mandatory artifact)
# ---------------------------------------------------------------------------
def _y_axis_range(standard_threshold, acceptable_threshold):
    """Fixed y-range covering both thresholds + all observed values + headroom.

    Cached into FIXED_Y_RANGE so subsequent runs share the same axis range for
    cross-iteration comparability.
    """
    global FIXED_Y_RANGE
    if FIXED_Y_RANGE is not None:
        return FIXED_Y_RANGE
    candidates = [0.0, standard_threshold, acceptable_threshold]
    for _, v in _SERIES:
        if v == v:
            candidates.append(v)
    lo = min(candidates)
    hi = max(candidates)
    span = max(hi - lo, 0.1)
    FIXED_Y_RANGE = [max(lo - 0.02, 0.0), hi + 0.5 * span + 0.02]
    return FIXED_Y_RANGE


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

    trend_txt = "improving" if result["delta"] < 0 else ("flat" if result["delta"] == 0 else "worsening")
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


def _save_matplotlib_summary():
    """Per-group recall lines + gap series -> result/mimic/fairness_recall_gap.png.

    Dashed horizontal line at 0.05 (standard threshold). One line per ethnicity
    group's recall, plus the gap series in black.
    """
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
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in ETH_NAMES:
        ys = [r["groups"][g]["recall"] for r in _GROUP_HISTORY]
        if any(y is not None for y in ys):
            ax.plot(epochs, ys, marker="o", label=f"{g} recall")
    ax.plot(epochs, [r["gap"] for r in _GROUP_HISTORY],
            marker="s", linestyle="--", color="black", label="recall gap")
    ax.axhline(0.05, color="red", linestyle=":", label="standard=0.05")
    ax.axhline(0.10, color="orange", linestyle=":", label="acceptable=0.10")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall / gap")
    ax.set_title("Equal-opportunity per-ethnicity recall over epochs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)
