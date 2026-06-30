"""Training-quality probe: per-epoch ethnicity subgroup AUROC gap.

Metric (lower is better):
    subgroup_auroc_gap = overall_AUROC - min(black_AUROC, hispanic_AUROC)

`record(epoch, model, loader, device)` runs a forward pass over the (unshuffled)
validation/test loader, buckets predictions by ethnicity group, and appends the
gap for this epoch. It also overwrites a live trajectory JSON for the UI.

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
from sklearn.metrics import recall_score, roc_auc_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKING_SPACE = Path(__file__).resolve().parent
PROBE_DIR = WORKING_SPACE / ".agent_probe"
LIVE_DIR = PROBE_DIR / "live"
METRIC_DIR = PROBE_DIR / "metric"
PLOT_DIR = PROBE_DIR / "plot"

METRIC_NAME = "ethnicity_subgroup_auroc_gap"
DIRECTION = "lower_is_better"

ETH_NAMES = ["white", "black", "hispanic", "asian", "other"]
MINORITY_GROUPS = ["black", "hispanic"]
MIN_SAMPLES = 20  # min positives AND min negatives required to score a group

# CSV time series + matplotlib summary plot (per the dev document)
LOG_DIR = WORKING_SPACE / "logs"
CSV_PATH = LOG_DIR / "fairness_subgroup.csv"
PNG_PATH = LOG_DIR / "fairness_subgroup.png"

# In-memory series: list of (epoch, gap)
_SERIES: list[tuple[int, float]] = []
# Per-epoch per-group detail for the matplotlib plot / CSV
_GROUP_HISTORY: list[dict] = []


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
    """Compute per-group AUROC/recall/support. Returns dict keyed by group name."""
    out: dict[str, dict] = {}
    preds = (y_score >= 0.5).astype(int)
    for gi, name in enumerate(ETH_NAMES):
        mask = group_idx == gi
        n = int(mask.sum())
        n_pos = int(y_true[mask].sum())
        n_neg = n - n_pos
        entry = {"support": n, "n_pos": n_pos, "n_neg": n_neg,
                 "auroc": None, "recall": None}
        if n_pos >= MIN_SAMPLES and n_neg >= MIN_SAMPLES:
            entry["auroc"] = float(roc_auc_score(y_true[mask], y_score[mask]))
            entry["recall"] = float(
                recall_score(y_true[mask], preds[mask], zero_division=0)
            )
        else:
            logger.info("Subgroup '%s' insufficient support (pos=%d neg=%d)",
                        name, n_pos, n_neg)
        out[name] = entry
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
# Public API
# ---------------------------------------------------------------------------
def record(epoch, model, loader, device,
           standard_threshold: float = 0.03,
           acceptable_threshold: float = 0.07):
    """Compute and store the subgroup AUROC gap for this epoch."""
    y_true, y_score, group_idx = _collect(model, loader, device)

    overall_auroc = float(roc_auc_score(y_true, y_score))
    groups = _group_metrics(y_true, y_score, group_idx)

    minority_aurocs = [
        groups[g]["auroc"] for g in MINORITY_GROUPS if groups[g]["auroc"] is not None
    ]
    if minority_aurocs:
        gap = overall_auroc - min(minority_aurocs)
    else:
        gap = float("nan")

    _SERIES.append((int(epoch), float(gap)))
    _GROUP_HISTORY.append({"epoch": int(epoch), "overall": overall_auroc,
                           "gap": float(gap), "groups": groups})

    level = logging.WARNING if (gap == gap and gap > acceptable_threshold) else logging.INFO
    logger.log(level, "[probe] epoch %d subgroup_auroc_gap=%.4f (overall=%.4f)",
               epoch, gap, overall_auroc)

    _write_live(standard_threshold, acceptable_threshold)
    _write_csv()
    return gap


def _write_csv() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["epoch", "overall_auroc", "gap"]
    for g in ETH_NAMES:
        cols += [f"{g}_auroc", f"{g}_recall", f"{g}_support"]
    lines = [",".join(cols)]
    for row in _GROUP_HISTORY:
        vals = [row["epoch"], f"{row['overall']:.6f}", f"{row['gap']:.6f}"]
        for g in ETH_NAMES:
            e = row["groups"][g]
            vals += [
                "" if e["auroc"] is None else f"{e['auroc']:.6f}",
                "" if e["recall"] is None else f"{e['recall']:.6f}",
                e["support"],
            ]
        lines.append(",".join(str(v) for v in vals))
    CSV_PATH.write_text("\n".join(lines) + "\n")


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
        f"Ethnicity subgroup AUROC gap (lower is better) went from {first_value:.4f} "
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
    """Fixed y-range covering both thresholds + headroom for comparability."""
    candidates = [0.0, standard_threshold, acceptable_threshold]
    for _, v in _SERIES:
        if v == v:
            candidates.append(v)
    lo = min(candidates)
    hi = max(candidates)
    span = max(hi - lo, 0.1)
    return [lo - 0.02, hi + 0.5 * span + 0.02]


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


def _save_matplotlib_summary():
    """Per-group AUROC lines + gap series -> logs/fairness_subgroup.png."""
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
        ys = [r["groups"][g]["auroc"] for r in _GROUP_HISTORY]
        if any(y is not None for y in ys):
            ax.plot(epochs, ys, marker="o", label=f"{g} AUROC")
    ax.plot(epochs, [r["gap"] for r in _GROUP_HISTORY],
            marker="s", linestyle="--", color="black", label="subgroup gap")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC / gap")
    ax.set_title("Ethnicity subgroup AUROC over epochs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)
