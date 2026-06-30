"""Training-quality probe: per-epoch equal-opportunity difference (EOD).

Metric (lower is better):
    EOD = max per-ethnicity recall - min per-ethnicity recall, evaluated at a
    single shared decision threshold that is fixed once and for all on the
    first epoch's validation set (argmax macro-F1 over a 0.01 grid, or 0.5
    if no grid is run).

The probe also emits:
  * a per-epoch CSV log (epoch, group, recall, precision, f1, n) plus a
    summary row (epoch, EOD, AOD, threshold_used).
  * two matplotlib time-series: per-group recall curves overlaid, and
    EOD-over-epochs.

`record(epoch, model, loader, device, standard_threshold, acceptable_threshold)`
runs a forward pass over the (unshuffled) validation loader, buckets
predictions by ethnicity, computes per-group TP/FP/FN/TN at the shared
threshold, derives EOD/AOD, and appends the per-epoch EOD value to the
in-memory series. It also overwrites a live trajectory JSON for the UI.

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKING_SPACE = Path(__file__).resolve().parent
PROBE_DIR = WORKING_SPACE / ".agent_probe"
LIVE_DIR = PROBE_DIR / "live"
METRIC_DIR = PROBE_DIR / "metric"
PLOT_DIR = PROBE_DIR / "plot"

METRIC_NAME = "equal_opportunity_difference"
DIRECTION = "lower_is_better"

ETH_NAMES = ["white", "black", "hispanic", "asian", "other"]
MIN_SAMPLES = 30  # groups with fewer val samples are pooled into "other"

# CSV time series + matplotlib summary plot (per the dev document)
LOG_DIR = WORKING_SPACE / "logs"
CSV_PATH = LOG_DIR / "fairness_eod.csv"
PNG_PATH = LOG_DIR / "fairness_eod.png"

# In-memory series: list of (epoch, EOD)
_SERIES: list[tuple[int, float]] = []
# Per-epoch per-group detail for the matplotlib plot / CSV
_GROUP_HISTORY: list[dict] = []
# Shared threshold fixed on first epoch
_SHARED_THRESHOLD: float | None = None


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

    y_score = np.concatenate(scores)
    y_true = np.concatenate(labels).astype(int)

    eth = loader.dataset.eth.cpu().numpy()
    group_idx = eth.argmax(axis=1)[: len(y_true)]
    return y_true, y_score, group_idx


def _pick_threshold(y_true, y_score) -> float:
    """Argmax macro-F1 over a 0.01 grid; fall back to 0.5."""
    try:
        from sklearn.metrics import f1_score
        grid = np.arange(0.01, 1.00, 0.01)
        best_t, best_f = 0.5, -1.0
        for t in grid:
            preds = (y_score >= t).astype(int)
            f = f1_score(y_true, preds, zero_division=0, average="macro")
            if f > best_f:
                best_f, best_t = float(f), float(t)
        return best_t
    except Exception as exc:
        logger.warning("[probe] threshold grid search failed: %s -> 0.5", exc)
        return 0.5


def _pool_small_groups(y_true, y_score, group_idx):
    """Map small (<MIN_SAMPLES) ethnicity buckets to the 'other' index.

    Returns (group_idx_pooled, names_in_order) where names_in_order is the
    list of ETH_NAMES actually used (with 'other' always last when pooled).
    """
    counts = np.bincount(group_idx, minlength=len(ETH_NAMES))
    keep = [i for i in range(len(ETH_NAMES)) if counts[i] >= MIN_SAMPLES]
    if len(keep) == len(ETH_NAMES):
        return group_idx, ETH_NAMES

    other_idx = ETH_NAMES.index("other") if "other" in ETH_NAMES else len(ETH_NAMES) - 1
    pooled = group_idx.copy()
    for i in range(len(ETH_NAMES)):
        if counts[i] < MIN_SAMPLES and i != other_idx:
            pooled[pooled == i] = other_idx
    names = [ETH_NAMES[i] for i in sorted(set(keep) | {other_idx})]
    return pooled, names


def _group_counts(y_true, y_pred, group_idx, names):
    """Per-group TP/FP/FN/TN + recall/precision/f1/support."""
    out: dict[str, dict] = {}
    for gi, name in enumerate(names):
        mask = group_idx == gi
        n = int(mask.sum())
        yt = y_true[mask]
        yp = y_pred[mask]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        out[name] = {
            "support": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": recall, "precision": precision, "f1": f1,
            "tpr": recall, "fpr": (fp / (fp + tn)) if (fp + tn) > 0 else float("nan"),
        }
    return out


def _eod_aod(group_metrics, names):
    """EOD = max recall - min recall; AOD = mean(|dTPR|+|dFPR|)/2 vs ref group."""
    recalls = [group_metrics[n]["recall"] for n in names
               if group_metrics[n]["recall"] == group_metrics[n]["recall"]]
    if not recalls:
        return float("nan"), float("nan")
    eod = float(max(recalls) - min(recalls))

    # reference group = the one with the most support
    ref = max(names, key=lambda n: group_metrics[n]["support"])
    aod_terms = []
    for n in names:
        if n == ref:
            continue
        dtpr = abs(group_metrics[n]["tpr"] - group_metrics[ref]["tpr"])
        dfpr = abs(group_metrics[n]["fpr"] - group_metrics[ref]["fpr"])
        if dtpr == dtpr and dfpr == dfpr:  # both non-nan
            aod_terms.append((dtpr + dfpr) / 2.0)
    aod = float(np.mean(aod_terms)) if aod_terms else float("nan")
    return eod, aod


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
# CSV + matplotlib (per dev doc: per-epoch CSV + two time-series plots)
# ---------------------------------------------------------------------------
def _write_csv(threshold_used: float) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["epoch,group,recall,precision,f1,n"]
    for row in _GROUP_HISTORY:
        e = row["epoch"]
        for name in row["names"]:
            g = row["groups"][name]
            lines.append(
                f"{e},{name},{g['recall']:.6f},{g['precision']:.6f},"
                f"{g['f1']:.6f},{g['support']}"
            )
        lines.append(f"{e},__summary__,EOD={row['eod']:.6f},AOD={row['aod']:.6f},"
                     f"threshold={threshold_used:.4f},ngroups={len(row['names'])}")
    CSV_PATH.write_text("\n".join(lines) + "\n")


def _save_matplotlib_summary():
    """Per-group recall curves + EOD-over-epochs -> logs/fairness_eod.png."""
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    all_names = sorted({n for r in _GROUP_HISTORY for n in r["names"]})
    for name in all_names:
        ys = [r["groups"].get(name, {}).get("recall", float("nan"))
              for r in _GROUP_HISTORY]
        ys = [y if y == y else None for y in ys]
        if any(y is not None for y in ys):
            ax1.plot(epochs, ys, marker="o", label=f"{name} recall")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Recall")
    ax1.set_title("Per-ethnicity recall at shared threshold")
    ax1.legend(fontsize=8)

    eod_ys = [r["eod"] for r in _GROUP_HISTORY]
    ax2.plot(epochs, eod_ys, marker="s", color="black", label="EOD")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("EOD")
    ax2.set_title("Equal-opportunity difference over epochs")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PNG_PATH)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def record(epoch, model, loader, device,
           standard_threshold: float = 0.05,
           acceptable_threshold: float = 0.10):
    """Compute and store the EOD for this epoch at the shared threshold."""
    global _SHARED_THRESHOLD
    y_true, y_score, group_idx = _collect(model, loader, device)

    if _SHARED_THRESHOLD is None:
        _SHARED_THRESHOLD = _pick_threshold(y_true, y_score)
        logger.info("[probe] shared threshold fixed at %.4f (epoch %d)",
                   _SHARED_THRESHOLD, epoch)
    t = _SHARED_THRESHOLD
    y_pred = (y_score >= t).astype(int)

    pooled_idx, names = _pool_small_groups(y_true, y_score, group_idx)
    groups = _group_counts(y_true, y_pred, pooled_idx, names)
    eod, aod = _eod_aod(groups, names)

    _SERIES.append((int(epoch), float(eod)))
    _GROUP_HISTORY.append({
        "epoch": int(epoch), "eod": float(eod), "aod": float(aod),
        "threshold": float(t), "names": names, "groups": groups,
    })

    level = logging.WARNING if (eod == eod and eod > acceptable_threshold) else logging.INFO
    logger.log(level,
               "[probe] epoch %d EOD=%.4f AOD=%.4f (t=%.3f, groups=%d)",
               epoch, eod, aod, t, len(names))

    _write_live(standard_threshold, acceptable_threshold)
    _write_csv(t)
    return eod


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
    acceptable_met = bool(_satisfies(tail_mean, acceptable_threshold))

    cross_ep = _first_cross_epoch(_SERIES, standard_threshold)
    trend = "improving" if delta < 0 else ("flat" if delta == 0 else "worsening")
    cross_txt = (f", first reaching the {standard_threshold} standard threshold at epoch {cross_ep}"
                 if cross_ep is not None else
                 f", never reaching the {standard_threshold} standard threshold")
    conclusion = (
        f"Equal-opportunity difference (lower is better) went from {first_value:.4f} "
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
        "acceptable_met": acceptable_met,
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
                  annotation_text=f"standard={std_thr}",
                  annotation_position="top right")
    fig.add_hline(y=acc_thr, line=dict(color="orange", dash="dash"),
                  annotation_text=f"acceptable={acc_thr}",
                  annotation_position="bottom right")
    if cross_ep is not None:
        fig.add_vline(x=cross_ep, line=dict(color="gray", dash="dot"),
                      annotation_text=f"crossed @ {cross_ep}",
                      annotation_position="top left")

    trend = ("improving" if result["delta"] < 0
             else ("flat" if result["delta"] == 0 else "worsening"))
    textbox = (
        f"min={result['min']:.4f}  max={result['max']:.4f}<br>"
        f"mean={result['mean']:.4f}  std={result['std']:.4f}<br>"
        f"delta={result['delta']:.4f}  trend={trend}<br>"
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
