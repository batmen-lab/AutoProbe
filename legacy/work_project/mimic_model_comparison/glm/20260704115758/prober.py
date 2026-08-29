"""Training-quality probe: equal-opportunity TPR gap across ethnicity groups.

Metric (lower is better):
    gap = max_g TPR_g - min_g TPR_g   over non-skipped ethnicity groups
    where TPR_g = TP_g / P_g  (P_g = count of death_label==1 in group g)

Groups with P_g < 20 are skipped (small-denominator guard).

The decision threshold is HARD-CODED as a module-level literal (THRESHOLD = 0.5)
and is NOT exposed as a function parameter, config field, or CLI flag. This is
the frozen operating point — the fix-loop is not allowed to move it.

Public API:
    record(epoch, model, loader, device, anchor_metrics=None,
           standard_threshold=0.05, acceptable_threshold=0.10)
        Called once per epoch. Computes the TPR gap on the validation loader,
        appends to the in-memory series, overwrites the live trajectory JSON,
        appends a CSV row, and prints a one-line stdout summary.

    conclude(standard_threshold, acceptable_threshold)
        Called once after training. Writes probe_result_N.json, the Plotly PDF
        chart, and a matplotlib summary PNG.
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
# Frozen operating point — DO NOT make this tunable anywhere.
# A code reviewer or static check must be able to confirm it is not exposed
# as a function parameter, config field, or CLI flag.
# ---------------------------------------------------------------------------
THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKING_SPACE = Path(__file__).resolve().parent
PROBE_DIR = WORKING_SPACE / ".agent_probe"
LIVE_DIR = PROBE_DIR / "live"
METRIC_DIR = PROBE_DIR / "metric"
PLOT_DIR = PROBE_DIR / "plot"

LOG_DIR = WORKING_SPACE / "logs"
CSV_PATH = LOG_DIR / "fairness_tpr_gap.csv"
PNG_PATH = LOG_DIR / "fairness_tpr_gap.png"

METRIC_NAME = "equal_opportunity_tpr_gap"
DIRECTION = "lower_is_better"

ETH_NAMES = ["white", "black", "hispanic", "asian", "other"]
MIN_POSITIVES = 20  # small-denominator guard: skip groups with P_g < 20

# In-memory series: list of (epoch, gap)
_SERIES: list[tuple[int, float]] = []
# Per-epoch per-group detail for the matplotlib plot / CSV
_GROUP_HISTORY: list[dict] = []
# Last-seen anchor metrics (train.py's own final-epoch loss/eval). conclude()
# emits these as original_train_metric_* alongside the probe fields.
_LAST_ANCHORS: list[dict] = []


# ---------------------------------------------------------------------------
# Eval pass
# ---------------------------------------------------------------------------
@torch.no_grad()
def _collect(model, loader, device):
    """Return (y_true, y_pred, group_idx) arrays aligned across the loader.

    The loader is assumed unshuffled so accumulated predictions line up with
    ``loader.dataset.eth`` row-for-row. y_pred is the 0/1 decision at the
    frozen THRESHOLD (not the sigmoid score) — the metric is defined at the
    operating point, not as a rank-quality measure.
    """
    model.eval()
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for batch in loader:
        features = batch["features"].to(device)
        logits = model(features)
        probs_list.append(logits.sigmoid().cpu().numpy())
        labels_list.append(batch["label"].cpu().numpy())

    y_score = np.concatenate(probs_list)
    y_true = np.concatenate(labels_list).astype(int)
    y_pred = (y_score >= THRESHOLD).astype(int)

    eth = loader.dataset.eth.cpu().numpy()
    group_idx = eth.argmax(axis=1)[: len(y_true)]
    return y_true, y_pred, group_idx


def _per_group_tpr(y_true, y_pred, group_idx):
    """Compute TPR_g = TP_g / P_g for each ethnicity group. Skip groups with
    P_g < MIN_POSITIVES (small-denominator guard). Returns dict keyed by name
    plus the overall TPR and the max deviation from overall."""
    out: dict[str, dict] = {}
    for gi, name in enumerate(ETH_NAMES):
        mask = group_idx == gi
        n = int(mask.sum())
        p_g = int(y_true[mask].sum())  # positives in group
        tp_g = int(((y_pred[mask] == 1) & (y_true[mask] == 1)).sum())
        entry = {"support": n, "n_pos": p_g, "TP": tp_g, "tpr": None}
        if p_g >= MIN_POSITIVES:
            entry["tpr"] = float(tp_g / p_g)
        else:
            logger.warning(
                "[probe] group '%s' skipped: P_g=%d < MIN_POSITIVES=%d",
                name, p_g, MIN_POSITIVES,
            )
        out[name] = entry

    # overall TPR across the full validation set
    p_all = int(y_true.sum())
    tp_all = int(((y_pred == 1) & (y_true == 1)).sum())
    tpr_all = float(tp_all / p_all) if p_all > 0 else float("nan")
    return out, tpr_all


def _gap_and_maxdev(per_group: dict, tpr_all: float) -> tuple[float, float]:
    tprs = [per_group[g]["tpr"] for g in ETH_NAMES if per_group[g]["tpr"] is not None]
    if not tprs:
        return float("nan"), float("nan")
    gap = float(max(tprs) - min(tprs))
    max_dev = float(max(abs(t - tpr_all) for t in tprs)) if tpr_all == tpr_all else float("nan")
    return gap, max_dev


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
    cols = ["epoch",
            "tpr_white", "tpr_black", "tpr_hispanic", "tpr_asian", "tpr_other",
            "tpr_all", "gap", "max_dev_from_overall"]
    lines = [",".join(cols)]
    for row in _GROUP_HISTORY:
        vals = [row["epoch"]]
        for g in ETH_NAMES:
            t = row["groups"][g]["tpr"]
            vals.append("" if t is None else f"{t:.6f}")
        vals.append(f"{row['tpr_all']:.6f}")
        vals.append(f"{row['gap']:.6f}")
        vals.append(f"{row['max_dev']:.6f}")
        lines.append(",".join(str(v) for v in vals))
    CSV_PATH.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def record(epoch, model, loader, device,
           anchor_metrics: list[dict] | None = None,
           standard_threshold: float = 0.05,
           acceptable_threshold: float = 0.10):
    """Compute and store the equal-opportunity TPR gap for this epoch.

    Args:
        epoch: 1-indexed epoch number.
        model, loader, device: model + unshuffled validation loader.
        anchor_metrics: optional list of train.py's own final-epoch metrics,
            each {"name": str, "value": float, "direction": str}. The last
            call's values are emitted by conclude() as original_train_metric_*.
        standard_threshold / acceptable_threshold: only used for the live JSON
            header. The frozen THRESHOLD constant is what defines the operating
            point and is not affected by these.
    """
    y_true, y_pred, group_idx = _collect(model, loader, device)
    per_group, tpr_all = _per_group_tpr(y_true, y_pred, group_idx)
    gap, max_dev = _gap_and_maxdev(per_group, tpr_all)

    _SERIES.append((int(epoch), float(gap)))
    _GROUP_HISTORY.append({
        "epoch": int(epoch), "groups": per_group,
        "tpr_all": tpr_all, "gap": gap, "max_dev": max_dev,
    })

    if anchor_metrics is not None:
        _LAST_ANCHORS.clear()
        _LAST_ANCHORS.extend(anchor_metrics)

    # Stdout summary — exactly the format requested in the dev document.
    print(f"fairness_tpr_gap epoch={epoch} gap={gap:.3f} "
          f"tpr_all={tpr_all:.3f} max_dev={max_dev:.3f}")

    level = logging.WARNING if (gap == gap and gap > acceptable_threshold) else logging.INFO
    logger.log(level, "[probe] epoch %d gap=%.4f tpr_all=%.4f max_dev=%.4f",
               epoch, gap, tpr_all, max_dev)

    _write_live(standard_threshold, acceptable_threshold)
    _write_csv()
    return gap


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

    # Anchor(s): train.py's own final-epoch loss/eval, recorded ALONGSIDE the
    # probe metric. If more than one anchor is present, emit one
    # original_train_metric_{i} field per anchor (eval/selection first, loss
    # next). If exactly one, emit original_train_metric. Zero -> omit.
    if len(_LAST_ANCHORS) == 1:
        result["original_train_metric"] = _LAST_ANCHORS[0]
    else:
        for i, a in enumerate(_LAST_ANCHORS):
            result[f"original_train_metric_{i}"] = a

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
    """Fixed y-range covering both thresholds + headroom for comparability."""
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


# ---------------------------------------------------------------------------
# Matplotlib summary: gap on primary axis, per-group TPR on secondary axis,
# plus two horizontal dashed threshold lines.
# ---------------------------------------------------------------------------
def _save_matplotlib_summary(standard_threshold, acceptable_threshold):
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

    fig, ax1 = plt.subplots(figsize=(9, 5))
    gaps = [r["gap"] for r in _GROUP_HISTORY]
    ax1.plot(epochs, gaps, marker="s", color="black", linestyle="-",
             label="TPR gap (max-min)", linewidth=2)
    ax1.axhline(standard_threshold, color="red", linestyle="--", linewidth=1,
                label=f"standard={standard_threshold}")
    ax1.axhline(acceptable_threshold, color="orange", linestyle="--", linewidth=1,
                label=f"acceptable={acceptable_threshold}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("TPR gap (max - min)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    ax2 = ax1.twinx()
    for g in ETH_NAMES:
        ys = [r["groups"][g]["tpr"] for r in _GROUP_HISTORY]
        if any(y is not None for y in ys):
            ax2.plot(epochs, ys, marker="o", linestyle=":", alpha=0.7,
                     label=f"{g} TPR")
    ax2.set_ylabel("Per-group TPR", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    ax1.set_title("Equal-opportunity TPR gap + per-group TPR over epochs")
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=130)
    plt.close(fig)
