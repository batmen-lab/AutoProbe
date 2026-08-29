"""Equal-opportunity recall-gap probe for MIMIC-III mortality.

Metric: delta_recall = max_g TPR_g - min_g TPR_g over the 5 ethnicity buckets
{white, black, hispanic, asian, other}, computed on the validation split at a
shared operating threshold (0.5) each epoch. Direction: lower_is_better.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)

METRIC_NAME = "delta_recall"
DIRECTION = "lower_is_better"
SHARED_THRESHOLD = 0.5
MIN_POS_PER_GROUP = 20
ETH_NAMES = ["white", "black", "hispanic", "asian", "other"]

_WORKSPACE = Path(__file__).resolve().parent
_PROBE_DIR = _WORKSPACE / ".agent_probe"
_LIVE_DIR = _PROBE_DIR / "live"
_METRIC_DIR = _PROBE_DIR / "metric"
_PLOT_DIR = _PROBE_DIR / "plot"

# Fixed y-axis range for cross-run comparability. Includes both thresholds
# (0.10 standard, 0.15 acceptable) and leaves ample headroom.
_Y_MIN = 0.0
_Y_MAX = 0.5

# In-memory series and anchor cache filled by record().
_SERIES: list[tuple[int, float]] = []
_ANCHORS: dict[str, dict] = {}


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _per_group_recall(
    probs: np.ndarray,
    labels: np.ndarray,
    group: np.ndarray,
    threshold: float = SHARED_THRESHOLD,
) -> dict[str, float | None]:
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)
    out: dict[str, float | None] = {}
    for gi, name in enumerate(ETH_NAMES):
        mask = group == gi
        pos = mask & (labels == 1)
        n_pos = int(pos.sum())
        if n_pos < MIN_POS_PER_GROUP:
            out[name] = None
            continue
        tp = int((preds[pos] == 1).sum())
        out[name] = tp / n_pos
    return out


def record(epoch, probs, labels, eth, anchors=None):
    """Record one epoch's per-ethnicity recall gap (delta_recall)."""
    probs = np.asarray(probs).ravel().astype(float)
    labels = np.asarray(labels).ravel().astype(int)
    eth = np.asarray(eth)
    group = eth.argmax(axis=1) if eth.ndim == 2 else eth.astype(int)
    per_group = _per_group_recall(probs, labels, group)
    valid = [v for v in per_group.values() if v is not None]
    delta = float(max(valid) - min(valid)) if len(valid) >= 2 else 0.0
    _SERIES.append((int(epoch), delta))
    if anchors:
        _ANCHORS.update(anchors)
    _LIVE_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(_LIVE_DIR / "probe_live.json", {
        "metric_name": METRIC_NAME, "direction": DIRECTION,
        "values": [{"epoch": e, "value": v} for e, v in _SERIES],
        "per_group_recall": per_group,
    })
    try:
        _METRIC_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = _PROBE_DIR / "fairness_recall_gap.csv"
        header = not csv_path.exists()
        with open(csv_path, "a") as f:
            if header:
                f.write("epoch,thr," + ",".join("tpr_" + n for n in ETH_NAMES) + ",delta_recall\n")
            row = [str(int(epoch)), str(SHARED_THRESHOLD)]
            row += ["" if per_group[n] is None else f"{per_group[n]:.6f}" for n in ETH_NAMES]
            row += [f"{delta:.6f}"]
            f.write(",".join(row) + "\n")
    except Exception:
        pass
    logger.info("[probe] epoch %d delta_recall=%.4f", int(epoch), delta)
    return delta


def _next_metric_index() -> int:
    _METRIC_DIR.mkdir(parents=True, exist_ok=True)
    idxs = []
    for p in _METRIC_DIR.glob("probe_result_*.json"):
        try:
            idxs.append(int(p.stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return (max(idxs) + 1) if idxs else 1


def conclude(standard_threshold, acceptable_threshold):
    """Summarise the run: tail-mean vs thresholds, write probe_result_N.json + plot."""
    vals = [v for _e, v in _SERIES] or [0.0]
    arr = np.asarray(vals, dtype=float)
    tail_mean = float(arr[-5:].mean())
    lower_better = DIRECTION == "lower_is_better"
    meets = (tail_mean <= standard_threshold) if lower_better else (tail_mean >= standard_threshold)
    acc = (tail_mean <= acceptable_threshold) if lower_better else (tail_mean >= acceptable_threshold)
    status = "PASS" if meets else "FAIL"
    first_value, final_value = float(arr[0]), float(arr[-1])
    doc = {
        "metric_name": METRIC_NAME,
        "standard_threshold": float(standard_threshold),
        "acceptable_threshold": float(acceptable_threshold),
        "direction": DIRECTION,
        "values": [{"epoch": e, "value": v} for e, v in _SERIES],
        "min": float(arr.min()), "max": float(arr.max()),
        "mean": float(arr.mean()), "std": float(arr.std()),
        "first_value": first_value, "final_value": final_value,
        "delta": final_value - first_value,
        "tail_mean": tail_mean,
        "status": status, "acceptable_met": bool(acc),
        "conclusion": (f"{METRIC_NAME} moved from {first_value:.4f} to {final_value:.4f}; "
                       f"tail_mean {tail_mean:.4f} {'met' if meets else 'did not meet'} the "
                       f"{standard_threshold} standard threshold, status={status}."),
    }
    for i, key in enumerate(("val_auprc", "train_loss")):
        a = _ANCHORS.get(key)
        if a:
            doc[f"original_train_metric_{i}"] = {"name": a["name"], "value": float(a["value"]), "direction": a["direction"]}
    idx = _next_metric_index()
    _atomic_write_json(_METRIC_DIR / f"probe_result_{idx}.json", doc)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([e for e, _ in _SERIES], [v for _, v in _SERIES], marker="o")
        ax.axhline(standard_threshold, ls="--", color="green", label="standard")
        ax.axhline(acceptable_threshold, ls="--", color="orange", label="acceptable")
        ax.set_ylim(_Y_MIN, _Y_MAX); ax.set_xlabel("epoch"); ax.set_ylabel(METRIC_NAME); ax.legend()
        _PLOT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(_PLOT_DIR / f"probe_result_{idx}.pdf"); plt.close(fig)
    except Exception:
        pass
    return doc
