"""Probe: gender-conditional leakage on the CelebA multi-attribute classifier.

Metric — mean over ATTRS of the normalized conditional mutual information
    I(pred_a ; Male | y_a_true) / H(pred_a | y_a_true)
computed on the validation set at the end of every epoch. Lower is better.

Public entry points:
    compute_conditional_leakage(model, val_loader, device, ...)
        -> (mean_leakage, per_attr_dict)
    record(epoch, value, anchor=None)
    conclude(standard_threshold, acceptable_threshold, final_anchor=None)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATTRS: List[str] = [
    "Heavy_Makeup", "Wearing_Lipstick", "Attractive",
    "Wavy_Hair", "No_Beard", "Arched_Eyebrows",
]
# CelebA column indices for the attrs above (matches utils.constant.ATTRIBUTES).
ATTR_IDX: List[int] = [18, 36, 2, 33, 24, 1]
MALE_IDX: int = 20

METRIC_NAME = "mean_gender_conditional_leakage"
DIRECTION = "lower_is_better"

_WORKING_SPACE = Path(__file__).resolve().parent
_LIVE_PATH = _WORKING_SPACE / ".agent_probe" / "live" / "probe_live.json"
_METRIC_DIR = _WORKING_SPACE / ".agent_probe" / "metric"
_PLOT_DIR = _WORKING_SPACE / ".agent_probe" / "plot"
_LEAKAGE_CSV = _WORKING_SPACE / ".agent_probe" / "leakage.csv"

# Fixed y-axis range (comparability across rounds). Includes both thresholds
# with generous headroom for ERM-style leakage (~0.15-0.35) and a small floor
# below zero for visual breathing room.
_Y_RANGE = (-0.02, 0.60)


# In-memory series + caches filled by record().
_SERIES: list[tuple[int, float]] = []
_ANCHORS: dict[str, dict] = {}
_PER_ATTR: dict[str, float] = {}
_STD_THR: float = 0.05
_ACC_THR: float = 0.15


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


def _entropy(counts: np.ndarray) -> float:
    p = counts.astype(float)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p[p > 0] / s
    return float(-(p * np.log(p)).sum())


def _mutual_info(x: np.ndarray, y: np.ndarray) -> float:
    """Discrete MI(x; y) in nats for two binary/int arrays."""
    if len(x) == 0:
        return 0.0
    xs = np.unique(x); ys = np.unique(y)
    n = len(x)
    mi = 0.0
    for xv in xs:
        px = np.mean(x == xv)
        for yv in ys:
            py = np.mean(y == yv)
            pxy = np.mean((x == xv) & (y == yv))
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * np.log(pxy / (px * py))
    return float(mi)


def _conditional_leakage(pred_a: np.ndarray, male: np.ndarray, y_a: np.ndarray) -> float:
    """Normalized conditional MI  I(pred_a ; Male | y_a) / H(pred_a | y_a)."""
    pred_a = pred_a.astype(int); male = male.astype(int); y_a = y_a.astype(int)
    n = len(y_a)
    if n == 0:
        return 0.0
    cmi = 0.0; cond_H = 0.0
    for v in np.unique(y_a):
        m = y_a == v
        w = m.mean()
        if m.sum() < 2:
            continue
        cmi += w * _mutual_info(pred_a[m], male[m])
        _, c = np.unique(pred_a[m], return_counts=True)
        cond_H += w * _entropy(c)
    return float(cmi / cond_H) if cond_H > 1e-12 else 0.0


def compute_conditional_leakage(model, val_loader, device, max_batches=None):
    """Return (mean_leakage, per_attr_dict): mean over ATTRS of the normalized
    conditional MI I(pred_a ; Male | y_a) / H(pred_a | y_a) on the val split."""
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if max_batches is not None and bi >= max_batches:
                break
            x, y = batch[0], batch[1]
            logits = model(x.to(device))
            probs_all.append(torch.sigmoid(logits).float().cpu().numpy())
            labels_all.append(np.asarray(y))
    P = np.concatenate(probs_all); Y = np.concatenate(labels_all).astype(int)
    male = Y[:, MALE_IDX]
    per_attr: dict[str, float] = {}
    for name, ti in zip(ATTRS, ATTR_IDX):
        pred_a = (P[:, ti] >= 0.5).astype(int)
        per_attr[name] = round(_conditional_leakage(pred_a, male, Y[:, ti]), 6)
    mean_leak = float(np.mean(list(per_attr.values()))) if per_attr else 0.0
    return mean_leak, per_attr


def record(epoch, value, anchor=None, per_attr=None,
           standard_threshold=None, acceptable_threshold=None):
    global _STD_THR, _ACC_THR
    _SERIES.append((int(epoch), float(value)))
    if anchor:
        _ANCHORS.update(anchor)
    if per_attr:
        _PER_ATTR.clear(); _PER_ATTR.update(per_attr)
    if standard_threshold is not None:
        _STD_THR = float(standard_threshold)
    if acceptable_threshold is not None:
        _ACC_THR = float(acceptable_threshold)
    _LIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(_LIVE_PATH, {
        "metric_name": METRIC_NAME, "direction": DIRECTION,
        "standard_threshold": _STD_THR, "acceptable_threshold": _ACC_THR,
        "values": [{"epoch": e, "value": v} for e, v in _SERIES],
        "per_attr": _PER_ATTR,
    })
    try:
        _LEAKAGE_CSV.parent.mkdir(parents=True, exist_ok=True)
        header = not _LEAKAGE_CSV.exists()
        with open(_LEAKAGE_CSV, "a") as f:
            if header:
                f.write("epoch,mean_leakage," + ",".join(ATTRS) + "\n")
            f.write(",".join([str(int(epoch)), f"{float(value):.6f}"] +
                             [f"{_PER_ATTR.get(a, float('nan')):.6f}" for a in ATTRS]) + "\n")
    except Exception:
        pass
    return float(value)


def _next_metric_index() -> int:
    _METRIC_DIR.mkdir(parents=True, exist_ok=True)
    idxs = []
    for p in _METRIC_DIR.glob("probe_result_*.json"):
        try:
            idxs.append(int(p.stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    return (max(idxs) + 1) if idxs else 1


def conclude(standard_threshold, acceptable_threshold, final_anchor=None):
    vals = [v for _e, v in _SERIES] or [0.0]
    arr = np.asarray(vals, dtype=float)
    tail_mean = float(arr[-5:].mean())
    meets = tail_mean <= standard_threshold
    acc = tail_mean <= acceptable_threshold
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
        "delta": final_value - first_value, "tail_mean": tail_mean,
        "status": status, "acceptable_met": bool(acc),
        "conclusion": (f"{METRIC_NAME} moved from {first_value:.4f} to {final_value:.4f}; "
                       f"tail_mean {tail_mean:.4f} {'met' if meets else 'did not meet'} the "
                       f"{standard_threshold} standard threshold, status={status}."),
    }
    anchors_list = final_anchor if final_anchor else list(_ANCHORS.values() if isinstance(_ANCHORS, dict) else [])
    if isinstance(_ANCHORS, dict) and not final_anchor:
        anchors_list = []
        for k, v in _ANCHORS.items():
            if isinstance(v, dict):
                anchors_list.append(v)
            else:
                dirn = "higher_is_better" if "f1" in k.lower() else "lower_is_better"
                anchors_list.append({"name": k, "value": float(v), "direction": dirn})
    for i, a in enumerate(anchors_list[:2]):
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
        ax.set_ylim(*_Y_RANGE); ax.set_xlabel("epoch"); ax.set_ylabel(METRIC_NAME); ax.legend()
        _PLOT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(_PLOT_DIR / f"probe_result_{idx}.pdf"); plt.close(fig)
    except Exception:
        pass
    return doc
