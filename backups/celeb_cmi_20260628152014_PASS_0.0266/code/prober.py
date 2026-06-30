"""Training-quality probe: residual shortcut leakage via conditional mutual information.

Metric (per epoch): mean normalized conditional mutual information
    I(pred_target ; Male | true_target)
across a fixed set of gender-correlated CelebA attribute pairs on the validation
split. Lower is better — near 0 means once the true attribute is known, the model's
prediction carries no extra information about the spurious proxy (gender).
"""

import json
import os
import tempfile

import numpy as np
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import mutual_info_score

from utils.constant import ATTRIBUTES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORKING_SPACE = os.path.dirname(os.path.abspath(__file__))
_PROBE_ROOT = os.path.join(WORKING_SPACE, ".agent_probe")
_LIVE_PATH = os.path.join(_PROBE_ROOT, "live", "probe_live.json")
_METRIC_DIR = os.path.join(_PROBE_ROOT, "metric")
_PLOT_DIR = os.path.join(_PROBE_ROOT, "plot")

METRIC_NAME = "mean_normalized_conditional_MI"
DIRECTION = "lower_is_better"

_NAME_TO_IDX = {v: k for k, v in ATTRIBUTES.items()}
_SPURIOUS = "Male"
# Standard CelebA gender-correlated targets paired against the Male proxy.
_TARGET_ATTRS = [
    "Heavy_Makeup",
    "Wearing_Lipstick",
    "Attractive",
    "Wavy_Hair",
    "No_Beard",
    "Arched_Eyebrows",
]
_PAIRS = [(t, _SPURIOUS) for t in _TARGET_ATTRS]

# In-memory metric series: list of (epoch, value)
_series = []


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def _entropy(labels):
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def _conditional_mi(pred_t, y_s, y_t):
    """I(pred_t ; y_s | y_t): stratify on y_t, weighted-average within-stratum MI."""
    n = len(y_t)
    if n == 0:
        return 0.0
    total = 0.0
    for v in np.unique(y_t):
        mask = y_t == v
        w = mask.sum() / n
        if mask.sum() < 2:
            continue
        total += w * mutual_info_score(pred_t[mask], y_s[mask])
    return total


def _compute_metric(preds, labels):
    """preds, labels: (N, 40) binary arrays. Returns mean normalized conditional MI."""
    preds = np.asarray(preds).astype(int)
    labels = np.asarray(labels).astype(int)
    s_idx = _NAME_TO_IDX[_SPURIOUS]
    y_s = labels[:, s_idx]
    h_s = _entropy(y_s)
    scores = []
    for t_name, _ in _PAIRS:
        t_idx = _NAME_TO_IDX[t_name]
        pred_t = preds[:, t_idx]
        y_t = labels[:, t_idx]
        cmi = _conditional_mi(pred_t, y_s, y_t)
        norm = cmi / h_s if h_s > 1e-12 else 0.0
        scores.append(norm)
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Live trajectory writer
# ---------------------------------------------------------------------------
def _write_live(standard_threshold, acceptable_threshold):
    os.makedirs(os.path.dirname(_LIVE_PATH), exist_ok=True)
    payload = {
        "metric_name": METRIC_NAME,
        "standard_threshold": float(standard_threshold),
        "acceptable_threshold": float(acceptable_threshold),
        "direction": DIRECTION,
        "values": [{"epoch": int(e), "value": float(v)} for e, v in _series],
    }
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(_LIVE_PATH), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, _LIVE_PATH)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# Thresholds carried for the live writer (set when probe is constructed).
_STD_TH = 0.03
_ACC_TH = 0.07


def record(epoch, preds, labels):
    """Called once per epoch with validation binary predictions and true labels.

    preds, labels: array-like of shape (N, n_attributes).
    """
    value = _compute_metric(preds, labels)
    _series.append((int(epoch), float(value)))
    _write_live(_STD_TH, _ACC_TH)
    return value


# ---------------------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------------------
def _next_index(directory):
    os.makedirs(directory, exist_ok=True)
    n = 0
    for fn in os.listdir(directory):
        if fn.startswith("probe_result_") and fn.endswith(".json"):
            try:
                n = max(n, int(fn[len("probe_result_"):-len(".json")]))
            except ValueError:
                pass
    return n + 1


def _satisfies(value, threshold):
    return value <= threshold if DIRECTION == "lower_is_better" else value >= threshold


def conclude(standard_threshold, acceptable_threshold):
    standard_threshold = float(standard_threshold)
    acceptable_threshold = float(acceptable_threshold)

    epochs = [e for e, _ in _series]
    values = [v for _, v in _series]
    if not values:
        values = [0.0]
        epochs = [0]

    arr = np.asarray(values, dtype=float)
    vmin, vmax = float(arr.min()), float(arr.max())
    vmean, vstd = float(arr.mean()), float(arr.std())
    first_value, final_value = float(values[0]), float(values[-1])
    delta = final_value - first_value
    tail = values[-5:] if len(values) >= 5 else values
    tail_mean = float(np.mean(tail))

    status = "PASS" if _satisfies(tail_mean, standard_threshold) else "FAIL"
    acceptable_met = bool(_satisfies(tail_mean, acceptable_threshold))

    # First epoch crossing the standard threshold (lower_is_better => value <= th)
    cross_epoch = None
    for e, v in zip(epochs, values):
        if _satisfies(v, standard_threshold):
            cross_epoch = e
            break

    trend = "improving" if delta < 0 else ("degrading" if delta > 0 else "flat")
    if cross_epoch is not None:
        cross_txt = f"crossing the {standard_threshold} standard threshold at epoch {cross_epoch}"
    else:
        cross_txt = f"never crossing the {standard_threshold} standard threshold"
    conclusion = (
        f"{METRIC_NAME} went from {first_value:.4f} to {final_value:.4f} "
        f"(tail mean {tail_mean:.4f}), {cross_txt}; status {status}."
    )

    n = _next_index(_METRIC_DIR)
    result = {
        "metric_name": METRIC_NAME,
        "standard_threshold": standard_threshold,
        "acceptable_threshold": acceptable_threshold,
        "direction": DIRECTION,
        "values": [{"epoch": int(e), "value": float(v)} for e, v in zip(epochs, values)],
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

    os.makedirs(_METRIC_DIR, exist_ok=True)
    json_path = os.path.join(_METRIC_DIR, f"probe_result_{n}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    _save_plot(n, epochs, values, standard_threshold, acceptable_threshold,
               status, acceptable_met, vmin, vmax, vmean, vstd, delta, trend,
               cross_epoch)

    return result


def _save_plot(n, epochs, values, standard_threshold, acceptable_threshold,
               status, acceptable_met, vmin, vmax, vmean, vstd, delta, trend,
               cross_epoch):
    import plotly.graph_objects as go

    os.makedirs(_PLOT_DIR, exist_ok=True)

    line_color = "green" if status == "PASS" else "red"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=values, mode="lines+markers",
        name=METRIC_NAME, line=dict(color=line_color, width=2),
    ))

    fig.add_hline(y=standard_threshold, line=dict(color="red", dash="dash"),
                  annotation_text=f"standard={standard_threshold}",
                  annotation_position="top right")
    fig.add_hline(y=acceptable_threshold, line=dict(color="orange", dash="dash"),
                  annotation_text=f"acceptable={acceptable_threshold}",
                  annotation_position="bottom right")

    if cross_epoch is not None:
        fig.add_vline(x=cross_epoch, line=dict(color="gray", dash="dot"),
                      annotation_text=f"cross @ {cross_epoch}",
                      annotation_position="top left")

    stats_txt = (
        f"min={vmin:.4f}  max={vmax:.4f}<br>"
        f"mean={vmean:.4f}  std={vstd:.4f}<br>"
        f"delta={delta:.4f}  trend={trend}<br>"
        f"status={status}  acceptable_met={acceptable_met}"
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.02, showarrow=False,
        align="left", text=stats_txt,
        bordercolor="black", borderwidth=1, bgcolor="white", opacity=0.85,
    )

    # Fixed y-range for comparability across iterations: includes both thresholds
    # plus headroom for future runs that may leak more.
    y_top = max(0.20, acceptable_threshold, standard_threshold, max(values) if values else 0)
    fig.update_yaxes(range=[-0.01, y_top * 1.1])

    fig.update_layout(
        title=METRIC_NAME,
        xaxis_title="Epoch",
        yaxis_title=METRIC_NAME,
    )

    pdf_path = os.path.join(_PLOT_DIR, f"probe_result_{n}.pdf")
    fig.write_image(pdf_path)


# ---------------------------------------------------------------------------
# Lightning integration callback
# ---------------------------------------------------------------------------
class ProbeCallback(Callback):
    """Accumulates validation predictions/labels each epoch and calls record()."""

    def __init__(self):
        self._preds = []
        self._labels = []

    def on_validation_epoch_start(self, trainer, pl_module):
        self._preds = []
        self._labels = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx=0):
        import torch
        import torch.nn as nn
        _, y = batch
        logits = outputs["logits"].detach().cpu()
        preds = torch.round(nn.Sigmoid()(logits))
        self._preds.append(preds.numpy())
        self._labels.append(y.detach().cpu().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self._preds:
            return
        preds = np.concatenate(self._preds, axis=0)
        labels = np.concatenate(self._labels, axis=0)
        record(trainer.current_epoch, preds, labels)
