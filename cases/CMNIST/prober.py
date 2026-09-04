"""Probe for the CMNIST case — per-epoch worst-group accuracy tracker.

Stage-3 contract: this file lives in the workspace root alongside train.py and
exposes two entry points that train.py calls:

    record(epoch, wg, overall, adjusted)   # once per epoch, after validation
    conclude(standard_threshold, acceptable_threshold)  # once, after training

The tracked metric is worst-group accuracy — `val["min_group"]["accuracy"]`,
the min over the four (y, a) groups. The orchestrator's decision value is the
tail-mean of this series (mean of the last 5 epochs), gated against the frozen
thresholds below (0.68 standard / 0.52 acceptable). Those thresholds are the
measured operating point of the probe and are HARD-CODED here — they must NOT
be moved by any fix-loop edit to train.py, because the fix-loop can never touch
prober.py.

The anchor (original_train_metric) is the final-epoch overall validation
accuracy (val_accuracy), which the orchestrator's auto-revert guard uses so a
fix round cannot sacrifice the average to pad the minority groups.

WORKING_SPACE points at the workspace root (the dir this file sits in), so the
probe always writes into the correct `.agent_probe/` tree regardless of cwd.
"""
import json
import math
import os
import tempfile

import numpy as np

WORKING_SPACE = os.path.dirname(os.path.abspath(__file__))

METRIC_NAME = "worst_group_accuracy"
DIRECTION = "higher_is_better"
STANDARD_THRESHOLD = 0.68
ACCEPTABLE_THRESHOLD = 0.52

TAIL_WINDOW = 5

# ── in-memory trajectory ─────────────────────────────────────────────────────
values = []          # list of {"epoch": int, "value": float} for wg
gap_series = []      # per-epoch (overall - wg) / overall, diagnostic only
overall_series = []  # per-epoch overall accuracy, keeps the log line honest
adjusted_series = []  # per-epoch adjusted accuracy, diagnostic only
gap_streak = 0       # consecutive epochs with gap > 0.20 (spurious-signal alert)
printed_group_counts = False


def _atomic_write_json(path, payload):
    """Small atomic write: tempfile in the same dir + os.replace, so a partial
    read never sees half-written JSON."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _mean(seq):
    return float(np.mean(seq)) if len(seq) else float("nan")


def _std(seq):
    return float(np.std(seq)) if len(seq) else float("nan")


def _tail_mean(seq):
    return _mean(seq[-TAIL_WINDOW:] if len(seq) >= TAIL_WINDOW else seq)


def record(epoch, wg, overall, adjusted):
    """Collect one worst-group accuracy reading for the epoch.

    Args:
        epoch:     int, 1-based epoch number.
        wg:        float, worst-group accuracy `val["min_group"]["accuracy"]`.
        overall:   float, overall accuracy `val["overall"]["accuracy"]`.
        adjusted:  float, unweighted mean over groups
                   `val["adjusted_accuracy"]` (diagnostic).
    """
    global gap_streak, printed_group_counts
    values.append({"epoch": int(epoch), "value": float(wg)})
    overall_series.append(float(overall))
    adjusted_series.append(float(adjusted))

    gap = (float(overall) - float(wg)) / float(overall) if float(overall) > 0 else float("nan")
    gap_series.append(gap)

    # Consecutive-epoch spurious-signal alert to stderr (one-line, not a log flood).
    gap_streak = gap_streak + 1 if gap > 0.20 else 0
    if gap_streak == 3:
        import sys
        print(
            f"[prober] WARNING: spurious-correlation gap (overall-wg)/overall "
            f"exceeded 0.20 for 3 consecutive epochs (last gap={gap:.3f}) — "
            f"the model is leaning on the colour shortcut while the minority "
            f"(y,a) groups collapse.",
            file=sys.stderr, flush=True,
        )
    elif gap_streak > 3:
        import sys
        print(
            f"[prober] WARNING: spurious-correlation gap still > 0.20 "
            f"({gap_streak} consecutive epochs, last gap={gap:.3f}).",
            file=sys.stderr, flush=True,
        )

    # Log one line per epoch with the running tail-mean for quick scanning.
    print(
        f"  [probe] epoch {int(epoch):>4}  worst_group={float(wg):.4f}  "
        f"overall={float(overall):.4f}  adjusted={float(adjusted):.4f}  "
        f"gap={gap:.3f}  running_tail_mean={_tail_mean([v['value'] for v in values]):.4f}",
        flush=True,
    )

    # Overwrite the live trajectory so the orchestrator's UI can draw while the
    # run is still going. Use the frozen thresholds so live and final agree.
    live = {
        "metric_name": METRIC_NAME,
        "standard_threshold": STANDARD_THRESHOLD,
        "acceptable_threshold": ACCEPTABLE_THRESHOLD,
        "direction": DIRECTION,
        "values": list(values),
    }
    _atomic_write_json(os.path.join(WORKING_SPACE, ".agent_probe", "live", "probe_live.json"), live)


def _first_crossing_epoch(series, standard_threshold):
    """Epoch at which the metric first meets the standard threshold, if ever."""
    for entry in series:
        if entry["value"] >= standard_threshold:
            return entry["epoch"]
    return None


def _conclusion(series, tail, delta, overall_mean, gap_series_):
    """One-sentence plain-English summary of what the probe found."""
    first, last = series[0], series[-1]
    if tail >= STANDARD_THRESHOLD:
        gap_tail = _tail_mean(gap_series_)
        return (
            f"Worst-group accuracy {'improved steadily from' if first < last else 'ended at'} "
            f"{first:.2f} to {last:.2f}, holding >= {STANDARD_THRESHOLD:.2f} across the final "
            f"{TAIL_WINDOW} epochs with a mean spurious-correlation gap of {gap_tail:.3f}."
        )
    if tail < ACCEPTABLE_THRESHOLD and overall_mean > 0.75:
        return (
            f"Classic spurious-correlation failure: overall validation accuracy stayed healthy "
            f"at {overall_mean:.2f} while worst-group collapsed to {tail:.2f} — the minority "
            f"(y,a) groups are sacrificed to the colour shortcut."
        )
    return (
        f"Worst-group accuracy {'rose from' if last > first else 'ended around'} "
        f"{first:.2f} to {last:.2f} (tail mean {tail:.2f}), below the {STANDARD_THRESHOLD:.2f} "
        f"standard bar."
    )


def conclude(standard_threshold, acceptable_threshold):
    """Finalize the probe: stats, JSON result, and Plotly chart.

    Called once after training with BOTH thresholds (standard first, then
    acceptable). Loads the final-epoch anchor value from the recorded series and
    writes the full probe_result_N.json plus probe_result_N.pdf.

    Args:
        standard_threshold:   strict PASS bar for the tail-mean.
        acceptable_threshold: loose "we'd settle for this" bar (recorded only).
    """
    stats = _compute_stats(standard_threshold, acceptable_threshold)

    # Anchor: train.py's own final-epoch primary eval metric (overall val
    # accuracy). Recorded alongside the probe fields, never folded into them.
    anchor = {
        "name": "val_accuracy",
        "value": overall_series[-1] if overall_series else float("nan"),
        "direction": "higher_is_better",
    }
    stats["original_train_metric"] = anchor

    n = _next_index()
    _write_metric_json(n, stats)
    _write_plot(n, stats)
    _print_summary(stats)
    return n


def _compute_stats(standard_threshold, acceptable_threshold):
    """Everything derived from the recorded series + thresholds. Extracted so a
    headless test can check it without writing files."""
    raw = [v["value"] for v in values]
    first = raw[0] if raw else float("nan")
    final = raw[-1] if raw else float("nan")
    tail = _tail_mean(raw)
    status = "PASS" if tail >= standard_threshold else "FAIL"
    acceptable_met = bool(tail >= acceptable_threshold)
    # Inequality convention for the direction (higher_is_better, positive = improving).
    delta = final - first
    overall_mean = _mean(overall_series)
    return {
        "metric_name": METRIC_NAME,
        "standard_threshold": float(standard_threshold),
        "acceptable_threshold": float(acceptable_threshold),
        "direction": DIRECTION,
        "values": list(values),
        "min": min(raw) if raw else float("nan"),
        "max": max(raw) if raw else float("nan"),
        "mean": _mean(raw),
        "std": _std(raw),
        "first_value": first,
        "final_value": final,
        "delta": delta,
        "tail_mean": tail,
        "status": status,
        "acceptable_met": acceptable_met,
        "conclusion": _conclusion(raw, tail, delta, overall_mean, gap_series),
    }


def _next_index():
    """Smallest unused probe_result index (N starts at 1)."""
    metric_dir = os.path.join(WORKING_SPACE, ".agent_probe", "metric")
    used = set()
    if os.path.isdir(metric_dir):
        for fn in os.listdir(metric_dir):
            m = fn.startswith("probe_result_") and fn.endswith(".json")
            if not m:
                continue
            stem = fn[len("probe_result_"):-len(".json")]
            if stem.isdigit():
                used.add(int(stem))
    n = 1
    while n in used:
        n += 1
    return n


def _write_metric_json(n, stats):
    path = os.path.join(WORKING_SPACE, ".agent_probe", "metric", f"probe_result_{n}.json")
    _atomic_write_json(path, stats)


def _write_plot(n, stats):
    """Plotly line chart of the metric over epochs, saved to PDF. The y-axis is
    pinned to [0.0, 1.0] on every chart so successive iterations are directly
    comparable (covers both thresholds with headroom)."""
    if not values:
        return
    first_cross = _first_crossing_epoch(values, stats["standard_threshold"])
    line_color = "#1a9850" if stats["status"] == "PASS" else "#d62728"
    epochs = [v["epoch"] for v in values]
    y_vals = [v["value"] for v in values]

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=y_vals, mode="lines+markers",
        name=METRIC_NAME, line=dict(color=line_color, width=2),
        marker=dict(size=5, color=line_color),
    ))

    fig.add_hline(
        y=stats["standard_threshold"], line_dash="dash", line_color="#d62728",
        annotation_text=f"standard={stats['standard_threshold']:.2f}",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=stats["acceptable_threshold"], line_dash="dash", line_color="#ffa500",
        annotation_text=f"acceptable={stats['acceptable_threshold']:.2f}",
        annotation_position="top right",
    )

    if first_cross is not None:
        fig.add_vline(
            x=first_cross, line_dash="dash", line_color="#7f7f7f",
            annotation_text=f"first crossing @ epoch {first_cross}",
            annotation_position="top left",
        )

    info = (
        f"min={stats['min']:.4f}  max={stats['max']:.4f}<br>"
        f"mean={stats['mean']:.4f}  std={stats['std']:.4f}<br>"
        f"delta={stats['delta']:+.4f}  tail_mean={stats['tail_mean']:.4f}<br>"
        f"status=<b>{stats['status']}</b>  acceptable_met={str(stats['acceptable_met'])}"
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98, xanchor="left", yanchor="top",
        text=info, showarrow=False, bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#666", borderwidth=1, font=dict(size=11),
    )

    fig.update_layout(
        title=METRIC_NAME,
        xaxis_title="Epoch",
        yaxis_title=METRIC_NAME,
        yaxis=dict(range=[0.0, 1.0]),
        xaxis=dict(range=[min(epochs) - 0.5, max(epochs) + 0.5]),
        height=520, width=820,
        margin=dict(l=60, r=30, t=60, b=50),
        template="plotly_white",
    )

    plot_dir = os.path.join(WORKING_SPACE, ".agent_probe", "plot")
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, f"probe_result_{n}.pdf")
    fig.write_image(path)


def _print_summary(stats):
    print("\n[probe] conclude:")
    print(f"  values recorded: {len(values)} epochs")
    print(f"  min={stats['min']:.4f} max={stats['max']:.4f} "
          f"mean={stats['mean']:.4f} std={stats['std']:.4f}")
    print(f"  first={stats['first_value']:.4f} final={stats['final_value']:.4f} "
          f"delta={stats['delta']:+.4f} tail_mean={stats['tail_mean']:.4f}")
    print(f"  status={stats['status']} (standard {stats['standard_threshold']:.2f})  "
          f"acceptable_met={stats['acceptable_met']} (acceptable {stats['acceptable_threshold']:.2f})")
    print(f"  conclusion: {stats['conclusion']}")
    print(f"  wrote probe_result_{_next_index() - 1}.json / .pdf")


def log_validation_group_counts(val_eval_metrics):
    """Print the validation (y, a) group sample counts once — per CASE.md, so the
    noise band of `min_group` (which is set by the smallest group) is on the
    record. Pure diagnostics; does not alter the metric."""
    global printed_group_counts
    if printed_group_counts or not val_eval_metrics:
        return
    per_group = val_eval_metrics.get("per_group", {})
    counts = []
    for g in sorted(per_group.keys()):
        n = per_group[g].get("n_samples")
        if n is not None:
            counts.append(f"{g}:{n}")
    if counts:
        print("[probe] validation group sample counts (noise band for min_group): " + " ".join(counts), flush=True)
    printed_group_counts = True


# math is imported for parity with numpy's NaN handling if ever used directly.
_ = math