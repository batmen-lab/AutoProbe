"""Performance probe for M5 Forecasting LightGBM training.

Tracks per-boosting-round validation RMSE (root mean squared error) on the
28-day forecast horizon. Writes a live trajectory JSON during training and a
final summary JSON after training completes.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
from pathlib import Path

_METRIC_NAME = "val_rmse"
_DIRECTION = "lower_is_better"

_BASE_DIR = Path(__file__).resolve().parent
_PROBE_DIR = _BASE_DIR / ".agent_probe"
_LIVE_DIR = _PROBE_DIR / "live"
_METRIC_DIR = _PROBE_DIR / "metric"
_LIVE_FILE = _LIVE_DIR / "probe_live.json"

_series: list[dict] = []


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as h:
            json.dump(payload, h, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def record(epoch: int, value: float) -> None:
    """Record the metric value for one epoch/round and refresh the live JSON."""
    _series.append({"epoch": int(epoch), "value": float(value)})
    payload = {
        "metric_name": _METRIC_NAME,
        "direction": _DIRECTION,
        "values": list(_series),
    }
    try:
        _atomic_write_json(_LIVE_FILE, payload)
    except Exception:
        pass


def _next_result_path() -> Path:
    _METRIC_DIR.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for entry in _METRIC_DIR.glob("probe_result_*.json"):
        stem = entry.stem
        suffix = stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            max_n = max(max_n, int(suffix))
    return _METRIC_DIR / f"probe_result_{max_n + 1}.json"


def conclude() -> None:
    """Compute summary statistics over the recorded series and persist them."""
    try:
        values = [pt["value"] for pt in _series]
        if values:
            v_min = float(min(values))
            v_max = float(max(values))
            v_mean = float(statistics.fmean(values))
            v_std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
            first_value = float(values[0])
            final_value = float(values[-1])
            delta = final_value - first_value
            tail = values[-5:] if len(values) >= 5 else values
            tail_mean = float(statistics.fmean(tail))
            direction_word = "rose" if delta > 0 else ("fell" if delta < 0 else "held steady")
            conclusion = (
                f"Validation RMSE {direction_word} from {first_value:.6f} to "
                f"{final_value:.6f} across {len(values)} boosting rounds "
                f"(min {v_min:.6f}, max {v_max:.6f})."
            )
        else:
            v_min = v_max = v_mean = v_std = 0.0
            first_value = final_value = delta = tail_mean = 0.0
            conclusion = "No metric values were recorded during training."

        payload = {
            "metric_name": _METRIC_NAME,
            "direction": _DIRECTION,
            "values": list(_series),
            "min": v_min,
            "max": v_max,
            "mean": v_mean,
            "std": v_std,
            "first_value": first_value,
            "final_value": final_value,
            "delta": delta,
            "tail_mean": tail_mean,
            "conclusion": conclusion,
        }

        out_path = _next_result_path()
        _atomic_write_json(out_path, payload)
    except Exception:
        return
