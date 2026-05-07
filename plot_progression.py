"""Plot per-round metric progression for all four auto-research projects.

For each project, reads probe_result_{3..12}.json (the 10 iteration outputs after
the 2 pre-loop checks), applies a best-so-far envelope (so a regression-then-revert
keeps the line flat at the prior round's value), and saves a PDF line chart.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent
RESPONSE_DIR = REPO_ROOT / "response"
RESPONSE_DIR.mkdir(parents=True, exist_ok=True)

PROJECTS = [
    "home_credit",
    "ieee_cis_fraud_detection",
    "m5_forecast",
    "rossmann",
]
N_ROUNDS = 10
FIRST_ITERATION_PROBE_INDEX = 3  # probe_result_1 = post-setup, _2 = post-commentor, _3 = iter 1


def load_round_values(project: str) -> tuple[list[float], str, str]:
    metric_dir = REPO_ROOT / project / ".agent_probe" / "metric"
    values: list[float] = []
    metric_name = ""
    direction = ""
    for round_idx in range(1, N_ROUNDS + 1):
        probe_path = metric_dir / f"probe_result_{FIRST_ITERATION_PROBE_INDEX + round_idx - 1}.json"
        if not probe_path.exists():
            break
        with probe_path.open() as fh:
            data = json.load(fh)
        values.append(float(data["tail_mean"]))
        if not metric_name:
            metric_name = str(data.get("metric_name", "metric"))
        if not direction:
            direction = str(data.get("direction", "higher_is_better"))
    if not values:
        raise FileNotFoundError(f"No iteration probe results found under {metric_dir}")
    return values, metric_name, direction


def best_so_far(values: list[float], direction: str) -> list[float]:
    envelope: list[float] = []
    best = values[0]
    for v in values:
        if direction == "higher_is_better":
            best = max(best, v)
        else:
            best = min(best, v)
        envelope.append(best)
    return envelope


def plot_one(project: str) -> Path:
    raw, metric_name, direction = load_round_values(project)
    envelope = best_so_far(raw, direction)
    rounds = list(range(1, len(raw) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, envelope, marker="o", linewidth=2, color="#1f77b4", label="best-so-far")
    ax.plot(rounds, raw, marker="x", linestyle="--", linewidth=1, alpha=0.5, color="gray", label="raw per-round")

    ax.set_xticks(rounds)
    ax.set_xlabel("Iteration round")
    ax.set_ylabel(metric_name)
    arrow = "↑" if direction == "higher_is_better" else "↓"
    ax.set_title(f"{project} — {metric_name} ({direction.replace('_', ' ')} {arrow})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    output = RESPONSE_DIR / f"{project}_progression.pdf"
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def main() -> None:
    for project in PROJECTS:
        path = plot_one(project)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
