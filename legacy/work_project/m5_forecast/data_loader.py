"""Data loading for M5 Forecasting - Accuracy.

Ensures the raw Kaggle CSVs are present under ``data/`` (downloading them via the
Kaggle API on first use). The competition rules must be accepted once in the
browser and a valid ``~/.kaggle/kaggle.json`` must exist before download works.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
COMPETITION = "m5-forecasting-accuracy"
REQUIRED_FILES = ["sales_train_validation.csv", "calendar.csv", "sell_prices.csv"]


def ensure_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED_FILES if not (DATA_DIR / name).exists()]
    if not missing:
        return

    print(f"[data_loader] missing {missing}; downloading via Kaggle API ...")
    cmd = [
        "kaggle", "competitions", "download",
        "-c", COMPETITION,
        "-p", str(DATA_DIR),
        "--force",
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            f"[data_loader] Kaggle download failed ({exc}).\n"
            f"  - Put valid credentials in ~/.kaggle/kaggle.json\n"
            f"  - Accept the rules at https://www.kaggle.com/c/{COMPETITION}\n"
            f"  - Then re-run. Required files under {DATA_DIR}: {REQUIRED_FILES}"
        ) from exc

    for archive in DATA_DIR.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(DATA_DIR)
        archive.unlink()


def load_sales() -> pd.DataFrame:
    """Return the wide sales_train_validation dataframe (one row per series)."""
    ensure_data()
    return pd.read_csv(DATA_DIR / "sales_train_validation.csv")
