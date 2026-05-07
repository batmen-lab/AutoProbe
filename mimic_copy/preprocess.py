"""Preprocess MIMIC-III CSV files into per-ICU-stay datasets for mortality prediction.

Replicates the SQL-based pipeline from the original MIMIC-III benchmarking code
using pandas operations on flat CSV (or CSV.GZ) files.

Outputs (saved to the same directory as input):
    Per-split feature files (efficient binary formats):
        - {split}_tfidf.npz  : sparse TF-IDF matrix  (scipy CSR)
        - {split}_meta.npz   : dense arrays  (eth features + labels)
    Shared artefacts:
        - tfidf_vectorizer.pkl : fitted TfidfVectorizer (for inference)
        - dataset_statistics.json : per-ethnicity & token-length stats
    Splits are stratified by mort_icu and fixed via SEED for reproducibility.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_ICU_LOS_HOURS = 48
MAX_AGE_YEARS = 300
NOTES_WINDOW_HOURS = 48
PEDIATRIC_UNITS = frozenset({"PICU", "NICU"})

ETHNICITY_PATTERNS: dict[str, str] = {
    "white": r"^white",
    "black": r"^black",
    "hispanic": r"^hisp|^latin",
    "asian": r"^asia",
}

SEED = 42
TRAIN_RATIO = 0.50  # 2:1:1 → 50% train, 25% val, 25% test
VAL_RATIO = 0.25
LABEL_COL = "mort_icu"
MAX_TFIDF_FEATURES = 10_000


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_table(
    data_dir: Path,
    name: str,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Load a MIMIC-III table, preferring uncompressed CSV over .csv.gz."""
    csv_path = data_dir / f"{name}.csv"
    gz_path = data_dir / f"{name}.csv.gz"

    path = csv_path if csv_path.exists() else gz_path if gz_path.exists() else None
    if path is None:
        raise FileNotFoundError(
            f"Neither {csv_path} nor {gz_path} found in {data_dir}"
        )

    logger.info("Loading %-15s from %s", name, path.name)
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    logger.info("  -> %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------
def build_demographics(
    icustays: pd.DataFrame,
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
) -> pd.DataFrame:
    """Join ICU stays with admissions and patients; compute derived features."""
    df = (
        icustays
        .merge(admissions, on=["SUBJECT_ID", "HADM_ID"], how="inner")
        .merge(patients, on="SUBJECT_ID", how="inner")
    )
    df = df[df["HAS_CHARTEVENTS_DATA"] == 1].copy()

    # Parse datetimes
    dt_cols = ["ADMITTIME", "DISCHTIME", "DEATHTIME", "INTIME", "OUTTIME", "DOB"]
    for col in dt_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Length of stay (fractional days / hours)
    df["los_hospital"] = (df["DISCHTIME"] - df["ADMITTIME"]).dt.total_seconds() / 86400
    df["los_icu"] = (df["OUTTIME"] - df["INTIME"]).dt.total_seconds() / 86400
    df["los_icu_hr"] = (df["OUTTIME"] - df["INTIME"]).dt.total_seconds() / 3600

    # Age in years — computed via year/month/day components to avoid
    # nanosecond overflow (MIMIC shifts DOB ~300 years for patients >89).
    admit_year = df["ADMITTIME"].dt.year
    admit_month = df["ADMITTIME"].dt.month
    admit_day = df["ADMITTIME"].dt.day
    dob_year = df["DOB"].dt.year
    dob_month = df["DOB"].dt.month
    dob_day = df["DOB"].dt.day
    df["age"] = (
        admit_year - dob_year
        - ((admit_month < dob_month) | ((admit_month == dob_month) & (admit_day < dob_day))).astype(int)
    )

    # Mortality flags
    death = df["DEATHTIME"]
    df["mort_hosp"] = (
        death.notna()
        & death.between(df["ADMITTIME"], df["DISCHTIME"])
    ).astype(int)

    df["mort_icu"] = (
        death.notna()
        & death.between(df["INTIME"], df["OUTTIME"])
    ).astype(int)

    df["mort_oneyr"] = (
        death.notna()
        & death.between(df["ADMITTIME"], df["ADMITTIME"] + pd.Timedelta(days=365))
    ).astype(int)

    # Sequencing
    df["hospstay_seq"] = (
        df.groupby("SUBJECT_ID")["ADMITTIME"].rank(method="dense").astype(int)
    )
    df["first_hosp_stay"] = (df["hospstay_seq"] == 1).astype(int)

    df["icustay_seq"] = (
        df.groupby("HADM_ID")["INTIME"].rank(method="dense").astype(int)
    )
    df["first_icu_stay"] = (df["icustay_seq"] == 1).astype(int)

    return df


def apply_inclusion_criteria(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to adult ICU stays >= 48 h with plausible age."""
    n = len(df)
    df = df[df["los_icu_hr"] >= MIN_ICU_LOS_HOURS]
    logger.info("ICU LOS >= %dh: %d -> %d", MIN_ICU_LOS_HOURS, n, len(df))

    n = len(df)
    df = df[df["age"] < MAX_AGE_YEARS]
    logger.info("Age < %d: %d -> %d", MAX_AGE_YEARS, n, len(df))

    df = df.copy()
    df["adult_icu"] = (~df["FIRST_CAREUNIT"].isin(PEDIATRIC_UNITS)).astype(int)
    n = len(df)
    df = df[df["adult_icu"] == 1]
    logger.info("Adult ICU only: %d -> %d", n, len(df))

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode gender, ethnicity, and admission type into numeric features."""
    df = df.copy()

    # Binary gender: M=1, F=0
    df["gender"] = (df["GENDER"] == "M").astype(int)

    # Collapse ethnicity into 5 groups
    eth = df["ETHNICITY"].str.lower()
    for label, pattern in ETHNICITY_PATTERNS.items():
        eth = eth.where(~eth.str.contains(pattern, na=False), label)
    eth = eth.where(eth.isin(ETHNICITY_PATTERNS.keys()), "other")

    eth_dummies = pd.get_dummies(eth, prefix="eth", dtype=int)
    adm_dummies = pd.get_dummies(df["ADMISSION_TYPE"], prefix="admType", dtype=int)

    df = pd.concat([df, eth_dummies, adm_dummies], axis=1)

    # Keep only final feature columns
    id_cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]
    feature_cols = [
        "gender", "INSURANCE",
        "mort_hosp", "mort_icu", "mort_oneyr",
        "first_hosp_stay", "first_icu_stay", "adult_icu",
    ]
    dummy_cols = [c for c in df.columns if c.startswith(("eth_", "admType_"))]

    return df[id_cols + feature_cols + dummy_cols].dropna()


# ---------------------------------------------------------------------------
# Clinical notes
# ---------------------------------------------------------------------------
def extract_first48h_notes(
    noteevents: pd.DataFrame,
    icustays: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate clinical notes from the first 48 hours of each ICU stay.

    Excludes discharge summaries, matching the original pipeline.
    """
    noteevents = noteevents.copy()
    noteevents["CHARTTIME"] = pd.to_datetime(noteevents["CHARTTIME"], errors="coerce")

    icu = icustays[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"]].copy()
    icu["INTIME"] = pd.to_datetime(icu["INTIME"], errors="coerce")

    # Filter out discharge summaries and notes without a charttime
    notes = noteevents[
        (noteevents["CATEGORY"] != "Discharge summary")
        & noteevents["CHARTTIME"].notna()
    ]

    merged = icu.merge(
        notes[["SUBJECT_ID", "HADM_ID", "CHARTTIME", "TEXT"]],
        on=["SUBJECT_ID", "HADM_ID"],
        how="inner",
    )

    # Keep only notes charted within the first 48 hours
    cutoff = merged["INTIME"] + pd.Timedelta(hours=NOTES_WINDOW_HOURS)
    merged = merged[merged["CHARTTIME"].between(merged["INTIME"], cutoff)]

    # Aggregate text per ICU stay
    grouped = (
        merged
        .groupby(["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"])["TEXT"]
        .agg(lambda texts: " ".join(texts.dropna()))
        .reset_index()
        .rename(columns={"TEXT": "chartext"})
    )

    logger.info("Notes extracted for %d ICU stays", len(grouped))
    return grouped


# ---------------------------------------------------------------------------
# Stratified splitting
# ---------------------------------------------------------------------------
def stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    label_col: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / val / test preserving class ratio. Remainder goes to test."""
    pos = df[df[label_col] == 1].sample(frac=1, random_state=seed).reset_index(drop=True)
    neg = df[df[label_col] == 0].sample(frac=1, random_state=seed).reset_index(drop=True)

    def _cut(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n_train = int(len(group) * train_ratio)
        n_val = int(len(group) * val_ratio)
        return group.iloc[:n_train], group.iloc[n_train:n_train + n_val], group.iloc[n_train + n_val:]

    train_pos, val_pos, test_pos = _cut(pos)
    train_neg, val_neg, test_neg = _cut(neg)

    train = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    val = pd.concat([val_pos, val_neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    test = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, val, test


ETH_COLS = [c for pat in ETHNICITY_PATTERNS for c in [f"eth_{pat}"]] + ["eth_other"]
TEXT_COL = "chartext"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(data_path: str) -> None:
    data_dir = Path(data_path)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"{data_dir} is not a valid directory")
    logger.info("Data directory: %s", data_dir.resolve())

    # ---- Load tables (only the columns we need) ----
    admissions = load_table(data_dir, "ADMISSIONS")
    icustays = load_table(data_dir, "ICUSTAYS")
    patients = load_table(
        data_dir, "PATIENTS",
        usecols=["SUBJECT_ID", "GENDER", "DOB"],
    )
    noteevents = load_table(
        data_dir, "NOTEEVENTS",
        usecols=["SUBJECT_ID", "HADM_ID", "CHARTTIME", "CATEGORY", "TEXT"],
    )

    # ---- Build demographics ----
    demographics = build_demographics(icustays, admissions, patients)
    demographics = apply_inclusion_criteria(demographics)
    demographics = encode_categoricals(demographics)
    logger.info("Demographics table: %d rows", len(demographics))

    # ---- Extract clinical notes ----
    notes = extract_first48h_notes(noteevents, icustays)

    # ---- Merge ----
    output = notes.merge(
        demographics,
        on=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"],
        how="inner",
    )
    logger.info("Final dataset: %d rows", len(output))

    # ---- Stratified train / val / test split (2:1:1) ----
    train_df, val_df, test_df = stratified_split(
        output, TRAIN_RATIO, VAL_RATIO, LABEL_COL, SEED,
    )

    # ---- Fit TF-IDF on training text, transform all splits ----
    logger.info("Fitting TF-IDF (max_features=%d) on train split ...", MAX_TFIDF_FEATURES)
    vectorizer = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, stop_words="english")
    vectorizer.fit(train_df[TEXT_COL].fillna(""))
    logger.info("TF-IDF vocabulary size: %d", len(vectorizer.vocabulary_))

    # ---- Save per-split features (sparse tfidf + dense meta) ----
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        tfidf_matrix = vectorizer.transform(split_df[TEXT_COL].fillna(""))
        eth = split_df[ETH_COLS].values.astype(np.float32)
        labels = split_df[LABEL_COL].values.astype(np.float32)

        sparse.save_npz(data_dir / f"{name}_tfidf.npz", tfidf_matrix)
        np.savez(data_dir / f"{name}_meta.npz", eth=eth, labels=labels)

        n_pos = int(labels.sum())
        logger.info(
            "Saved %-5s -> %s_tfidf.npz + %s_meta.npz  (%d rows, %d pos / %d neg)",
            name, name, name, len(split_df), n_pos, len(split_df) - n_pos,
        )

    # ---- Save fitted vectorizer for inference ----
    vec_path = data_dir / "tfidf_vectorizer.pkl"
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Saved TF-IDF vectorizer to %s", vec_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess MIMIC-III CSV data for mortality prediction.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to directory containing MIMIC-III CSV (or .csv.gz) files",
    )
    args = parser.parse_args()
    main(args.data_path)