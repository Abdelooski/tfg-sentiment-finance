"""
src/reports/summary_tables.py

Lee los resultados de todos los modelos entrenados y construye una tabla
resumen con el mejor modelo por activo, junto con un flag de overfitting
y una interpretación textual automática.

Entrada : reports/tables/model_results.csv
Salida  : reports/tables/best_model_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "reports" / "tables"

RESULTS_PATH = TABLES_DIR / "model_results.csv"
SUMMARY_PATH = TABLES_DIR / "best_model_summary.csv"

# ── Thresholds ────────────────────────────────────────────────────────────────
# A gap between train accuracy and test accuracy above this value is considered
# a sign of overfitting — the model memorised training data rather than learning
# generalisable patterns.
OVERFIT_THRESHOLD = 0.20

# F1 tiers used to build the human-readable interpretation
F1_HIGH   = 0.65   # strong predictive signal
F1_MEDIUM = 0.55   # moderate — better than chance but not reliable

# Columns to keep in the final summary, in display order
SUMMARY_COLS = [
    "asset",
    "best_model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "train_acc",
    "overfitting_flag",
    "interpretation",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_results(path: Path) -> pd.DataFrame:
    """Load and minimally validate the model results CSV."""
    df = pd.read_csv(path)
    required = {"asset", "model", "f1", "accuracy", "precision", "recall", "train_acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"model_results.csv is missing columns: {missing}")
    return df


def _select_best_per_asset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per asset: the row with the highest F1-score.

    When two models tie on F1, accuracy is used as a secondary criterion.
    Using F1 as primary metric is more robust than accuracy for imbalanced
    datasets because it weights precision and recall equally.
    """
    return (
        df.sort_values(["asset", "f1", "accuracy"], ascending=[True, False, False])
        .groupby("asset", sort=False)
        .first()
        .reset_index()
    )


def _overfitting_flag(train_acc: float, test_acc: float) -> bool:
    """
    Return True when the gap between training and test accuracy
    exceeds OVERFIT_THRESHOLD, indicating the model has memorised
    training data rather than learning generalisable rules.
    """
    return (train_acc - test_acc) > OVERFIT_THRESHOLD


def _interpret(overfit: bool, f1: float, accuracy: float) -> str:
    """
    Build a concise, human-readable interpretation string from three signals:
      1. Whether overfitting was detected.
      2. The F1-score tier (strong / moderate / low).
      3. Whether accuracy beats the 50 % random-classifier baseline.
    """
    # F1 tier
    if f1 >= F1_HIGH:
        f1_label = "strong F1"
    elif f1 >= F1_MEDIUM:
        f1_label = "moderate F1"
    else:
        f1_label = "low F1"

    # Accuracy vs random baseline
    acc_label = "above" if accuracy > 0.50 else "at or below"

    if overfit:
        return (
            f"Overfitting detected (gap > {OVERFIT_THRESHOLD:.0%}). "
            f"{f1_label.capitalize()}, accuracy {acc_label} random baseline."
        )
    return (
        f"No overfitting. "
        f"{f1_label.capitalize()}, accuracy {acc_label} random baseline."
    )


# ── Public API ────────────────────────────────────────────────────────────────

def generate_summary(
    results_path: Path = RESULTS_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> pd.DataFrame:
    """
    Build and save the best-model summary table.

    Steps
    -----
    1. Load model_results.csv.
    2. Select the best model per asset (highest F1, accuracy as tiebreaker).
    3. Compute overfitting_flag for each selected model.
    4. Generate a human-readable interpretation string.
    5. Print the summary to the console.
    6. Save best_model_summary.csv.

    Returns
    -------
    pd.DataFrame with columns defined in SUMMARY_COLS.
    """
    print(f"  Loading  ->  {results_path.relative_to(ROOT)}")
    results_df = _load_results(results_path)

    # ── Select best model per asset ───────────────────────────────────────────
    best_df = _select_best_per_asset(results_df)

    # ── Rename 'model' to 'best_model' ────────────────────────────────────────
    best_df = best_df.rename(columns={"model": "best_model"})

    # ── Overfitting flag ──────────────────────────────────────────────────────
    best_df["overfitting_flag"] = best_df.apply(
        lambda row: _overfitting_flag(row["train_acc"], row["accuracy"]),
        axis=1,
    )

    # ── Interpretation ────────────────────────────────────────────────────────
    best_df["interpretation"] = best_df.apply(
        lambda row: _interpret(row["overfitting_flag"], row["f1"], row["accuracy"]),
        axis=1,
    )

    # ── Keep only display columns ─────────────────────────────────────────────
    summary_df = best_df[SUMMARY_COLS].copy()

    # ── Console output ────────────────────────────────────────────────────────
    print("\n  Best model per asset")
    print("  " + "-" * 70)
    for _, row in summary_df.iterrows():
        print(f"\n  Asset        : {row['asset']}")
        print(f"  Best model   : {row['best_model']}")
        print(f"  Accuracy     : {row['accuracy']:.4f}   "
              f"(train: {row['train_acc']:.4f})")
        print(f"  Precision    : {row['precision']:.4f}")
        print(f"  Recall       : {row['recall']:.4f}")
        print(f"  F1-score     : {row['f1']:.4f}")
        flag_str = "YES" if row["overfitting_flag"] else "NO"
        print(f"  Overfitting  : {flag_str}")
        print(f"  Interpretation: {row['interpretation']}")
    print("  " + "-" * 70)

    # ── Save ──────────────────────────────────────────────────────────────────
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved  ->  {summary_path.relative_to(ROOT)}")

    return summary_df
