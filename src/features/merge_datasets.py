"""
src/features/merge_datasets.py

Une el sentimiento diario de VADER (Reddit WSB) con los datos de precio
de cada activo usando Date como clave de cruce (inner join).

Entradas:
  data/processed/reddit_wsb_vader_daily.csv
  data/processed/btc_processed.csv
  data/processed/sp500_processed.csv

Salidas:
  data/processed/btc_vader_merged.csv
  data/processed/sp500_vader_merged.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

VADER_PATH = PROCESSED_DIR / "reddit_wsb_vader_daily.csv"

# Maps asset name -> (input CSV, output CSV)
ASSET_CONFIG: dict[str, tuple[Path, Path]] = {
    "btc": (
        PROCESSED_DIR / "btc_processed.csv",
        PROCESSED_DIR / "btc_vader_merged.csv",
    ),
    "sp500": (
        PROCESSED_DIR / "sp500_processed.csv",
        PROCESSED_DIR / "sp500_vader_merged.csv",
    ),
}


# ── Private helpers ───────────────────────────────────────────────────────────

def _load_csv(path: Path, label: str) -> pd.DataFrame:
    """Load a CSV, parse Date as string, and validate it is not empty."""
    df = pd.read_csv(path, dtype={"Date": str})
    if df.empty:
        raise ValueError(f"{label} file is empty: {path}")
    return df


def _merge(vader_df: pd.DataFrame, asset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Inner-join the daily VADER sentiment table with an asset price table on Date.

    Inner join ensures only dates present in BOTH datasets are kept,
    which avoids NaN sentiments or NaN prices in the final features.
    The result is sorted chronologically.
    """
    merged = asset_df.merge(vader_df, on="Date", how="inner")
    return merged.sort_values("Date").reset_index(drop=True)


def _print_summary(name: str, df: pd.DataFrame, out_path: Path) -> None:
    """Print a concise diagnostic summary for a merged dataset."""
    print(f"\n  [{name.upper()}] merged dataset")
    print(f"    Rows       : {len(df):,}")
    print(f"    Date range : {df['Date'].min()}  to  {df['Date'].max()}")
    print(f"    Columns    : {list(df.columns)}")
    print(f"    Saved  ->    {out_path.relative_to(ROOT)}")


# ── Public API ────────────────────────────────────────────────────────────────

def merge_all(
    vader_path: Path = VADER_PATH,
    asset_config: dict[str, tuple[Path, Path]] = ASSET_CONFIG,
) -> dict[str, pd.DataFrame]:
    """
    Merge the daily VADER sentiment table with each configured asset dataset.

    Steps
    -----
    1. Load the daily VADER CSV once (shared across all assets).
    2. For each asset, load its processed price CSV.
    3. Inner-join on Date.
    4. Print a summary (rows, date range, columns).
    5. Save the merged CSV.

    Parameters
    ----------
    vader_path   : Path to reddit_wsb_vader_daily.csv.
    asset_config : Mapping of asset name -> (input_path, output_path).

    Returns
    -------
    dict[str, pd.DataFrame]  — keys are asset names ('btc', 'sp500').
    """
    print(f"  Loading VADER daily  ->  {vader_path.relative_to(ROOT)}")
    vader_df = _load_csv(vader_path, "VADER daily")
    print(f"  VADER rows : {len(vader_df):,}  |  "
          f"dates : {vader_df['Date'].min()} to {vader_df['Date'].max()}")

    results: dict[str, pd.DataFrame] = {}

    for name, (asset_path, out_path) in asset_config.items():
        print(f"\n  Loading {name.upper()} prices  ->  {asset_path.relative_to(ROOT)}")
        asset_df = _load_csv(asset_path, name.upper())

        merged = _merge(vader_df, asset_df)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)

        _print_summary(name, merged, out_path)
        results[name] = merged

    return results
