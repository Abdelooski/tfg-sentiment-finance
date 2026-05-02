"""
src/features/feature_engineering.py

Construye features predictivas a partir de los datasets merged
(precio + sentimiento VADER) sin introducir data leakage.

Regla anti-leakage: en el instante t predecimos Direction[t],
que vale 1 si Return[t+1] > 0. Por tanto, cualquier información
conocida al cierre del día t es válida como feature:
  - Return[t], vader_compound[t], n_posts[t]   ← datos del día actual, no son leakage
  - Return[t-k], vader_compound[t-k]            ← lags (días anteriores), seguros
  - rolling sobre [t-w+1 … t]                   ← ventana pasada incluido hoy, segura

Entradas : data/processed/btc_vader_merged.csv
           data/processed/sp500_vader_merged.csv
Salidas  : data/processed/btc_features.csv
           data/processed/sp500_features.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

# Maps asset name -> (input, output)
ASSET_CONFIG: dict[str, tuple[Path, Path]] = {
    "btc": (
        PROCESSED_DIR / "btc_vader_merged.csv",
        PROCESSED_DIR / "btc_features.csv",
    ),
    "sp500": (
        PROCESSED_DIR / "sp500_vader_merged.csv",
        PROCESSED_DIR / "sp500_features.csv",
    ),
}

TARGET_COL = "Direction"

# Feature columns created by this module (used for the summary print)
_ENGINEERED_FEATURES = [
    "return_lag_1", "return_lag_2", "return_lag_3",
    "vader_compound_lag_1", "vader_compound_lag_2", "vader_compound_lag_3",
    "vader_compound_rolling_3", "vader_compound_rolling_7",
    "n_posts_log",
    "return_rolling_3", "return_rolling_7",
    "volatility_7",
]


# ── Feature construction ──────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append all engineered feature columns to df and drop any rows
    that contain NaN (produced by lag/rolling operations at the start
    of the time series).

    No future information is used — every computation is grounded at
    day t or earlier.
    """
    out = df.copy()

    # ── Return lags ───────────────────────────────────────────────────────────
    # return_lag_k[t] = Return[t-k]  → strictly past, no leakage
    for k in (1, 2, 3):
        out[f"return_lag_{k}"] = out["Return"].shift(k)

    # ── VADER compound lags ───────────────────────────────────────────────────
    for k in (1, 2, 3):
        out[f"vader_compound_lag_{k}"] = out["vader_compound"].shift(k)

    # ── VADER compound rolling mean ───────────────────────────────────────────
    # rolling(w) at position t uses [t-w+1 … t] → includes today, no leakage
    out["vader_compound_rolling_3"] = out["vader_compound"].rolling(3).mean()
    out["vader_compound_rolling_7"] = out["vader_compound"].rolling(7).mean()

    # ── n_posts log transform ─────────────────────────────────────────────────
    # log1p handles the rare case of n_posts == 0 without producing -inf
    out["n_posts_log"] = np.log1p(out["n_posts"])

    # ── Return rolling mean ───────────────────────────────────────────────────
    out["return_rolling_3"] = out["Return"].rolling(3).mean()
    out["return_rolling_7"] = out["Return"].rolling(7).mean()

    # ── Volatility: rolling std of returns over 7 days ───────────────────────
    # ddof=1 gives the sample standard deviation (standard in finance)
    out["volatility_7"] = out["Return"].rolling(7).std(ddof=1)

    # ── Drop NaN rows ─────────────────────────────────────────────────────────
    # The largest rolling window is 7, so the first 6 rows will have NaN.
    # Lag-3 also requires 3 rows of history → max(7-1, 3) = 6 rows dropped.
    out = out.dropna().reset_index(drop=True)

    return out


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(name: str, df: pd.DataFrame, out_path: Path) -> None:
    all_cols = list(df.columns)
    feature_cols = [c for c in all_cols if c not in ("Date", TARGET_COL)]

    print(f"\n  [{name.upper()}] feature dataset")
    print(f"    Shape          : {df.shape}")
    print(f"    Date range     : {df['Date'].min()}  to  {df['Date'].max()}")
    print(f"    Target column  : '{TARGET_COL}'  "
          f"(1 = next-day positive return, 0 = otherwise)")
    print(f"    Feature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"      - {col}")
    print(f"    Saved  ->  {out_path.relative_to(ROOT)}")


# ── Public API ────────────────────────────────────────────────────────────────

def engineer_all(
    asset_config: dict[str, tuple[Path, Path]] = ASSET_CONFIG,
) -> dict[str, pd.DataFrame]:
    """
    Build and save the full feature matrix for each configured asset.

    Steps
    -----
    1. Load the merged (price + VADER) CSV for each asset.
    2. Compute all lag, rolling, log, and volatility features.
    3. Drop NaN rows produced by windowed operations.
    4. Print a diagnostic summary (shape, features, target).
    5. Save the feature CSV to data/processed/.

    Returns
    -------
    dict[str, pd.DataFrame] — keys are asset names ('btc', 'sp500').
    """
    results: dict[str, pd.DataFrame] = {}

    for name, (in_path, out_path) in asset_config.items():
        print(f"\n  Loading {name.upper()}  ->  {in_path.relative_to(ROOT)}")
        df = pd.read_csv(in_path, dtype={"Date": str})

        featured = _build_features(df)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        featured.to_csv(out_path, index=False)

        _print_summary(name, featured, out_path)
        results[name] = featured

    return results
