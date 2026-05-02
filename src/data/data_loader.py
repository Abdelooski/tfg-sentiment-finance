"""
src/data/data_loader.py

Downloads historical daily price data for BTC-USD and S&P 500 (^GSPC),
engineers return and next-day direction labels, and persists both raw
and processed CSVs to disk.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── Path constants ────────────────────────────────────────────────────────────
# ROOT is two levels up from this file (project root)
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# ── Download parameters ───────────────────────────────────────────────────────
START_DATE = "2020-01-01"
END_DATE = datetime.date.today().isoformat()

# Maps a short asset name to its Yahoo Finance ticker symbol
TICKERS: dict[str, str] = {
    "btc": "BTC-USD",
    "sp500": "^GSPC",
}


# ── Private helpers ───────────────────────────────────────────────────────────

def _download_raw(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from Yahoo Finance and return a two-column
    DataFrame: Date (datetime.date) and Close (float).

    Uses auto_adjust=True so Close already reflects splits and dividends.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. "
                         "Check the symbol and your internet connection.")

    # yfinance >= 0.2 may return MultiIndex columns even for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only Close; bring Date from the index into a plain column
    df = df[["Close"]].copy()
    df.index.name = "Date"
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date   # strip timezone / time component

    return df


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two engineered columns to a raw Close DataFrame.

    Return    – daily percentage change in Close price (row t).
    Direction – binary next-day label: 1 if Return[t+1] > 0, else 0.
                This is the prediction target for next-day direction.

    Rows that cannot be labelled are dropped:
      - The first row (no previous Close to compute Return).
      - The last row  (no future Close to compute the Direction label).
    """
    df = df.copy()

    # Percentage change: (Close_t - Close_{t-1}) / Close_{t-1} * 100
    df["Return"] = df["Close"].pct_change() * 100

    # Shift Return one step into the past so Direction[t] = sign(Return[t+1])
    df["Direction"] = (df["Return"].shift(-1) > 0).astype(int)

    # Remove unlabellable rows (first row: NaN Return; last row: no future return)
    df = df.iloc[1:-1].reset_index(drop=True)

    return df


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame to CSV, creating parent directories if necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"    Saved {len(df):>5} rows  ->  {path.relative_to(ROOT)}")


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_save_all() -> dict[str, pd.DataFrame]:
    """
    Run the full ingestion pipeline for every configured ticker:
      1. Download raw daily Close prices from Yahoo Finance.
      2. Save raw CSV (Date, Close) to data/raw/.
      3. Engineer Return and Direction features.
      4. Save processed CSV to data/processed/.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are asset names ('btc', 'sp500'); values are the processed DataFrames.
    """
    results: dict[str, pd.DataFrame] = {}

    for name, ticker in TICKERS.items():
        print(f"\n[{name.upper()}]  {ticker}  |  {START_DATE} to {END_DATE}")

        # Step 1 — download
        raw = _download_raw(ticker, START_DATE, END_DATE)

        # Step 2 — persist raw (Date + Close only)
        _save_csv(raw, RAW_DIR / f"{name}.csv")

        # Step 3 — engineer features
        processed = _add_features(raw)

        # Step 4 — persist processed (Date, Close, Return, Direction)
        _save_csv(processed, PROCESSED_DIR / f"{name}_processed.csv")

        results[name] = processed

    return results
