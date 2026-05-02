"""
src/nlp/reddit_preprocessing.py

Carga y limpia un dataset de Reddit WallStreetBets (CSV) para
prepararlo para análisis de sentimiento.

Entrada : data/raw/reddit_wsb.csv
Salida  : data/processed/reddit_wsb_processed.csv
Columnas de salida: Date | text | clean_text
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "reddit_wsb.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "reddit_wsb_processed.csv"

# ── Column name candidates ────────────────────────────────────────────────────
# Ordered by preference: first match wins
_TEXT_CANDIDATES: list[str] = ["title", "selftext", "body", "text"]
_DATE_CANDIDATES: list[str] = ["date", "created_utc", "timestamp", "created"]

# ── Regex patterns ────────────────────────────────────────────────────────────
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_USER = re.compile(r"u/\w+", re.IGNORECASE)
_RE_WHITESPACE = re.compile(r"\s+")


# ── Column detection ──────────────────────────────────────────────────────────

def _detect_columns(columns: list[str]) -> tuple[list[str], str]:
    """
    Scan the CSV header and return:
      - a list of detected text column names (in discovery order)
      - the name of the date column

    Raises ValueError if no usable text or date column is found.
    """
    lower_map = {c.lower(): c for c in columns}   # case-insensitive lookup

    text_cols = [
        lower_map[candidate]
        for candidate in _TEXT_CANDIDATES
        if candidate in lower_map
    ]
    if not text_cols:
        raise ValueError(
            f"No text column found. Expected one of {_TEXT_CANDIDATES}. "
            f"Got: {columns}"
        )

    date_candidates_found = [
        lower_map[candidate]
        for candidate in _DATE_CANDIDATES
        if candidate in lower_map
    ]
    if not date_candidates_found:
        raise ValueError(
            f"No date column found. Expected one of {_DATE_CANDIDATES}. "
            f"Got: {columns}"
        )

    return text_cols, date_candidates_found[0]


# ── Text assembly ─────────────────────────────────────────────────────────────

def _assemble_text(df: pd.DataFrame, text_cols: list[str]) -> pd.Series:
    """
    Build a single 'text' Series from the detected columns.

    If both 'title' and 'selftext' are present, they are concatenated
    with a space so the model sees the full post content.
    Otherwise the single available column is used directly.
    """
    cols = [c for c in text_cols if c in df.columns]

    if len(cols) >= 2 and "title" in [c.lower() for c in cols] \
            and "selftext" in [c.lower() for c in cols]:
        title_col = next(c for c in cols if c.lower() == "title")
        body_col = next(c for c in cols if c.lower() == "selftext")
        combined = (
            df[title_col].fillna("").astype(str)
            + " "
            + df[body_col].fillna("").astype(str)
        )
        return combined.str.strip()

    # Single column case
    return df[cols[0]].fillna("").astype(str).str.strip()


# ── Date parsing ──────────────────────────────────────────────────────────────

def _parse_date(series: pd.Series, col_name: str) -> pd.Series:
    """
    Convert a date/timestamp column to a YYYY-MM-DD string.

    Handles both Unix epoch integers (e.g. created_utc from Pushshift)
    and human-readable date strings automatically.
    """
    # Unix timestamp: numeric column or a column named 'created_utc'
    if pd.api.types.is_numeric_dtype(series) or col_name.lower() == "created_utc":
        try:
            parsed = pd.to_datetime(series, unit="s", errors="coerce")
            if parsed.notna().sum() > 0:
                return parsed.dt.strftime("%Y-%m-%d")
        except Exception:
            pass  # fall through to string parsing

    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """
    Apply a deterministic cleaning pipeline to a single text string:
      1. Lowercase
      2. Remove URLs
      3. Remove Reddit usernames  (u/handle)
      4. Collapse whitespace
      5. Strip leading/trailing spaces
    """
    text = text.lower()
    text = _RE_URL.sub(" ", text)
    text = _RE_USER.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text)
    return text.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess_reddit(input_path: Path = RAW_PATH,
                      output_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """
    Full preprocessing pipeline for a Reddit WallStreetBets CSV.

    Steps
    -----
    1. Load CSV and auto-detect text + date columns.
    2. Assemble a single 'text' field (combines title + selftext when both exist).
    3. Parse and normalise dates to YYYY-MM-DD.
    4. Clean text (lowercase, strip URLs, usernames, whitespace).
    5. Drop rows with empty clean_text.
    6. Save the processed file and return the DataFrame.

    Parameters
    ----------
    input_path  : Path to the raw CSV (default: data/raw/reddit_wsb.csv).
    output_path : Destination path   (default: data/processed/reddit_wsb_processed.csv).

    Returns
    -------
    pd.DataFrame with columns: Date, text, clean_text
    """
    print(f"  Loading  ->  {input_path.relative_to(ROOT)}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Rows loaded : {len(df):,}")

    # ── 1. Detect columns ─────────────────────────────────────────────────────
    text_cols, date_col = _detect_columns(list(df.columns))
    print(f"  Text columns detected : {text_cols}")
    print(f"  Date column detected  : {date_col!r}")

    # ── 2. Assemble text ──────────────────────────────────────────────────────
    df["text"] = _assemble_text(df, text_cols)

    # ── 3. Parse date ─────────────────────────────────────────────────────────
    df["Date"] = _parse_date(df[date_col], date_col)

    # ── 4. Clean text ─────────────────────────────────────────────────────────
    df["clean_text"] = df["text"].apply(_clean)

    # ── 5. Drop empty rows ────────────────────────────────────────────────────
    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with empty text after cleaning.")

    # ── 6. Keep only the three output columns ─────────────────────────────────
    result = df[["Date", "text", "clean_text"]].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"  Saved {len(result):,} rows  ->  {output_path.relative_to(ROOT)}")

    return result
