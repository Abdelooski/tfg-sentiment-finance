"""
src/nlp/vader_sentiment.py

Aplica VADER sentiment analysis a los posts de Reddit WallStreetBets
ya preprocesados y genera dos archivos de salida:
  - reddit_wsb_vader_posts.csv  : puntuaciones por post
  - reddit_wsb_vader_daily.csv  : agregado diario (media + conteo)
"""

from __future__ import annotations

from pathlib import Path

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_PATH = PROCESSED_DIR / "reddit_wsb_processed.csv"
POSTS_OUTPUT = PROCESSED_DIR / "reddit_wsb_vader_posts.csv"
DAILY_OUTPUT = PROCESSED_DIR / "reddit_wsb_vader_daily.csv"

# VADER score column names
_SCORE_COLS = ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]


# ── Setup ─────────────────────────────────────────────────────────────────────

def _ensure_vader_lexicon() -> None:
    """Download the VADER lexicon if it is not already present."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        print("  Downloading VADER lexicon …")
        nltk.download("vader_lexicon", quiet=True)


# ── Core steps ────────────────────────────────────────────────────────────────

def _apply_vader(df: pd.DataFrame, sia: SentimentIntensityAnalyzer) -> pd.DataFrame:
    """
    Score every row in df using VADER and append four new columns.

    The analyser returns a dict with keys neg / neu / pos / compound.
    We rename them to vader_* for clarity in the final dataset.
    """
    scores = df["clean_text"].apply(
        lambda text: sia.polarity_scores(text if isinstance(text, str) else "")
    )

    scores_df = pd.DataFrame(scores.tolist(), index=df.index)
    scores_df = scores_df.rename(columns={
        "neg": "vader_neg",
        "neu": "vader_neu",
        "pos": "vader_pos",
        "compound": "vader_compound",
    })

    return pd.concat([df, scores_df[_SCORE_COLS]], axis=1)


def _aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by Date and compute:
      - mean of each VADER score across posts published that day
      - n_posts: number of posts published that day
    """
    agg = (
        df.groupby("Date")[_SCORE_COLS]
        .mean()
        .round(4)
        .reset_index()
    )
    counts = df.groupby("Date").size().reset_index(name="n_posts")
    return agg.merge(counts, on="Date").sort_values("Date").reset_index(drop=True)


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved {len(df):>6,} rows  ->  {path.relative_to(ROOT)}")


# ── Public API ────────────────────────────────────────────────────────────────

def analyse_sentiment(
    input_path: Path = INPUT_PATH,
    posts_output: Path = POSTS_OUTPUT,
    daily_output: Path = DAILY_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full VADER pipeline for the preprocessed Reddit WSB dataset.

    Steps
    -----
    1. Ensure the VADER lexicon is downloaded.
    2. Load reddit_wsb_processed.csv.
    3. Score each post — produces vader_neg, vader_neu, vader_pos, vader_compound.
    4. Save per-post scores to reddit_wsb_vader_posts.csv.
    5. Aggregate scores by Date (mean) and count posts per day.
    6. Save daily aggregate to reddit_wsb_vader_daily.csv.

    Returns
    -------
    (posts_df, daily_df) — both as DataFrames.
    """
    _ensure_vader_lexicon()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"  Loading  ->  {input_path.relative_to(ROOT)}")
    df = pd.read_csv(input_path)
    print(f"  Rows loaded  : {len(df):,}")
    print(f"  Date range   : {df['Date'].min()}  to  {df['Date'].max()}")

    # ── 2. Score each post ────────────────────────────────────────────────────
    sia = SentimentIntensityAnalyzer()
    posts_df = _apply_vader(df, sia)
    print(f"  VADER scores computed for {len(posts_df):,} posts.")

    # ── 3. Save per-post file ─────────────────────────────────────────────────
    _save(posts_df, posts_output)

    # ── 4. Daily aggregation ──────────────────────────────────────────────────
    daily_df = _aggregate_daily(posts_df)
    print(f"  Daily aggregation: {len(daily_df):,} unique dates.")

    # ── 5. Save daily file ────────────────────────────────────────────────────
    _save(daily_df, daily_output)

    return posts_df, daily_df
