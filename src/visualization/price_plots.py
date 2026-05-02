"""
src/visualization/price_plots.py

Generates and saves Close-price line charts for each configured asset.
Reads from data/processed/, writes PNGs to reports/figures/.
Pure matplotlib — no seaborn dependency.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

# ── Plot style ────────────────────────────────────────────────────────────────
STYLE: dict = {
    "figure.figsize": (14, 5),
    "axes.facecolor": "#f8f9fa",
    "figure.facecolor": "#ffffff",
    "axes.grid": True,
    "grid.color": "#d0d0d0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}

# Maps asset name to (CSV filename, chart colour, y-axis label, chart title)
ASSET_CONFIG: dict[str, tuple[str, str, str, str]] = {
    "btc": (
        "btc_processed.csv",
        "#f7931a",          # Bitcoin orange
        "Price (USD)",
        "Bitcoin (BTC-USD) — Daily Close Price",
    ),
    "sp500": (
        "sp500_processed.csv",
        "#1f77b4",          # Steel blue
        "Index Value",
        "S&P 500 (^GSPC) — Daily Close Price",
    ),
}


# ── Private helpers ───────────────────────────────────────────────────────────

def _load(csv_path: Path) -> pd.DataFrame:
    """Read a processed CSV and parse the Date column as datetime."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    if df.empty:
        raise ValueError(f"Processed file is empty: {csv_path}")
    return df


def _plot_price(
    df: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    color: str,
    out_path: Path,
) -> None:
    """
    Draw a single Close-price line chart and save it as a PNG.

    Parameters
    ----------
    df       : DataFrame with at least 'Date' and 'Close' columns.
    title    : Chart title string.
    ylabel   : Label for the y-axis.
    color    : Hex or named colour for the price line.
    out_path : Destination file path (must end in .png).
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()

        # Main price line with a light fill beneath it
        ax.plot(df["Date"], df["Close"], color=color, linewidth=1.4, label="Close")
        ax.fill_between(df["Date"], df["Close"], alpha=0.08, color=color)

        # X-axis: one label per year, minor tick every quarter
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        # Y-axis: comma-separated thousands
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"
        ))

        ax.set_title(title, pad=14)
        ax.set_xlabel("Date")
        ax.set_ylabel(ylabel)
        ax.set_xlim(df["Date"].min(), df["Date"].max())
        ax.legend(frameon=False)

        # Annotate the last closing value
        last_date = df["Date"].iloc[-1]
        last_close = df["Close"].iloc[-1]
        ax.annotate(
            f"{last_close:,.2f}",
            xy=(last_date, last_close),
            xytext=(-60, 12),
            textcoords="offset points",
            fontsize=9,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"    Saved figure  ->  {out_path.relative_to(ROOT)}")


# ── Public API ────────────────────────────────────────────────────────────────

def generate_price_plots() -> None:
    """
    Generate one Close-price chart per configured asset and save as PNG.
    Reads from data/processed/, writes to reports/figures/.
    """
    for name, (csv_file, color, ylabel, title) in ASSET_CONFIG.items():
        csv_path = PROCESSED_DIR / csv_file
        print(f"\n[{name.upper()}]  Reading {csv_path.relative_to(ROOT)}")

        df = _load(csv_path)

        _plot_price(
            df,
            title=title,
            ylabel=ylabel,
            color=color,
            out_path=FIGURES_DIR / f"{name}_close_price.png",
        )
