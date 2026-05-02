"""
src/visualization/model_plots.py

Genera gráficas de barras agrupadas que comparan las métricas de
clasificación (F1 y Accuracy) entre modelos y activos financieros.

Entrada : reports/tables/model_results.csv
Salidas : reports/figures/model_f1_comparison.png
          reports/figures/model_accuracy_comparison.png

Sólo usa matplotlib — sin seaborn.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"

RESULTS_PATH = TABLES_DIR / "model_results.csv"

# ── Visual constants ──────────────────────────────────────────────────────────
# Asset colours: consistent with price_plots.py
ASSET_COLORS: dict[str, str] = {
    "BTC":   "#f7931a",   # Bitcoin orange
    "SP500": "#1f77b4",   # Steel blue
}

STYLE: dict = {
    "figure.figsize": (10, 5),
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
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
}

BAR_WIDTH = 0.30   # width of each individual bar
BASELINE = 0.50    # random-classifier baseline for binary classification


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_results(path: Path) -> pd.DataFrame:
    """Load the model results CSV and validate required columns."""
    df = pd.read_csv(path)
    required = {"asset", "model", "accuracy", "f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"model_results.csv is missing columns: {missing}")
    return df


def _pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Reshape the results table so rows are models and columns are assets.

    Example output (metric='f1'):
      asset                   BTC    SP500
      model
      Logistic Regression  0.6400   0.7097
      Random Forest        0.5652   0.6875
      XGBoost              0.5957   0.5333
    """
    return df.pivot(index="model", columns="asset", values=metric)


def _bar_label(ax, bars, fmt: str = "{:.3f}") -> None:
    """Place a value label just above each bar."""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.008,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
        )


def _plot_metric(
    pivot_df: pd.DataFrame,
    *,
    metric_label: str,
    title: str,
    out_path: Path,
) -> None:
    """
    Draw a grouped bar chart comparing one metric across models and assets.

    Layout: one cluster of bars per model, one bar per asset inside each
    cluster. A dashed horizontal line marks the 0.50 random-baseline.

    Parameters
    ----------
    pivot_df     : DataFrame with index=model and columns=asset.
    metric_label : Y-axis label (e.g. 'F1-Score').
    title        : Chart title.
    out_path     : Destination PNG path.
    """
    assets = list(pivot_df.columns)          # e.g. ['BTC', 'SP500']
    models = list(pivot_df.index)            # e.g. ['Logistic Regression', ...]
    n_models = len(models)
    n_assets = len(assets)

    # X positions: one tick per model, bars offset left/right within each cluster
    x = np.arange(n_models)
    # Centre the asset bars symmetrically around each model tick
    offsets = np.linspace(
        -(n_assets - 1) / 2 * BAR_WIDTH,
         (n_assets - 1) / 2 * BAR_WIDTH,
        n_assets,
    )

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()

        for i, asset in enumerate(assets):
            values = pivot_df[asset].values.astype(float)
            color  = ASSET_COLORS.get(asset, "#999999")
            bars = ax.bar(
                x + offsets[i],
                values,
                width=BAR_WIDTH,
                label=asset,
                color=color,
                alpha=0.88,
                edgecolor="white",
                linewidth=0.6,
            )
            _bar_label(ax, bars)

        # Random-baseline reference line
        ax.axhline(
            BASELINE,
            linestyle="--",
            color="#888888",
            linewidth=1.1,
            label=f"Random baseline ({BASELINE:.0%})",
        )

        # Horizontal grid only — cleaner for bar charts
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

        # Axes formatting
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, ha="center")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric_label)
        ax.set_title(title, pad=14)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:.0%}"
        ))
        ax.legend(frameon=False, loc="upper right")

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved figure  ->  {out_path.relative_to(ROOT)}")


# ── Public API ────────────────────────────────────────────────────────────────

def generate_model_plots(
    results_path: Path = RESULTS_PATH,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    """
    Generate and save two grouped bar charts from model_results.csv:
      1. F1-score comparison across models and assets.
      2. Accuracy comparison across models and assets.
    """
    print(f"  Loading  ->  {results_path.relative_to(ROOT)}")
    df = _load_results(results_path)

    charts = [
        ("f1",       "F1-Score",  "F1-Score Comparison by Model and Asset",
         figures_dir / "model_f1_comparison.png"),
        ("accuracy", "Accuracy",  "Accuracy Comparison by Model and Asset",
         figures_dir / "model_accuracy_comparison.png"),
    ]

    for metric, ylabel, title, out_path in charts:
        pivot_df = _pivot(df, metric)
        _plot_metric(pivot_df, metric_label=ylabel, title=title, out_path=out_path)
