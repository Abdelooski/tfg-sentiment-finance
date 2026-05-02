"""
src/visualization/tfg_figures.py

Generates all figures required for the TFG report.
Each function is independent and skips gracefully if its input file is missing.

Output: reports/figures/  (PNG, dpi=300)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
PROC_DIR    = ROOT / "data" / "processed"
TABLE_DIR   = ROOT / "reports" / "tables"
FIG_DIR     = ROOT / "reports" / "figures"

# ── Style ──────────────────────────────────────────────────────────────────────
ASSET_COLORS = {"BTC": "#f7931a", "SP500": "#1f77b4"}
MODEL_COLORS = {
    "Logistic Regression": "#4c72b0",
    "Random Forest":       "#55a868",
    "XGBoost":             "#c44e52",
}
SENTIMENT_COLOR = "#8e44ad"
FINBERT_COLOR   = "#e67e22"

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [SKIP] File not found: {path.relative_to(ROOT)}")
        return None
    return pd.read_csv(path, dtype={"Date": str})


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out.relative_to(ROOT)}")


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


# ── Figure 1: BTC Price + Sentiment Overlay ────────────────────────────────────

def fig_btc_price_sentiment() -> None:
    print("\n[1] BTC Price + Sentiment Overlay")
    btc   = _load(PROC_DIR / "btc_processed.csv")
    vader = _load(PROC_DIR / "reddit_wsb_vader_daily.csv")
    if btc is None or vader is None:
        return

    btc   = _parse_dates(btc)
    vader = _parse_dates(vader)
    merged = btc.merge(vader[["Date", "vader_compound"]], on="Date", how="inner")

    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(merged["Date"], merged["Close"], color=ASSET_COLORS["BTC"],
             linewidth=1.4, label="BTC Close Price")
    ax1.fill_between(merged["Date"], merged["Close"],
                     alpha=0.07, color=ASSET_COLORS["BTC"])
    ax1.set_ylabel("Close Price (USD)", color=ASSET_COLORS["BTC"], fontsize=11)
    ax1.tick_params(axis="y", labelcolor=ASSET_COLORS["BTC"])
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax2 = ax1.twinx()
    ax2.bar(merged["Date"], merged["vader_compound"],
            color=SENTIMENT_COLOR, alpha=0.35, width=1.5, label="VADER Compound")
    ax2.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax2.set_ylabel("VADER Compound Sentiment", color=SENTIMENT_COLOR, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=SENTIMENT_COLOR)
    ax2.set_ylim(-1, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

    ax1.set_title("Bitcoin (BTC-USD) Close Price and Reddit VADER Sentiment (2020–2023)",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Date")
    fig.tight_layout()
    _save(fig, "btc_price_sentiment.png")


# ── Figure 2: SP500 Price + Sentiment Overlay ──────────────────────────────────

def fig_sp500_price_sentiment() -> None:
    print("\n[2] SP500 Price + Sentiment Overlay")
    sp5   = _load(PROC_DIR / "sp500_processed.csv")
    vader = _load(PROC_DIR / "reddit_wsb_vader_daily.csv")
    if sp5 is None or vader is None:
        return

    sp5   = _parse_dates(sp5)
    vader = _parse_dates(vader)
    merged = sp5.merge(vader[["Date", "vader_compound"]], on="Date", how="inner")

    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(merged["Date"], merged["Close"], color=ASSET_COLORS["SP500"],
             linewidth=1.4, label="S&P 500 Close")
    ax1.fill_between(merged["Date"], merged["Close"],
                     alpha=0.07, color=ASSET_COLORS["SP500"])
    ax1.set_ylabel("Index Value", color=ASSET_COLORS["SP500"], fontsize=11)
    ax1.tick_params(axis="y", labelcolor=ASSET_COLORS["SP500"])
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax2 = ax1.twinx()
    ax2.bar(merged["Date"], merged["vader_compound"],
            color=SENTIMENT_COLOR, alpha=0.35, width=1.5, label="VADER Compound")
    ax2.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax2.set_ylabel("VADER Compound Sentiment", color=SENTIMENT_COLOR, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=SENTIMENT_COLOR)
    ax2.set_ylim(-1, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

    ax1.set_title("S&P 500 Close Price and Reddit VADER Sentiment (2020–2023)",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Date")
    fig.tight_layout()
    _save(fig, "sp500_price_sentiment.png")


# ── Figure 3: Sentiment Evolution (VADER + FinBERT) ───────────────────────────

def fig_sentiment_evolution() -> None:
    print("\n[3] Sentiment Evolution (VADER vs FinBERT)")
    vader   = _load(PROC_DIR / "reddit_wsb_vader_daily.csv")
    finbert = _load(PROC_DIR / "reddit_wsb_finbert_daily.csv")
    if vader is None:
        return

    vader = _parse_dates(vader)

    fig, ax = plt.subplots(figsize=(14, 4))

    # 30-day rolling mean for readability
    vader["rolling"] = vader["vader_compound"].rolling(30, min_periods=1).mean()
    ax.plot(vader["Date"], vader["vader_compound"],
            color=SENTIMENT_COLOR, alpha=0.25, linewidth=0.8, label="VADER (daily)")
    ax.plot(vader["Date"], vader["rolling"],
            color=SENTIMENT_COLOR, linewidth=2.0, label="VADER (30-day MA)")

    if finbert is not None:
        finbert = _parse_dates(finbert)
        finbert["rolling"] = finbert["finbert_compound"].rolling(30, min_periods=1).mean()
        ax.plot(finbert["Date"], finbert["finbert_compound"],
                color=FINBERT_COLOR, alpha=0.20, linewidth=0.8, label="FinBERT (daily)")
        ax.plot(finbert["Date"], finbert["rolling"],
                color=FINBERT_COLOR, linewidth=2.0, label="FinBERT (30-day MA)")

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Sentiment Score")
    ax.set_xlabel("Date")
    ax.set_title("Reddit WSB Sentiment Evolution — VADER and FinBERT (2020–2023)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, "sentiment_evolution.png")


# ── Figure 4: Posts Volume Over Time ──────────────────────────────────────────

def fig_posts_volume() -> None:
    print("\n[4] Posts Volume Over Time")
    vader = _load(PROC_DIR / "reddit_wsb_vader_daily.csv")
    if vader is None:
        return

    vader = _parse_dates(vader)
    vader["month"] = vader["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = vader.groupby("month")["n_posts"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(monthly["month"], monthly["n_posts"], width=20,
           color="#2ecc71", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    ax.set_ylabel("Number of Posts")
    ax.set_xlabel("Month")
    ax.set_title("Reddit r/WallStreetBets — Monthly Post Volume (2020–2023)",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    _save(fig, "posts_volume.png")


# ── Figure 5: Feature Importance (Sentiment Features) ─────────────────────────

def fig_feature_importance() -> None:
    print("\n[5] Feature Importance — Sentiment Features")
    fi = _load(TABLE_DIR / "feature_importance_sentiment.csv")
    if fi is None:
        return

    assets = fi["asset"].unique()
    fig, axes = plt.subplots(1, len(assets), figsize=(14, 5), sharey=False)
    if len(assets) == 1:
        axes = [axes]

    for ax, asset in zip(axes, sorted(assets)):
        sub = (fi[fi["asset"] == asset]
               .sort_values("importance", ascending=True)
               .tail(10))
        color = ASSET_COLORS.get(asset, "#555555")
        bars = ax.barh(sub["feature"], sub["importance"],
                       color=color, alpha=0.8, edgecolor="white")
        ax.set_xlabel("Importance (Gini Impurity Reduction)")
        ax.set_title(f"{asset} — Sentiment Feature Importance",
                     fontsize=11, fontweight="bold")
        # Value labels
        for bar, val in zip(bars, sub["importance"]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left", fontsize=8)

    fig.suptitle("Random Forest Feature Importance — Sentiment Features",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "feature_importance_sentiment.png")


# ── Figure 6: Ablation Results (F1 Delta) ─────────────────────────────────────

def fig_ablation() -> None:
    print("\n[6] Ablation Results")
    ab = _load(TABLE_DIR / "ablation_summary.csv")
    if ab is None:
        return

    assets  = sorted(ab["asset"].unique())
    models  = ab["model"].unique()
    x       = np.arange(len(models))
    width   = 0.35
    fig, axes = plt.subplots(1, len(assets), figsize=(13, 5), sharey=True)
    if len(assets) == 1:
        axes = [axes]

    for ax, asset in zip(axes, assets):
        sub = ab[ab["asset"] == asset].set_index("model").reindex(models)
        colors = [ASSET_COLORS.get(asset, "#888") if v >= 0 else "#c0392b"
                  for v in sub["f1_delta"]]
        bars = ax.bar(x, sub["f1_delta"], width=0.55,
                      color=colors, alpha=0.85, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.set_title(f"{asset}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Model")

        for bar, val in zip(bars, sub["f1_delta"]):
            sign = "+" if val >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (0.002 if val >= 0 else -0.004),
                    f"{sign}{val:.3f}",
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=9, fontweight="bold")

    axes[0].set_ylabel("F1 Delta (Price+Sentiment − Price Only)")
    fig.suptitle("Ablation Analysis — Sentiment Contribution to F1-Score",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "ablation_f1_delta.png")


# ── Figure 7: Windowed Analysis (F1 Delta by Horizon) ─────────────────────────

def fig_windowed() -> None:
    print("\n[7] Windowed Analysis")
    wa = _load(TABLE_DIR / "windowed_ablation_summary.csv")
    if wa is None:
        return

    # Mean delta per (asset, target)
    summary = (wa.groupby(["asset", "target"])["f1_delta"]
                 .mean()
                 .reset_index())
    summary["horizon"] = summary["target"].str.replace("Direction_", "").str.upper()

    assets   = sorted(summary["asset"].unique())
    horizons = ["3D", "5D", "7D"]
    x        = np.arange(len(horizons))
    width    = 0.35
    fig, ax  = plt.subplots(figsize=(10, 5))

    offsets = np.linspace(-width / 2 * (len(assets) - 1),
                          width / 2 * (len(assets) - 1),
                          len(assets))

    for offset, asset in zip(offsets, assets):
        vals = [summary[(summary["asset"] == asset) &
                        (summary["horizon"] == h)]["f1_delta"].values
                for h in horizons]
        vals = [v[0] if len(v) else 0.0 for v in vals]
        color = ASSET_COLORS.get(asset, "#888")
        bars = ax.bar(x + offset, vals, width=width,
                      label=asset, color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            sign = "+" if val >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (0.003 if val >= 0 else -0.005),
                    f"{sign}{val:.3f}",
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=8.5, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} Horizon" for h in horizons], fontsize=11)
    ax.set_ylabel("Mean F1 Delta (Price+Sentiment − Price Only)")
    ax.set_xlabel("Prediction Horizon")
    ax.set_title("Multi-Horizon Sentiment Impact on F1-Score — BTC vs S&P 500",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "windowed_f1_delta.png")


# ── Figure 8: VADER vs FinBERT Comparison ─────────────────────────────────────

def fig_vader_finbert() -> None:
    print("\n[8] VADER vs FinBERT Comparison")
    vf = _load(TABLE_DIR / "vader_finbert_summary.csv")
    if vf is None:
        return

    assets = sorted(vf["asset"].unique())
    models = vf["model"].unique()
    x      = np.arange(len(models))
    width  = 0.35

    fig, axes = plt.subplots(1, len(assets), figsize=(13, 5), sharey=True)
    if len(assets) == 1:
        axes = [axes]

    for ax, asset in zip(axes, assets):
        sub = vf[vf["asset"] == asset].set_index("model").reindex(models)
        pos_color = "#27ae60"
        neg_color = "#c0392b"
        colors = [pos_color if v >= 0 else neg_color
                  for v in sub["f1_delta_finbert_minus_vader"]]
        bars = ax.bar(x, sub["f1_delta_finbert_minus_vader"],
                      width=0.55, color=colors, alpha=0.85, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.set_title(f"{asset}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Model")

        for bar, val in zip(bars, sub["f1_delta_finbert_minus_vader"]):
            sign = "+" if val >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (0.002 if val >= 0 else -0.004),
                    f"{sign}{val:.3f}",
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=9, fontweight="bold")

    axes[0].set_ylabel("F1 Delta (FinBERT − VADER)")
    fig.suptitle("Sentiment Model Comparison: FinBERT vs VADER — F1-Score Difference",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "vader_finbert_comparison.png")


# ── Figure 9: Correlation Heatmap (BTC Features) ──────────────────────────────

def fig_correlation_heatmap() -> None:
    print("\n[9] Correlation Heatmap (BTC Features)")
    feat = _load(PROC_DIR / "btc_features.csv")
    if feat is None:
        return

    cols = [
        "Return", "return_lag_1", "return_lag_2", "return_lag_3",
        "return_rolling_3", "return_rolling_7", "volatility_7",
        "vader_compound", "vader_compound_lag_1", "vader_compound_lag_2",
        "vader_compound_lag_3", "vader_compound_rolling_3", "vader_compound_rolling_7",
        "n_posts_log",
    ]
    cols = [c for c in cols if c in feat.columns]
    corr = feat[cols].corr()

    labels = {
        "Return":                    "Return",
        "return_lag_1":              "Return (t-1)",
        "return_lag_2":              "Return (t-2)",
        "return_lag_3":              "Return (t-3)",
        "return_rolling_3":          "Return MA-3",
        "return_rolling_7":          "Return MA-7",
        "volatility_7":              "Volatility-7",
        "vader_compound":            "Sentiment",
        "vader_compound_lag_1":      "Sentiment (t-1)",
        "vader_compound_lag_2":      "Sentiment (t-2)",
        "vader_compound_lag_3":      "Sentiment (t-3)",
        "vader_compound_rolling_3":  "Sentiment MA-3",
        "vader_compound_rolling_7":  "Sentiment MA-7",
        "n_posts_log":               "Log(Posts)",
    }
    corr.index   = [labels.get(c, c) for c in corr.index]
    corr.columns = [labels.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.4, linecolor="white",
        square=True, ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title("Feature Correlation Matrix — BTC Price and Sentiment Features",
                 fontsize=13, fontweight="bold", pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()
    _save(fig, "correlation_heatmap_btc.png")


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_all_figures() -> None:
    """Generate all TFG figures. Each function skips safely if inputs are missing."""
    print("=" * 60)
    print("  Generating TFG figures")
    print("=" * 60)

    fig_btc_price_sentiment()
    fig_sp500_price_sentiment()
    fig_sentiment_evolution()
    fig_posts_volume()
    fig_feature_importance()
    fig_ablation()
    fig_windowed()
    fig_vader_finbert()
    fig_correlation_heatmap()

    print("\n" + "=" * 60)
    print("  Done. All figures saved to reports/figures/")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_figures()
