"""
Entry point for the sentiment-finance pipeline.
Orchestrates data ingestion, NLP processing, feature engineering, and model training.
"""

from src.data.data_loader import load_and_save_all
from src.data.fear_greed_loader import OUTPUT_PATH as FNG_RAW_PATH
from src.data.fear_greed_loader import load_fear_greed
from src.features.feature_engineering import ASSET_CONFIG as FEATURES_CONFIG
from src.features.feature_engineering import engineer_all
from src.features.merge_datasets import VADER_PATH as VADER_DAILY
from src.features.merge_datasets import merge_all
from src.features.btc_fear_greed_features import BTC_FEATURES_PATH as FNG_BTC_FEAT
from src.features.btc_fear_greed_features import FNG_PATH
from src.features.btc_fear_greed_features import run_btc_fng_analysis
from src.features.finbert_feature_engineering import ASSET_CONFIG as FINBERT_FEAT_CONFIG
from src.features.finbert_feature_engineering import engineer_finbert_all
from src.features.merge_finbert import ASSET_CONFIG as FINBERT_MERGE_CONFIG
from src.features.merge_finbert import FINBERT_DAILY_PATH
from src.features.merge_finbert import merge_finbert_all
from src.models.compare_vader_finbert import FINBERT_FEAT_CONFIG as CMP_FINBERT
from src.models.compare_vader_finbert import VADER_FEAT_CONFIG as CMP_VADER
from src.models.compare_vader_finbert import run_vader_finbert_comparison
from src.features.window_targets import ASSET_CONFIG as WINDOW_FEAT_CONFIG
from src.features.window_targets import generate_windowed_targets
from src.analysis.granger_causality import ASSET_CONFIG as GRANGER_FEAT_CONFIG
from src.analysis.granger_causality import run_granger_analysis
from src.models.windowed_model_analysis import ASSET_CONFIG as WINDOWED_MODEL_CONFIG
from src.models.windowed_model_analysis import run_windowed_analysis
from src.models.ablation_analysis import ASSET_CONFIG as ABLATION_FEAT_CONFIG
from src.models.ablation_analysis import run_ablation
from src.models.feature_importance_analysis import ASSET_CONFIG as FI_FEAT_CONFIG
from src.models.feature_importance_analysis import run_feature_importance
from src.models.statistical_analysis import ASSET_CONFIG as STATS_FEAT_CONFIG
from src.models.statistical_analysis import RESULTS_PATH as STATS_MODEL_RESULTS
from src.models.statistical_analysis import run_analysis
from src.models.train_models import ASSET_CONFIG as MODELS_CONFIG
from src.models.train_models import train_all
from src.nlp.finbert_sentiment import INPUT_PATH as FINBERT_INPUT
from src.nlp.finbert_sentiment import apply_finbert
from src.nlp.reddit_preprocessing import RAW_PATH as REDDIT_RAW
from src.nlp.reddit_preprocessing import preprocess_reddit
from src.nlp.vader_sentiment import INPUT_PATH as VADER_INPUT
from src.nlp.vader_sentiment import analyse_sentiment
from src.reports.summary_tables import RESULTS_PATH as SUMMARY_INPUT
from src.reports.summary_tables import generate_summary
from src.visualization.model_plots import RESULTS_PATH as MODEL_RESULTS
from src.visualization.model_plots import generate_model_plots
from src.visualization.price_plots import generate_price_plots


def _section(n: int, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Step {n} -- {title}")
    print(f"{'=' * 60}")


def main() -> None:
    # ── Step 1: Data ingestion ────────────────────────────────────────────────
    _section(1, "Data ingestion")
    data = load_and_save_all()

    print("\nSummary")
    print("-" * 40)
    for name, df in data.items():
        print(f"  {name.upper():<8}  shape={df.shape}  columns={list(df.columns)}")
    print("-" * 40)

    # ── Step 2: Price visualisations ─────────────────────────────────────────
    _section(2, "Price plots")
    generate_price_plots()

    # ── Step 3: Reddit preprocessing (optional) ───────────────────────────────
    _section(3, "Reddit WSB preprocessing")
    if REDDIT_RAW.exists():
        preprocess_reddit()
    else:
        print(f"  [SKIPPED] {REDDIT_RAW.name} not found in data/raw/")
        print("  Place reddit_wsb.csv in data/raw/ and re-run to enable this step.")

    # ── Step 4: VADER sentiment analysis (optional) ───────────────────────────
    _section(4, "VADER sentiment analysis")
    if VADER_INPUT.exists():
        posts_df, daily_df = analyse_sentiment()
        print(f"\n  Posts df shape : {posts_df.shape}")
        print(f"  Daily df shape : {daily_df.shape}")
        print(f"\n  Daily preview (first 3 rows):")
        print(daily_df.head(3).to_string(index=False))
    else:
        print(f"  [SKIPPED] {VADER_INPUT.name} not found in data/processed/")
        print("  Run Step 3 first (requires reddit_wsb.csv in data/raw/).")

    # ── Step 3b: FinBERT inference on Reddit posts (optional) ────────────────
    _section("3b", "FinBERT sentiment inference")
    if FINBERT_INPUT.exists():
        fb_posts, fb_daily = apply_finbert()
        print(f"\n  Posts df shape : {fb_posts.shape}")
        print(f"  Daily df shape : {fb_daily.shape}")
    else:
        print(f"  [SKIPPED] {FINBERT_INPUT.name} not found in data/processed/")
        print("  Run Step 3 first (requires reddit_wsb.csv in data/raw/).")

    # ── Step 4b: Merge FinBERT sentiment with assets (optional) ──────────────
    _section("4b", "Merge FinBERT sentiment with assets")
    if FINBERT_DAILY_PATH.exists():
        finbert_merged = merge_finbert_all()
        for name, df in finbert_merged.items():
            print(f"  {name:<6}  shape={df.shape}")
    else:
        print(f"  [SKIPPED] {FINBERT_DAILY_PATH.name} not found in data/processed/")
        print("  Run Step 19 (apply_finbert) first to generate FinBERT daily scores.")

    # ── Step 4c: FinBERT feature engineering (optional) ───────────────────────
    _section("4c", "FinBERT feature engineering")
    finbert_merge_exist = any(
        in_path.exists() for in_path, _ in FINBERT_MERGE_CONFIG.values()
    )
    if finbert_merge_exist:
        finbert_features = engineer_finbert_all()
        for name, df in finbert_features.items():
            print(f"  {name:<6}  shape={df.shape}")
    else:
        print("  [SKIPPED] FinBERT merged CSVs not found in data/processed/")
        print("  Run Step 4b first to generate the merged FinBERT datasets.")

    # ── Step 4d: VADER vs FinBERT model comparison (optional) ────────────────
    _section("4d", "VADER vs FinBERT model comparison")
    vader_exist   = any(p.exists() for p in CMP_VADER.values())
    finbert_exist = any(p.exists() for p in CMP_FINBERT.values())
    if vader_exist and finbert_exist:
        cmp_df, cmp_summary = run_vader_finbert_comparison()
        print(f"\n  Rows in vader_finbert_comparison.csv : {len(cmp_df)}")
        print(f"  Rows in vader_finbert_summary.csv    : {len(cmp_summary)}")
    else:
        print("  [SKIPPED] VADER or FinBERT feature CSVs not found.")
        print("  Run Steps 3-6 (VADER) and Steps 4b-4c (FinBERT) first.")

    # ── Step 5: Merge sentiment with price data (optional) ───────────────────
    _section(5, "Merge VADER sentiment with BTC and S&P500")
    if VADER_DAILY.exists():
        merged = merge_all()
        print(f"\n  Assets merged: {list(merged.keys())}")
    else:
        print(f"  [SKIPPED] {VADER_DAILY.name} not found in data/processed/")
        print("  Run Steps 3 and 4 first to generate the VADER daily file.")

    # ── Step 6: Feature engineering (optional) ────────────────────────────────
    _section(6, "Feature engineering")
    merged_files_exist = any(in_path.exists() for in_path, _ in FEATURES_CONFIG.values())
    if merged_files_exist:
        engineer_all()
    else:
        print("  [SKIPPED] No merged datasets found in data/processed/")
        print("  Run Steps 3-5 first to generate the merged files.")

    # ── Step 7: Model training and evaluation (optional) ─────────────────────
    _section(7, "Model training and evaluation")
    feature_files_exist = any(feat.exists() for feat, _ in MODELS_CONFIG.values())
    if feature_files_exist:
        results_df = train_all()
        print(f"\n  Full results table:")
        print(results_df.to_string(index=False))
    else:
        print("  [SKIPPED] No feature datasets found in data/processed/")
        print("  Run Steps 3-6 first to generate the feature files.")

    # ── Step 8: Model comparison plots (optional) ────────────────────────────
    _section(8, "Model comparison plots")
    if MODEL_RESULTS.exists():
        generate_model_plots()
    else:
        print(f"  [SKIPPED] {MODEL_RESULTS.name} not found in reports/tables/")
        print("  Run Step 7 first to generate model results.")

    # ── Step 9: Best-model summary table (optional) ───────────────────────────
    _section(9, "Best-model summary")
    if SUMMARY_INPUT.exists():
        generate_summary()
    else:
        print(f"  [SKIPPED] {SUMMARY_INPUT.name} not found in reports/tables/")
        print("  Run Step 7 first to generate model results.")

    # ── Step 10: Statistical analysis (optional) ──────────────────────────────
    _section(10, "Statistical analysis (correlations + baseline)")
    stats_feat_exist = any(p.exists() for p in STATS_FEAT_CONFIG.values())
    if stats_feat_exist and STATS_MODEL_RESULTS.exists():
        stats_df = run_analysis()
        print(f"\n  Rows in statistical_tests.csv: {len(stats_df)}")
    else:
        print("  [SKIPPED] Feature CSVs or model_results.csv not found.")
        print("  Run Steps 3-7 first to generate the required inputs.")

    # ── Step 11: Ablation analysis (optional) ─────────────────────────────────
    _section(11, "Ablation analysis (price-only vs price+sentiment)")
    ablation_feat_exist = any(p.exists() for p in ABLATION_FEAT_CONFIG.values())
    if ablation_feat_exist:
        abl_results, abl_summary = run_ablation()
        print(f"\n  Rows in ablation_results.csv : {len(abl_results)}")
        print(f"  Rows in ablation_summary.csv : {len(abl_summary)}")
    else:
        print("  [SKIPPED] Feature CSVs not found in data/processed/")
        print("  Run Steps 3-6 first to generate the feature files.")

    # ── Step 12: Feature importance analysis (optional) ───────────────────────
    _section(12, "Feature importance analysis (Random Forest)")
    fi_feat_exist = any(p.exists() for p in FI_FEAT_CONFIG.values())
    if fi_feat_exist:
        fi_df, fi_sentiment_df = run_feature_importance()
        print(f"\n  Rows in feature_importance.csv          : {len(fi_df)}")
        print(f"  Rows in feature_importance_sentiment.csv : {len(fi_sentiment_df)}")
    else:
        print("  [SKIPPED] Feature CSVs not found in data/processed/")
        print("  Run Steps 3-6 first to generate the feature files.")

    # ── Step 12b: Download Fear & Greed Index (optional) ─────────────────────
    _section("12b", "Fear & Greed Index download")
    if not FNG_RAW_PATH.exists():
        try:
            fng_raw = load_fear_greed()
            print(f"\n  Rows downloaded : {len(fng_raw):,}")
        except RuntimeError as exc:
            print(f"  [AVISO] No se pudo descargar el Fear & Greed Index: {exc}")
            print("  El Step 13 se omitirá si fear_greed.csv no existe.")
    else:
        print(f"  [OK] {FNG_RAW_PATH.name} ya existe — omitiendo descarga.")

    # ── Step 13: BTC Fear & Greed integration (optional) ──────────────────────
    _section(13, "BTC Fear & Greed Index integration")
    if FNG_BTC_FEAT.exists() and FNG_PATH.exists():
        fng_results = run_btc_fng_analysis()
        print(f"\n  Rows in btc_fng_model_results.csv : {len(fng_results)}")
    else:
        print("  [SKIPPED] btc_features.csv or fear_greed.csv not found.")
        print("  Run Steps 3-6 and fear_greed_loader first.")

    # ── Step 14: Multi-horizon target generation (optional) ───────────────────
    _section(14, "Multi-horizon target generation (3d / 5d / 7d)")
    window_feat_exist = any(p.exists() for p, _ in WINDOW_FEAT_CONFIG.values())
    if window_feat_exist:
        windowed = generate_windowed_targets()
        for name, df in windowed.items():
            print(f"  {name:<6}  shape={df.shape}")
    else:
        print("  [SKIPPED] Feature CSVs not found in data/processed/")
        print("  Run Steps 3-6 first to generate the feature files.")

    # ── Step 15: Windowed model analysis (optional) ────────────────────────────
    _section(15, "Windowed model analysis (sentiment over 3d / 5d / 7d)")
    windowed_exist = any(p.exists() for p in WINDOWED_MODEL_CONFIG.values())
    if windowed_exist:
        win_results, win_summary = run_windowed_analysis()
        print(f"\n  Rows in windowed_model_results.csv    : {len(win_results)}")
        print(f"  Rows in windowed_ablation_summary.csv : {len(win_summary)}")
    else:
        print("  [SKIPPED] Windowed feature CSVs not found in data/processed/")
        print("  Run Step 14 first to generate windowed datasets.")

    # ── Step 16: Granger causality analysis (optional) ────────────────────────
    _section(16, "Granger causality analysis")
    granger_feat_exist = any(p.exists() for p in GRANGER_FEAT_CONFIG.values())
    if granger_feat_exist:
        granger_df = run_granger_analysis()
        print(f"\n  Rows in granger_results.csv : {len(granger_df)}")
    else:
        print("  [SKIPPED] Feature CSVs not found in data/processed/")
        print("  Run Steps 3-6 first to generate the feature files.")

    print("\nPipeline complete.\n")


if __name__ == "__main__":
    main()
