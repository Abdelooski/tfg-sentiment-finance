"""
src/models/train_models.py

Entrena tres clasificadores (Logistic Regression, Random Forest, XGBoost)
sobre las features temporales de BTC y S&P500 para predecir la dirección
del precio del día siguiente (Direction = 1 si Return[t+1] > 0).

Split temporal: 80% train / 20% test — sin shuffle para respetar la
causalidad de la serie temporal (nunca usamos datos futuros para entrenar).

Salidas:
  reports/tables/model_results.csv   — métricas de evaluación
  data/processed/btc_predictions.csv
  data/processed/sp500_predictions.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "reports" / "tables"

RESULTS_PATH = TABLES_DIR / "model_results.csv"

# Columns that are not features: identifier + target
_NON_FEATURE_COLS = {"Date", "Direction"}

TARGET_COL = "Direction"
TRAIN_RATIO = 0.80

# Maps asset name -> (feature CSV, predictions CSV)
ASSET_CONFIG: dict[str, tuple[Path, Path]] = {
    "btc": (
        PROCESSED_DIR / "btc_features.csv",
        PROCESSED_DIR / "btc_predictions.csv",
    ),
    "sp500": (
        PROCESSED_DIR / "sp500_features.csv",
        PROCESSED_DIR / "sp500_predictions.csv",
    ),
}


# ── Model definitions ─────────────────────────────────────────────────────────

def _build_models() -> dict[str, object]:
    """
    Return a fresh dict of untrained models.

    Logistic Regression is wrapped in a Pipeline with StandardScaler because
    it is sensitive to feature magnitude. Tree-based models (RF, XGB) are
    scale-invariant and need no preprocessing.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
    }


# ── Data loading and splitting ────────────────────────────────────────────────

def _load(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load a feature CSV and return (X, y, dates).

    X     : DataFrame of all feature columns (everything except Date and Direction).
    y     : Series with the Direction target.
    dates : Series with the Date column for labelling predictions.
    """
    df = pd.read_csv(path, dtype={"Date": str})
    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    return df[feature_cols], df[TARGET_COL], df["Date"]


def _temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    ratio: float = TRAIN_RATIO,
) -> tuple:
    """
    Split (X, y, dates) into train and test preserving chronological order.

    Using iloc[:split] / iloc[split:] guarantees that the model never
    sees future information during training — essential for time series.
    """
    split = int(len(X) * ratio)
    return (
        X.iloc[:split], X.iloc[split:],
        y.iloc[:split], y.iloc[split:],
        dates.iloc[split:],   # only test dates are needed for the predictions CSV
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(y_true: pd.Series, y_pred) -> dict[str, float]:
    """Compute the four classification metrics on the test set."""
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred),  4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred,        zero_division=0), 4),
    }


# ── Per-asset training loop ───────────────────────────────────────────────────

def _train_asset(
    name: str,
    feat_path: Path,
    pred_path: Path,
) -> list[dict]:
    """
    Train all three models on one asset, print a results table, save predictions.

    Returns a list of result dicts (one per model) to be aggregated later.
    """
    print(f"\n  Loading {name.upper()}  ->  {feat_path.relative_to(ROOT)}")
    X, y, dates = _load(feat_path)

    X_train, X_test, y_train, y_test, test_dates = _temporal_split(X, y, dates)

    n_train, n_test = len(X_train), len(X_test)
    print(f"  Train rows : {n_train}  ({X_train.index[0]}..{X_train.index[-1]})")
    print(f"  Test  rows : {n_test}  ({X_test.index[0]}..{X_test.index[-1]})")
    print(f"  Features   : {X.shape[1]}")

    models = _build_models()
    results = []
    predictions_df = pd.DataFrame({"Date": test_dates.values, "Direction": y_test.values})

    # Header for the per-asset console table
    print(f"\n  {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = _evaluate(y_test, y_pred)

        # Append prediction column (one per model)
        col_key = model_name.lower().replace(" ", "_")
        predictions_df[f"pred_{col_key}"] = y_pred

        # Train accuracy — quick sanity check for overfitting
        train_acc = round(accuracy_score(y_train, model.predict(X_train)), 4)

        print(
            f"  {model_name:<22} "
            f"{metrics['accuracy']:>6.4f} "
            f"{metrics['precision']:>6.4f} "
            f"{metrics['recall']:>6.4f} "
            f"{metrics['f1']:>6.4f}"
            f"  (train acc: {train_acc:.4f})"
        )

        results.append({
            "asset":     name.upper(),
            "model":     model_name,
            "accuracy":  metrics["accuracy"],
            "precision": metrics["precision"],
            "recall":    metrics["recall"],
            "f1":        metrics["f1"],
            "train_acc": train_acc,
            "n_train":   n_train,
            "n_test":    n_test,
        })

    # Save predictions for this asset
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(pred_path, index=False)
    print(f"\n  Predictions saved  ->  {pred_path.relative_to(ROOT)}")

    return results


# ── Public API ────────────────────────────────────────────────────────────────

def train_all(
    asset_config: dict[str, tuple[Path, Path]] = ASSET_CONFIG,
    results_path: Path = RESULTS_PATH,
) -> pd.DataFrame:
    """
    Train and evaluate all models on all configured assets.

    Steps
    -----
    1. For each asset: load features, split temporally, train 3 models,
       print metrics, save predictions CSV.
    2. Concatenate all results into a single DataFrame.
    3. Save the results table to reports/tables/model_results.csv.

    Returns
    -------
    pd.DataFrame with one row per (asset, model) combination.
    """
    all_results: list[dict] = []

    for name, (feat_path, pred_path) in asset_config.items():
        asset_results = _train_asset(name, feat_path, pred_path)
        all_results.extend(asset_results)

    results_df = pd.DataFrame(all_results)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\n  Results table saved  ->  {results_path.relative_to(ROOT)}")

    return results_df
