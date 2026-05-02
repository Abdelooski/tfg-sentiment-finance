"""
src/models/windowed_model_analysis.py

Evalúa si el sentimiento de Reddit mejora la predicción de dirección de
precio en horizontes multi-día: 3d, 5d y 7d para BTC y S&P500.

La hipótesis de fondo es que el sentimiento puede tener efecto acumulado:
puede que no prediga bien el siguiente día pero sí la tendencia a una semana.
Si BTC muestra deltas positivos en horizontes largos y S&P500 no, refuerza
la conclusión central del TFG.

Entradas:
  data/processed/btc_features_windowed.csv
  data/processed/sp500_features_windowed.csv  (generados por window_targets.py)

Salidas:
  reports/tables/windowed_model_results.csv   — métricas por (activo, target, feature_set, modelo)
  reports/tables/windowed_ablation_summary.csv — delta F1 sentimiento por (activo, target, modelo)
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

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR    = ROOT / "reports" / "tables"

WINDOWED_RESULTS_PATH = TABLES_DIR / "windowed_model_results.csv"
WINDOWED_SUMMARY_PATH = TABLES_DIR / "windowed_ablation_summary.csv"

# Mapea nombre de activo -> CSV con targets multi-horizonte
ASSET_CONFIG: dict[str, Path] = {
    "BTC":   PROCESSED_DIR / "btc_features_windowed.csv",
    "SP500": PROCESSED_DIR / "sp500_features_windowed.csv",
}

# Tres targets: dirección de precio en 3, 5 y 7 días vista
TARGETS: list[str] = ["Direction_3d", "Direction_5d", "Direction_7d"]

TRAIN_RATIO = 0.80

# ── Conjuntos de features ─────────────────────────────────────────────────────

_PRICE_FEATURES: list[str] = [
    "Return",
    "return_lag_1", "return_lag_2", "return_lag_3",
    "return_rolling_3", "return_rolling_7",
    "volatility_7",
]

_SENTIMENT_FEATURES: list[str] = [
    "vader_compound",
    "vader_neg", "vader_neu", "vader_pos",
    "vader_compound_lag_1", "vader_compound_lag_2", "vader_compound_lag_3",
    "vader_compound_rolling_3", "vader_compound_rolling_7",
    "n_posts_log",
]

FEATURE_SETS: dict[str, list[str]] = {
    "price_only":           _PRICE_FEATURES,
    "price_plus_sentiment": _PRICE_FEATURES + _SENTIMENT_FEATURES,
}

# ── Columnas de salida ────────────────────────────────────────────────────────
_RESULTS_COLS: list[str] = [
    "asset", "target", "feature_set", "model",
    "accuracy", "precision", "recall", "f1",
    "n_train", "n_test",
]

_SUMMARY_COLS: list[str] = [
    "asset", "target", "model",
    "f1_price_only", "f1_price_plus_sentiment",
    "f1_delta", "sentiment_improves_model",
]


# ── Modelos ───────────────────────────────────────────────────────────────────

def _build_models() -> dict[str, object]:
    """
    Instancias frescas de los tres clasificadores.
    Hiperparámetros idénticos a los demás módulos del proyecto.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
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


# ── Carga, split y evaluación ─────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame:
    """Carga el CSV windowed y valida que contiene los targets esperados."""
    df = pd.read_csv(path, dtype={"Date": str})
    missing_targets = [t for t in TARGETS if t not in df.columns]
    if missing_targets:
        raise ValueError(
            f"{path.name} no contiene los targets: {missing_targets}. "
            "Ejecuta primero window_targets.py (Step 14)."
        )
    return df


def _temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split cronológico 80/20 sin shuffle — preserva causalidad temporal."""
    split = int(len(X) * ratio)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def _evaluate(y_true: pd.Series, y_pred) -> dict[str, float]:
    """Cuatro métricas de clasificación redondeadas a 4 decimales."""
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred),                   4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred,        zero_division=0), 4),
    }


# ── Bucle de entrenamiento ────────────────────────────────────────────────────

def _train_asset(asset: str, path: Path) -> list[dict]:
    """
    Entrena los 3 modelos x 2 conjuntos de features x 3 horizontes para un activo.

    Itera los targets en orden (3d → 5d → 7d) para que la tabla de consola
    muestre claramente el efecto del horizonte sobre las métricas.

    Devuelve una lista de dicts con todos los resultados del activo.
    """
    print(f"\n  {'='*58}")
    print(f"  Activo: {asset}  ({path.relative_to(ROOT)})")
    print(f"  {'='*58}")

    df       = _load(path)
    all_rows: list[dict] = []

    for target in TARGETS:
        horizon = target.split("_")[1]   # "3d", "5d", "7d"
        print(f"\n  -- Target: {target} --")
        print(f"  {'Conjunto':<22} {'Modelo':<22} {'Acc':>6} {'Prec':>6} "
              f"{'Rec':>6} {'F1':>6}")
        print(f"  {'-'*22} {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

        for set_name, feature_cols in FEATURE_SETS.items():
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"[{asset}/{target}] Columnas ausentes en '{set_name}': {missing}"
                )

            X = df[feature_cols]
            y = df[target]
            X_train, X_test, y_train, y_test = _temporal_split(X, y)
            n_train, n_test = len(X_train), len(X_test)

            for model_name, model in _build_models().items():
                model.fit(X_train, y_train)
                metrics = _evaluate(y_test, model.predict(X_test))

                print(
                    f"  {set_name:<22} {model_name:<22} "
                    f"{metrics['accuracy']:>6.4f} {metrics['precision']:>6.4f} "
                    f"{metrics['recall']:>6.4f} {metrics['f1']:>6.4f}"
                )

                all_rows.append({
                    "asset":       asset,
                    "target":      target,
                    "feature_set": set_name,
                    "model":       model_name,
                    **metrics,
                    "n_train":     n_train,
                    "n_test":      n_test,
                })

    return all_rows


# ── Tabla resumen ─────────────────────────────────────────────────────────────

def _build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye windowed_ablation_summary.csv a partir de windowed_model_results.csv.

    Pivota el DataFrame para tener f1_price_only y f1_price_plus_sentiment en
    columnas separadas, calcula el delta y el flag de mejora.
    """
    pivot = results_df.pivot_table(
        index=["asset", "target", "model"],
        columns="feature_set",
        values="f1",
    ).reset_index()

    pivot.columns.name = None

    pivot = pivot.rename(columns={
        "price_only":           "f1_price_only",
        "price_plus_sentiment": "f1_price_plus_sentiment",
    })

    pivot["f1_delta"]               = round(
        pivot["f1_price_plus_sentiment"] - pivot["f1_price_only"], 4
    )
    pivot["sentiment_improves_model"] = pivot["f1_delta"] > 0

    return pivot[_SUMMARY_COLS].sort_values(
        ["asset", "target", "model"]
    ).reset_index(drop=True)


# ── Impresión de resumen ──────────────────────────────────────────────────────

def _print_summary(results_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """
    Imprime el mejor resultado por activo y horizonte, y el delta medio de F1.
    """
    print("\n\n  " + "=" * 70)
    print("  Mejor resultado por activo y horizonte (F1 maximo)")
    print("  " + "-" * 70)
    print(f"  {'Activo':<7} {'Target':<14} {'Conjunto':<22} "
          f"{'Modelo':<22} {'F1':>6}")
    print(f"  {'-'*7} {'-'*14} {'-'*22} {'-'*22} {'-'*6}")

    for (asset, target), grp in results_df.groupby(["asset", "target"]):
        best = grp.loc[grp["f1"].idxmax()]
        print(
            f"  {asset:<7} {target:<14} {best['feature_set']:<22} "
            f"{best['model']:<22} {best['f1']:>6.4f}"
        )

    print("\n\n  Delta F1 medio de sentimiento por activo y horizonte")
    print("  " + "-" * 50)
    print(f"  {'Activo':<7} {'Target':<14} {'Delta F1 medio':>16} {'Mejoras':>8}")
    print(f"  {'-'*7} {'-'*14} {'-'*16} {'-'*8}")

    for (asset, target), grp in summary_df.groupby(["asset", "target"]):
        mean_delta = grp["f1_delta"].mean()
        n_improved = int(grp["sentiment_improves_model"].sum())
        n_total    = len(grp)
        print(
            f"  {asset:<7} {target:<14} {mean_delta:>+16.4f} "
            f"{n_improved}/{n_total:>5}"
        )

    print("  " + "=" * 70)


# ── API pública ───────────────────────────────────────────────────────────────

def run_windowed_analysis(
    asset_config:   dict[str, Path] = ASSET_CONFIG,
    results_path:   Path            = WINDOWED_RESULTS_PATH,
    summary_path:   Path            = WINDOWED_SUMMARY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el análisis multi-horizonte completo para todos los activos.

    Pasos
    -----
    1. Para cada activo: carga el CSV windowed, entrena 3 modelos x 2 conjuntos
       de features x 3 horizontes de predicción.
    2. Agrega en windowed_model_results.csv.
    3. Calcula deltas F1 por sentimiento en windowed_ablation_summary.csv.
    4. Imprime el mejor resultado por horizonte y los deltas medios.

    Devuelve
    --------
    (results_df, summary_df) — DataFrames completos.
    """
    all_rows: list[dict] = []

    for asset, path in asset_config.items():
        if not path.exists():
            print(f"\n  [SKIP] {path.name} no encontrado — omitiendo {asset}")
            continue
        all_rows.extend(_train_asset(asset, path))

    if not all_rows:
        raise RuntimeError(
            "No se procesó ningún activo. "
            "Ejecuta primero generate_windowed_targets() (Step 14)."
        )

    results_df = pd.DataFrame(all_rows)[_RESULTS_COLS]
    summary_df = _build_summary(results_df)

    _print_summary(results_df, summary_df)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n  Guardado  ->  {results_path.relative_to(ROOT)}")
    print(f"  Guardado  ->  {summary_path.relative_to(ROOT)}")

    return results_df, summary_df
