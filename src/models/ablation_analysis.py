"""
src/models/ablation_analysis.py

Análisis de ablación: evalúa si añadir features de sentimiento mejora
la capacidad predictiva respecto a usar solo features de precio.

Para cada activo (BTC, S&P500) se entrenan los mismos tres clasificadores
sobre dos conjuntos de features distintos:

  price_only           — solo indicadores técnicos de precio
  price_plus_sentiment — indicadores de precio + scores VADER

El split temporal es idéntico al de train_models.py (80% train / 20% test,
sin shuffle) para que los resultados sean directamente comparables.

Salidas:
  reports/tables/ablation_results.csv  — métricas por (activo, conjunto, modelo)
  reports/tables/ablation_summary.csv  — comparativa F1 y delta por sentimiento
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

ABLATION_RESULTS_PATH = TABLES_DIR / "ablation_results.csv"
ABLATION_SUMMARY_PATH = TABLES_DIR / "ablation_summary.csv"

# Mapea nombre de activo -> CSV de features generado por feature_engineering.py
ASSET_CONFIG: dict[str, Path] = {
    "BTC":   PROCESSED_DIR / "btc_features.csv",
    "SP500": PROCESSED_DIR / "sp500_features.csv",
}

TARGET_COL  = "Direction"
TRAIN_RATIO = 0.80

# ── Conjuntos de features ──────────────────────────────────────────────────────

# Indicadores técnicos de precio: retorno actual, lags, medias móviles y volatilidad.
# Representan la información que cualquier modelo cuantitativo básico tendría.
_PRICE_FEATURES: list[str] = [
    "Return",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_rolling_3",
    "return_rolling_7",
    "volatility_7",
]

# Scores VADER del sentimiento de Reddit: compuesto, componentes negativos/neutros/positivos,
# lags temporales y medias móviles. Representan la señal de sentimiento del mercado.
_SENTIMENT_FEATURES: list[str] = [
    "vader_compound",
    "vader_neg",
    "vader_neu",
    "vader_pos",
    "vader_compound_lag_1",
    "vader_compound_lag_2",
    "vader_compound_lag_3",
    "vader_compound_rolling_3",
    "vader_compound_rolling_7",
    "n_posts_log",
]

# Diccionario ordenado: primero price_only para que el delta tenga sentido
FEATURE_SETS: dict[str, list[str]] = {
    "price_only":           _PRICE_FEATURES,
    "price_plus_sentiment": _PRICE_FEATURES + _SENTIMENT_FEATURES,
}

# ── Columnas de los CSV de salida ─────────────────────────────────────────────
_RESULTS_COLS = [
    "asset", "feature_set", "model",
    "accuracy", "precision", "recall", "f1",
    "n_train", "n_test",
]

_SUMMARY_COLS = [
    "asset", "model",
    "f1_price_only", "f1_price_plus_sentiment",
    "f1_delta", "sentiment_improves_model",
]


# ── Modelos ───────────────────────────────────────────────────────────────────

def _build_models() -> dict[str, object]:
    """
    Devuelve instancias nuevas de los tres clasificadores.

    Logistic Regression va envuelta en un Pipeline con StandardScaler porque
    es sensible a la escala de las features. RF y XGBoost son invariantes a escala.
    Los hiperparámetros son idénticos a train_models.py para comparabilidad directa.
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


# ── Carga y split ─────────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame:
    """Carga el CSV de features validando que existe la columna objetivo."""
    df = pd.read_csv(path, dtype={"Date": str})
    if TARGET_COL not in df.columns:
        raise ValueError(f"{path.name} no contiene la columna '{TARGET_COL}'")
    return df


def _validate_features(df: pd.DataFrame, feature_cols: list[str], path: Path) -> list[str]:
    """
    Filtra la lista de features a las que realmente existen en el DataFrame.

    Si falta alguna feature obligatoria se lanza ValueError. En la práctica,
    todas deberían estar presentes porque el CSV proviene de feature_engineering.py.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} no contiene las features requeridas: {missing}"
        )
    return feature_cols


def _temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split cronológico sin shuffle: los primeros `ratio` % de filas son train,
    el resto test. Garantiza que el modelo nunca ve datos futuros durante el
    entrenamiento (principio de no-fuga de datos en series temporales).
    """
    split = int(len(X) * ratio)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# ── Evaluación ────────────────────────────────────────────────────────────────

def _evaluate(y_true: pd.Series, y_pred) -> dict[str, float]:
    """Calcula las cuatro métricas de clasificación sobre el conjunto de test."""
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred),                       4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0),     4),
        "recall":    round(recall_score(y_true, y_pred,    zero_division=0),     4),
        "f1":        round(f1_score(y_true, y_pred,        zero_division=0),     4),
    }


# ── Bucle de entrenamiento ────────────────────────────────────────────────────

def _train_feature_set(
    asset: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    set_name: str,
) -> list[dict]:
    """
    Entrena los tres modelos para un activo y un conjunto de features concreto.

    Devuelve una lista de dicts (uno por modelo) con todas las métricas y
    metadatos necesarios para construir ablation_results.csv.
    """
    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = _temporal_split(X, y)
    n_train, n_test = len(X_train), len(X_test)

    models  = _build_models()
    results = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        metrics = _evaluate(y_test, y_pred)

        results.append({
            "asset":       asset,
            "feature_set": set_name,
            "model":       model_name,
            **metrics,
            "n_train":     n_train,
            "n_test":      n_test,
        })

    return results


def _train_asset(asset: str, feat_path: Path) -> list[dict]:
    """
    Carga el CSV de un activo y entrena todos los modelos sobre ambos conjuntos
    de features. Imprime una tabla comparativa por consola.
    """
    print(f"\n  Activo: {asset}")
    print(f"  Cargando  ->  {feat_path.relative_to(ROOT)}")

    df = _load(feat_path)
    all_results: list[dict] = []

    # Cabecera de la tabla de consola
    print(f"\n  {'Conjunto':<22} {'Modelo':<22} {'Acc':>6} {'Prec':>6} "
          f"{'Rec':>6} {'F1':>6}")
    print(f"  {'-'*22} {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for set_name, feature_cols in FEATURE_SETS.items():
        _validate_features(df, feature_cols, feat_path)
        rows = _train_feature_set(asset, df, feature_cols, set_name)
        all_results.extend(rows)

        for r in rows:
            print(
                f"  {r['feature_set']:<22} {r['model']:<22} "
                f"{r['accuracy']:>6.4f} {r['precision']:>6.4f} "
                f"{r['recall']:>6.4f} {r['f1']:>6.4f}"
            )

    return all_results


# ── Tabla resumen de ablación ─────────────────────────────────────────────────

def _build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye ablation_summary.csv a partir de ablation_results.csv.

    Para cada (activo, modelo) compara el F1 entre price_only y
    price_plus_sentiment y calcula:
      f1_delta = f1_price_plus_sentiment - f1_price_only
      sentiment_improves_model = f1_delta > 0

    Un delta positivo indica que añadir sentimiento mejora la predicción;
    si BTC muestra deltas mayores que S&P500 en todos los modelos, se confirma
    la hipótesis central del TFG.
    """
    pivot = results_df.pivot_table(
        index=["asset", "model"],
        columns="feature_set",
        values="f1",
    ).reset_index()

    pivot.columns.name = None  # elimina el nombre del índice de columnas

    pivot = pivot.rename(columns={
        "price_only":           "f1_price_only",
        "price_plus_sentiment": "f1_price_plus_sentiment",
    })

    pivot["f1_delta"] = round(
        pivot["f1_price_plus_sentiment"] - pivot["f1_price_only"], 4
    )
    pivot["sentiment_improves_model"] = pivot["f1_delta"] > 0

    return pivot[_SUMMARY_COLS]


# ── API pública ───────────────────────────────────────────────────────────────

def run_ablation(
    asset_config: dict[str, Path] = ASSET_CONFIG,
    results_path: Path            = ABLATION_RESULTS_PATH,
    summary_path: Path            = ABLATION_SUMMARY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el análisis de ablación completo para todos los activos configurados.

    Pasos
    -----
    1. Para cada activo: entrena 3 modelos x 2 conjuntos de features.
    2. Agrega todos los resultados en ablation_results.csv.
    3. Construye ablation_summary.csv con los deltas de F1.
    4. Imprime el resumen en consola destacando si el sentimiento ayuda.

    Devuelve
    --------
    (results_df, summary_df) — ambos DataFrames completos.
    """
    all_results: list[dict] = []

    for asset, feat_path in asset_config.items():
        if not feat_path.exists():
            print(f"\n  [SKIP] {feat_path.name} no encontrado — omitiendo {asset}")
            continue
        asset_results = _train_asset(asset, feat_path)
        all_results.extend(asset_results)

    if not all_results:
        raise RuntimeError(
            "No se procesó ningún activo. "
            "Verifica que existan los CSVs de features en data/processed/."
        )

    results_df = pd.DataFrame(all_results)[_RESULTS_COLS]
    summary_df = _build_summary(results_df)

    # ── Resumen en consola ────────────────────────────────────────────────────
    print("\n\n  " + "=" * 68)
    print("  Resumen de ablación: impacto del sentimiento sobre F1")
    print("  " + "-" * 68)
    print(f"  {'Activo':<8} {'Modelo':<22} {'F1 precio':>10} "
          f"{'F1 +senti':>10} {'Delta':>8} {'Mejora':>7}")
    print(f"  {'-'*8} {'-'*22} {'-'*10} {'-'*10} {'-'*8} {'-'*7}")

    for _, r in summary_df.iterrows():
        mejora = "SI" if r["sentiment_improves_model"] else "NO"
        delta_str = f"{r['f1_delta']:+.4f}"
        print(
            f"  {r['asset']:<8} {r['model']:<22} "
            f"{r['f1_price_only']:>10.4f} "
            f"{r['f1_price_plus_sentiment']:>10.4f} "
            f"{delta_str:>8} {mejora:>7}"
        )

    print("  " + "=" * 68)

    # Conclusión rápida por activo
    for asset in summary_df["asset"].unique():
        sub = summary_df[summary_df["asset"] == asset]
        n_improved = sub["sentiment_improves_model"].sum()
        n_total    = len(sub)
        mean_delta = sub["f1_delta"].mean()
        print(
            f"\n  {asset}: el sentimiento mejora {n_improved}/{n_total} modelos "
            f"(delta F1 medio = {mean_delta:+.4f})"
        )

    # ── Guardado ──────────────────────────────────────────────────────────────
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n  Guardado  ->  {results_path.relative_to(ROOT)}")
    print(f"  Guardado  ->  {summary_path.relative_to(ROOT)}")

    return results_df, summary_df
