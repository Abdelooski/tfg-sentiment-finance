"""
src/models/compare_vader_finbert.py

Compara directamente el poder predictivo del sentimiento VADER (basado en reglas)
frente al sentimiento FinBERT (basado en transformer preentrenado en textos financieros).

Para cada activo se entrenan los mismos tres clasificadores sobre dos conjuntos
de features simétricos — uno con VADER y otro con FinBERT — usando exactamente
las mismas features de precio, la misma ventana temporal y los mismos modelos.
Esto aísla el efecto del modelo de sentimiento como única variable cambiante.

Hipótesis: FinBERT, al entender la jerga financiera y el contexto, debería
generar scores más predictivos que VADER para textos de foros de inversión.

Entradas:
  data/processed/btc_features.csv         — features VADER de BTC
  data/processed/sp500_features.csv       — features VADER de S&P500
  data/processed/btc_finbert_features.csv — features FinBERT de BTC
  data/processed/sp500_finbert_features.csv

Salidas:
  reports/tables/vader_finbert_comparison.csv — métricas por (activo, fuente, modelo)
  reports/tables/vader_finbert_summary.csv    — delta F1 FinBERT - VADER por (activo, modelo)
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

COMPARISON_PATH = TABLES_DIR / "vader_finbert_comparison.csv"
SUMMARY_PATH    = TABLES_DIR / "vader_finbert_summary.csv"

# Configs independientes para facilitar la comprobación de existencia en main.py
VADER_FEAT_CONFIG: dict[str, Path] = {
    "BTC":   PROCESSED_DIR / "btc_features.csv",
    "SP500": PROCESSED_DIR / "sp500_features.csv",
}

FINBERT_FEAT_CONFIG: dict[str, Path] = {
    "BTC":   PROCESSED_DIR / "btc_finbert_features.csv",
    "SP500": PROCESSED_DIR / "sp500_finbert_features.csv",
}

TARGET_COL  = "Direction"
TRAIN_RATIO = 0.80

# ── Conjuntos de features ─────────────────────────────────────────────────────

# Features de precio compartidas por ambas fuentes de sentimiento.
# Son idénticas en ambos CSV de entrada, lo que garantiza que la única
# diferencia entre los dos experimentos es el modelo de sentimiento.
_PRICE_FEATURES: list[str] = [
    "Return",
    "return_lag_1", "return_lag_2", "return_lag_3",
    "return_rolling_3", "return_rolling_7",
    "volatility_7",
]

_VADER_SENT: list[str] = [
    "vader_compound",
    "vader_neg", "vader_neu", "vader_pos",
    "vader_compound_lag_1", "vader_compound_lag_2", "vader_compound_lag_3",
    "vader_compound_rolling_3", "vader_compound_rolling_7",
    "n_posts_log",
]

_FINBERT_SENT: list[str] = [
    "finbert_compound",
    "finbert_neg", "finbert_neu", "finbert_pos",
    "finbert_compound_lag_1", "finbert_compound_lag_2", "finbert_compound_lag_3",
    "finbert_compound_rolling_3", "finbert_compound_rolling_7",
    "n_posts_log",
]

# Ambos conjuntos tienen exactamente el mismo número de features (17)
SENTIMENT_SOURCES: dict[str, list[str]] = {
    "vader":   _PRICE_FEATURES + _VADER_SENT,
    "finbert": _PRICE_FEATURES + _FINBERT_SENT,
}

_COMPARISON_COLS: list[str] = [
    "asset", "sentiment_source", "model",
    "accuracy", "precision", "recall", "f1",
    "n_train", "n_test",
]

_SUMMARY_COLS: list[str] = [
    "asset", "model",
    "f1_vader", "f1_finbert",
    "f1_delta_finbert_minus_vader", "finbert_beats_vader",
]


# ── Modelos ───────────────────────────────────────────────────────────────────

def _build_models() -> dict[str, object]:
    """Instancias frescas de los tres clasificadores con hiperparámetros estándar."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
    }


# ── Split y evaluación ────────────────────────────────────────────────────────

def _temporal_split(
    X: pd.DataFrame, y: pd.Series, ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split = int(len(X) * ratio)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def _evaluate(y_true: pd.Series, y_pred) -> dict[str, float]:
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred),                   4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred,        zero_division=0), 4),
    }


# ── Entrenamiento por fuente de sentimiento ───────────────────────────────────

def _load_and_validate(path: Path, feature_cols: list[str], source: str) -> pd.DataFrame:
    """Carga el CSV y verifica que contiene todas las columnas requeridas."""
    df = pd.read_csv(path, dtype={"Date": str})
    missing = [c for c in feature_cols + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{source}] {path.name} no contiene las columnas: {missing}"
        )
    return df


def _train_source(
    asset: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    source_name: str,
) -> list[dict]:
    """
    Entrena los tres modelos para un activo y una fuente de sentimiento.
    Devuelve una lista de dicts con las métricas de cada modelo.
    """
    X = df[feature_cols]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = _temporal_split(X, y)
    n_train, n_test = len(X_train), len(X_test)

    rows: list[dict] = []
    for model_name, model in _build_models().items():
        model.fit(X_train, y_train)
        metrics = _evaluate(y_test, model.predict(X_test))

        print(
            f"  {source_name:<10} {model_name:<22} "
            f"{metrics['accuracy']:>6.4f} {metrics['precision']:>6.4f} "
            f"{metrics['recall']:>6.4f} {metrics['f1']:>6.4f}"
        )

        rows.append({
            "asset":            asset,
            "sentiment_source": source_name,
            "model":            model_name,
            **metrics,
            "n_train":          n_train,
            "n_test":           n_test,
        })
    return rows


def _train_asset(
    asset: str,
    vader_path: Path,
    finbert_path: Path,
) -> list[dict]:
    """
    Carga los CSVs de VADER y FinBERT para un activo y entrena ambas fuentes.

    Los dos CSV pueden tener distinto número de filas (el inner join con
    VADER o FinBERT diario produce ventanas temporales distintas), por lo que
    el split 80/20 se aplica independientemente a cada uno.
    """
    print(f"\n  {'='*60}")
    print(f"  Activo: {asset}")
    print(f"  {'='*60}")
    print(f"  {'Fuente':<10} {'Modelo':<22} {'Acc':>6} {'Prec':>6} "
          f"{'Rec':>6} {'F1':>6}")
    print(f"  {'-'*10} {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    all_rows: list[dict] = []

    vader_cols   = SENTIMENT_SOURCES["vader"]
    finbert_cols = SENTIMENT_SOURCES["finbert"]

    vader_df   = _load_and_validate(vader_path,   vader_cols,   "vader")
    finbert_df = _load_and_validate(finbert_path, finbert_cols, "finbert")

    all_rows.extend(_train_source(asset, vader_df,   vader_cols,   "vader"))
    all_rows.extend(_train_source(asset, finbert_df, finbert_cols, "finbert"))

    return all_rows


# ── Tablas resumen ────────────────────────────────────────────────────────────

def _build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivota los resultados para poner f1_vader y f1_finbert en columnas separadas
    y calcula el delta y el flag de superioridad de FinBERT.

    f1_delta_finbert_minus_vader > 0 indica que FinBERT es más predictivo.
    """
    pivot = results_df.pivot_table(
        index=["asset", "model"],
        columns="sentiment_source",
        values="f1",
    ).reset_index()

    pivot.columns.name = None
    pivot = pivot.rename(columns={
        "vader":   "f1_vader",
        "finbert": "f1_finbert",
    })

    pivot["f1_delta_finbert_minus_vader"] = round(
        pivot["f1_finbert"] - pivot["f1_vader"], 4
    )
    pivot["finbert_beats_vader"] = pivot["f1_delta_finbert_minus_vader"] > 0

    return pivot[_SUMMARY_COLS].sort_values(
        ["asset", "model"]
    ).reset_index(drop=True)


def _print_summary(results_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """
    Imprime la mejor fuente por activo y modelo, y el F1 medio por fuente y activo.
    """
    print("\n\n  " + "=" * 66)
    print("  Mejor fuente de sentimiento por activo y modelo")
    print("  " + "-" * 66)
    print(f"  {'Activo':<7} {'Modelo':<22} {'Ganador':<10} "
          f"{'F1 VADER':>9} {'F1 FinBERT':>11} {'Delta':>8}")
    print(f"  {'-'*7} {'-'*22} {'-'*10} {'-'*9} {'-'*11} {'-'*8}")

    for _, r in summary_df.iterrows():
        winner = "FinBERT" if r["finbert_beats_vader"] else "VADER  "
        print(
            f"  {r['asset']:<7} {r['model']:<22} {winner:<10} "
            f"{r['f1_vader']:>9.4f} {r['f1_finbert']:>11.4f} "
            f"{r['f1_delta_finbert_minus_vader']:>+8.4f}"
        )

    print("\n\n  F1 medio por fuente y activo")
    print("  " + "-" * 40)
    for (asset, source), grp in results_df.groupby(["asset", "sentiment_source"]):
        mean_f1 = grp["f1"].mean()
        print(f"  {asset:<7} {source:<10}  F1 medio = {mean_f1:.4f}")

    print("  " + "=" * 66)


# ── API pública ───────────────────────────────────────────────────────────────

def run_vader_finbert_comparison(
    vader_config:   dict[str, Path] = VADER_FEAT_CONFIG,
    finbert_config: dict[str, Path] = FINBERT_FEAT_CONFIG,
    comparison_path: Path           = COMPARISON_PATH,
    summary_path:    Path           = SUMMARY_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta la comparación VADER vs FinBERT para todos los activos configurados.

    Pasos
    -----
    1. Para cada activo: carga los CSVs de VADER y FinBERT.
    2. Entrena 3 modelos x 2 fuentes de sentimiento.
    3. Guarda vader_finbert_comparison.csv con todas las métricas.
    4. Construye vader_finbert_summary.csv con los deltas de F1.
    5. Imprime la tabla comparativa y el F1 medio por fuente.

    Devuelve
    --------
    (comparison_df, summary_df) — DataFrames completos.
    """
    assets = sorted(set(vader_config) & set(finbert_config))
    if not assets:
        raise RuntimeError("No hay activos comunes entre vader_config y finbert_config.")

    all_rows: list[dict] = []

    for asset in assets:
        vpath = vader_config[asset]
        fpath = finbert_config[asset]

        if not vpath.exists():
            print(f"  [SKIP] {vpath.name} no encontrado — omitiendo {asset} VADER")
            continue
        if not fpath.exists():
            print(f"  [SKIP] {fpath.name} no encontrado — omitiendo {asset} FinBERT")
            continue

        all_rows.extend(_train_asset(asset, vpath, fpath))

    if not all_rows:
        raise RuntimeError(
            "No se procesó ningún activo. "
            "Verifica que existen los CSVs de features VADER y FinBERT."
        )

    comparison_df = pd.DataFrame(all_rows)[_COMPARISON_COLS]
    summary_df    = _build_summary(comparison_df)

    _print_summary(comparison_df, summary_df)

    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n  Guardado  ->  {comparison_path.relative_to(ROOT)}")
    print(f"  Guardado  ->  {summary_path.relative_to(ROOT)}")

    return comparison_df, summary_df
