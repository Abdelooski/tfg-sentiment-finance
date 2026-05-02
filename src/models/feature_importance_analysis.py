"""
src/models/feature_importance_analysis.py

Analiza la importancia de cada feature en la predicción de la dirección
del precio para BTC y S&P500 usando Random Forest.

Random Forest calcula feature_importances_ a partir de la reducción media
de impureza (Gini) en cada árbol. Es el método más directo para entender
qué variables aportan más información al modelo.

Entradas:
  data/processed/btc_features.csv
  data/processed/sp500_features.csv

Salidas:
  reports/tables/feature_importance.csv          — todas las features, ambos activos
  reports/tables/feature_importance_sentiment.csv — solo features de sentimiento
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR    = ROOT / "reports" / "tables"

FI_PATH           = TABLES_DIR / "feature_importance.csv"
FI_SENTIMENT_PATH = TABLES_DIR / "feature_importance_sentiment.csv"

ASSET_CONFIG: dict[str, Path] = {
    "BTC":   PROCESSED_DIR / "btc_features.csv",
    "SP500": PROCESSED_DIR / "sp500_features.csv",
}

TARGET_COL  = "Direction"
TRAIN_RATIO = 0.80
TOP_N       = 10   # número de features a mostrar por consola

# Hiperparámetros idénticos a train_models.py para comparabilidad
_RF_PARAMS: dict = dict(
    n_estimators=100,
    max_depth=4,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# Columnas que no son features: identificador temporal y variable objetivo
_NON_FEATURE_COLS = {"Date", "Direction"}

# Palabras clave para identificar features de sentimiento en el nombre de columna
_SENTIMENT_KEYWORDS = ("vader", "n_posts")

OUTPUT_COLS = ["asset", "feature", "importance"]


# ── Carga y split ─────────────────────────────────────────────────────────────

def _load(path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Carga el CSV de features y devuelve (X, y, feature_names).

    Todas las columnas excepto Date y Direction se tratan como features,
    igual que en train_models.py, para que la importancia refleje el conjunto
    completo de variables disponibles para el modelo.
    """
    df = pd.read_csv(path, dtype={"Date": str})
    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    missing = [c for c in [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} no contiene la columna '{TARGET_COL}'")
    return df[feature_cols], df[TARGET_COL], feature_cols


def _temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split cronológico 80/20 sin shuffle.

    Solo se necesita X_train e y_train para extraer importancias; X_test e y_test
    se devuelven igualmente para mantener la misma interfaz que los demás módulos.
    """
    split = int(len(X) * ratio)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# ── Análisis por activo ───────────────────────────────────────────────────────

def _analyse_asset(asset: str, feat_path: Path) -> pd.DataFrame:
    """
    Entrena un Random Forest sobre los datos de train y extrae la importancia
    de cada feature por reducción media de impureza (Gini).

    Devuelve un DataFrame con columnas [asset, feature, importance] ordenado
    de mayor a menor importancia.
    """
    print(f"\n  Activo: {asset}")
    print(f"  Cargando  ->  {feat_path.relative_to(ROOT)}")

    X, y, feature_names = _load(feat_path)
    X_train, _X_test, y_train, _y_test = _temporal_split(X, y)

    print(f"  Features   : {len(feature_names)}")
    print(f"  Train rows : {len(X_train)}")

    rf = RandomForestClassifier(**_RF_PARAMS)
    rf.fit(X_train, y_train)

    fi_df = (
        pd.DataFrame({
            "asset":      asset,
            "feature":    feature_names,
            "importance": rf.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # ── Top N en consola ──────────────────────────────────────────────────────
    print(f"\n  Top {TOP_N} features por importancia (Gini):")
    print(f"  {'#':>3}  {'Feature':<35} {'Importancia':>12}")
    print(f"  {'-'*3}  {'-'*35} {'-'*12}")
    for rank, row in fi_df.head(TOP_N).iterrows():
        print(f"  {rank+1:>3}  {row['feature']:<35} {row['importance']:>12.6f}")

    return fi_df


# ── API pública ───────────────────────────────────────────────────────────────

def run_feature_importance(
    asset_config: dict[str, Path] = ASSET_CONFIG,
    fi_path: Path                 = FI_PATH,
    fi_sentiment_path: Path       = FI_SENTIMENT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el análisis de importancia de features para todos los activos.

    Pasos
    -----
    1. Para cada activo: carga features, entrena RF, extrae importancias.
    2. Concatena los resultados de todos los activos.
    3. Filtra las features de sentimiento (vader, n_posts).
    4. Imprime un resumen comparativo BTC vs S&P500 para las features de sentimiento.
    5. Guarda feature_importance.csv y feature_importance_sentiment.csv.

    Devuelve
    --------
    (fi_df, fi_sentiment_df) — DataFrames completos.
    """
    all_frames: list[pd.DataFrame] = []

    for asset, feat_path in asset_config.items():
        if not feat_path.exists():
            print(f"\n  [SKIP] {feat_path.name} no encontrado — omitiendo {asset}")
            continue
        asset_fi = _analyse_asset(asset, feat_path)
        all_frames.append(asset_fi)

    if not all_frames:
        raise RuntimeError(
            "No se procesó ningún activo. "
            "Verifica que existan los CSVs de features en data/processed/."
        )

    fi_df = pd.concat(all_frames, ignore_index=True)[OUTPUT_COLS]

    # ── Tabla de sentimiento ──────────────────────────────────────────────────
    # Filtra por nombre de columna: cualquier feature que contenga "vader" o "n_posts".
    # Esto captura vader_compound, vader_compound_lag_*, vader_compound_rolling_*,
    # vader_neg/neu/pos, n_posts_log, etc.
    sentiment_mask  = fi_df["feature"].str.contains("|".join(_SENTIMENT_KEYWORDS))
    fi_sentiment_df = (
        fi_df[sentiment_mask]
        .sort_values(["asset", "importance"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # ── Resumen comparativo de sentimiento en consola ────────────────────────
    print("\n\n  " + "=" * 62)
    print("  Importancia de features de sentimiento por activo")
    print("  " + "-" * 62)

    assets = fi_df["asset"].unique()
    for asset in assets:
        sub = fi_sentiment_df[fi_sentiment_df["asset"] == asset]
        total_sentiment_imp = sub["importance"].sum()

        # Importancia acumulada de TODAS las features del activo
        total_imp = fi_df[fi_df["asset"] == asset]["importance"].sum()  # == 1.0 siempre
        pct = total_sentiment_imp / total_imp * 100

        print(f"\n  {asset}  —  sentimiento explica el {pct:.1f}% de la importancia total")
        print(f"  {'Feature':<35} {'Importancia':>12}")
        print(f"  {'-'*35} {'-'*12}")
        for _, row in sub.iterrows():
            print(f"  {row['feature']:<35} {row['importance']:>12.6f}")

    print("\n  " + "=" * 62)

    # ── Guardado ──────────────────────────────────────────────────────────────
    fi_path.parent.mkdir(parents=True, exist_ok=True)
    fi_df.to_csv(fi_path, index=False)
    fi_sentiment_df.to_csv(fi_sentiment_path, index=False)

    print(f"\n  Guardado  ->  {fi_path.relative_to(ROOT)}")
    print(f"  Guardado  ->  {fi_sentiment_path.relative_to(ROOT)}")

    return fi_df, fi_sentiment_df
