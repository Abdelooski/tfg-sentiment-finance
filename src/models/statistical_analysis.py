"""
src/models/statistical_analysis.py

Calcula estadísticas de asociación entre el sentimiento VADER y el retorno
del día siguiente para BTC y S&P500, y compara los modelos entrenados contra
un clasificador naive (siempre predice la clase mayoritaria).

Entradas:
  data/processed/btc_features.csv       — features con vader_compound y Return
  data/processed/sp500_features.csv
  reports/tables/model_results.csv      — métricas de los modelos entrenados

Salida:
  reports/tables/statistical_tests.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR    = ROOT / "reports" / "tables"

RESULTS_PATH  = TABLES_DIR / "model_results.csv"
STATS_PATH    = TABLES_DIR / "statistical_tests.csv"

# Mapea nombre de activo -> CSV de features
ASSET_CONFIG: dict[str, Path] = {
    "BTC":   PROCESSED_DIR / "btc_features.csv",
    "SP500": PROCESSED_DIR / "sp500_features.csv",
}

# Columnas del CSV de salida, en orden de visualización
OUTPUT_COLS = [
    "asset",
    "pearson_corr",
    "spearman_corr",
    "baseline_accuracy",
    "majority_class",
    "best_model",
    "best_model_accuracy",
    "model_beats_baseline",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_features(path: Path) -> pd.DataFrame:
    """
    Carga el CSV de features y verifica que contenga las columnas necesarias.

    Se requieren al menos: 'vader_compound', 'Return' y 'Direction'.
    Estas columnas provienen del dataset merged (antes de la ingeniería de
    features) y se conservan en el CSV de salida de feature_engineering.py.
    """
    df = pd.read_csv(path, dtype={"Date": str})
    required = {"vader_compound", "Return", "Direction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} no contiene las columnas requeridas: {missing}"
        )
    return df


def _pearson(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """
    Correlación de Pearson entre dos series alineadas.

    Mide la asociación lineal (magnitud + dirección). Sensible a outliers.
    Devuelve (correlación, p-valor).
    """
    corr, pval = stats.pearsonr(x, y)
    return float(corr), float(pval)


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """
    Correlación de Spearman (basada en rangos) entre dos series alineadas.

    Más robusta que Pearson para distribuciones no normales o relaciones
    monótonas no lineales. Devuelve (correlación, p-valor).
    """
    corr, pval = stats.spearmanr(x, y)
    return float(corr), float(pval)


def _naive_baseline(direction: pd.Series) -> tuple[float, int]:
    """
    Clasificador naive: siempre predice la clase mayoritaria de Direction.

    La exactitud (accuracy) del naive baseline es la proporción de la clase
    más frecuente. Sirve como umbral mínimo que cualquier modelo real debe
    superar para añadir valor predictivo.

    Devuelve (accuracy, majority_class).
    """
    majority_class = int(direction.mode().iloc[0])
    accuracy = round(float((direction == majority_class).mean()), 4)
    return accuracy, majority_class


def _best_model(results_df: pd.DataFrame, asset: str) -> tuple[str, float]:
    """
    Selecciona el mejor modelo para un activo dado según F1 (y accuracy como
    criterio de desempate), coherente con la lógica de summary_tables.py.

    Devuelve (nombre_del_modelo, accuracy_en_test).
    """
    asset_rows = results_df[results_df["asset"] == asset]
    if asset_rows.empty:
        raise ValueError(
            f"No se encontraron resultados para el activo '{asset}' "
            f"en model_results.csv"
        )
    best = (
        asset_rows
        .sort_values(["f1", "accuracy"], ascending=False)
        .iloc[0]
    )
    return str(best["model"]), round(float(best["accuracy"]), 4)


def _analyse_asset(
    asset: str,
    feat_path: Path,
    results_df: pd.DataFrame,
) -> dict:
    """
    Ejecuta todos los análisis estadísticos para un activo.

    Pasos:
    1. Carga el CSV de features.
    2. Construye la serie de retorno del día siguiente (shift -1) y la
       alinea con vader_compound. La última fila se descarta porque su
       retorno siguiente es desconocido.
    3. Calcula Pearson y Spearman entre sentimiento y retorno siguiente.
    4. Calcula la exactitud del clasificador naive.
    5. Recupera el mejor modelo de model_results.csv y compara con el baseline.
    """
    print(f"\n  Activo: {asset}")
    print(f"  Cargando  ->  {feat_path.relative_to(ROOT)}")
    df = _load_features(feat_path)

    # Retorno del día siguiente (variable continua, más informativa que
    # Direction para correlación porque preserva la magnitud del movimiento)
    next_return = df["Return"].shift(-1).dropna()
    sentiment   = df["vader_compound"].loc[next_return.index]

    n_pairs = len(next_return)
    print(f"  Pares (vader_compound, retorno_siguiente) : {n_pairs}")

    # ── Correlaciones ─────────────────────────────────────────────────────────
    pearson_r,  pearson_p  = _pearson(sentiment, next_return)
    spearman_r, spearman_p = _spearman(sentiment, next_return)

    print(f"  Pearson  r={pearson_r:+.4f}  p={pearson_p:.4f}")
    print(f"  Spearman r={spearman_r:+.4f}  p={spearman_p:.4f}")

    # ── Naive baseline ────────────────────────────────────────────────────────
    baseline_acc, majority_cls = _naive_baseline(df["Direction"])
    print(f"  Naive baseline  ->  clase={majority_cls}  acc={baseline_acc:.4f}")

    # ── Mejor modelo ─────────────────────────────────────────────────────────
    best_name, best_acc = _best_model(results_df, asset)
    beats = best_acc > baseline_acc
    sign  = ">" if beats else "<="
    print(f"  Mejor modelo    ->  {best_name}  acc={best_acc:.4f}  "
          f"({best_acc:.4f} {sign} baseline {baseline_acc:.4f})")

    return {
        "asset":               asset,
        "pearson_corr":        round(pearson_r,  4),
        "spearman_corr":       round(spearman_r, 4),
        "baseline_accuracy":   baseline_acc,
        "majority_class":      majority_cls,
        "best_model":          best_name,
        "best_model_accuracy": best_acc,
        "model_beats_baseline": beats,
    }


# ── API pública ───────────────────────────────────────────────────────────────

def run_analysis(
    asset_config: dict[str, Path] = ASSET_CONFIG,
    results_path: Path = RESULTS_PATH,
    stats_path: Path = STATS_PATH,
) -> pd.DataFrame:
    """
    Ejecuta el análisis estadístico completo para todos los activos configurados.

    Pasos
    -----
    1. Carga model_results.csv (métricas de los modelos entrenados).
    2. Para cada activo: calcula correlaciones, baseline y compara modelos.
    3. Construye la tabla resumen y la guarda en reports/tables/.

    Devuelve
    --------
    pd.DataFrame con una fila por activo y las columnas de OUTPUT_COLS.
    """
    if not results_path.exists():
        raise FileNotFoundError(
            f"No se encontró model_results.csv en {results_path}. "
            "Ejecuta primero el Step 7 (entrenamiento de modelos)."
        )

    print(f"  Cargando resultados  ->  {results_path.relative_to(ROOT)}")
    results_df = pd.read_csv(results_path)

    rows: list[dict] = []
    for asset, feat_path in asset_config.items():
        if not feat_path.exists():
            print(f"\n  [SKIP] {feat_path.name} no encontrado — omitiendo {asset}")
            continue
        row = _analyse_asset(asset, feat_path, results_df)
        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No se procesó ningún activo. "
            "Verifica que existan los CSVs de features en data/processed/."
        )

    stats_df = pd.DataFrame(rows)[OUTPUT_COLS]

    # ── Impresión del resumen ─────────────────────────────────────────────────
    print("\n  " + "=" * 70)
    print("  Tabla resumen de análisis estadístico")
    print("  " + "-" * 70)
    for _, r in stats_df.iterrows():
        print(f"\n  Activo               : {r['asset']}")
        print(f"  Pearson  (r)         : {r['pearson_corr']:+.4f}")
        print(f"  Spearman (r)         : {r['spearman_corr']:+.4f}")
        print(f"  Naive baseline acc   : {r['baseline_accuracy']:.4f}  "
              f"(clase mayoritaria = {r['majority_class']})")
        print(f"  Mejor modelo         : {r['best_model']}  "
              f"acc={r['best_model_accuracy']:.4f}")
        verdict = "SI supera" if r["model_beats_baseline"] else "NO supera"
        print(f"  Supera baseline      : {verdict}")
    print("  " + "=" * 70)

    # ── Guardado ──────────────────────────────────────────────────────────────
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(stats_path, index=False)
    print(f"\n  Guardado  ->  {stats_path.relative_to(ROOT)}")

    return stats_df
