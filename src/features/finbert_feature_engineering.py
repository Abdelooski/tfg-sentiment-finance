"""
src/features/finbert_feature_engineering.py

Genera el conjunto de features de FinBERT para BTC y S&P500.

El proceso es idéntico al de feature_engineering.py (que usa VADER), pero
usando finbert_compound en lugar de vader_compound como fuente de sentimiento.
Mantener la misma estructura de features permite comparar directamente la
capacidad predictiva de VADER vs FinBERT en los mismos modelos.

Features creadas a partir de finbert_compound y Return:
  - finbert_compound_lag_1/2/3       → valores del día t-1, t-2, t-3
  - finbert_compound_rolling_3/7     → media móvil de 3 y 7 días
  - n_posts_log                      → log(1 + n_posts), escala la actividad
  - return_lag_1/2/3                 → retorno rezagado 1, 2, 3 días
  - return_rolling_3/7               → retorno medio de 3 y 7 días
  - volatility_7                     → desviación estándar de Return en 7 días

Anti-leakage: todos los valores usan datos de t o anteriores (shift positivo
hacia el futuro del índice, shift negativo del precio ya calculado en el CSV).

Entradas:
  data/processed/btc_finbert_merged.csv
  data/processed/sp500_finbert_merged.csv

Salidas:
  data/processed/btc_finbert_features.csv
  data/processed/sp500_finbert_features.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

# Mapea nombre de activo -> (CSV fusionado de entrada, CSV de features de salida)
ASSET_CONFIG: dict[str, tuple[Path, Path]] = {
    "BTC": (
        PROCESSED_DIR / "btc_finbert_merged.csv",
        PROCESSED_DIR / "btc_finbert_features.csv",
    ),
    "SP500": (
        PROCESSED_DIR / "sp500_finbert_merged.csv",
        PROCESSED_DIR / "sp500_finbert_features.csv",
    ),
}

# Columnas base que se conservan en el CSV de salida (sin modificar)
_BASE_COLS = [
    "Date", "Close", "Return", "Direction",
    "finbert_neg", "finbert_neu", "finbert_pos", "finbert_compound",
    "n_posts",
]

# Ventana máxima usada en features de rolling/lag. Las primeras (MAX_WINDOW - 1)
# filas tendrán NaN y serán descartadas con dropna.
_MAX_WINDOW = 7


# ── Ingeniería de features ────────────────────────────────────────────────────

def _add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Añade todas las features derivadas de finbert_compound y Return.

    Devuelve el DataFrame enriquecido y la lista de columnas nuevas.
    """
    df = df.copy()
    new_cols: list[str] = []

    # ── Lags de sentimiento FinBERT ───────────────────────────────────────────
    for lag in [1, 2, 3]:
        col = f"finbert_compound_lag_{lag}"
        df[col] = df["finbert_compound"].shift(lag)
        new_cols.append(col)

    # ── Medias móviles de sentimiento FinBERT ─────────────────────────────────
    for window in [3, 7]:
        col = f"finbert_compound_rolling_{window}"
        df[col] = df["finbert_compound"].rolling(window).mean()
        new_cols.append(col)

    # ── Actividad de posts (escala logarítmica) ───────────────────────────────
    # log1p evita log(0) cuando n_posts = 0 y comprime la escala de la variable.
    df["n_posts_log"] = np.log1p(df["n_posts"])
    new_cols.append("n_posts_log")

    # ── Lags de retorno de precio ─────────────────────────────────────────────
    for lag in [1, 2, 3]:
        col = f"return_lag_{lag}"
        df[col] = df["Return"].shift(lag)
        new_cols.append(col)

    # ── Medias móviles de retorno de precio ───────────────────────────────────
    for window in [3, 7]:
        col = f"return_rolling_{window}"
        df[col] = df["Return"].rolling(window).mean()
        new_cols.append(col)

    # ── Volatilidad (desviación estándar móvil de Return) ─────────────────────
    # ddof=1 usa varianza muestral (Bessel's correction), estándar en finanzas.
    df["volatility_7"] = df["Return"].rolling(7).std(ddof=1)
    new_cols.append("volatility_7")

    return df, new_cols


def _engineer_asset(asset: str, in_path: Path, out_path: Path) -> pd.DataFrame:
    """
    Carga el CSV fusionado de un activo, crea todas las features y guarda el resultado.
    """
    print(f"\n  Activo : {asset}")
    print(f"  Entrada  ->  {in_path.relative_to(ROOT)}")

    df = pd.read_csv(in_path, dtype={"Date": str})

    # Validar columnas requeridas
    missing = [c for c in _BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{in_path.name} no contiene las columnas requeridas: {missing}. "
            "Ejecuta primero merge_finbert_all()."
        )

    df, new_cols = _add_features(df)

    # Elimina filas iniciales con NaN por el lag/rolling máximo (_MAX_WINDOW - 1 filas)
    df_clean = df.dropna(subset=new_cols).reset_index(drop=True)
    rows_dropped = len(df) - len(df_clean)

    # Columnas de salida: base + features nuevas
    output_cols = _BASE_COLS + new_cols
    df_out = df_clean[output_cols].copy()

    print(f"  Filas originales : {len(df)}")
    print(f"  Filas eliminadas : {rows_dropped}  "
          f"(NaN por ventana máxima = {_MAX_WINDOW} días)")
    print(f"  Filas finales    : {len(df_out)}")
    print(f"  Rango de fechas  : {df_out['Date'].iloc[0]}  a  {df_out['Date'].iloc[-1]}")
    print(f"  Columnas totales : {len(df_out.columns)}")
    print(f"  Features nuevas  : {new_cols}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"  Guardado  ->  {out_path.relative_to(ROOT)}")

    return df_out


# ── API pública ───────────────────────────────────────────────────────────────

def engineer_finbert_all(
    asset_config: dict[str, tuple[Path, Path]] = ASSET_CONFIG,
) -> dict[str, pd.DataFrame]:
    """
    Genera los datasets de features FinBERT para todos los activos configurados.

    Pasos
    -----
    1. Para cada activo: carga el CSV fusionado (FinBERT + precio).
    2. Añade lags, medias móviles, log de posts y volatilidad.
    3. Elimina filas con NaN causadas por el lag/rolling máximo.
    4. Guarda el dataset de features y devuelve un dict con los resultados.

    Devuelve
    --------
    dict con clave = nombre del activo y valor = DataFrame de features.
    """
    results: dict[str, pd.DataFrame] = {}

    for asset, (in_path, out_path) in asset_config.items():
        if not in_path.exists():
            print(f"\n  [SKIP] {in_path.name} no encontrado — omitiendo {asset}")
            continue
        results[asset] = _engineer_asset(asset, in_path, out_path)

    if not results:
        raise RuntimeError(
            "No se procesó ningún activo. "
            "Verifica que existen los CSV fusionados en data/processed/. "
            "Ejecuta primero merge_finbert_all()."
        )

    print(f"\n  Activos procesados : {list(results.keys())}")
    return results
