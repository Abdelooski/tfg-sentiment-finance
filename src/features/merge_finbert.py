"""
src/features/merge_finbert.py

Fusiona el sentimiento diario de FinBERT con los datasets de precio procesados
de BTC y S&P500 mediante un inner join sobre la columna Date.

El resultado es análogo a merge_datasets.py (que fusiona VADER con los precios),
pero usando los scores de FinBERT en lugar de los de VADER. Esto permite
comparar directamente ambos modelos de sentimiento sobre la misma ventana temporal.

Entradas:
  data/processed/reddit_wsb_finbert_daily.csv
  data/processed/btc_processed.csv
  data/processed/sp500_processed.csv

Salidas:
  data/processed/btc_finbert_merged.csv
  data/processed/sp500_finbert_merged.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

FINBERT_DAILY_PATH = PROCESSED_DIR / "reddit_wsb_finbert_daily.csv"

# Mapea nombre de activo -> (precio procesado, salida del merge)
ASSET_CONFIG: dict[str, tuple[Path, Path]] = {
    "BTC": (
        PROCESSED_DIR / "btc_processed.csv",
        PROCESSED_DIR / "btc_finbert_merged.csv",
    ),
    "SP500": (
        PROCESSED_DIR / "sp500_processed.csv",
        PROCESSED_DIR / "sp500_finbert_merged.csv",
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: Path, label: str) -> pd.DataFrame:
    """
    Carga un CSV forzando Date como string y normalizando su formato a YYYY-MM-DD.

    Forzar dtype=str en Date evita que pandas lo convierta a Timestamp con zona
    horaria, lo que causaría discrepancias al hacer el merge.
    """
    df = pd.read_csv(path, dtype={"Date": str})
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    print(f"  {label:<22}  {len(df):>5} filas  "
          f"({df['Date'].min()} .. {df['Date'].max()})")
    return df


def _merge_asset(
    asset:      str,
    price_path: Path,
    out_path:   Path,
    finbert_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fusiona el dataset de precio de un activo con el sentimiento FinBERT diario.

    Se usa inner join para conservar solo los días en que ambos datasets tienen
    datos — igual que en merge_datasets.py para mantener consistencia metodológica.
    """
    price_df = _load_csv(price_path, asset)
    merged   = price_df.merge(finbert_df, on="Date", how="inner")

    print(f"\n  {asset}:")
    print(f"    Filas fusionadas : {len(merged)}")
    print(f"    Rango de fechas  : {merged['Date'].min()}  a  {merged['Date'].max()}")
    print(f"    Columnas         : {list(merged.columns)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"    Guardado  ->  {out_path.relative_to(ROOT)}")

    return merged


# ── API pública ───────────────────────────────────────────────────────────────

def merge_finbert_all(
    finbert_path: Path                          = FINBERT_DAILY_PATH,
    asset_config: dict[str, tuple[Path, Path]] = ASSET_CONFIG,
) -> dict[str, pd.DataFrame]:
    """
    Fusiona el sentimiento FinBERT diario con cada activo configurado.

    Pasos
    -----
    1. Carga reddit_wsb_finbert_daily.csv y normaliza fechas.
    2. Para cada activo: carga el CSV de precio y hace inner join por Date.
    3. Guarda el resultado en data/processed/.
    4. Imprime resumen de filas, rango y columnas por activo.

    Devuelve
    --------
    dict con clave = nombre del activo y valor = DataFrame fusionado.
    """
    if not finbert_path.exists():
        raise FileNotFoundError(
            f"No se encontró {finbert_path.name}. "
            "Ejecuta primero apply_finbert() (Step 19)."
        )

    print(f"  Cargando sentimiento FinBERT  ->  {finbert_path.relative_to(ROOT)}")
    finbert_df = _load_csv(finbert_path, "finbert_daily")
    print()

    results: dict[str, pd.DataFrame] = {}

    for asset, (price_path, out_path) in asset_config.items():
        if not price_path.exists():
            print(f"  [SKIP] {price_path.name} no encontrado — omitiendo {asset}")
            continue
        results[asset] = _merge_asset(asset, price_path, out_path, finbert_df)

    if not results:
        raise RuntimeError(
            "No se fusionó ningún activo. "
            "Verifica que btc_processed.csv y sp500_processed.csv existen."
        )

    return results
