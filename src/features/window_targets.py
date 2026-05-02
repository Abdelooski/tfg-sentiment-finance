"""
src/features/window_targets.py

Genera targets multi-horizonte para BTC y S&P500.

En lugar de predecir solo la dirección del día siguiente (Direction, horizonte 1d),
este módulo crea tres horizontes adicionales:
  - 3 días : ¿sube el precio en los próximos 3 días?
  - 5 días : ¿sube el precio en los próximos 5 días?
  - 7 días : ¿sube el precio en los próximos 7 días?

Para cada horizonte X se calculan dos columnas nuevas:
  Return_Xd    = (Close[t+X] / Close[t]) - 1   → retorno continuo X días adelante
  Direction_Xd = 1 si Return_Xd > 0, 0 en caso contrario  → variable objetivo binaria

Los horizontes más largos permiten estudiar si el sentimiento tiene efecto
acumulado (no solo a 1 día), algo relevante para la hipótesis del TFG.

Entradas:
  data/processed/btc_features.csv
  data/processed/sp500_features.csv

Salidas:
  data/processed/btc_features_windowed.csv
  data/processed/sp500_features_windowed.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

# Mapea nombre de activo -> (CSV de entrada, CSV de salida)
ASSET_CONFIG: dict[str, tuple[Path, Path]] = {
    "BTC": (
        PROCESSED_DIR / "btc_features.csv",
        PROCESSED_DIR / "btc_features_windowed.csv",
    ),
    "SP500": (
        PROCESSED_DIR / "sp500_features.csv",
        PROCESSED_DIR / "sp500_features_windowed.csv",
    ),
}

# Horizontes en días para los que se crean targets adicionales
HORIZONS: list[int] = [3, 5, 7]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_window_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Añade columnas de retorno y dirección para cada horizonte definido en HORIZONS.

    Fórmula:
      Return_Xd    = (Close[t+X] / Close[t]) - 1
      Direction_Xd = 1 si Return_Xd > 0, 0 en caso contrario

    El shift(-X) desplaza los valores de Close X posiciones hacia arriba,
    de modo que cada fila t accede al precio futuro sin mirar el futuro durante
    el entrenamiento — siempre que el modelo solo use filas hasta t.

    Las últimas HORIZONS[-1] filas (7 por defecto) tendrán NaN en las columnas
    de mayor horizonte y se eliminan con dropna para mantener el dataset limpio.

    Devuelve el DataFrame enriquecido y la lista de columnas nuevas.
    """
    if "Close" not in df.columns:
        raise ValueError(
            "El dataset no contiene la columna 'Close'. "
            "Verifica que btc_features.csv / sp500_features.csv provienen "
            "de feature_engineering.py y no han sido modificados."
        )

    df = df.copy()
    new_cols: list[str] = []

    for h in HORIZONS:
        ret_col = f"Return_{h}d"
        dir_col = f"Direction_{h}d"

        df[ret_col] = (df["Close"].shift(-h) / df["Close"]) - 1
        df[dir_col] = (df[ret_col] > 0).astype(int)

        new_cols.extend([ret_col, dir_col])

    # Elimina las últimas filas donde el precio futuro es desconocido (NaN)
    df = df.dropna(subset=new_cols).reset_index(drop=True)

    return df, new_cols


def _process_asset(asset: str, in_path: Path, out_path: Path) -> pd.DataFrame:
    """
    Carga el CSV de un activo, añade los targets multi-horizonte y guarda el resultado.

    No modifica el archivo de entrada — lee de in_path y escribe en out_path.
    """
    print(f"\n  Activo : {asset}")
    print(f"  Entrada  ->  {in_path.relative_to(ROOT)}")

    df_in = pd.read_csv(in_path, dtype={"Date": str})
    print(f"  Filas originales  : {len(df_in)}")

    df_out, new_cols = _add_window_targets(df_in)

    print(f"  Filas tras shift  : {len(df_out)}  "
          f"(eliminadas {len(df_in) - len(df_out)} por NaN en horizonte {max(HORIZONS)}d)")
    print(f"  Rango de fechas   : {df_out['Date'].iloc[0]}  a  {df_out['Date'].iloc[-1]}")
    print(f"  Columnas nuevas   : {new_cols}")

    # Distribución de clases por horizonte
    for h in HORIZONS:
        dir_col = f"Direction_{h}d"
        n_up = int(df_out[dir_col].sum())
        n_tot = len(df_out)
        print(f"    {dir_col:<15}  subidas={n_up}/{n_tot}  ({n_up/n_tot:.1%})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"  Guardado  ->  {out_path.relative_to(ROOT)}")

    return df_out


# ── API pública ───────────────────────────────────────────────────────────────

def generate_windowed_targets(
    asset_config: dict[str, tuple[Path, Path]] = ASSET_CONFIG,
) -> dict[str, pd.DataFrame]:
    """
    Genera datasets con targets multi-horizonte para todos los activos configurados.

    Pasos
    -----
    1. Para cada activo: carga el CSV de features, añade Return_Xd y Direction_Xd
       para cada horizonte en HORIZONS.
    2. Elimina filas con NaN causadas por el shift del precio futuro.
    3. Guarda el dataset enriquecido sin modificar el archivo original.

    Devuelve
    --------
    dict con clave = nombre del activo y valor = DataFrame enriquecido.
    """
    results: dict[str, pd.DataFrame] = {}

    for asset, (in_path, out_path) in asset_config.items():
        if not in_path.exists():
            print(f"\n  [SKIP] {in_path.name} no encontrado — omitiendo {asset}")
            continue
        results[asset] = _process_asset(asset, in_path, out_path)

    if not results:
        raise RuntimeError(
            "No se procesó ningún activo. "
            "Verifica que existan los CSVs de features en data/processed/."
        )

    print(f"\n  Activos procesados : {list(results.keys())}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_windowed_targets()
