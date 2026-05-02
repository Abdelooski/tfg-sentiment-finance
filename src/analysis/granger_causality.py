"""
src/analysis/granger_causality.py

Contrasta la causalidad de Granger entre el sentimiento de Reddit (vader_compound)
y los retornos de precio (Return) para BTC y S&P500.

La causalidad de Granger responde a:
  "¿Los valores pasados de X ayudan a predecir Y más allá de los propios
   valores pasados de Y?"

Se prueban ambas direcciones para cada activo:
  A) sentiment_to_return : ¿vader_compound causa Return?
  B) return_to_sentiment : ¿Return causa vader_compound?

La dirección A es la hipótesis central del TFG: el sentimiento anticipa el precio.
La dirección B sirve de control: ¿es el precio quien mueve el sentimiento?
Si solo A es significativa, el sentimiento es una señal *leading* (adelantada).

Nota sobre estacionariedad: el test de Granger asume series estacionarias.
Return (variación porcentual diaria) es aproximadamente estacionaria.
vader_compound es una puntuación acotada en [-1, 1], también estacionaria.
Para un análisis de mayor rigor podría añadirse un test ADF previo.

Entradas:
  data/processed/btc_features.csv
  data/processed/sp500_features.csv

Salida:
  reports/tables/granger_results.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR    = ROOT / "reports" / "tables"

GRANGER_PATH  = TABLES_DIR / "granger_results.csv"

ASSET_CONFIG: dict[str, Path] = {
    "BTC":   PROCESSED_DIR / "btc_features.csv",
    "SP500": PROCESSED_DIR / "sp500_features.csv",
}

# Número máximo de lags a probar.
# Con datos diarios, 5 lags = 5 días hábiles ≈ 1 semana de mercado.
MAXLAG = 5

# Nivel de significación estadística estándar
ALPHA  = 0.05

# El test ssr_ftest (F-test basado en suma de residuos) es el más usado en
# la literatura para reportar resultados de causalidad de Granger.
_TEST_KEY = "ssr_ftest"

# Para cada dirección, la lista [y, x] define qué variable se predice (y)
# y cuál es el potencial causante (x). grangercausalitytests espera [y, x].
_DIRECTIONS: dict[str, list[str]] = {
    "sentiment_to_return": ["Return",        "vader_compound"],
    "return_to_sentiment": ["vader_compound", "Return"],
}

OUTPUT_COLS: list[str] = [
    "asset", "direction", "lag", "p_value", "significant",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame:
    """
    Carga el CSV de features y devuelve solo las columnas necesarias,
    eliminando filas con NaN en cualquiera de las dos variables.
    """
    required = {"vader_compound", "Return"}
    df = pd.read_csv(path, dtype={"Date": str})

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} no contiene las columnas: {missing}. "
            "Verifica que el CSV proviene de feature_engineering.py."
        )

    return df[list(required)].dropna().reset_index(drop=True)


def _run_granger_direction(
    df: pd.DataFrame,
    asset: str,
    direction: str,
    col_order: list[str],
    maxlag: int,
) -> list[dict]:
    """
    Ejecuta grangercausalitytests para una dirección concreta y devuelve
    una lista de dicts con el p-valor de cada lag.

    col_order = [y, x]  donde y es la variable a predecir y x el causante.

    Si el test falla (por ejemplo, por colinealidad o datos insuficientes)
    se registra p_value=None y significant=False para ese lag.
    """
    data   = df[col_order].values   # array 2D: columna 0 = y, columna 1 = x
    rows: list[dict] = []

    try:
        # verbose=False suprime la salida de tabla de statsmodels;
        # el FutureWarning de statsmodels sobre verbose se silencia aquí.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            results = grangercausalitytests(data, maxlag=maxlag, verbose=False)
    except Exception as exc:
        print(f"  [ERROR] {asset} / {direction}: {exc}")
        # Registra todos los lags como no calculables
        for lag in range(1, maxlag + 1):
            rows.append({
                "asset":       asset,
                "direction":   direction,
                "lag":         lag,
                "p_value":     None,
                "significant": False,
            })
        return rows

    for lag in range(1, maxlag + 1):
        try:
            p_value = float(results[lag][0][_TEST_KEY][1])
        except (KeyError, IndexError, TypeError):
            p_value = None

        rows.append({
            "asset":       asset,
            "direction":   direction,
            "lag":         lag,
            "p_value":     round(p_value, 6) if p_value is not None else None,
            "significant": (p_value is not None) and (p_value < ALPHA),
        })

    return rows


def _analyse_asset(asset: str, path: Path) -> list[dict]:
    """
    Carga los datos de un activo y ejecuta el test de Granger en ambas
    direcciones. Imprime un resumen por consola.
    """
    print(f"\n  Activo : {asset}")
    print(f"  Cargando  ->  {path.relative_to(ROOT)}")

    df = _load(path)
    n  = len(df)
    min_obs = 2 * MAXLAG + 1
    if n < min_obs:
        raise ValueError(
            f"{asset}: solo {n} observaciones disponibles; "
            f"se necesitan al menos {min_obs} para maxlag={MAXLAG}."
        )
    print(f"  Observaciones : {n}")

    all_rows: list[dict] = []

    for direction, col_order in _DIRECTIONS.items():
        rows = _run_granger_direction(df, asset, direction, col_order, MAXLAG)
        all_rows.extend(rows)

        # Resumen de consola para esta dirección
        cause, effect = col_order[1], col_order[0]
        print(f"\n  Test: {cause} -> {effect}  ({direction})")
        print(f"  {'Lag':>4}  {'p-valor':>10}  {'Significativo':>14}")
        print(f"  {'-'*4}  {'-'*10}  {'-'*14}")
        for r in rows:
            pv_str  = f"{r['p_value']:.6f}" if r["p_value"] is not None else "    N/A   "
            sig_str = "SI  *" if r["significant"] else "no"
            print(f"  {r['lag']:>4}  {pv_str:>10}  {sig_str:>14}")

    return all_rows


def _print_summary(granger_df: pd.DataFrame) -> None:
    """
    Imprime para cada (activo, dirección): el lag con menor p-valor y si
    al menos un lag resulta significativo.
    """
    print("\n\n  " + "=" * 62)
    print("  Resumen de causalidad de Granger")
    print("  " + "-" * 62)

    for (asset, direction), grp in granger_df.groupby(["asset", "direction"]):
        valid  = grp.dropna(subset=["p_value"])
        if valid.empty:
            print(f"\n  {asset} | {direction}: sin resultados válidos")
            continue

        best    = valid.loc[valid["p_value"].idxmin()]
        any_sig = grp["significant"].any()

        cause, effect = (
            ("vader_compound", "Return")
            if direction == "sentiment_to_return"
            else ("Return", "vader_compound")
        )

        print(f"\n  {asset} -- {cause} -> {effect}")
        print(f"    Mejor lag    : {int(best['lag'])} "
              f"(p = {best['p_value']:.6f})")
        if any_sig:
            sig_lags = grp[grp["significant"]]["lag"].tolist()
            print(f"    Significativo: SI  (lags {sig_lags}, alpha={ALPHA})")
        else:
            print(f"    Significativo: NO  (ningun lag < {ALPHA})")

    print("  " + "=" * 62)


# ── API pública ───────────────────────────────────────────────────────────────

def run_granger_analysis(
    asset_config: dict[str, Path] = ASSET_CONFIG,
    granger_path: Path            = GRANGER_PATH,
    maxlag: int                   = MAXLAG,
) -> pd.DataFrame:
    """
    Ejecuta el análisis de causalidad de Granger para todos los activos.

    Pasos
    -----
    1. Carga vader_compound y Return de cada CSV de features.
    2. Prueba ambas direcciones (sentiment->return y return->sentiment).
    3. Extrae el p-valor del F-test (ssr_ftest) para cada lag 1..maxlag.
    4. Guarda granger_results.csv e imprime el resumen.

    Devuelve
    --------
    pd.DataFrame con columnas: asset, direction, lag, p_value, significant.
    """
    all_rows: list[dict] = []

    for asset, path in asset_config.items():
        if not path.exists():
            print(f"\n  [SKIP] {path.name} no encontrado — omitiendo {asset}")
            continue
        all_rows.extend(_analyse_asset(asset, path))

    if not all_rows:
        raise RuntimeError(
            "No se procesó ningún activo. "
            "Verifica que existan los CSVs de features en data/processed/."
        )

    granger_df = pd.DataFrame(all_rows)[OUTPUT_COLS]

    _print_summary(granger_df)

    granger_path.parent.mkdir(parents=True, exist_ok=True)
    granger_df.to_csv(granger_path, index=False)
    print(f"\n  Guardado  ->  {granger_path.relative_to(ROOT)}")

    return granger_df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_granger_analysis()
