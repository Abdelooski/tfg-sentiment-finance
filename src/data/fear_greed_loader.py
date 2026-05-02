"""
src/data/fear_greed_loader.py

Descarga el Crypto Fear & Greed Index desde la API pública de alternative.me
y lo guarda como CSV.

El índice oscila entre 0 (Miedo Extremo) y 100 (Codicia Extrema) y se publica
diariamente. Es un proxy ampliamente usado en la literatura académica sobre
sentimiento en mercados de criptomonedas.

API     : https://api.alternative.me/fng/?limit=2000
Sin API key — petición HTTP GET estándar.

Salida  : data/raw/fear_greed.csv
Columnas: Date, fng_value, fng_classification
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
RAW_DIR     = ROOT / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "fear_greed.csv"

# ── Constantes ────────────────────────────────────────────────────────────────
API_URL     = "https://api.alternative.me/fng/?limit=2000"
TIMEOUT     = 30          # segundos de espera máxima por la respuesta HTTP
OUTPUT_COLS = ["Date", "fng_value", "fng_classification"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch(url: str, timeout: int) -> dict:
    """
    Realiza la petición GET y devuelve el JSON parseado.

    Lanza requests.HTTPError si el servidor devuelve un código de error (4xx/5xx).
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _parse(payload: dict) -> pd.DataFrame:
    """
    Convierte el JSON de la API en un DataFrame limpio.

    Estructura esperada del payload:
      {
        "data": [
          {
            "value":               "72",
            "value_classification": "Greed",
            "timestamp":           "1714435200"
          },
          ...
        ],
        "metadata": {"error": null}
      }

    Los timestamps son Unix epoch en segundos (cadena de texto).
    Se convierten a fecha YYYY-MM-DD en UTC para consistencia con el resto
    del pipeline (BTC y S&P500 también usan fechas en UTC).
    """
    # Validación básica de la respuesta
    if "metadata" in payload and payload["metadata"].get("error"):
        raise ValueError(f"La API devolvió un error: {payload['metadata']['error']}")

    records = payload.get("data")
    if not records:
        raise ValueError("La respuesta de la API no contiene el campo 'data'.")

    rows: list[dict] = []
    skipped = 0

    for entry in records:
        try:
            date_str = pd.to_datetime(
                int(entry["timestamp"]), unit="s", utc=True
            ).strftime("%Y-%m-%d")

            rows.append({
                "Date":               date_str,
                "fng_value":          int(entry["value"]),
                "fng_classification": str(entry["value_classification"]).strip(),
            })
        except (KeyError, ValueError, TypeError):
            skipped += 1
            continue

    if skipped:
        print(f"  [AVISO] {skipped} entradas omitidas por campos inesperados.")

    if not rows:
        raise ValueError("No se pudo parsear ninguna entrada de la API.")

    df = (
        pd.DataFrame(rows, columns=OUTPUT_COLS)
        .sort_values("Date", ascending=True)
        .drop_duplicates(subset="Date")   # la API puede repetir la fecha actual
        .reset_index(drop=True)
    )

    return df


# ── API pública ───────────────────────────────────────────────────────────────

def load_fear_greed(
    url: str         = API_URL,
    output_path: Path = OUTPUT_PATH,
    timeout: int      = TIMEOUT,
) -> pd.DataFrame:
    """
    Descarga el Fear & Greed Index y lo guarda en data/raw/fear_greed.csv.

    Pasos
    -----
    1. GET a la API de alternative.me (sin autenticación).
    2. Parsea el JSON y convierte timestamps Unix a fechas YYYY-MM-DD.
    3. Construye DataFrame con Date, fng_value, fng_classification.
    4. Ordena por fecha ascendente y elimina duplicados de fecha.
    5. Guarda el CSV e imprime resumen.

    Devuelve
    --------
    pd.DataFrame con los datos descargados.
    """
    print(f"  Consultando  ->  {url}")

    try:
        payload = _fetch(url, timeout)
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"La petición a la API superó el límite de {timeout}s. "
            "Comprueba la conexión a Internet e inténtalo de nuevo."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "No se pudo conectar con la API de alternative.me. "
            "Comprueba la conexión a Internet."
        )
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"La API respondió con error HTTP: {exc}")

    df = _parse(payload)

    # ── Resumen ───────────────────────────────────────────────────────────────
    print(f"  Filas descargadas : {len(df):,}")
    print(f"  Rango de fechas   : {df['Date'].iloc[0]}  a  {df['Date'].iloc[-1]}")
    print(f"  Valor medio       : {df['fng_value'].mean():.1f}  "
          f"(min={df['fng_value'].min()}  max={df['fng_value'].max()})")

    clasificaciones = df["fng_classification"].value_counts().to_dict()
    print(f"  Clasificaciones   : {clasificaciones}")

    # ── Guardado ──────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Guardado  ->  {output_path.relative_to(ROOT)}")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_fear_greed()
