"""
src/nlp/finbert_sentiment.py

Aplica el modelo FinBERT (ProsusAI/finbert) al dataset de Reddit WSB
para generar scores de sentimiento financiero a nivel de post y diario.

FinBERT es un modelo BERT preentrenado en textos financieros (noticias,
informes, foros de inversión). A diferencia de VADER (basado en reglas),
FinBERT comprende el contexto y la jerga financiera, lo que lo hace más
preciso para textos de Reddit sobre mercados y criptomonedas.

Entradas:
  data/processed/reddit_wsb_processed.csv

Salidas:
  data/processed/reddit_wsb_finbert_posts.csv  — scores por post
  data/processed/reddit_wsb_finbert_daily.csv  — scores agregados por día
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_PATH = PROCESSED_DIR / "reddit_wsb_processed.csv"
POSTS_PATH = PROCESSED_DIR / "reddit_wsb_finbert_posts.csv"
DAILY_PATH = PROCESSED_DIR / "reddit_wsb_finbert_daily.csv"

# ── Parámetros ────────────────────────────────────────────────────────────────
MODEL_NAME   = "ProsusAI/finbert"
SAMPLE_SIZE  = 100_000   # máximo de posts a procesar
BATCH_SIZE   = 32        # posts por batch de inferencia
RANDOM_STATE = 42

TEXT_COL = "clean_text"
DATE_COL = "Date"

# Columnas del CSV de entrada que se conservan en la salida
_INPUT_KEEP = ["Date", "text", "clean_text"]

# Columnas de sentimiento generadas por FinBERT
_SENT_COLS = ["finbert_neg", "finbert_neu", "finbert_pos", "finbert_compound"]

# Mapeo canónico de label FinBERT -> nombre de columna de salida
_LABEL_TO_COL: dict[str, str] = {
    "positive": "finbert_pos",
    "negative": "finbert_neg",
    "neutral":  "finbert_neu",
}


# ── Verificación de dependencias ──────────────────────────────────────────────

def _check_dependencies() -> None:
    """
    Verifica que transformers, torch y tqdm están instalados.
    Si alguno falta, lanza ImportError con instrucciones claras.
    """
    missing = []
    for pkg in ("transformers", "torch", "tqdm"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        pkgs = " ".join(missing)
        raise ImportError(
            f"Faltan las siguientes dependencias: {missing}\n"
            f"Instálalas con:\n"
            f"  pip install {pkgs}\n"
            f"Para soporte GPU añade la versión CUDA de torch:\n"
            f"  https://pytorch.org/get-started/locally/"
        )


# ── Carga y muestreo ──────────────────────────────────────────────────────────

def _load_and_sample(path: Path, sample_size: int, random_state: int) -> pd.DataFrame:
    """
    Carga el CSV procesado de Reddit y devuelve una muestra aleatoria
    estratificada por fecha si supera sample_size filas.

    El muestreo aleatorio con random_state fijo garantiza reproducibilidad:
    ejecutar el módulo dos veces produce exactamente el mismo subconjunto.
    """
    required = {DATE_COL, TEXT_COL, "text"}

    df = pd.read_csv(path, dtype={DATE_COL: str})
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{path.name} no contiene las columnas requeridas: {missing_cols}"
        )

    n_total = len(df)
    print(f"  Filas cargadas   : {n_total:,}")

    if n_total > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"  Filas muestreadas: {len(df):,}  (random_state={random_state})")
    else:
        print(f"  Filas muestreadas: {n_total:,}  (dataset completo, < {sample_size:,})")

    df = df[_INPUT_KEEP].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=[TEXT_COL]).reset_index(drop=True)

    min_date = df[DATE_COL].min()
    max_date = df[DATE_COL].max()
    print(f"  Rango de fechas  : {min_date}  a  {max_date}")

    return df


# ── Carga del modelo ──────────────────────────────────────────────────────────

def _load_model(model_name: str, device):
    """
    Descarga (si no está en caché) y carga el tokenizador y el modelo FinBERT.

    El modelo se descarga de HuggingFace Hub la primera vez (~500 MB).
    Las ejecuciones posteriores usan la caché local de HuggingFace.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
        print(f"  Cargando modelo  : {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
    except OSError as exc:
        raise RuntimeError(
            f"No se pudo cargar el modelo '{model_name}'.\n"
            f"Comprueba la conexión a Internet o el nombre del modelo.\n"
            f"Detalle: {exc}"
        ) from exc

    # Verificar que el modelo tiene las etiquetas esperadas
    id2label = model.config.id2label
    label_set = {v.lower() for v in id2label.values()}
    expected  = {"positive", "negative", "neutral"}
    if not expected.issubset(label_set):
        warnings.warn(
            f"El modelo tiene etiquetas inesperadas: {label_set}. "
            f"Se esperaban {expected}. Los scores pueden ser incorrectos.",
            UserWarning,
            stacklevel=2,
        )

    return tokenizer, model


# ── Inferencia por batch ──────────────────────────────────────────────────────

def _score_batch(
    texts: list[str],
    model,
    tokenizer,
    device,
) -> list[dict[str, float]]:
    """
    Ejecuta FinBERT sobre un batch de textos y devuelve las probabilidades
    softmax para cada etiqueta (positive, negative, neutral).

    Se usa truncation=True porque FinBERT (BERT-base) tiene límite de 512 tokens.
    Los textos más largos se truncan por la derecha.
    """
    import torch
    import torch.nn.functional as F

    # Sustituye textos vacíos por un espacio para evitar entradas degeneradas
    texts = [t if t and t.strip() else " " for t in texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs    = F.softmax(logits, dim=-1).cpu().numpy()
    id2label = model.config.id2label

    results: list[dict[str, float]] = []
    for row in probs:
        scores: dict[str, float] = {}
        for idx, prob in enumerate(row):
            label   = id2label[idx].lower()
            col_key = _LABEL_TO_COL.get(label)
            if col_key:
                scores[col_key] = float(prob)
        results.append(scores)

    return results


def _run_inference(df: pd.DataFrame, model, tokenizer, device) -> pd.DataFrame:
    """
    Itera sobre todos los posts en batches de BATCH_SIZE, acumula los scores
    y los añade como columnas al DataFrame.

    Si un batch falla (por ejemplo, por un texto muy raro) se asignan scores
    neutros (finbert_neu=1.0) para ese batch y se continúa con el siguiente.
    """
    from tqdm import tqdm

    texts    = df[TEXT_COL].fillna("").tolist()
    n        = len(texts)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    all_scores: list[dict[str, float]] = []

    with tqdm(total=n_batches, desc="  FinBERT inference", unit="lote") as pbar:
        for start in range(0, n, BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            try:
                scores = _score_batch(batch, model, tokenizer, device)
            except Exception as exc:
                batch_idx = start // BATCH_SIZE + 1
                print(f"\n  [WARN] Lote {batch_idx} fallido ({exc}). "
                      f"Asignando neutral=1.0 al lote.")
                scores = [
                    {"finbert_pos": 0.0, "finbert_neg": 0.0, "finbert_neu": 1.0}
                    for _ in batch
                ]
            all_scores.extend(scores)
            pbar.update(1)

    scores_df = pd.DataFrame(all_scores)

    # Columnas no calculadas (raro si el modelo tiene etiquetas distintas)
    for col in ["finbert_pos", "finbert_neg", "finbert_neu"]:
        if col not in scores_df.columns:
            scores_df[col] = 0.0

    scores_df["finbert_compound"] = (
        scores_df["finbert_pos"] - scores_df["finbert_neg"]
    )

    return pd.concat(
        [df.reset_index(drop=True), scores_df.reset_index(drop=True)],
        axis=1,
    )


# ── Agregación diaria ─────────────────────────────────────────────────────────

def _aggregate_daily(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega los scores de sentimiento FinBERT por fecha.

    Para cada día calcula la media de los cuatro scores y el número de posts.
    El resultado es análogo al generado por vader_sentiment.py para permitir
    comparaciones directas entre VADER y FinBERT.
    """
    daily = (
        posts_df
        .groupby(DATE_COL)
        .agg(
            finbert_neg=("finbert_neg",      "mean"),
            finbert_neu=("finbert_neu",      "mean"),
            finbert_pos=("finbert_pos",      "mean"),
            finbert_compound=("finbert_compound", "mean"),
            n_posts=(DATE_COL,              "count"),
        )
        .reset_index()
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    return daily


# ── API pública ───────────────────────────────────────────────────────────────

def apply_finbert(
    input_path:  Path = INPUT_PATH,
    posts_path:  Path = POSTS_PATH,
    daily_path:  Path = DAILY_PATH,
    sample_size: int  = SAMPLE_SIZE,
    batch_size:  int  = BATCH_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completo de análisis de sentimiento FinBERT sobre Reddit WSB.

    Pasos
    -----
    1. Verifica dependencias (transformers, torch, tqdm).
    2. Carga y muestrea reddit_wsb_processed.csv.
    3. Detecta dispositivo (CUDA si disponible, si no CPU).
    4. Descarga y carga ProsusAI/finbert.
    5. Inferencia en batches con barra de progreso.
    6. Agrega scores por fecha.
    7. Guarda los dos CSV de salida.

    Devuelve
    --------
    (posts_df, daily_df) — DataFrames de posts y diario.
    """
    global BATCH_SIZE
    BATCH_SIZE = batch_size   # permite sobreescribir desde CLI / tests

    _check_dependencies()

    import torch

    # ── Dispositivo ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Dispositivo      : {device.type.upper()}")
    if device.type == "cpu":
        print("  [INFO] GPU no disponible. La inferencia en CPU puede tardar "
              "varios minutos para 100,000 posts.")

    # ── Carga de datos ────────────────────────────────────────────────────────
    print(f"\n  Entrada  ->  {input_path.relative_to(ROOT)}")
    df = _load_and_sample(input_path, sample_size, random_state)

    # ── Carga del modelo ──────────────────────────────────────────────────────
    print()
    tokenizer, model = _load_model(MODEL_NAME, device)
    print(f"  Etiquetas modelo : {model.config.id2label}")

    # ── Inferencia ────────────────────────────────────────────────────────────
    print(f"\n  Posts a analizar : {len(df):,}  |  Batch size: {BATCH_SIZE}")
    posts_df = _run_inference(df, model, tokenizer, device)

    # Libera memoria GPU si se usó
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Agregación diaria ─────────────────────────────────────────────────────
    daily_df = _aggregate_daily(posts_df)

    # ── Guardado ──────────────────────────────────────────────────────────────
    posts_path.parent.mkdir(parents=True, exist_ok=True)
    posts_df[_INPUT_KEEP + _SENT_COLS].to_csv(posts_path, index=False)
    daily_df.to_csv(daily_path, index=False)

    print(f"\n  Posts guardados  ->  {posts_path.relative_to(ROOT)}  "
          f"({len(posts_df):,} filas)")
    print(f"  Diario guardado  ->  {daily_path.relative_to(ROOT)}  "
          f"({len(daily_df):,} días)")

    return posts_df, daily_df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    apply_finbert()
