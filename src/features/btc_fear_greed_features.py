"""
src/features/btc_fear_greed_features.py

Integra el Crypto Fear & Greed Index en el dataset de BTC y evalúa si
el sentimiento específico de criptomonedas mejora la capacidad predictiva
más allá del sentimiento de Reddit (VADER).

Compara tres conjuntos de features progresivos:
  1. price_only               — solo indicadores técnicos de precio
  2. price_plus_reddit        — precio + sentimiento VADER de Reddit (WSB)
  3. price_plus_reddit_plus_fng — precio + Reddit + Fear & Greed Index

Si el tercer conjunto supera al segundo en F1, el índice FNG añade valor
predictivo incremental sobre el sentimiento de Reddit.

Entradas:
  data/processed/btc_features.csv   — features de BTC con VADER
  data/raw/fear_greed.csv           — Fear & Greed Index diario

Salidas:
  data/processed/btc_features_fng.csv      — dataset enriquecido con FNG
  reports/tables/btc_fng_model_results.csv — métricas por feature set y modelo
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
RAW_DIR       = ROOT / "data" / "raw"
TABLES_DIR    = ROOT / "reports" / "tables"

BTC_FEATURES_PATH    = PROCESSED_DIR / "btc_features.csv"
FNG_PATH             = RAW_DIR       / "fear_greed.csv"
OUTPUT_FEATURES_PATH = PROCESSED_DIR / "btc_features_fng.csv"
RESULTS_PATH         = TABLES_DIR    / "btc_fng_model_results.csv"

TARGET_COL  = "Direction"
TRAIN_RATIO = 0.80

# ── Conjuntos de features ─────────────────────────────────────────────────────

_PRICE_FEATURES: list[str] = [
    "Return",
    "return_lag_1", "return_lag_2", "return_lag_3",
    "return_rolling_3", "return_rolling_7",
    "volatility_7",
]

# Sentimiento de Reddit / WallStreetBets vía VADER
_REDDIT_FEATURES: list[str] = [
    "vader_compound",
    "vader_neg", "vader_neu", "vader_pos",
    "vader_compound_lag_1", "vader_compound_lag_2", "vader_compound_lag_3",
    "vader_compound_rolling_3", "vader_compound_rolling_7",
    "n_posts_log",
]

# Fear & Greed Index: valor diario + lags + medias móviles
_FNG_FEATURES: list[str] = [
    "fng_value",
    "fng_lag_1", "fng_lag_2", "fng_lag_3",
    "fng_rolling_3", "fng_rolling_7",
]

# Orden deliberado: cada conjunto es un superconjunto del anterior,
# lo que permite comparar el valor incremental de cada fuente de sentimiento.
FEATURE_SETS: dict[str, list[str]] = {
    "price_only":                 _PRICE_FEATURES,
    "price_plus_reddit":          _PRICE_FEATURES + _REDDIT_FEATURES,
    "price_plus_reddit_plus_fng": _PRICE_FEATURES + _REDDIT_FEATURES + _FNG_FEATURES,
}

RESULTS_COLS: list[str] = [
    "feature_set", "model",
    "accuracy", "precision", "recall", "f1",
    "n_train", "n_test",
]


# ── Carga y fusión ────────────────────────────────────────────────────────────

def _load_and_merge(
    btc_path: Path,
    fng_path: Path,
) -> pd.DataFrame:
    """
    Carga btc_features.csv y fear_greed.csv y los une por Date (inner join).

    La columna Date se fuerza a string YYYY-MM-DD en ambos datasets antes del
    merge para evitar discrepancias de formato (datetime vs string).
    """
    btc_df = pd.read_csv(btc_path, dtype={"Date": str})
    fng_df = pd.read_csv(fng_path, dtype={"Date": str})

    # Normaliza el formato de fecha a YYYY-MM-DD en ambos lados
    btc_df["Date"] = pd.to_datetime(btc_df["Date"]).dt.strftime("%Y-%m-%d")
    fng_df["Date"] = pd.to_datetime(fng_df["Date"]).dt.strftime("%Y-%m-%d")

    # Solo necesitamos fng_value del CSV del índice; las otras columnas
    # (fng_classification) no se usan como feature numérica.
    fng_df = fng_df[["Date", "fng_value"]]

    merged = btc_df.merge(fng_df, on="Date", how="inner")
    return merged


def _add_fng_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade lags y medias móviles del Fear & Greed Index al DataFrame.

    Los lags y ventanas son idénticos a los usados con vader_compound para
    que la comparación entre fuentes de sentimiento sea simétrica.
    Anti-leakage: todos los valores son de t o anteriores (shift hacia delante).
    """
    df = df.copy()
    df["fng_lag_1"]     = df["fng_value"].shift(1)
    df["fng_lag_2"]     = df["fng_value"].shift(2)
    df["fng_lag_3"]     = df["fng_value"].shift(3)
    df["fng_rolling_3"] = df["fng_value"].rolling(3).mean()
    df["fng_rolling_7"] = df["fng_value"].rolling(7).mean()

    # La ventana máxima es 7, por lo que las primeras 6 filas tendrán NaN
    df = df.dropna(subset=_FNG_FEATURES).reset_index(drop=True)
    return df


# ── Modelos ───────────────────────────────────────────────────────────────────

def _build_models() -> dict[str, object]:
    """
    Instancias frescas de los tres clasificadores con hiperparámetros
    idénticos a train_models.py para garantizar comparabilidad directa.
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


# ── Split y evaluación ────────────────────────────────────────────────────────

def _temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split cronológico 80/20 sin shuffle — preserva causalidad temporal."""
    split = int(len(X) * ratio)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def _evaluate(y_true: pd.Series, y_pred) -> dict[str, float]:
    """Cuatro métricas de clasificación redondeadas a 4 decimales."""
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred),                   4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred,        zero_division=0), 4),
    }


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def _train_all_sets(df: pd.DataFrame) -> list[dict]:
    """
    Entrena los tres modelos sobre los tres conjuntos de features.

    Itera en el orden de FEATURE_SETS (price_only → price_plus_reddit →
    price_plus_reddit_plus_fng) para que la tabla de consola muestre
    el progreso incremental de cada fuente de sentimiento.
    """
    all_results: list[dict] = []

    print(f"\n  {'Conjunto':<30} {'Modelo':<22} {'Acc':>6} {'Prec':>6} "
          f"{'Rec':>6} {'F1':>6}")
    print(f"  {'-'*30} {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for set_name, feature_cols in FEATURE_SETS.items():
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"El dataset no contiene las features requeridas "
                f"para '{set_name}': {missing}"
            )

        X = df[feature_cols]
        y = df[TARGET_COL]
        X_train, X_test, y_train, y_test = _temporal_split(X, y)
        n_train, n_test = len(X_train), len(X_test)

        models = _build_models()
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            metrics = _evaluate(y_test, model.predict(X_test))

            print(
                f"  {set_name:<30} {model_name:<22} "
                f"{metrics['accuracy']:>6.4f} {metrics['precision']:>6.4f} "
                f"{metrics['recall']:>6.4f} {metrics['f1']:>6.4f}"
            )

            all_results.append({
                "feature_set": set_name,
                "model":       model_name,
                **metrics,
                "n_train":     n_train,
                "n_test":      n_test,
            })

    return all_results


# ── Conclusión ────────────────────────────────────────────────────────────────

def _print_conclusion(results_df: pd.DataFrame) -> None:
    """
    Compara el mejor F1 de price_plus_reddit vs price_plus_reddit_plus_fng
    e imprime una conclusión clara sobre el valor añadido del FNG.
    """
    def best_f1(feature_set: str) -> float:
        return results_df[results_df["feature_set"] == feature_set]["f1"].max()

    f1_reddit = best_f1("price_plus_reddit")
    f1_fng    = best_f1("price_plus_reddit_plus_fng")
    delta     = round(f1_fng - f1_reddit, 4)

    print("\n  " + "=" * 58)
    print("  Conclusión: impacto del Fear & Greed Index en BTC")
    print("  " + "-" * 58)
    print(f"  Mejor F1 (Reddit only)      : {f1_reddit:.4f}")
    print(f"  Mejor F1 (Reddit + FNG)     : {f1_fng:.4f}")
    print(f"  Delta F1                    : {delta:+.4f}")
    print()

    if delta > 0:
        print("  >> FNG MEJORA la prediccion de BTC")
        print(f"     El Fear & Greed Index añade +{delta:.4f} F1 sobre Reddit solo.")
        print("     Conclusion: BTC responde a multiples señales de sentimiento crypto.")
    else:
        print("  >> FNG NO mejora la prediccion de BTC")
        print(f"     El Fear & Greed Index resta {delta:.4f} F1 sobre Reddit solo.")
        print("     Conclusion: el sentimiento de Reddit ya captura la informacion del FNG.")

    print("  " + "=" * 58)


# ── API pública ───────────────────────────────────────────────────────────────

def run_btc_fng_analysis(
    btc_path: Path    = BTC_FEATURES_PATH,
    fng_path: Path    = FNG_PATH,
    out_feat: Path    = OUTPUT_FEATURES_PATH,
    out_results: Path = RESULTS_PATH,
) -> pd.DataFrame:
    """
    Pipeline completo de integración FNG para BTC.

    Pasos
    -----
    1. Carga y fusiona btc_features.csv con fear_greed.csv (inner join por Date).
    2. Genera features de FNG (lags y medias móviles).
    3. Guarda btc_features_fng.csv.
    4. Entrena 3 modelos x 3 conjuntos de features.
    5. Guarda btc_fng_model_results.csv.
    6. Imprime tabla comparativa y conclusión.

    Devuelve
    --------
    pd.DataFrame con los resultados de evaluación.
    """
    # ── Carga y merge ─────────────────────────────────────────────────────────
    print(f"  BTC features  ->  {btc_path.relative_to(ROOT)}")
    print(f"  Fear & Greed  ->  {fng_path.relative_to(ROOT)}")

    merged = _load_and_merge(btc_path, fng_path)
    print(f"\n  Filas tras inner join         : {len(merged)}")

    df = _add_fng_features(merged)
    print(f"  Filas tras lags/rolling FNG   : {len(df)}")
    print(f"  Rango de fechas               : {df['Date'].iloc[0]}  a  {df['Date'].iloc[-1]}")
    print(f"  Columnas totales              : {len(df.columns)}")

    # ── Guarda dataset enriquecido ────────────────────────────────────────────
    out_feat.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_feat, index=False)
    print(f"  Guardado  ->  {out_feat.relative_to(ROOT)}")

    # ── Entrena modelos ───────────────────────────────────────────────────────
    print()
    all_results = _train_all_sets(df)

    results_df = pd.DataFrame(all_results)[RESULTS_COLS]

    # ── Conclusión ────────────────────────────────────────────────────────────
    _print_conclusion(results_df)

    # ── Guarda resultados ─────────────────────────────────────────────────────
    out_results.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_results, index=False)
    print(f"\n  Guardado  ->  {out_results.relative_to(ROOT)}")

    return results_df
