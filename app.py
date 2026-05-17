"""
app.py  —  Streamlit dashboard for the sentiment-finance pipeline.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Layout ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment & Finance TFG",
    page_icon="📈",
    layout="wide",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
PROC_DIR  = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "reports" / "tables"

PATHS: dict[str, Path] = {
    "btc_price":          PROC_DIR  / "btc_processed.csv",
    "sp500_price":        PROC_DIR  / "sp500_processed.csv",
    "vader_daily":        PROC_DIR  / "reddit_wsb_vader_daily.csv",
    "model_results":      TABLE_DIR / "model_results.csv",
    "ablation_results":   TABLE_DIR / "ablation_results.csv",
    "ablation_summary":   TABLE_DIR / "ablation_summary.csv",
    "windowed_results":   TABLE_DIR / "windowed_model_results.csv",
    "windowed_summary":   TABLE_DIR / "windowed_ablation_summary.csv",
    "vader_finbert_cmp":  TABLE_DIR / "vader_finbert_comparison.csv",
    "vader_finbert_sum":  TABLE_DIR / "vader_finbert_summary.csv",
    "granger":            TABLE_DIR / "granger_results.csv",
    "feature_imp":        TABLE_DIR / "feature_importance.csv",
    "feature_imp_sent":   TABLE_DIR / "feature_importance_sentiment.csv",
    "statistical_tests":  TABLE_DIR / "statistical_tests.csv",
}

ASSET_COLORS: dict[str, str] = {
    "BTC":   "#f7931a",
    "SP500": "#1f77b4",
}


# ── Data loader ────────────────────────────────────────────────────────────────

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"Date": str})


# ── Sidebar navigation ─────────────────────────────────────────────────────────

SECTIONS = [
    "Introducción",
    "Evolución del precio",
    "Sentimiento VADER",
    "Resultados de modelos",
    "Ablación (precio vs sentimiento)",
    "Análisis multi-horizonte",
    "VADER vs FinBERT",
    "Causalidad de Granger",
    "Importancia de features",
    "Conclusión",
]

with st.sidebar:
    st.title("TFG · Sentimiento & Finanzas")
    st.markdown("---")
    section = st.radio("Sección", SECTIONS)
    st.markdown("---")
    st.caption("Universidad Europea de Madrid")


# ── Helper ─────────────────────────────────────────────────────────────────────

def _missing(key: str) -> None:
    st.warning(
        f"Archivo no encontrado: `{PATHS[key].relative_to(ROOT)}`  \n"
        "Ejecuta el pipeline completo (`python main.py`) para generar este archivo."
    )


# ── Sections ───────────────────────────────────────────────────────────────────

def section_intro() -> None:
    st.title("Análisis comparativo del sentimiento financiero")
    st.subheader("Bitcoin y S&P 500 mediante NLP y Machine Learning")
    st.markdown("""
Este dashboard visualiza los resultados del pipeline completo del TFG:

| Paso | Descripción |
|------|-------------|
| 1  | Descarga de precios históricos (yfinance) |
| 3–4 | Preprocesamiento de Reddit WSB + sentimiento VADER |
| 4b–4d | Sentimiento FinBERT + comparación VADER vs FinBERT |
| 5–6 | Fusión y feature engineering |
| 7  | Entrenamiento de modelos (LR, RF, XGBoost) |
| 10 | Análisis estadístico (correlaciones + baseline) |
| 11 | Ablación: precio vs precio+sentimiento |
| 12 | Importancia de features (Random Forest) |
| 13 | Fear & Greed Index (BTC) |
| 14–15 | Objetivos multi-horizonte (3d, 5d, 7d) |
| 16 | Causalidad de Granger |

Navega entre secciones usando el menú lateral.
""")

    col1, col2 = st.columns(2)
    with col1:
        df_btc = load_csv(PATHS["btc_price"])
        if df_btc is not None:
            st.metric("Filas BTC", f"{len(df_btc):,}")
    with col2:
        df_sp = load_csv(PATHS["sp500_price"])
        if df_sp is not None:
            st.metric("Filas S&P500", f"{len(df_sp):,}")


def section_price() -> None:
    st.header("Evolución del precio")

    for key, asset, color in [
        ("btc_price",   "BTC",   ASSET_COLORS["BTC"]),
        ("sp500_price", "SP500", ASSET_COLORS["SP500"]),
    ]:
        df = load_csv(PATHS[key])
        if df is None:
            _missing(key)
            continue

        st.subheader(asset)
        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", f"{len(df):,}")
        col2.metric("Inicio", df["Date"].min())
        col3.metric("Fin",    df["Date"].max())

        fig = px.line(
            df, x="Date", y="Close",
            title=f"{asset} — Precio de cierre",
            color_discrete_sequence=[color],
            labels={"Close": "Precio (USD)", "Date": "Fecha"},
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        if "Return" in df.columns:
            fig2 = px.bar(
                df, x="Date", y="Return",
                title=f"{asset} — Retorno diario",
                color_discrete_sequence=[color],
                labels={"Return": "Retorno", "Date": "Fecha"},
            )
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)


def section_sentiment() -> None:
    st.header("Sentimiento VADER diario (Reddit WSB)")

    df = load_csv(PATHS["vader_daily"])
    if df is None:
        _missing("vader_daily")
        return

    st.write(f"**{len(df):,} días** · {df['Date'].min()} → {df['Date'].max()}")

    if "vader_compound" in df.columns:
        fig = px.line(
            df, x="Date", y="vader_compound",
            title="Sentimiento compuesto VADER (media diaria)",
            labels={"vader_compound": "Compound", "Date": "Fecha"},
            color_discrete_sequence=["#2ca02c"],
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    cols_sent = [c for c in ["vader_pos", "vader_neu", "vader_neg"] if c in df.columns]
    if cols_sent:
        fig2 = px.area(
            df.melt(id_vars="Date", value_vars=cols_sent,
                    var_name="Componente", value_name="Score"),
            x="Date", y="Score", color="Componente",
            title="Desglose de sentimiento VADER (pos / neu / neg)",
            labels={"Date": "Fecha"},
        )
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)

    if "n_posts" in df.columns:
        fig3 = px.bar(
            df, x="Date", y="n_posts",
            title="Número de posts por día",
            labels={"n_posts": "Posts", "Date": "Fecha"},
            color_discrete_sequence=["#9467bd"],
        )
        fig3.update_xaxes(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Vista de tabla")
    st.dataframe(df.head(20), use_container_width=True)


def section_models() -> None:
    st.header("Resultados de modelos")

    df = load_csv(PATHS["model_results"])
    if df is None:
        _missing("model_results")
        return

    st.write(f"**{len(df)} combinaciones** (activo × modelo × conjunto de features)")

    # Best model per asset
    best = (
        df.sort_values(["f1", "accuracy"], ascending=False)
        .groupby("asset", as_index=False)
        .first()
    )
    st.subheader("Mejor modelo por activo")
    st.dataframe(best, use_container_width=True)

    # F1 bar chart
    fig = px.bar(
        df,
        x="model", y="f1", color="asset",
        barmode="group",
        facet_col="feature_set" if "feature_set" in df.columns else None,
        title="F1-score por modelo y activo",
        color_discrete_map=ASSET_COLORS,
        labels={"f1": "F1", "model": "Modelo"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy / Precision / Recall / F1 per asset
    for asset in df["asset"].unique():
        sub = df[df["asset"] == asset]
        st.subheader(f"Métricas — {asset}")
        metrics_cols = [c for c in ["accuracy", "precision", "recall", "f1"] if c in sub.columns]
        if metrics_cols:
            fig2 = px.bar(
                sub.melt(id_vars=["model"], value_vars=metrics_cols,
                         var_name="Métrica", value_name="Valor"),
                x="model", y="Valor", color="Métrica", barmode="group",
                title=f"{asset} — Métricas por modelo",
                labels={"model": "Modelo"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Tabla completa")
    st.dataframe(df, use_container_width=True)

    # Statistical tests
    df_stats = load_csv(PATHS["statistical_tests"])
    if df_stats is not None:
        st.subheader("Tests estadísticos")
        st.dataframe(df_stats, use_container_width=True)


def section_ablation() -> None:
    st.header("Ablación: precio solo vs precio + sentimiento")

    df_sum = load_csv(PATHS["ablation_summary"])
    df_res = load_csv(PATHS["ablation_results"])

    if df_sum is None:
        _missing("ablation_summary")
    else:
        st.subheader("Resumen (delta F1)")
        st.dataframe(df_sum, use_container_width=True)

        if "f1_delta" in df_sum.columns:
            fig = px.bar(
                df_sum, x="model", y="f1_delta",
                color="asset", barmode="group",
                color_discrete_map=ASSET_COLORS,
                title="Delta F1 (precio+sentimiento − precio_solo) por modelo",
                labels={"f1_delta": "ΔF1", "model": "Modelo"},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

    if df_res is None:
        _missing("ablation_results")
    else:
        st.subheader("Resultados detallados")
        fig2 = px.bar(
            df_res, x="model", y="f1",
            color="feature_set", barmode="group",
            facet_col="asset",
            title="F1 por modelo, conjunto de features y activo",
            labels={"f1": "F1", "model": "Modelo", "feature_set": "Features"},
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(df_res, use_container_width=True)


def section_windowed() -> None:
    st.header("Análisis multi-horizonte (3d / 5d / 7d)")

    df_sum = load_csv(PATHS["windowed_summary"])
    df_res = load_csv(PATHS["windowed_results"])

    if df_sum is None:
        _missing("windowed_summary")
    else:
        st.subheader("Resumen ablación por horizonte")
        st.dataframe(df_sum, use_container_width=True)

        if {"f1_delta", "target", "model"}.issubset(df_sum.columns):
            fig = px.bar(
                df_sum, x="model", y="f1_delta",
                color="target", barmode="group",
                facet_col="asset" if "asset" in df_sum.columns else None,
                title="Delta F1 por horizonte temporal y modelo",
                labels={"f1_delta": "ΔF1", "model": "Modelo", "target": "Horizonte"},
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

    if df_res is None:
        _missing("windowed_results")
    else:
        st.subheader("Resultados completos")
        fig2 = px.scatter(
            df_res,
            x="model", y="f1",
            color="target",
            symbol="asset" if "asset" in df_res.columns else None,
            facet_col="feature_set" if "feature_set" in df_res.columns else None,
            title="F1 por modelo, horizonte y conjunto de features",
            labels={"f1": "F1", "model": "Modelo", "target": "Horizonte"},
            size_max=12,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(df_res, use_container_width=True)


def section_vader_finbert() -> None:
    st.header("VADER vs FinBERT")

    df_sum = load_csv(PATHS["vader_finbert_sum"])
    df_cmp = load_csv(PATHS["vader_finbert_cmp"])

    if df_sum is None:
        _missing("vader_finbert_sum")
    else:
        st.subheader("Resumen: ¿FinBERT supera a VADER?")
        st.dataframe(df_sum, use_container_width=True)

        if {"f1_vader", "f1_finbert", "model"}.issubset(df_sum.columns):
            df_melt = df_sum.melt(
                id_vars=["asset", "model"] if "asset" in df_sum.columns else ["model"],
                value_vars=["f1_vader", "f1_finbert"],
                var_name="Fuente", value_name="F1",
            )
            fig = px.bar(
                df_melt, x="model", y="F1", color="Fuente", barmode="group",
                facet_col="asset" if "asset" in df_melt.columns else None,
                title="F1 — VADER vs FinBERT por modelo",
                labels={"F1": "F1", "model": "Modelo"},
                color_discrete_map={
                    "f1_vader":   "#1f77b4",
                    "f1_finbert": "#ff7f0e",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        if "f1_delta_finbert_minus_vader" in df_sum.columns:
            fig2 = px.bar(
                df_sum, x="model", y="f1_delta_finbert_minus_vader",
                color="asset" if "asset" in df_sum.columns else None,
                barmode="group",
                color_discrete_map=ASSET_COLORS,
                title="Delta F1 FinBERT − VADER (>0 = FinBERT mejor)",
                labels={
                    "f1_delta_finbert_minus_vader": "ΔF1 (FinBERT−VADER)",
                    "model": "Modelo",
                },
            )
            fig2.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig2, use_container_width=True)

    if df_cmp is None:
        _missing("vader_finbert_cmp")
    else:
        st.subheader("Comparación detallada")
        st.dataframe(df_cmp, use_container_width=True)


def section_granger() -> None:
    st.header("Causalidad de Granger")

    df = load_csv(PATHS["granger"])
    if df is None:
        _missing("granger")
        return

    st.write(
        "El test de Granger evalúa si los valores pasados de la variable X ayudan "
        "a predecir Y (más allá de los propios valores pasados de Y).  \n"
        "**Hipótesis nula**: X no Granger-causa Y."
    )

    if "significant" in df.columns:
        sig_count = df["significant"].sum()
        st.metric("Combinaciones significativas (p < 0.05)", f"{sig_count} / {len(df)}")

    if {"direction", "lag", "p_value", "asset"}.issubset(df.columns):
        for asset in df["asset"].unique():
            sub = df[df["asset"] == asset]
            st.subheader(asset)
            fig = px.scatter(
                sub, x="lag", y="p_value",
                color="direction", symbol="significant" if "significant" in sub.columns else None,
                title=f"{asset} — p-valor por lag y dirección",
                labels={"p_value": "p-valor", "lag": "Lag (días)", "direction": "Dirección"},
            )
            fig.add_hline(y=0.05, line_dash="dash", line_color="red",
                          annotation_text="α = 0.05")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tabla de resultados")
    st.dataframe(df, use_container_width=True)


def section_feature_importance() -> None:
    st.header("Importancia de features (Random Forest)")

    df = load_csv(PATHS["feature_imp"])
    df_sent = load_csv(PATHS["feature_imp_sent"])

    if df is None:
        _missing("feature_imp")
    else:
        st.subheader("Top features por activo")
        for asset in df["asset"].unique() if "asset" in df.columns else []:
            sub = df[df["asset"] == asset].head(15)
            fig = px.bar(
                sub, x="importance", y="feature",
                orientation="h",
                title=f"{asset} — Top {len(sub)} features",
                color="importance",
                color_continuous_scale="Blues",
                labels={"importance": "Importancia (Gini)", "feature": "Feature"},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tabla completa")
        st.dataframe(df, use_container_width=True)

    if df_sent is None:
        _missing("feature_imp_sent")
    else:
        st.subheader("Features de sentimiento (filtradas)")
        if "asset" in df_sent.columns:
            fig2 = px.bar(
                df_sent, x="importance", y="feature",
                color="asset", orientation="h",
                barmode="group",
                color_discrete_map=ASSET_COLORS,
                title="Importancia de features de sentimiento por activo",
                labels={"importance": "Importancia", "feature": "Feature"},
            )
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(df_sent, use_container_width=True)


def section_conclusion() -> None:
    st.header("Conclusión")

    st.markdown("""
### Hallazgos principales

**1. Resultados de los modelos base**
El mejor resultado global se obtiene en Bitcoin con XGBoost, con una accuracy de 0,5508 y un F1-score
de 0,5691. Este resultado supera moderadamente el baseline de mayoría de clase en Bitcoin. En cambio,
en el S&P 500 ningún modelo supera el baseline en accuracy, por lo que la capacidad predictiva es más
limitada bajo la configuración experimental utilizada.

**2. Contribución del sentimiento**
El análisis de ablación muestra que la incorporación de variables de sentimiento no mejora todos los
modelos de forma uniforme. La mejora aparece principalmente en XGBoost, tanto para Bitcoin como para
el S&P 500. En Regresión Logística y Random Forest, el sentimiento no aporta una mejora clara en el
horizonte diario.

**3. Análisis multi-horizonte**
En Bitcoin, el sentimiento mejora 8 de las 9 combinaciones analizadas y todos los modelos mejoran en
los horizontes de 5 y 7 días. En el S&P 500, en cambio, la incorporación de sentimiento no mejora
ninguna de las 9 combinaciones multi-horizonte. Esto sugiere que Bitcoin puede ser más sensible a
señales sociales agregadas que un índice bursátil amplio y más institucionalizado.

**4. VADER vs FinBERT**
FinBERT mejora el rendimiento de Regresión Logística y Random Forest en ambos activos, pero XGBoost
obtiene mejores resultados con VADER. Por tanto, no puede afirmarse que FinBERT supere de forma general
a VADER. El mejor resultado global en Bitcoin sigue correspondiendo a VADER con XGBoost.

**5. Causalidad de Granger**
El test de causalidad de Granger no muestra evidencia significativa de que el sentimiento preceda a
los retornos futuros. En cambio, los retornos sí preceden al sentimiento en varios rezagos. Esto sugiere
que el sentimiento de Reddit funciona más como una reacción al mercado que como una señal anticipatoria clara.

---
### Limitaciones

- WallStreetBets representa una comunidad concreta de inversores minoristas y no al conjunto del mercado.
- El sentimiento se agrega a nivel diario, por lo que no captura dinámicas intradía.
- FinBERT se aplica sobre una muestra aleatoria reproducible de 100.000 publicaciones debido a su coste computacional.
- El análisis se limita a Bitcoin y al S&P 500.
- Los resultados corresponden al periodo analizado y no deben generalizarse automáticamente a otros activos
  o contextos de mercado.

---
### Trabajo futuro

- Ampliar el análisis a más fuentes textuales, como otros subreddits, noticias financieras o Google Trends.
- Construir señales de sentimiento específicas por activo.
- Aplicar FinBERT al corpus completo o usar un muestreo estratificado por fecha.
- Incorporar validación walk-forward y modelos temporales más avanzados.
- Ampliar la comparación a otras criptomonedas, acciones individuales o índices sectoriales.
""")

    # Quick stats panel
    st.subheader("Resumen numérico rápido")
    cols = st.columns(4)

    df_res = load_csv(PATHS["model_results"])
    if df_res is not None and "f1" in df_res.columns:
        cols[0].metric("F1 máximo", f"{df_res['f1'].max():.4f}")
        cols[1].metric("F1 medio",  f"{df_res['f1'].mean():.4f}")

    df_abl = load_csv(PATHS["ablation_summary"])
    if df_abl is not None and "f1_delta" in df_abl.columns:
        cols[2].metric("ΔF1 medio (ablación)", f"{df_abl['f1_delta'].mean():+.4f}")

    df_gr = load_csv(PATHS["granger"])
    if df_gr is not None and "significant" in df_gr.columns:
        n_sig = int(df_gr["significant"].sum())
        cols[3].metric("Tests Granger significativos", f"{n_sig} / {len(df_gr)}")


# ── Router ─────────────────────────────────────────────────────────────────────

_DISPATCH = {
    "Introducción":                   section_intro,
    "Evolución del precio":           section_price,
    "Sentimiento VADER":              section_sentiment,
    "Resultados de modelos":          section_models,
    "Ablación (precio vs sentimiento)": section_ablation,
    "Análisis multi-horizonte":       section_windowed,
    "VADER vs FinBERT":               section_vader_finbert,
    "Causalidad de Granger":          section_granger,
    "Importancia de features":        section_feature_importance,
    "Conclusión":                     section_conclusion,
}

_DISPATCH[section]()
