# -*- coding: utf-8 -*-
"""
Genera las tres figuras para la memoria del TFG.
Ejecutar desde la raiz del proyecto:
    python reports/figures/memoria/_gen_memoria_figures.py
"""

from __future__ import annotations
from pathlib import Path
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np

OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

WHITE   = "#ffffff"
C_BLUE  = "#2c5f8a"
C_LBLUE = "#4a90c4"
C_GRAY  = "#6c757d"
C_LGRAY = "#e9ecef"
C_GREEN = "#2e7d52"
C_LGREEN= "#52b788"
C_ORANGE= "#b5460f"
C_LORANG= "#e07b39"
C_PURP  = "#5a3f7a"
C_LPURP = "#9b72cf"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "figure.facecolor": WHITE,
    "axes.facecolor":   WHITE,
})


# =============================================================================
# FIGURA 1 - Cronograma Gantt
# =============================================================================

def figura_1_gantt():
    phases = [
        ("1. Revisión bibliográfica\n    y diseño metodológico",
         datetime.date(2026, 2,  1), datetime.date(2026, 2, 28), C_BLUE),
        ("2. Recopilación y\n    preparación de datos",
         datetime.date(2026, 2, 15), datetime.date(2026, 3, 31), C_GREEN),
        ("3. Análisis de sentimiento\n    (VADER y FinBERT)",
         datetime.date(2026, 3,  1), datetime.date(2026, 3, 31), C_LBLUE),
        ("4. Ingeniería de características\n    y modelado ML",
         datetime.date(2026, 4,  1), datetime.date(2026, 4, 30), C_ORANGE),
        ("5. Análisis experimentales\n    complementarios",
         datetime.date(2026, 4, 15), datetime.date(2026, 5, 10), C_PURP),
        ("6. Dashboard interactivo\n    (Streamlit)",
         datetime.date(2026, 5,  1), datetime.date(2026, 5, 18), C_LGREEN),
        ("7. Redacción de la memoria\n    y defensa",
         datetime.date(2026, 4, 15), datetime.date(2026, 6, 19), C_LORANG),
    ]

    fig, ax = plt.subplots(figsize=(13, 6.0))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor("#f8f9fa")

    y_pos = list(range(len(phases) - 1, -1, -1))

    for i, (label, start, end, color) in enumerate(phases):
        y = y_pos[i]
        s = mdates.date2num(start)
        e = mdates.date2num(end)
        ax.barh(y, e - s, left=s, height=0.55,
                color=color, alpha=0.85, edgecolor="white", linewidth=1.2)
        dur = (end - start).days
        ax.text(s + (e - s) / 2, y, f"{dur}d",
                ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

    # Hitos — etiquetas dentro del ylim, justo debajo del limite superior
    milestones = [
        (datetime.date(2026, 5,  4), "Entrega\nintermedia", C_ORANGE),
        (datetime.date(2026, 5, 18), "Entrega\nfinal",      C_BLUE),
        (datetime.date(2026, 6, 19), "Defensa",             C_PURP),
    ]

    # ylim superior: dejar margen para las etiquetas de hito
    TOP_Y = len(phases) + 1.0   # = 8.0  (ylim va hasta 8.0)

    for mdate, mlabel, mcolor in milestones:
        mx = mdates.date2num(mdate)
        ax.axvline(mx, color=mcolor, linewidth=1.4, linestyle="--", alpha=0.75,
                   ymin=0, ymax=(len(phases) - 0.5) / TOP_Y)
        ax.text(mx, len(phases) - 0.3, mlabel,
                ha="center", va="bottom",
                fontsize=7.5, color=mcolor, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=WHITE, ec=mcolor,
                          lw=0.8, alpha=0.9))

    ax.set_yticks(y_pos)
    ax.set_yticklabels([p[0] for p in phases], fontsize=9)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8.5)

    ax.set_xlim(
        mdates.date2num(datetime.date(2026, 1, 26)),
        mdates.date2num(datetime.date(2026, 6, 28)),
    )
    ax.set_ylim(-0.7, TOP_Y)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, color="#cccccc", alpha=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.set_title("Cronograma general del proyecto",
                 fontsize=13, fontweight="bold", pad=14)

    fig.tight_layout()
    out = OUT / "figura_1_cronograma_tfg.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"  Guardada -> {out}")


# =============================================================================
# FIGURA 2 - Pipeline experimental
# =============================================================================

def _box2(ax, x, y, w, h, text, fc, tc="white", fs=8.5, bold=False):
    patch = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        linewidth=0.8, edgecolor="#999999",
        facecolor=fc, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y, text,
            ha="center", va="center",
            fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal",
            multialignment="center", linespacing=1.35, zorder=4)


def _arr(ax, x0, y0, x1, y1, color="#666666"):
    ax.annotate("",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=1.1, mutation_scale=11),
        zorder=2,
    )


def figura_2_pipeline():
    fig, ax = plt.subplots(figsize=(14, 9.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10.5)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    # Cabeceras de columna
    ax.text(3.5, 10.2, "FUENTE FINANCIERA",
            ha="center", fontsize=8.5, color=C_BLUE, fontweight="bold")
    ax.text(10.5, 10.2, "FUENTE TEXTUAL (Reddit WSB)",
            ha="center", fontsize=8.5, color=C_GREEN, fontweight="bold")

    # Inputs
    _box2(ax, 3.5,  9.55, 3.2, 0.7,
          "Datos financieros\nBTC-USD · S&P 500 (^GSPC)",
          C_BLUE, fs=8.5)
    _box2(ax, 10.5, 9.55, 3.4, 0.7,
          "Reddit r/WallStreetBets\n2.275.310 publicaciones (2012-2023)",
          C_GREEN, fs=8.5)

    # Preprocesamiento
    _box2(ax, 3.5,  8.35, 3.1, 0.7,
          "Preprocesamiento financiero\n(yfinance · retornos · dirección)",
          C_LBLUE, fs=8)
    _box2(ax, 10.5, 8.35, 3.1, 0.7,
          "Preprocesamiento textual\n(limpieza · normalización)",
          C_LGREEN, fs=8)

    _arr(ax, 3.5, 9.19, 3.5, 8.71)
    _arr(ax, 10.5, 9.19, 10.5, 8.71)

    # NLP
    _box2(ax, 9.1,  7.1, 2.6, 0.7,
          "VADER\n(reglas, 2,27 M publicaciones)",
          C_BLUE, fs=8)
    _box2(ax, 12.0, 7.1, 2.6, 0.7,
          "FinBERT\n(transformer, 100.000 publicaciones)",
          C_PURP, fs=8)

    _arr(ax, 10.5, 7.99, 9.5,  7.46)
    _arr(ax, 10.5, 7.99, 11.5, 7.46)

    # Agregacion diaria
    _box2(ax, 10.55, 6.05, 3.2, 0.65,
          "Agregación diaria de sentimiento\n(media ponderada por número de publicaciones)",
          C_GRAY, fs=8)

    _arr(ax, 9.1,  6.74, 9.9,  6.38)
    _arr(ax, 12.0, 6.74, 11.2, 6.38)

    # Integracion temporal
    _box2(ax, 7.0, 5.05, 3.2, 0.7,
          "Integración temporal\n(merge por fecha - Date)",
          "#8B4513", fs=8.5, bold=True)

    # flecha financiera baja hasta merge
    _arr(ax, 3.5, 7.99, 3.5, 5.05)
    ax.annotate("", xy=(5.4, 5.05), xytext=(3.5, 5.05),
        arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.1, mutation_scale=11))

    # flecha sentimiento baja y va a la izquierda
    ax.annotate("", xy=(10.55, 5.40), xytext=(10.55, 5.72),
        arrowprops=dict(arrowstyle="-", color="#666666", lw=1.1))
    ax.annotate("", xy=(8.6, 5.05), xytext=(10.55, 5.05),
        arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.1, mutation_scale=11))

    # Ingenieria de caracteristicas
    _box2(ax, 7.0, 4.0, 3.3, 0.7,
          "Ingeniería de características\n(lags · rolling · volatilidad · log(posts))",
          C_ORANGE, fs=8)
    _arr(ax, 7.0, 4.7, 7.0, 4.36)

    # Modelos ML
    _box2(ax, 4.1, 2.9, 2.3, 0.65, "Logistic\nRegression", C_LBLUE, fs=8.5)
    _box2(ax, 7.0, 2.9, 2.3, 0.65, "Random\nForest",       C_GREEN,  fs=8.5)
    _box2(ax, 9.9, 2.9, 2.3, 0.65, "XGBoost",              C_ORANGE, fs=8.5, bold=True)

    _arr(ax, 5.8, 4.0, 4.7, 3.23)
    _arr(ax, 7.0, 3.65, 7.0, 3.23)
    _arr(ax, 8.2, 4.0, 9.3, 3.23)

    # Evaluacion
    _box2(ax, 7.0, 1.85, 7.0, 0.68,
          "Evaluación · Ablación · Multi-horizonte (3d/5d/7d) · Granger",
          C_PURP, fs=8.5)

    _arr(ax, 4.1, 2.57, 4.8, 2.19)
    _arr(ax, 7.0, 2.57, 7.0, 2.19)
    _arr(ax, 9.9, 2.57, 9.2, 2.19)

    # Outputs
    _box2(ax, 3.8, 0.75, 2.8, 0.62,
          "reports/tables\ntablas CSV de resultados", C_GRAY, fs=8)
    _box2(ax, 7.0, 0.75, 2.8, 0.62,
          "reports/figures\nfiguras y gráficos",      C_GRAY, fs=8)
    _box2(ax, 10.2, 0.75, 2.8, 0.62,
          "Dashboard\nStreamlit (app.py)",            C_BLUE, fs=8)

    _arr(ax, 5.5, 1.51, 4.4, 1.06)
    _arr(ax, 7.0, 1.51, 7.0, 1.06)
    _arr(ax, 8.5, 1.51, 9.6, 1.06)
    _arr(ax, 8.5, 1.51, 10.2, 1.06)

    ax.set_title("Arquitectura general del pipeline experimental",
                 fontsize=13, fontweight="bold", pad=6)
    fig.tight_layout()

    out = OUT / "figura_2_pipeline_experimental.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"  Guardada -> {out}")


# =============================================================================
# FIGURA 3 - Estructura del repositorio (bloques)
# =============================================================================

def figura_3_estructura():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    def draw_block(x0, y0, x1, y1, title, items,
                   hdr_color, content_color="#f4f6f9", item_fs=10):
        bw = x1 - x0
        bh = y1 - y0
        hdr_h = 0.62

        # Cabecera
        ax.add_patch(plt.Rectangle(
            (x0, y1 - hdr_h), bw, hdr_h,
            facecolor=hdr_color, edgecolor="none", zorder=3, clip_on=False))
        ax.text((x0 + x1) / 2, y1 - hdr_h / 2, title,
                ha="center", va="center", fontsize=11.5,
                color="white", fontweight="bold", zorder=4)

        # Fondo contenido
        ax.add_patch(plt.Rectangle(
            (x0, y0), bw, bh - hdr_h,
            facecolor=content_color, edgecolor="none", zorder=2))

        # Borde exterior
        ax.add_patch(plt.Rectangle(
            (x0, y0), bw, bh,
            fill=False, edgecolor=hdr_color, lw=2.2, zorder=5))

        # Items centrados en el area de contenido
        n = len(items)
        content_h = bh - hdr_h
        for i, item in enumerate(items):
            iy = y1 - hdr_h - content_h * (i + 0.5) / n
            ax.text((x0 + x1) / 2, iy, item,
                    ha="center", va="center",
                    fontsize=item_fs, color="#1a1a2e",
                    fontfamily="monospace", zorder=4)

    ax.set_title("Estructura modular del repositorio",
                 fontsize=14, fontweight="bold", pad=14)

    PAD = 0.25   # margen entre bloques

    # Fila superior
    # Bloque 1: Archivos principales
    draw_block(PAD, 5.1, 6.4, 8.7,
               "Archivos principales",
               ["main.py", "app.py", "requirements.txt", ".gitignore"],
               C_BLUE, item_fs=11)

    # Bloque 2: data/
    draw_block(6.6, 5.1, 12.75, 8.7,
               "data/",
               ["raw/", "processed/"],
               C_GREEN, item_fs=11)

    # Fila inferior
    # Bloque 3: src/
    draw_block(PAD, 0.3, 8.6, 4.85,
               "src/",
               ["src/data/", "src/nlp/", "src/features/",
                "src/models/", "src/analysis/", "src/visualization/"],
               C_ORANGE, item_fs=10.5)

    # Bloque 4: reports/
    draw_block(8.85, 0.3, 12.75, 4.85,
               "reports/",
               ["tables/", "figures/"],
               C_PURP, item_fs=11)

    fig.tight_layout()
    out = OUT / "figura_3_estructura_repositorio.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"  Guardada -> {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generando figuras para la memoria...")
    figura_1_gantt()
    figura_2_pipeline()
    figura_3_estructura()
    print("Listo.")
