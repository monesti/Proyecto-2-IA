"""Generate methodological diagrams for the final report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "resultados" / "diagramas"

COLORS = {
    "input": "#DCEEFF",
    "process": "#E8E2FF",
    "result": "#DDF3E4",
    "audit": "#FFF0CC",
    "text": "#17324D",
    "arrow": "#42657C",
}


def box(axis, center, width, height, text, color):
    x, y = center
    axis.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor=color,
            edgecolor=COLORS["text"],
            linewidth=1.2,
        )
    )
    axis.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=COLORS["text"],
        fontsize=9,
        linespacing=1.22,
    )


def arrow(axis, start, end):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.4,
            color=COLORS["arrow"],
            shrinkA=8,
            shrinkB=8,
        )
    )


def canvas(title, xlim=(0, 13), ylim=(0, 6), figsize=(13, 5.5)):
    figure, axis = plt.subplots(figsize=figsize)
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(title, fontsize=14, color=COLORS["text"], pad=16, fontweight="bold")
    return figure, axis


def pipeline_diagram() -> None:
    figure, axis = canvas(
        "Pipeline de preparación y transiciones",
        xlim=(0, 10),
        ylim=(0, 6),
        figsize=(10, 5.2),
    )
    top = [
        ((1.55, 4.35), "Dataset Kaggle\n204.000 × 47\ncolumnas"),
        ((5.0, 4.35), "Limpieza\n20 variables\nfecha-hora"),
        ((8.45, 4.35), "trafico_limpio.csv\n204.000 × 20\n0 nulos"),
    ]
    bottom = [
        ((3.25, 1.45), "Orden temporal\n100 pares\ncamino–cruce"),
        ((6.75, 1.45), "Transiciones\nX_t → X_t+1\nΔt = 1 hora"),
    ]
    for center, text in top:
        box(axis, center, 2.35, 1.65, text, COLORS["input"])
    box(axis, bottom[0][0], 2.55, 1.65, bottom[0][1], COLORS["result"])
    box(axis, bottom[1][0], 2.55, 1.65, bottom[1][1], COLORS["process"])
    arrow(axis, (2.75, 4.35), (3.8, 4.35))
    arrow(axis, (6.2, 4.35), (7.25, 4.35))
    arrow(axis, (8.45, 3.52), (8.45, 2.55))
    arrow(axis, (8.45, 2.55), (7.35, 1.45))
    arrow(axis, (4.52, 1.45), (5.48, 1.45))
    figure.savefig(
        OUTPUT_DIR / "diagrama_1_pipeline_datos.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.18,
    )
    plt.close(figure)


def architecture_diagram() -> None:
    figure, axis = canvas(
        "Arquitectura CMTD–colas",
        xlim=(0, 13),
        ylim=(0, 6),
        figsize=(12, 5.5),
    )
    box(axis, (1.25, 3), 2.05, 1.45, "Datos procesados\ntrafico_limpio.csv", COLORS["input"])
    box(axis, (3.85, 4.35), 2.15, 1.45, "Variables\nexógenas\n(hora, clima, día)", COLORS["input"])
    box(axis, (6.35, 4.35), 2.15, 1.45, "Matrices no\nhomogéneas P^(r)\npor régimen", COLORS["process"])
    box(axis, (8.85, 4.35), 2.15, 1.45, "Prueba χ² de\nhomogeneidad\n(hora, clima, día)", COLORS["audit"])
    box(axis, (3.85, 1.65), 2.15, 1.45, "Variables\noperacionales\nλ, L, v, verde g/C", COLORS["input"])
    box(axis, (6.35, 1.65), 2.15, 1.45, "Capacidad μ_ef\ny utilización\nρ = λ / μ_ef", COLORS["process"])
    box(axis, (8.85, 1.65), 2.15, 1.45, "Espera y cola\nno lineales\nvalidación SimPy", COLORS["result"])
    box(axis, (11.55, 3), 2.15, 1.45, "Diagnóstico\noperacional\npor régimen", COLORS["result"])
    arrow(axis, (2.28, 3.25), (2.78, 4.35))
    arrow(axis, (2.28, 2.75), (2.78, 1.65))
    arrow(axis, (4.93, 4.35), (5.28, 4.35))
    arrow(axis, (7.43, 4.35), (7.78, 4.35))
    arrow(axis, (4.93, 1.65), (5.28, 1.65))
    arrow(axis, (7.43, 1.65), (7.78, 1.65))
    arrow(axis, (9.93, 4.35), (10.48, 3.25))
    arrow(axis, (9.93, 1.65), (10.48, 2.75))
    figure.savefig(
        OUTPUT_DIR / "diagrama_2_arquitectura_modelo.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.18,
    )
    plt.close(figure)


def traceability_diagram() -> None:
    figure, axis = canvas(
        "Trazabilidad y auditoría",
        xlim=(0, 10),
        ylim=(0, 6),
        figsize=(10, 5.2),
    )
    top = [
        ((1.55, 4.35), "CSV limpio\nfuente\nde datos"),
        ((5.0, 4.35), "Análisis\nmétricas\ny tablas"),
        ((8.45, 4.35), "dibujar-grafo.py\ngrafos de\ntransición"),
    ]
    bottom = [
        ((3.25, 1.45), "recalcular_valores.py\nauditoría JSON"),
        ((6.75, 1.45), "revisar_fuentes.py\nInforme LaTeX"),
    ]
    for center, text in top:
        box(axis, center, 2.35, 1.65, text, COLORS["process"])
    box(axis, bottom[0][0], 2.55, 1.65, "recalcular_\nvalores.py\nauditoría JSON", COLORS["audit"])
    box(axis, bottom[1][0], 2.55, 1.65, bottom[1][1], COLORS["result"])
    arrow(axis, (2.72, 4.35), (3.82, 4.35))
    arrow(axis, (6.18, 4.35), (7.28, 4.35))
    arrow(axis, (8.45, 3.52), (8.45, 2.55))
    arrow(axis, (8.45, 2.55), (7.35, 1.45))
    arrow(axis, (4.52, 1.45), (5.48, 1.45))
    figure.savefig(
        OUTPUT_DIR / "diagrama_3_trazabilidad_auditoria.png",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.18,
    )
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_diagram()
    architecture_diagram()
    traceability_diagram()
    print("diagramas_generados=3")


if __name__ == "__main__":
    main()
