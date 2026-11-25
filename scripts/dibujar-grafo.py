"""Draw a directed transition graph from a transition matrix CSV or JSON."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "resultados" / "gráficos"


def load_matrix(path: Path, regime: str | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    preview = pd.read_csv(path, nrows=1)
    if {"regime", "from_state", "to_state", "probability"} <= set(preview.columns):
        long_matrix = pd.read_csv(path)
        selected = long_matrix[
            long_matrix["regime"] == (regime or long_matrix["regime"].iloc[0])
        ]
        return selected.pivot(
            index="from_state", columns="to_state", values="probability"
        )
    return pd.read_csv(path, index_col=0)


def draw_graph(
    matrix: pd.DataFrame,
    instant: str,
    output_dir: Path = OUTPUT_DIR,
    title: str | None = None,
) -> Path:
    matrix = matrix.apply(pd.to_numeric)
    states = ["Low", "Moderate", "High", "Severe"]
    matrix = matrix.reindex(index=states, columns=states, fill_value=0)
    edges = {}
    for source in matrix.index:
        for target in matrix.columns:
            probability = float(matrix.loc[source, target])
            if probability >= 0.001:
                edges[(str(source), str(target))] = probability
    for source, target in [("Low", "Moderate"), ("Moderate", "Low")]:
        if (source, target) in edges:
            print(f"{source} -> {target}: {edges[(source, target)]:.3f}")

    positions = {
        "Low": (-1.5, 1.5),
        "Moderate": (1.5, 1.5),
        "High": (1.5, -1.5),
        "Severe": (-1.5, -1.5),
    }
    figure, axis = plt.subplots(figsize=(11, 10))
    for state, (x, y) in positions.items():
        axis.scatter(x, y, s=3100, color="#d9eaf7", edgecolor="#38556e", linewidth=1.5, zorder=3)
        axis.text(x, y, state, ha="center", va="center", fontsize=9.5, fontweight="bold", zorder=4)

    node_radius = 0.37
    for (source, target), probability in edges.items():
        if source == target:
            continue
        start = np.asarray(positions[source], dtype=float)
        end = np.asarray(positions[target], dtype=float)
        direction = end - start
        unit = direction / np.linalg.norm(direction)
        p0 = start + unit * node_radius
        p1 = end - unit * node_radius
        right_normal = np.array((unit[1], -unit[0]))
        control = (p0 + p1) / 2 + right_normal * np.linalg.norm(p1 - p0) * 0.20
        t_values = np.linspace(0, 1, 80)
        curve = (
            (1 - t_values[:, None]) ** 2 * p0
            + 2 * (1 - t_values[:, None]) * t_values[:, None] * control
            + t_values[:, None] ** 2 * p1
        )
        axis.plot(curve[:, 0], curve[:, 1], color="#3f4f5f", linewidth=1.4, zorder=1)
        curve_at_95 = (
            0.05**2 * p0 + 2 * 0.05 * 0.95 * control + 0.95**2 * p1
        )
        axis.annotate(
            "",
            xy=p1,
            xytext=curve_at_95,
            arrowprops=dict(arrowstyle="-|>", color="#3f4f5f", lw=1.4, mutation_scale=16),
            zorder=2,
        )
        label_fraction = (
            0.28
            if (source, target) in {("Severe", "Moderate"), ("High", "Low")}
            else 0.50
        )
        label_point = (
            (1 - label_fraction) ** 2 * p0
            + 2 * (1 - label_fraction) * label_fraction * control
            + label_fraction**2 * p1
        )
        axis.text(
            label_point[0],
            label_point[1],
            f"{probability:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88),
            zorder=5,
        )

    for state in states:
        if (state, state) not in edges:
            continue
        point = np.asarray(positions[state], dtype=float)
        outward = point / np.linalg.norm(point)
        angle = math.degrees(math.atan2(outward[1], outward[0]))
        start_angle = math.radians(angle - 28)
        end_angle = math.radians(angle + 28)
        start = point + node_radius * np.array(
            [math.cos(start_angle), math.sin(start_angle)]
        )
        end = point + node_radius * np.array(
            [math.cos(end_angle), math.sin(end_angle)]
        )
        control_start = point + 2.4 * node_radius * np.array(
            [math.cos(start_angle), math.sin(start_angle)]
        )
        control_end = point + 2.4 * node_radius * np.array(
            [math.cos(end_angle), math.sin(end_angle)]
        )
        path = MplPath(
            [start, control_start, control_end, end],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        axis.add_patch(
            FancyArrowPatch(
                path=path,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.5,
                color="#333333",
                zorder=1,
            )
        )
        label_point = point + outward * 0.98
        axis.text(
            label_point[0], label_point[1],
            f"{edges[(state, state)]:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88),
            zorder=5,
        )
    axis.set_title(title or f"Grafo de transición: {instant}", fontsize=14, pad=18)
    axis.set_xlim(-2.55, 2.55)
    axis.set_ylim(-2.55, 2.55)
    axis.set_axis_off()
    axis.set_aspect("equal")
    figure.subplots_adjust(top=0.86, bottom=0.04, left=0.04, right=0.96)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"grafo_{instant}.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matriz", type=Path)
    parser.add_argument("--instante", required=True)
    parser.add_argument("--regime")
    parser.add_argument("--title")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    draw_graph(
        load_matrix(args.matriz, args.regime),
        args.instante,
        args.output_dir,
        args.title,
    )


if __name__ == "__main__":
    main()
