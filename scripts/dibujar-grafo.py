"""Draw a directed transition graph from a transition matrix CSV or JSON."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "resultados" / "gráficos"


def load_matrix(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return pd.read_csv(path, index_col=0)


def draw_graph(matrix: pd.DataFrame, instant: str, output_dir: Path = OUTPUT_DIR) -> Path:
    matrix = matrix.apply(pd.to_numeric)
    graph = nx.DiGraph()
    graph.add_nodes_from(matrix.index.astype(str))
    for source in matrix.index:
        for target in matrix.columns:
            probability = float(matrix.loc[source, target])
            if probability > 0:
                graph.add_edge(str(source), str(target), probability=probability)

    is_planar, _ = nx.check_planarity(graph)
    positions = nx.planar_layout(graph) if is_planar else nx.spring_layout(
        graph, seed=42
    )
    figure, axis = plt.subplots(figsize=(10, 8))
    nx.draw_networkx(
        graph,
        positions,
        ax=axis,
        node_color="#d9eaf7",
        node_size=1800,
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.12",
    )
    labels = {
        edge: f"{details['probability']:.3f}"
        for edge, details in graph.edges.items()
    }
    nx.draw_networkx_edge_labels(
        graph, positions, edge_labels=labels, ax=axis, label_pos=0.5
    )
    axis.set_axis_off()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"grafo_{instant}.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matriz", type=Path)
    parser.add_argument("--instante", required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    draw_graph(load_matrix(args.matriz), args.instante, args.output_dir)


if __name__ == "__main__":
    main()
