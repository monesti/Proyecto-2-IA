"""Recalculate report claims from the processed traffic CSV.

The script accepts a claim type and prints a compact JSON result, allowing the
source-audit workflow to compare report values without exposing the dataset.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_DATASET = (
    Path(__file__).resolve().parents[1] / "datos_procesados" / "trafico_limpio.csv"
)
DEFAULT_ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "resultados"


def recalculate(data: pd.DataFrame, claim: str) -> object:
    if claim == "shape":
        return {"filas": int(data.shape[0]), "columnas": int(data.shape[1])}
    if claim == "null_counts":
        return {key: int(value) for key, value in data.isna().sum().items()}
    if claim == "unique_values":
        return {
            column: sorted(data[column].dropna().unique().tolist(), key=str)
            for column in (
                "congestion_level",
                "peak_period",
                "weather_condition",
            )
        }
    if claim == "road_intersection_pairs":
        return int(data[["road_id", "intersection_id"]].drop_duplicates().shape[0])
    raise ValueError(f"Unsupported claim type: {claim}")


def analysis_claim(analysis_dir: Path, claim: str, regime: str | None = None) -> object:
    tables = analysis_dir / "tablas"
    if claim == "transition_matrix":
        matrix = pd.read_csv(tables / "tabla_2_matrices_transicion.csv")
        if regime is not None:
            matrix = matrix[matrix["regime"] == regime]
        return matrix.pivot(
            index="from_state", columns="to_state", values="probability"
        ).to_dict()
    if claim == "homogeneity_tests":
        return pd.read_csv(tables / "tabla_3_pruebas_homogeneidad.csv").to_dict(
            orient="records"
        )
    if claim == "queue_model":
        queue = pd.read_csv(tables / "tabla_4_modelo_colas_rho.csv")
        return {
            column.removeprefix("modelo_"): queue[column].iloc[0]
            for column in queue.columns
            if column.startswith("modelo_")
        }
    if claim == "table_stats":
        return {
            path.name: int(pd.read_csv(path).shape[0])
            for path in sorted(tables.glob("tabla_*.csv"))
        }
    raise ValueError(f"Unsupported analysis claim: {claim}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--claim",
        choices=(
            "shape",
            "null_counts",
            "unique_values",
            "road_intersection_pairs",
            "transition_matrix",
            "homogeneity_tests",
            "queue_model",
            "table_stats",
        ),
        required=True,
    )
    parser.add_argument("--regime")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    args = parser.parse_args()
    if args.claim in {
        "transition_matrix",
        "homogeneity_tests",
        "queue_model",
        "table_stats",
    }:
        result = analysis_claim(args.analysis_dir, args.claim, args.regime)
    else:
        result = recalculate(pd.read_csv(args.dataset), args.claim)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        )
    )


if __name__ == "__main__":
    main()
