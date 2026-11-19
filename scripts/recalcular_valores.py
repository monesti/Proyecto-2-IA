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
                "day_of_week",
                "is_weekend",
            )
        }
    if claim == "road_intersection_pairs":
        return int(data[["road_id", "intersection_id"]].drop_duplicates().shape[0])
    raise ValueError(f"Unsupported claim type: {claim}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--claim",
        choices=("shape", "null_counts", "unique_values", "road_intersection_pairs"),
        required=True,
    )
    args = parser.parse_args()
    result = recalculate(pd.read_csv(args.dataset), args.claim)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
