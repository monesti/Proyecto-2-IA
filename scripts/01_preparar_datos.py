"""Prepare the traffic dataset for the non-homogeneous Markov-chain analysis."""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "smart_city_traffic_mobility copy.csv"
OUTPUT_PATH = BASE_DIR / "datos_procesados" / "trafico_limpio.csv"
COUNT_PATH = BASE_DIR / "resultados" / "conteo_dataset.txt"

RELEVANT_COLUMNS = [
    "road_id",
    "intersection_id",
    "timestamp",
    "congestion_level",
    "congestion_score",
    "hour",
    "day_of_week",
    "is_weekend",
    "peak_period",
    "weather_condition",
    "rainfall_mm",
    "average_wait_time",
    "queue_length",
    "vehicle_count",
    "traffic_flow_rate",
    "lanes",
    "signal_cycle_seconds",
    "green_light_duration",
    "speed_limit",
    "average_speed",
]


def main() -> None:
    data = pd.read_csv(INPUT_PATH)
    original_rows, original_columns = data.shape
    COUNT_PATH.write_text(
        f"filas={original_rows}\ncolumnas={original_columns}\n",
        encoding="utf-8",
    )

    missing_columns = sorted(set(RELEVANT_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    clean = data.loc[:, RELEVANT_COLUMNS].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="raise")
    clean = clean.sort_values(
        ["road_id", "intersection_id", "timestamp"], kind="stable"
    )
    clean.to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S")

    print(f"nulos={clean.isna().sum().to_dict()}")
    for column in (
        "congestion_level",
        "peak_period",
        "weather_condition",
        "day_of_week",
        "is_weekend",
    ):
        print(f"unicos_{column}={sorted(clean[column].dropna().unique().tolist(), key=str)}")
    print(
        "pares_unicos_road_intersection="
        f"{clean[['road_id', 'intersection_id']].drop_duplicates().shape[0]}"
    )


if __name__ == "__main__":
    main()
