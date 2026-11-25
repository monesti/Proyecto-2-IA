"""Model non-homogeneous congestion transitions and queue utilization."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2_contingency


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET = BASE_DIR / "datos_procesados" / "trafico_limpio.csv"
TABLE_DIR = BASE_DIR / "resultados" / "tablas"
GRAPH_DIR = BASE_DIR / "resultados" / "gráficos"
STATES = ["Low", "Moderate", "High", "Severe"]
PEAK_ORDER = ["Night", "Morning Peak", "Midday", "Evening Peak", "Off-Peak"]
WEATHER_ORDER = ["Clear", "Fog", "Rain", "Heavy Rain"]


def transition_rows(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(["road_id", "intersection_id", "timestamp"])
    group = data.groupby(["road_id", "intersection_id"], sort=False)
    next_data = group[["timestamp", "congestion_level"]].shift(-1)
    rows = data.copy()
    rows["next_timestamp"] = next_data["timestamp"]
    rows["next_state"] = next_data["congestion_level"]
    rows["continuous"] = (
        rows["next_timestamp"] - rows["timestamp"]
    ) == pd.Timedelta(hours=1)
    return rows[
        rows["continuous"]
        & rows["next_state"].isin(STATES)
        & rows["congestion_level"].isin(STATES)
    ].copy()


def transition_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(rows["congestion_level"], rows["next_state"]).reindex(
        index=STATES, columns=STATES, fill_value=0
    )
    return counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)


def stationary_distribution(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(dtype=float)
    eigenvalues, eigenvectors = np.linalg.eig(values.T)
    vector = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1))])
    vector = np.abs(vector)
    return vector / vector.sum() if vector.sum() else np.zeros(len(STATES))


def test_homogeneity(rows: pd.DataFrame, column: str, order: list) -> dict:
    groups = []
    labels = []
    for label in order:
        subset = rows[rows[column] == label]
        counts = pd.crosstab(subset["congestion_level"], subset["next_state"]).reindex(
            index=STATES, columns=STATES, fill_value=0
        )
        if counts.to_numpy().sum():
            groups.append(counts.to_numpy().ravel())
            labels.append(label)
    if len(groups) < 2:
        return {"variable": column, "chi2": np.nan, "df": np.nan, "p_value": np.nan}
    statistic, p_value, degrees, _ = chi2_contingency(np.vstack(groups))
    return {
        "variable": column,
        "chi2": statistic,
        "df": degrees,
        "p_value": p_value,
    }


def effective_utilization(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    green_fraction = (
        result["green_light_duration"] / result["signal_cycle_seconds"]
    ).clip(0, 1)
    # Base service rate: 60 vehicles per lane-hour per speed-limit unit,
    # reduced by the effective green fraction.
    result["mu_ef"] = (
        result["lanes"] * result["speed_limit"] * 60 * green_fraction
    ).replace(0, np.nan)
    result["lambda"] = result["traffic_flow_rate"].clip(lower=0)
    result["rho"] = (result["lambda"] / result["mu_ef"]).replace(
        [np.inf, -np.inf], np.nan
    )
    return result.dropna(subset=["rho", "average_wait_time", "queue_length"])


def r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.sum((observed - observed.mean()) ** 2)
    return float(1 - np.sum((observed - predicted) ** 2) / denominator) if denominator else np.nan


def queue_models(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    usable = data[data["rho"] < 0.98].copy()
    x = usable["rho"].to_numpy()
    y = usable["average_wait_time"].to_numpy()
    linear_coefficients = np.polyfit(x, y, 1)
    linear_predicted = np.polyval(linear_coefficients, x)

    def nonlinear(rho, a, b, k):
        return a + b * np.power(np.clip(rho, 0, 0.979), k) / (1 - np.clip(rho, 0, 0.979))

    try:
        parameters, _ = curve_fit(
            nonlinear,
            x,
            y,
            p0=[float(y.min()), float(max(y.mean(), 1)), 1.5],
            bounds=([-np.inf, 0, 0.1], [np.inf, np.inf, 8]),
            maxfev=20000,
        )
        nonlinear_predicted = nonlinear(x, *parameters)
    except (RuntimeError, ValueError):
        parameters = np.array([np.nan, np.nan, np.nan])
        nonlinear_predicted = np.full_like(y, np.nan)

    usable["rho_decile"] = pd.qcut(
        usable["rho"], q=10, duplicates="drop", labels=False
    ) + 1
    deciles = usable.groupby("rho_decile", observed=True).agg(
        rho_min=("rho", "min"),
        rho_max=("rho", "max"),
        rho_mean=("rho", "mean"),
        average_wait_time_mean=("average_wait_time", "mean"),
        queue_length_mean=("queue_length", "mean"),
        observations=("rho", "size"),
    ).reset_index()
    metrics = {
        "linear_slope": linear_coefficients[0],
        "linear_intercept": linear_coefficients[1],
        "linear_r2": r2(y, linear_predicted),
        "nonlinear_a": parameters[0],
        "nonlinear_b": parameters[1],
        "nonlinear_k": parameters[2],
        "nonlinear_r2": r2(y, nonlinear_predicted),
        "observations": len(usable),
    }
    return deciles, {
        "metrics": metrics,
        "usable": usable,
        "nonlinear": nonlinear,
        "deciles": deciles,
    }


def export_tables(data: pd.DataFrame, rows: pd.DataFrame, matrices: dict, tests: list, deciles: pd.DataFrame, queue: dict) -> None:
    summary = data.groupby(["congestion_level", "peak_period"], observed=True).agg(
        observations=("congestion_level", "size"),
        traffic_flow_rate_mean=("traffic_flow_rate", "mean"),
        average_wait_time_mean=("average_wait_time", "mean"),
        queue_length_mean=("queue_length", "mean"),
        rho_mean=("rho", "mean"),
        rho_median=("rho", "median"),
    ).reset_index()
    summary.to_csv(TABLE_DIR / "tabla_1_resumen_regimenes.csv", index=False)

    matrix_rows = []
    for regime, matrix in matrices.items():
        for source in STATES:
            for target in STATES:
                matrix_rows.append(
                    {"regime": regime, "from_state": source, "to_state": target,
                     "probability": matrix.loc[source, target]}
                )
    pd.DataFrame(matrix_rows).to_csv(
        TABLE_DIR / "tabla_2_matrices_transicion.csv", index=False
    )
    pd.DataFrame(tests).to_csv(TABLE_DIR / "tabla_3_pruebas_homogeneidad.csv", index=False)
    metrics = queue["metrics"]
    for key, value in metrics.items():
        deciles[f"modelo_{key}"] = value
    deciles.to_csv(TABLE_DIR / "tabla_4_modelo_colas_rho.csv", index=False)


def invoke_graph(matrix: pd.DataFrame, name: str, title: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", encoding="utf-8") as handle:
        matrix.to_csv(handle)
        handle.flush()
        subprocess.run(
            [
                sys.executable,
                str(BASE_DIR / "scripts" / "dibujar-grafo.py"),
                handle.name,
                "--instante",
                name,
                "--output-dir",
                str(GRAPH_DIR),
                "--title",
                title,
            ],
            check=True,
        )


def plot_heatmap(matrices: dict) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, (name, matrix) in zip(axes, list(matrices.items())[:2]):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd")
        axis.set_title(name)
        axis.set_xticks(range(4), STATES, rotation=30)
        axis.set_yticks(range(4), STATES)
        axis.set_xlabel("Estado siguiente")
        axis.set_ylabel("Estado actual")
        for i in range(4):
            for j in range(4):
                axis.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axes, label="Probabilidad")
    figure.suptitle(
        "La punta vespertina con lluvia desplaza las transiciones hacia estados altos"
    )
    figure.savefig(GRAPH_DIR / "heatmap_transiciones.png", dpi=200)
    plt.close(figure)


def plot_hourly(data: pd.DataFrame) -> None:
    counts = (
        data.groupby(["is_weekend", "hour", "congestion_level"], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    full_index = pd.MultiIndex.from_product(
        [[0, 1], range(24)], names=["is_weekend", "hour"]
    )
    counts = counts.reindex(full_index, fill_value=0).reindex(columns=STATES, fill_value=0)
    frame = (
        counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
        .fillna(0)
        .stack()
        .rename("probability")
        .reset_index()
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, weekend in zip(axes, [0, 1]):
        subset = frame[frame["is_weekend"] == weekend]
        for state in STATES:
            values = subset[subset["congestion_level"] == state]
            axis.plot(values["hour"], values["probability"], marker="o", label=state)
        axis.set_title("Fin de semana" if weekend else "Día laboral")
        axis.set_xlabel("Hora del día")
        axis.set_ylabel("Probabilidad empírica")
        axis.set_xticks(range(0, 24, 3))
        axis.grid(alpha=0.25)
    axes[1].legend(title="Estado")
    figure.suptitle(
        "La congestión severa se concentra en las horas punta de días laborales"
    )
    figure.tight_layout()
    figure.savefig(GRAPH_DIR / "evolucion_horaria_estados.png", dpi=200)
    plt.close(figure)


def plot_queue(queue: dict) -> None:
    usable = queue["usable"]
    model = queue["nonlinear"]
    figure, axis = plt.subplots(figsize=(9, 6))
    for label, subset in usable.assign(
        clima=np.where(usable["weather_condition"].isin(["Rain", "Heavy Rain"]), "Lluvia", "Seco")
    ).groupby("clima"):
        axis.scatter(subset["rho"], subset["average_wait_time"], s=8, alpha=0.2, label=label)
    grid = np.linspace(0, min(0.979, usable["rho"].max()), 200)
    metrics = queue["metrics"]
    curve = model(grid, metrics["nonlinear_a"], metrics["nonlinear_b"], metrics["nonlinear_k"])
    axis.plot(grid, curve, color="black", linewidth=2, label="Ajuste no lineal")
    axis.plot(
        grid,
        metrics["linear_intercept"] + metrics["linear_slope"] * grid,
        color="#555555",
        linestyle="--",
        linewidth=1.8,
        label="Ajuste lineal",
    )
    deciles = queue["deciles"]
    axis.scatter(
        deciles["rho_mean"],
        deciles["average_wait_time_mean"],
        color="#111111",
        marker="D",
        s=55,
        zorder=4,
        label="Media por decil",
    )
    axis.set_xlabel("Utilización ρ = λ / μef")
    axis.set_ylabel("Tiempo medio de espera (min)")
    axis.set_title(
        "El tiempo de espera crece de forma acelerada al aumentar la utilización ρ"
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(GRAPH_DIR / "espera_vs_utilizacion_rho.png", dpi=200)
    plt.close(figure)


def plot_boxplot(data: pd.DataFrame) -> None:
    data = data.copy()
    data["regime"] = (
        data["congestion_level"].astype(str) + " | " + data["weather_condition"].astype(str)
    )
    regimes = [f"{state} | {weather}" for state in STATES for weather in WEATHER_ORDER]
    wait_values = [data.loc[data["regime"] == regime, "average_wait_time"] for regime in regimes]
    queue_values = [data.loc[data["regime"] == regime, "queue_length"] for regime in regimes]
    figure, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    axes[0].boxplot(wait_values, showfliers=False)
    axes[1].boxplot(queue_values, showfliers=False)
    axes[0].set_ylabel("Espera (min)")
    axes[1].set_ylabel("Longitud de cola (vehículos)")
    axes[1].set_xlabel("Estado de congestión | clima")
    axes[1].set_xticks(range(1, len(regimes) + 1), regimes, rotation=75, ha="right")
    figure.suptitle(
        "El estado Severe dispara esperas y colas bajo cualquier condición climática"
    )
    figure.tight_layout()
    figure.savefig(GRAPH_DIR / "boxplot_espera_por_estado_clima.png", dpi=200)
    plt.close(figure)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATASET, parse_dates=["timestamp"])
    required = {"is_weekend", "peak_period", "weather_condition"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data["congestion_level"] = pd.Categorical(data["congestion_level"], STATES, ordered=True)
    print("correspondencia_hour_peak_period=")
    print(pd.crosstab(data["hour"], data["peak_period"]).to_string())
    rows = transition_rows(data)
    global_matrix = transition_matrix(rows)
    matrices = {
        "Night | Clear": transition_matrix(rows[(rows["peak_period"] == "Night") & (rows["weather_condition"] == "Clear")]),
        "Evening Peak | Rain": transition_matrix(rows[(rows["peak_period"] == "Evening Peak") & (rows["weather_condition"] == "Rain")]),
        "Weekday": transition_matrix(rows[rows["is_weekend"] == 0]),
        "Weekend": transition_matrix(rows[rows["is_weekend"] == 1]),
        "Clear": transition_matrix(rows[rows["weather_condition"] == "Clear"]),
        "Heavy Rain": transition_matrix(rows[rows["weather_condition"] == "Heavy Rain"]),
    }
    matrices["Global"] = global_matrix
    tests = [
        test_homogeneity(rows, "peak_period", PEAK_ORDER),
        test_homogeneity(rows, "weather_condition", WEATHER_ORDER),
        test_homogeneity(rows, "is_weekend", [0, 1]),
    ]
    tests[2]["variable"] = "is_weekend"
    data = effective_utilization(data)
    deciles, queue = queue_models(data)
    export_tables(data, rows, matrices, tests, deciles, queue)
    invoke_graph(
        matrices["Night | Clear"],
        "night_clear",
        "Noche despejada: el tráfico permanece en flujo libre (97,6%)",
    )
    invoke_graph(
        matrices["Evening Peak | Rain"],
        "evening_peak_rain",
        "Punta tarde con lluvia: emerge persistencia en congestión severa (12,8%)",
    )
    plot_heatmap({"Night | Clear": matrices["Night | Clear"], "Evening Peak | Rain": matrices["Evening Peak | Rain"]})
    plot_hourly(data)
    plot_queue(queue)
    plot_boxplot(data)
    for name, matrix in matrices.items():
        stationary = stationary_distribution(matrix)
        print(
            f"matriz={name}; persistencia_high={matrix.loc['High', 'High']:.6f}; "
            f"persistencia_severe={matrix.loc['Severe', 'Severe']:.6f}; "
            f"estacionaria={dict(zip(STATES, stationary.round(6)))}"
        )
    for test in tests:
        print(f"chi2={test}")
    print(f"colas={queue['metrics']}")


if __name__ == "__main__":
    main()
