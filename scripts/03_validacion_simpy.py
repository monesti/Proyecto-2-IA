"""Short SimPy validation of nonlinear queue growth at a traffic signal."""

from __future__ import annotations

import json
import random
from pathlib import Path

import simpy


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT = BASE_DIR / "resultados" / "validacion_simpy.json"
RHO_LEVELS = [0.2, 0.4, 0.6, 0.8, 0.9]
CYCLE_MINUTES = 2.0
GREEN_MINUTES = 1.0
SERVICE_MINUTES = 0.5
EFFECTIVE_SERVICE_RATE = GREEN_MINUTES / (CYCLE_MINUTES * SERVICE_MINUTES)
SIMULATION_MINUTES = 1800.0
WARMUP_MINUTES = 300.0
REPLICATIONS = 3


def minutes_until_green(now: float) -> float:
    phase = now % CYCLE_MINUTES
    return 0.0 if phase < GREEN_MINUTES else CYCLE_MINUTES - phase


def vehicle(env: simpy.Environment, server: simpy.Resource, waits: list[float]) -> None:
    arrival = env.now
    with server.request() as request:
        yield request
        yield env.timeout(minutes_until_green(env.now))
        wait = env.now - arrival
        if arrival >= WARMUP_MINUTES:
            waits.append(wait)
        yield env.timeout(SERVICE_MINUTES)


def arrivals(
    env: simpy.Environment,
    server: simpy.Resource,
    arrival_rate: float,
    rng: random.Random,
    waits: list[float],
) -> None:
    while True:
        yield env.timeout(rng.expovariate(arrival_rate))
        if env.now >= SIMULATION_MINUTES:
            return
        env.process(vehicle(env, server, waits))


def run_replication(rho: float, seed: int) -> float:
    rng = random.Random(seed)
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)
    waits: list[float] = []
    arrival_rate = rho * EFFECTIVE_SERVICE_RATE
    env.process(arrivals(env, server, arrival_rate, rng, waits))
    env.run(until=SIMULATION_MINUTES)
    return sum(waits) / len(waits) if waits else 0.0


def load_reference() -> dict:
    reference = BASE_DIR / "resultados" / "tablas" / "tabla_4_modelo_colas_rho.csv"
    if not reference.exists():
        return {}
    import pandas as pd

    data = pd.read_csv(reference)
    return {
        "nonlinear_r2": float(data["modelo_nonlinear_r2"].iloc[0]),
        "rho_decile_mean_min": float(data["rho_mean"].min()),
        "rho_decile_mean_max": float(data["rho_mean"].max()),
        "wait_decile_mean_min": float(data["average_wait_time_mean"].min()),
        "wait_decile_mean_max": float(data["average_wait_time_mean"].max()),
    }


def main() -> None:
    results = []
    for rho in RHO_LEVELS:
        waits = [run_replication(rho, seed=1000 + index) for index in range(REPLICATIONS)]
        results.append(
            {
                "rho": rho,
                "wait_mean_minutes": sum(waits) / len(waits),
                "replications": REPLICATIONS,
            }
        )
    increases = all(
        left["wait_mean_minutes"] < right["wait_mean_minutes"]
        for left, right in zip(results, results[1:])
    )
    output = {
        "parameters": {
            "cycle_minutes": CYCLE_MINUTES,
            "green_minutes": GREEN_MINUTES,
            "service_minutes": SERVICE_MINUTES,
            "effective_service_rate_per_minute": EFFECTIVE_SERVICE_RATE,
            "simulation_minutes": SIMULATION_MINUTES,
            "warmup_minutes": WARMUP_MINUTES,
        },
        "results": results,
        "monotonic_increase": increases,
        "reference_table_4": load_reference(),
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("rho | espera_media_simulada_min")
    for result in results:
        print(f"{result['rho']:.1f} | {result['wait_mean_minutes']:.3f}")
    print(f"crecimiento_monotono={increases}")


if __name__ == "__main__":
    main()
