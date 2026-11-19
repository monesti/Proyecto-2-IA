"""Audit quantitative claims in a report against the processed dataset."""

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

from recalcular_valores import recalculate


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = BASE_DIR / "datos_procesados" / "trafico_limpio.csv"
DEFAULT_OUTPUT = BASE_DIR / "resultados" / "tablas" / "auditoria_afirmaciones.csv"


def audit_report(report: Path, data: pd.DataFrame) -> list[dict[str, str]]:
    text = report.read_text(encoding="utf-8")
    claims: list[dict[str, str]] = []
    shape = recalculate(data, "shape")
    shape_pattern = re.compile(r"(\d+)\s*filas?.*?(\d+)\s*columnas?", re.I | re.S)
    for match in shape_pattern.finditer(text):
        shown = f"filas={match.group(1)}, columnas={match.group(2)}"
        expected = f"filas={shape['filas']}, columnas={shape['columnas']}"
        claims.append(
            {
                "afirmacion": shown,
                "ubicacion": f"caracteres {match.start()}-{match.end()}",
                "estado": "APROBADO" if shown == expected else "REVISAR",
            }
        )
    if not claims:
        claims.append(
            {
                "afirmacion": "No se detectaron afirmaciones cuantitativas compatibles",
                "ubicacion": "documento completo",
                "estado": "PENDIENTE",
            }
        )
    return claims


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("informe", type=Path)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    claims = audit_report(args.informe, pd.read_csv(args.dataset))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["afirmacion", "ubicacion", "estado"])
        writer.writeheader()
        writer.writerows(claims)


if __name__ == "__main__":
    main()
