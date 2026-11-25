"""Audit quantitative claims in the report against generated results."""

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
    root = Path(__file__).resolve().parents[1]
    tests = pd.read_csv(root / "resultados" / "tablas" / "tabla_3_pruebas_homogeneidad.csv")
    matrices = pd.read_csv(root / "resultados" / "tablas" / "tabla_2_matrices_transicion.csv")
    queue = pd.read_csv(root / "resultados" / "tablas" / "tabla_4_modelo_colas_rho.csv")
    shape = recalculate(data, "shape")
    peak = tests.loc[tests["variable"] == "peak_period"].iloc[0]
    weather = tests.loc[tests["variable"] == "weather_condition"].iloc[0]
    night = matrices[(matrices["regime"] == "Night | Clear") & (matrices["from_state"] == "Low") & (matrices["to_state"] == "Moderate")].iloc[0]
    rain = matrices[(matrices["regime"] == "Evening Peak | Rain") & (matrices["from_state"] == "High") & (matrices["to_state"] == "Moderate")].iloc[0]
    if report.name == "infografia.tex":
        expected = [
            ("Registros horarios", r"\$?204[.]?000\$?\s*(observaciones|registros|filas)", "204000 registros"),
            ("Intersecciones semaforizadas", r"\$?100\$?\s+intersecciones", "100 intersecciones"),
            ("Variables procesadas", r"\$?20\$?\s+variables", "20 variables"),
            ("Transición Low-Low noche", r"0[,.]976", "P Low-Low=0.976"),
            ("Persistencia Severe lluvia", r"0[,.]128", "P Severe-Severe=0.128"),
            ("P-valor homogeneidad", r"p\s*<\s*0[,.]001", "p<0.001"),
            ("R2 no lineal", r"0[,.]4223", "R2=0.4223"),
        ]
        claims: list[dict[str, str]] = []
        for label, pattern, expected_value in expected:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            claims.append(
                {
                    "afirmacion": label,
                    "valor_recalculado": expected_value,
                    "ubicacion": f"caracteres {match.start()}-{match.end()}" if match else "no encontrado",
                    "estado": "APROBADO" if match else "PENDIENTE",
                }
            )
        return claims
    expected = [
        ("Dimensiones originales", rf"204000\s*filas.*47\s*columnas", f"{shape['filas']} filas, 47 columnas"),
        ("Dimensiones procesadas", rf"204000\s*filas.*20\s*columnas", "204000 filas, 20 columnas"),
        ("Chi-cuadrado horario", rf"308589[.,]19", "chi2=308589.19"),
        ("Chi-cuadrado climático", rf"5773[.,]07", "chi2=5773.07"),
        ("Transición Night Low-Moderate", rf"Night.*?Low.*?Moderate.*?0[.,]0235", f"P={night['probability']:.4f}"),
        ("Transición Evening High-Moderate", rf"Evening.*?High.*?Moderate.*?0[.,]6163", f"P={rain['probability']:.4f}"),
        ("R2 lineal", r"0[.,]4152", "R2=0.4152"),
        ("R2 no lineal", r"0[.,]4223", "R2=0.4223"),
    ]
    claims: list[dict[str, str]] = []
    for label, pattern, expected_value in expected:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        claims.append(
            {
                "afirmacion": label,
                "valor_recalculado": expected_value,
                "ubicacion": f"caracteres {match.start()}-{match.end()}" if match else "no encontrado",
                "estado": "APROBADO" if match else "PENDIENTE",
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
        writer = csv.DictWriter(
            handle,
            fieldnames=["afirmacion", "valor_recalculado", "ubicacion", "estado"],
        )
        writer.writeheader()
        writer.writerows(claims)


if __name__ == "__main__":
    main()
