from pathlib import Path
import json

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "modelo_final.joblib"
REPORT_PATH = ROOT_DIR / "reports" / "model_final_report.json"
DATA_PATH = ROOT_DIR / "data" / "processed" / "datos_integrados.csv"


def test_modelo_final_existe():
    assert MODEL_PATH.exists(), "No existe el archivo del modelo final."


def test_reporte_final_existe():
    assert REPORT_PATH.exists(), "No existe el reporte final del modelo."


def test_modelo_supera_accuracy_minimo():
    with open(REPORT_PATH, encoding="utf-8") as f:
        report = json.load(f)

    assert report["test_accuracy"] >= 0.80, (
        f"El accuracy del modelo es {report['test_accuracy']:.4f}, "
        "por debajo del mínimo requerido de 0.80."
    )


def test_modelo_supera_f1_macro_minimo():
    with open(REPORT_PATH, encoding="utf-8") as f:
        report = json.load(f)

    assert report["test_f1_macro"] >= 0.70, (
        f"El F1 macro del modelo es {report['test_f1_macro']:.4f}, "
        "por debajo del mínimo definido de 0.70."
    )


def test_modelo_predice_clases_validas():
    modelo = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH, sep=";")
    X = df.drop(columns=["falta_pago"]).head(10)

    predicciones = modelo.predict(X)

    assert len(predicciones) == 10
    assert set(predicciones).issubset({"N", "Y"})
