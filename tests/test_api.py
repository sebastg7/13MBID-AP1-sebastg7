from pathlib import Path
import sys

from fastapi.testclient import TestClient


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_endpoint():
    payload = {
        "importe_solicitado": 5000,
        "duracion_credito": 3,
        "situacion_vivienda": "ALQUILER",
        "ingresos": 50000,
        "objetivo_credito": "PERSONAL",
        "pct_ingreso": 0.10,
        "tasa_interes": 12.5,
        "antiguedad_cliente": 36,
        "estado_cliente": "ACTIVO",
        "gastos_ult_12m": 2500,
        "genero": "M",
        "limite_credito_tc": 8000,
        "nivel_educativo": "UNIVERSITARIO_COMPLETO",
        "operaciones_ult_12m": 60,
        "personas_a_cargo": 1,
        "estado_civil_N": "S",
        "estado_credito_N": "P",
        "antiguedad_empleado_N": "5_a_10",
        "edad_N": "30_o_mas",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["prediccion"] in ["N", "Y"]
    assert "interpretacion" in data
    assert "probabilidad_mora" in data
