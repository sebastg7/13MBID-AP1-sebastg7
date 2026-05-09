from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "modelo_final.joblib"

app = FastAPI(
    title="API de predicción de mora crediticia",
    description="API para ejecutar predicciones utilizando el modelo entrenado en la AP2.",
    version="1.0.0",
)


class CreditRequest(BaseModel):
    importe_solicitado: float
    duracion_credito: int
    situacion_vivienda: str
    ingresos: float
    objetivo_credito: str
    pct_ingreso: float
    tasa_interes: float
    antiguedad_cliente: float
    estado_cliente: str
    gastos_ult_12m: float
    genero: str
    limite_credito_tc: float
    nivel_educativo: str
    operaciones_ult_12m: float
    personas_a_cargo: float
    estado_civil_N: str
    estado_credito_N: str
    antiguedad_empleado_N: str
    edad_N: str


class PredictionResponse(BaseModel):
    prediccion: str
    interpretacion: str
    probabilidad_mora: Optional[float] = None


def cargar_modelo():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


modelo = cargar_modelo()


@app.get("/")
def root():
    return {
        "mensaje": "API de predicción de mora crediticia",
        "estado": "activa",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_loaded": modelo is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: CreditRequest):
    try:
        input_df = pd.DataFrame([request.model_dump()])

        prediccion = modelo.predict(input_df)[0]

        probabilidad_mora = None
        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(input_df)[0]
            clases = list(modelo.classes_)
            if "Y" in clases:
                probabilidad_mora = float(probabilidades[clases.index("Y")])

        interpretacion = (
            "El crédito presenta riesgo de mora."
            if prediccion == "Y"
            else "El crédito no presenta riesgo de mora."
        )

        return PredictionResponse(
            prediccion=str(prediccion),
            interpretacion=interpretacion,
            probabilidad_mora=probabilidad_mora,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error al ejecutar la predicción: {exc}",
        )
