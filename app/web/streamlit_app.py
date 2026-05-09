import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "https://one3mbid-api-prediccion-mora.onrender.com")


st.set_page_config(
    page_title="Predicción de mora crediticia",
    page_icon="💳",
    layout="centered",
)

st.title("Predicción de mora crediticia")
st.write(
    "Aplicación de consulta para estimar si un crédito puede presentar riesgo de mora "
    "a partir del modelo entrenado en el proyecto."
)

st.sidebar.header("Configuración")
api_url = st.sidebar.text_input("URL de la API", value=API_URL)

st.subheader("Datos del crédito y del cliente")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        importe_solicitado = st.number_input("Importe solicitado", min_value=0.0, value=5000.0)
        duracion_credito = st.number_input("Duración del crédito", min_value=1, max_value=10, value=3)
        situacion_vivienda = st.selectbox(
            "Situación de vivienda",
            ["ALQUILER", "HIPOTECA", "OTROS", "PROPIA"],
        )
        ingresos = st.number_input("Ingresos", min_value=0.0, value=50000.0)
        objetivo_credito = st.selectbox(
            "Objetivo del crédito",
            ["EDUCACIÓN", "INVERSIONES", "MEJORAS_HOGAR", "PAGO_DEUDAS", "PERSONAL", "SALUD"],
        )
        pct_ingreso = st.number_input("Porcentaje de ingreso", min_value=0.0, max_value=1.0, value=0.10)
        tasa_interes = st.number_input("Tasa de interés", min_value=0.0, max_value=100.0, value=12.5)
        edad_N = st.selectbox("Rango de edad", ["18_a_24", "25_a_29", "30_o_mas"])
        antiguedad_empleado_N = st.selectbox(
            "Antigüedad del empleado",
            ["menor_5", "5_a_10", "mayor_10"],
        )

    with col2:
        antiguedad_cliente = st.number_input("Antigüedad del cliente", min_value=0.0, value=36.0)
        estado_cliente = st.selectbox("Estado del cliente", ["ACTIVO", "PASIVO"])
        gastos_ult_12m = st.number_input("Gastos últimos 12 meses", min_value=0.0, value=2500.0)
        genero = st.selectbox("Género", ["F", "M"])
        limite_credito_tc = st.number_input("Límite de crédito TC", min_value=0.0, value=8000.0)
        nivel_educativo = st.selectbox(
            "Nivel educativo",
            [
                "DESCONOCIDO",
                "POSGRADO_COMPLETO",
                "POSGRADO_INCOMPLETO",
                "SECUNDARIO_COMPLETO",
                "UNIVERSITARIO_COMPLETO",
                "UNIVERSITARIO_INCOMPLETO",
            ],
        )
        operaciones_ult_12m = st.number_input("Operaciones últimos 12 meses", min_value=0.0, value=60.0)
        personas_a_cargo = st.number_input("Personas a cargo", min_value=0.0, max_value=10.0, value=1.0)
        estado_civil_N = st.selectbox("Estado civil", ["C", "S", "N", "D"])
        estado_credito_N = st.selectbox("Estado del crédito", ["P", "C"])

    submitted = st.form_submit_button("Ejecutar predicción")


if submitted:
    payload = {
        "importe_solicitado": importe_solicitado,
        "duracion_credito": duracion_credito,
        "situacion_vivienda": situacion_vivienda,
        "ingresos": ingresos,
        "objetivo_credito": objetivo_credito,
        "pct_ingreso": pct_ingreso,
        "tasa_interes": tasa_interes,
        "antiguedad_cliente": antiguedad_cliente,
        "estado_cliente": estado_cliente,
        "gastos_ult_12m": gastos_ult_12m,
        "genero": genero,
        "limite_credito_tc": limite_credito_tc,
        "nivel_educativo": nivel_educativo,
        "operaciones_ult_12m": operaciones_ult_12m,
        "personas_a_cargo": personas_a_cargo,
        "estado_civil_N": estado_civil_N,
        "estado_credito_N": estado_credito_N,
        "antiguedad_empleado_N": antiguedad_empleado_N,
        "edad_N": edad_N,
    }

    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()

            st.success("Predicción ejecutada correctamente")

            st.metric("Predicción", result["prediccion"])
            st.write(result["interpretacion"])

            if result.get("probabilidad_mora") is not None:
                st.metric(
                    "Probabilidad estimada de mora",
                    f"{result['probabilidad_mora']:.2%}",
                )
        else:
            st.error(f"Error en la API: {response.status_code}")
            st.write(response.text)

    except requests.exceptions.RequestException as exc:
        st.error("No se pudo conectar con la API.")
        st.write(str(exc))
