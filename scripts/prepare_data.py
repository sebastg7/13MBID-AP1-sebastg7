from pathlib import Path
import json

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def imputar_por_grupo(df, columna, grupo):
    """Imputa valores nulos usando la mediana por grupo y, si queda algún nulo, la mediana global."""
    df[columna] = df[columna].fillna(df.groupby(grupo)[columna].transform("median"))
    df[columna] = df[columna].fillna(df[columna].median())
    return df


def preparar_datos():
    creditos = pd.read_csv(RAW_DIR / "datos_creditos.csv", sep=";")
    tarjetas = pd.read_csv(RAW_DIR / "datos_tarjetas.csv", sep=";")

    resumen = {
        "filas_creditos_original": int(creditos.shape[0]),
        "columnas_creditos_original": int(creditos.shape[1]),
        "filas_tarjetas_original": int(tarjetas.shape[0]),
        "columnas_tarjetas_original": int(tarjetas.shape[1]),
    }

    # Selección de atributos
    if "nivel_tarjeta" in tarjetas.columns:
        tarjetas = tarjetas.drop(columns=["nivel_tarjeta"])

    # Limpieza de valores fuera de rango
    filas_edad_fuera_rango = creditos[
        (creditos["edad"] < 18) | (creditos["edad"] > 90)
    ].shape[0]

    creditos = creditos[(creditos["edad"] >= 18) & (creditos["edad"] <= 90)].copy()

    filas_antiguedad_fuera_rango = creditos[
        (creditos["antiguedad_empleado"] < 0) | (creditos["antiguedad_empleado"] > 80)
    ].shape[0]

    creditos.loc[
        (creditos["antiguedad_empleado"] < 0) | (creditos["antiguedad_empleado"] > 80),
        "antiguedad_empleado",
    ] = pd.NA

    resumen["filas_eliminadas_por_edad_fuera_rango"] = int(filas_edad_fuera_rango)
    resumen["valores_antiguedad_fuera_rango_imputados"] = int(filas_antiguedad_fuera_rango)

    resumen["nulos_antes_imputacion"] = {
        "antiguedad_empleado": int(creditos["antiguedad_empleado"].isna().sum()),
        "tasa_interes": int(creditos["tasa_interes"].isna().sum()),
    }

    # Imputación de valores nulos
    creditos = imputar_por_grupo(creditos, "antiguedad_empleado", "edad")
    creditos = imputar_por_grupo(creditos, "tasa_interes", "objetivo_credito")

    resumen["nulos_despues_imputacion"] = {
        "antiguedad_empleado": int(creditos["antiguedad_empleado"].isna().sum()),
        "tasa_interes": int(creditos["tasa_interes"].isna().sum()),
    }

    # Integración de datasets
    datos_integrados = pd.merge(creditos, tarjetas, on="id_cliente", how="inner")

    resumen["filas_despues_integracion"] = int(datos_integrados.shape[0])
    resumen["columnas_despues_integracion"] = int(datos_integrados.shape[1])

    # Transformaciones sobre atributos categóricos
    cambios_estado_civil = {
        "CASADO": "C",
        "SOLTERO": "S",
        "DESCONOCIDO": "N",
        "DIVORCIADO": "D",
    }

    cambios_estado_credito = {
        0: "P",
        1: "C",
    }

    datos_integrados["estado_civil_N"] = datos_integrados["estado_civil"].map(cambios_estado_civil)
    datos_integrados["estado_credito_N"] = datos_integrados["estado_credito"].map(cambios_estado_credito)

    datos_integrados["antiguedad_empleado_N"] = pd.cut(
        datos_integrados["antiguedad_empleado"],
        bins=[0, 5, 10, 81],
        labels=["menor_5", "5_a_10", "mayor_10"],
        include_lowest=True,
        right=False,
    )

    datos_integrados["edad_N"] = pd.cut(
        datos_integrados["edad"],
        bins=[18, 25, 30, 91],
        labels=["18_a_24", "25_a_29", "30_o_mas"],
        include_lowest=True,
        right=False,
    )

    # Eliminación de atributos originales reemplazados o no utilizados para modelado
    columnas_a_eliminar = [
        "id_cliente",
        "estado_civil",
        "estado_credito",
        "antiguedad_empleado",
        "edad",
    ]

    datos_integrados = datos_integrados.drop(columns=columnas_a_eliminar)

    resumen["filas_dataset_final"] = int(datos_integrados.shape[0])
    resumen["columnas_dataset_final"] = int(datos_integrados.shape[1])
    resumen["nulos_dataset_final"] = {
        col: int(valor)
        for col, valor in datos_integrados.isna().sum().items()
        if valor > 0
    }

    output_path = PROCESSED_DIR / "datos_integrados.csv"
    report_path = REPORTS_DIR / "prepare_data_summary.json"

    datos_integrados.to_csv(output_path, sep=";", index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=4, ensure_ascii=False)

    print("Preparación de datos finalizada correctamente.")
    print(f"Dataset procesado generado en: {output_path}")
    print(f"Reporte generado en: {report_path}")
    print(f"Dimensiones finales: {datos_integrados.shape[0]} filas x {datos_integrados.shape[1]} columnas")


if __name__ == "__main__":
    preparar_datos()
