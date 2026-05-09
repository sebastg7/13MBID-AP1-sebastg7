from pathlib import Path

import pandas as pd
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"


@pytest.fixture
def creditos():
    return pd.read_csv(DATA_DIR / "datos_creditos.csv", sep=";")


@pytest.fixture
def tarjetas():
    return pd.read_csv(DATA_DIR / "datos_tarjetas.csv", sep=";")


def test_estructura_dataset_creditos(creditos):
    columnas_esperadas = [
        "id_cliente",
        "edad",
        "importe_solicitado",
        "duracion_credito",
        "antiguedad_empleado",
        "situacion_vivienda",
        "ingresos",
        "objetivo_credito",
        "pct_ingreso",
        "tasa_interes",
        "estado_credito",
        "falta_pago",
    ]

    assert creditos.shape == (10127, 12)
    assert list(creditos.columns) == columnas_esperadas


def test_estructura_dataset_tarjetas(tarjetas):
    columnas_esperadas = [
        "id_cliente",
        "antiguedad_cliente",
        "estado_civil",
        "estado_cliente",
        "gastos_ult_12m",
        "genero",
        "limite_credito_tc",
        "nivel_educativo",
        "nivel_tarjeta",
        "operaciones_ult_12m",
        "personas_a_cargo",
    ]

    assert tarjetas.shape == (10127, 11)
    assert list(tarjetas.columns) == columnas_esperadas


def test_unicidad_id_cliente(creditos, tarjetas):
    assert creditos["id_cliente"].duplicated().sum() == 0
    assert tarjetas["id_cliente"].duplicated().sum() == 0


def test_integridad_referencial(creditos, tarjetas):
    ids_creditos = set(creditos["id_cliente"])
    ids_tarjetas = set(tarjetas["id_cliente"])

    assert ids_creditos == ids_tarjetas


@pytest.mark.parametrize(
    "columna,tolerancia",
    [
        ("antiguedad_empleado", 0.05),
        ("tasa_interes", 0.05),
    ],
)
def test_tolerancia_valores_nulos_creditos(creditos, columna, tolerancia):
    porcentaje_nulos = creditos[columna].isna().mean()

    assert porcentaje_nulos <= tolerancia, (
        f"La columna {columna} tiene un {porcentaje_nulos:.2%} de nulos, "
        f"superando la tolerancia definida de {tolerancia:.2%}."
    )


@pytest.mark.parametrize(
    "columna,limite_inferior,limite_superior",
    [
        ("edad", 18, 90),
        ("antiguedad_empleado", 0, 80),
        ("pct_ingreso", 0, 1),
        ("tasa_interes", 0, 100),
        ("duracion_credito", 1, 10),
    ],
)
def test_rangos_numericos_creditos(creditos, columna, limite_inferior, limite_superior):
    fuera_de_rango = creditos[
        (creditos[columna] < limite_inferior) | (creditos[columna] > limite_superior)
    ]

    assert len(fuera_de_rango) == 0, (
        f"La columna {columna} tiene {len(fuera_de_rango)} valores fuera "
        f"del rango esperado [{limite_inferior}, {limite_superior}]."
    )


def test_valores_categoricos_creditos(creditos):
    assert set(creditos["situacion_vivienda"].dropna()).issubset(
        {"ALQUILER", "HIPOTECA", "OTROS", "PROPIA"}
    )
    assert set(creditos["estado_credito"].dropna()).issubset({0, 1})
    assert set(creditos["falta_pago"].dropna()).issubset({"N", "Y"})


def test_valores_categoricos_tarjetas(tarjetas):
    assert set(tarjetas["estado_cliente"].dropna()).issubset({"ACTIVO", "PASIVO"})
    assert set(tarjetas["genero"].dropna()).issubset({"F", "M"})
