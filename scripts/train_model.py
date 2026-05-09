from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "datos_integrados.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def cargar_datos():
    df = pd.read_csv(DATA_PATH, sep=";")
    target = "falta_pago"

    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def crear_particiones(X, y):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.22,
        random_state=42,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def crear_preprocesador(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    return preprocessor, num_cols, cat_cols


def main():
    X, y = cargar_datos()
    X_train, X_val, X_test, y_train, y_val, y_test = crear_particiones(X, y)
    preprocessor, num_cols, cat_cols = crear_preprocesador(X)

    modelos = {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "LinearSVC": LinearSVC(max_iter=5000),
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
    }

    mlflow.set_experiment("AP2_Modelado_Creditos")

    resultados = []
    mejor_modelo = None
    mejor_nombre = None
    mejor_accuracy_val = -1

    for nombre, modelo in modelos.items():
        pipeline = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", modelo),
            ]
        )

        with mlflow.start_run(run_name=nombre):
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=5,
                scoring="accuracy",
            )

            pipeline.fit(X_train, y_train)

            y_val_pred = pipeline.predict(X_val)
            y_test_pred = pipeline.predict(X_test)

            accuracy_val = accuracy_score(y_val, y_val_pred)
            accuracy_test = accuracy_score(y_test, y_test_pred)
            f1_macro_val = f1_score(y_val, y_val_pred, average="macro")
            f1_macro_test = f1_score(y_test, y_test_pred, average="macro")

            mlflow.log_param("modelo", nombre)
            mlflow.log_param("num_columnas_numericas", len(num_cols))
            mlflow.log_param("num_columnas_categoricas", len(cat_cols))
            mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
            mlflow.log_metric("cv_accuracy_std", cv_scores.std())
            mlflow.log_metric("validation_accuracy", accuracy_val)
            mlflow.log_metric("test_accuracy", accuracy_test)
            mlflow.log_metric("validation_f1_macro", f1_macro_val)
            mlflow.log_metric("test_f1_macro", f1_macro_test)

            resultados.append(
                {
                    "modelo": nombre,
                    "cv_accuracy_mean": cv_scores.mean(),
                    "cv_accuracy_std": cv_scores.std(),
                    "validation_accuracy": accuracy_val,
                    "test_accuracy": accuracy_test,
                    "validation_f1_macro": f1_macro_val,
                    "test_f1_macro": f1_macro_test,
                }
            )

            if nombre != "DummyClassifier" and accuracy_val > mejor_accuracy_val:
                mejor_accuracy_val = accuracy_val
                mejor_modelo = pipeline
                mejor_nombre = nombre

    resultados_df = pd.DataFrame(resultados).sort_values(
        by="validation_accuracy",
        ascending=False,
    )

    resultados_path = REPORTS_DIR / "model_experimentation_results.csv"
    resultados_df.to_csv(resultados_path, index=False)

    mejor_modelo.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    modelo_path = MODELS_DIR / "modelo_final.joblib"
    joblib.dump(mejor_modelo, modelo_path)

    y_test_pred_final = mejor_modelo.predict(X_test)

    reporte_final = {
        "modelo_seleccionado": mejor_nombre,
        "test_accuracy": accuracy_score(y_test, y_test_pred_final),
        "test_f1_macro": f1_score(y_test, y_test_pred_final, average="macro"),
        "classification_report": classification_report(
            y_test,
            y_test_pred_final,
            output_dict=True,
        ),
        "filas_train": int(X_train.shape[0]),
        "filas_validation": int(X_val.shape[0]),
        "filas_test": int(X_test.shape[0]),
        "columnas_x": int(X.shape[1]),
    }

    reporte_path = REPORTS_DIR / "model_final_report.json"
    with open(reporte_path, "w", encoding="utf-8") as f:
        json.dump(reporte_final, f, indent=4, ensure_ascii=False)

    with mlflow.start_run(run_name=f"modelo_final_{mejor_nombre}"):
        mlflow.log_param("modelo_final", mejor_nombre)
        mlflow.log_metric("test_accuracy", reporte_final["test_accuracy"])
        mlflow.log_metric("test_f1_macro", reporte_final["test_f1_macro"])
        mlflow.log_artifact(str(resultados_path))
        mlflow.log_artifact(str(reporte_path))
        mlflow.sklearn.log_model(mejor_modelo, artifact_path="modelo_final")

    print("Experimentación finalizada correctamente.")
    print(f"Resultados guardados en: {resultados_path}")
    print(f"Modelo final guardado en: {modelo_path}")
    print(f"Reporte final guardado en: {reporte_path}")
    print(f"Modelo seleccionado: {mejor_nombre}")
    print(f"Accuracy test: {reporte_final['test_accuracy']:.4f}")
    print(f"F1 macro test: {reporte_final['test_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
