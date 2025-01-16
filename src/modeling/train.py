# src/modeling/train.py

import sys
import os
import json
import pandas as pd

# Obtener la ruta absoluta del directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Añadir la ruta raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "..")))

from src.dataset import load_dataset
from src.modeling.regression import PairwiseLinearRegression
from src.modeling.change_point import ChangePointDetector
from src.plots import plot_residuals_with_changes, load_change_points


def main():
    # Configuración de rutas
    data_path = os.path.abspath(os.path.join(script_dir, "..", "..", "data", "raw"))
    filename = "epanet.csv"

    # Cargar y preprocesar datos
    dataset = load_dataset(data_path=data_path, filename=filename)

    # Acceder a los datos de presión
    pressures = dataset.pressures

    # Entrenar modelos de regresión
    regression = PairwiseLinearRegression(pressures)
    regression.train_models()

    print(
        "Modelos de regresión entrenados para todos los pares de sensores de presión."
    )

    # Calcular y guardar métricas
    metrics_df = regression.get_metrics()
    metrics_df.to_csv(os.path.join(data_path, "regression_metrics.csv"), index=False)

    print("Métricas de regresión calculadas y guardadas en regression_metrics.csv.")

    # Mostrar algunas métricas en consola
    print("\nEjemplo de Métricas Calculadas:")
    print(metrics_df.head())

    # Calcular residuales
    residuals = regression.calculate_residuals()

    # Guardar los residuales
    residuals_df = pd.DataFrame(residuals)
    residuals_df.to_csv(os.path.join(data_path, "residuals.csv"))

    print("Residuales calculados y guardados en residuals.csv.")

    # Verificar datos
    print("Dimensiones de residuals_df:", residuals_df.shape)
    print(residuals_df.head())

    # Inicializar y ejecutar el detector de puntos de cambio
    detector = ChangePointDetector(residuals=residuals_df, model="l2", penalty=3)
    detector.detect_changes()
    change_points = detector.get_change_points()

    # Guardar los puntos de cambio detectados
    change_points_serializable = {f"{k[0]}_{k[1]}": v for k, v in change_points.items()}
    with open(os.path.join(data_path, "change_points.json"), "w") as f:
        json.dump(change_points_serializable, f)

    print("Puntos de cambio detectados y guardados en change_points.json.")

    # Verificar puntos de cambio
    for key, value in change_points.items():
        print(f"Puntos de cambio para {key}: {value}")

    # Convertir columnas a cadenas si son tuplas
    residuals_df.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
        for col in residuals_df.columns
    ]

    # Verificar nombres de columnas
    print("Nombres de columnas en residuals_df:", residuals_df.columns.tolist())
    print("Claves en change_points:", list(change_points.keys()))

    # Definir la ruta correcta para guardar los gráficos
    output_dir = os.path.abspath(
        os.path.join(script_dir, "..", "..", "reports", "figures")
    )
    print(f"Ruta absoluta del directorio de salida: {output_dir}")

    plot_residuals_with_changes(
        residuals=residuals_df,
        change_points=change_points,
        output_dir=output_dir,
    )

    print("Gráficos de residuales generados y guardados en reports/figures.")

    # Opcional: Guardar el modelo de regresión y las métricas en un archivo JSON
    # Para facilitar la reutilización y el análisis posterior

    models_metrics = {
        "models": {
            f"{k[0]}_{k[1]}": {"coef": v.coef_.tolist(), "intercept": v.intercept_}
            for k, v in regression.models.items()
        },
        "metrics": metrics_df.to_dict(orient="records"),
    }

    with open(os.path.join(data_path, "models_metrics.json"), "w") as f:
        json.dump(models_metrics, f, indent=4)

    print("Modelos y métricas guardados en models_metrics.json.")


if __name__ == "__main__":
    main()
