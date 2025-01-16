# src/plots.py

import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
import re  # Importar el módulo de expresiones regulares


def plot_residuals_with_changes(
    residuals: pd.DataFrame,
    change_points: Dict[Tuple[str, str], List[int]],
    output_dir: str = "../reports/figures",
):
    """
    Genera y guarda gráficos de residuales con puntos de cambio.

    Args:
        residuals: DataFrame con los residuales para cada par de sensores.
        change_points: Diccionario con los puntos de cambio para cada par.
        output_dir: Directorio para guardar los gráficos.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Generando gráficos en el directorio: {output_dir}")

    for column in residuals.columns:
        # Utilizar regex para extraer Pressure_X y Pressure_Y
        match = re.match(r"^(Pressure_\d+)_(Pressure_\d+)$", column)
        if match:
            sensor_x, sensor_y = match.groups()
        else:
            print(f"Formato de columna inesperado: {column}")
            sensor_x, sensor_y = column, None

        if sensor_y is None:
            print(
                f"No se pudo extraer correctamente los sensores de la columna: {column}"
            )
            continue  # Saltar a la siguiente columna

        print(f"Generando gráfico para el par: {sensor_x}, {sensor_y}")

        plt.figure(figsize=(15, 5))
        plt.plot(residuals.index, residuals[column], label="Residual")

        # Añadir puntos de cambio si existen
        if (sensor_x, sensor_y) in change_points:
            cps = change_points[(sensor_x, sensor_y)]
            for cp in cps:
                if cp < len(residuals.index):
                    plt.axvline(
                        x=residuals.index[cp],
                        color="r",
                        linestyle="--",
                        label="Change Point" if cp == cps[0] else "",
                    )
        else:
            print(f"No hay puntos de cambio para el par: {sensor_x}, {sensor_y}")

        plt.title(f"Residuals between {sensor_x} and {sensor_y}")
        plt.xlabel("Time")
        plt.ylabel("Residual")
        plt.legend()
        plt.tight_layout()

        # Guardar el gráfico
        file_name = f"residual_{sensor_x}_{sensor_y}.png"
        save_path = Path(output_dir) / file_name
        plt.savefig(save_path)
        plt.close()
        print(f"Gráfico guardado en: {save_path}")


def load_change_points(change_points_path: str) -> Dict[Tuple[str, str], List[int]]:
    """
    Carga los puntos de cambio desde un archivo JSON.

    Args:
        change_points_path: Ruta al archivo JSON.

    Returns:
        Diccionario con puntos de cambio.
    """
    with open(change_points_path, "r") as f:
        data = json.load(f)
    change_points = {}
    for key, value in data.items():
        sensors = key.split("_")
        if len(sensors) == 2:
            sensor_x, sensor_y = sensors
            change_points[(sensor_x, sensor_y)] = value
        else:
            sensor_x = sensors[0]
            change_points[(sensor_x, None)] = value
    return change_points
