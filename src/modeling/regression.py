# src/modeling/regression.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Tuple


class PairwiseLinearRegression:
    """
    Clase para manejar regresiones lineales entre pares de sensores de presión
    y calcular métricas estadísticas.
    """

    def __init__(self, pressures: pd.DataFrame):
        self.pressures = pressures
        self.models: Dict[Tuple[str, str], LinearRegression] = {}
        self.metrics: Dict[Tuple[str, str], Dict[str, float]] = {}

    def train_models(self):
        """
        Entrena modelos de regresión lineal para cada par de sensores y calcula métricas.
        """
        sensor_names = self.pressures.columns
        for i in range(len(sensor_names)):
            for j in range(i + 1, len(sensor_names)):
                sensor_x = sensor_names[i]
                sensor_y = sensor_names[j]
                X = self.pressures[[sensor_x]].values
                y = self.pressures[sensor_y].values
                model = LinearRegression()
                model.fit(X, y)
                self.models[(sensor_x, sensor_y)] = model

                # Calcular métricas
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                mse = mean_squared_error(y, y_pred)

                self.metrics[(sensor_x, sensor_y)] = {"R2": r2, "MSE": mse}

    def get_model(self, sensor_x: str, sensor_y: str) -> LinearRegression:
        return self.models.get((sensor_x, sensor_y))

    def get_metrics(self) -> pd.DataFrame:
        """
        Retorna un DataFrame con las métricas de cada par de sensores.

        Returns:
            pd.DataFrame: DataFrame con columnas ['Sensor_X', 'Sensor_Y', 'R2', 'MSE']
        """
        data = []
        for (sensor_x, sensor_y), metric in self.metrics.items():
            data.append(
                {
                    "Sensor_X": sensor_x,
                    "Sensor_Y": sensor_y,
                    "R2": metric["R2"],
                    "MSE": metric["MSE"],
                }
            )
        return pd.DataFrame(data)

    def calculate_residuals(self) -> Dict[Tuple[str, str], pd.Series]:
        """
        Calcula los residuales para cada par de sensores.

        Returns:
            Dict[Tuple[str, str], pd.Series]: Residuales para cada par.
        """
        residuals = {}
        for (sensor_x, sensor_y), model in self.models.items():
            X = self.pressures[[sensor_x]].values
            y_true = self.pressures[sensor_y].values
            y_pred = model.predict(X)
            residual = y_true - y_pred
            residuals[(sensor_x, sensor_y)] = pd.Series(
                residual, index=self.pressures.index
            )
        return residuals
