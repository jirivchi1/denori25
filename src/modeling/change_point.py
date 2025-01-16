# src/modeling/change_point.py

import pandas as pd
import ruptures as rpt
from typing import Dict, Tuple, List


class ChangePointDetector:
    """
    Clase para detectar puntos de cambio en los residuales de los modelos de regresión.
    """

    def __init__(self, residuals: pd.DataFrame, model="l2", penalty=3):
        """
        Args:
            residuals: DataFrame con los residuales para cada par de sensores.
            model: Modelo de ruptures a utilizar.
            penalty: Penalización para el número de cambios.
        """
        self.residuals = residuals
        self.model = model
        self.penalty = penalty
        self.change_points: Dict[Tuple[str, str], List[int]] = {}

    def detect_changes(self):
        """
        Detecta puntos de cambio para cada par de sensores.
        """
        for column in self.residuals.columns:
            sensor_x, sensor_y = column
            algo = rpt.Pelt(model=self.model).fit(self.residuals[column].values)
            result = algo.predict(pen=self.penalty)
            # El método predict incluye el último punto, lo eliminamos
            result = result[:-1]
            self.change_points[(sensor_x, sensor_y)] = result

    def get_change_points(self) -> Dict[Tuple[str, str], List[int]]:
        return self.change_points
