import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List


class SCADADataset:
    """
    Class to handle SCADA data from hydraulic simulations or real measurements.
    Similar to LILA's SCADA_data class but with additional functionality.
    """

    def __init__(self, data_dir: str = "../data/raw"):
        self.data_dir = Path(data_dir)
        self.pressures: Optional[pd.DataFrame] = None
        self.flows: Optional[pd.DataFrame] = None
        self.demands: Optional[pd.DataFrame] = None
        self.levels: Optional[pd.DataFrame] = None
        self.time_step: Optional[float] = None

    def load_hydraulic_results(self, filename: str, time_unit: str = "hours") -> None:
        """
        Load hydraulic results from EPANET simulation

        Args:
            filename: Name of the CSV file with hydraulic results
            time_unit: Unit for time conversion ('hours' or 'minutes')
        """
        # Read the full dataset
        df = pd.read_csv(self.data_dir / filename)

        # Convert time based on specified unit
        time_div = 3600 if time_unit == "hours" else 60
        df["Time"] = df["Time"] / time_div

        # Separate data into corresponding attributes
        pressure_cols = [col for col in df.columns if "Pressure" in col]
        flow_cols = [col for col in df.columns if "Flow" in col]
        demand_cols = [col for col in df.columns if "Demand" in col]
        level_cols = [col for col in df.columns if "Head" in col]

        self.pressures = df[["Time"] + pressure_cols].set_index("Time")
        self.flows = df[["Time"] + flow_cols].set_index("Time")
        self.demands = df[["Time"] + demand_cols].set_index("Time")
        self.levels = df[["Time"] + level_cols].set_index("Time")

        # Calculate time step
        self.time_step = df["Time"].diff().mean()

    def get_pressure_nodes(self) -> List[str]:
        """Return list of pressure node names"""
        return list(self.pressures.columns)

    def get_time_range(self) -> tuple:
        """Return start and end times of the dataset"""
        return (self.pressures.index.min(), self.pressures.index.max())

    def validate_data(self) -> Dict[str, bool]:
        """
        Perform basic validation checks on the data

        Returns:
            Dictionary with validation results
        """
        validations = {
            "has_pressure_data": self.pressures is not None
            and not self.pressures.empty,
            "has_flow_data": self.flows is not None and not self.flows.empty,
            "has_demand_data": self.demands is not None and not self.demands.empty,
            "has_level_data": self.levels is not None and not self.levels.empty,
            "no_missing_values": not (
                self.pressures.isnull().any().any()
                if self.pressures is not None
                else True
            ),
        }
        return validations


def load_dataset(
    data_path: str = "../data/raw", filename: str = "epanet.csv"
) -> SCADADataset:
    """
    Helper function to load and initialize dataset

    Args:
        data_path: Path to data directory
        filename: Name of the hydraulic results file

    Returns:
        Initialized SCADADataset object
    """
    dataset = SCADADataset(data_path)
    dataset.load_hydraulic_results(filename)
    return dataset


if __name__ == "__main__":
    # Example usage
    dataset = load_dataset()
    print("Available pressure nodes:", dataset.get_pressure_nodes())
    print("Time range:", dataset.get_time_range())
    print("Data validation:", dataset.validate_data())
