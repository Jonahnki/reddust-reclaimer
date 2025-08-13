"""
Mars Environment Simulator
Comprehensive modeling of Martian environmental conditions for bacterial stress testing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarsEnvironmentParams:
    """Mars environmental parameters based on scientific data"""

    # Temperature parameters (°C)
    temp_min: float = -80.0
    temp_max: float = 20.0
    temp_mean: float = -63.0

    # Atmospheric parameters
    pressure_kpa: float = 0.6  # 0.6% of Earth's atmospheric pressure
    co2_concentration: float = 95.0  # % CO2 in atmosphere
    n2_concentration: float = 2.7  # % N2
    ar_concentration: float = 1.6  # % Ar

    # Radiation parameters
    uv_intensity: float = 1.5  # Relative to Earth (higher on Mars)
    cosmic_radiation: float = 2.0  # Relative to Earth

    # Soil composition
    perchlorate_concentration: float = 0.007  # 0.7% by weight in regolith
    iron_oxide_concentration: float = 18.0  # % Fe2O3
    silica_concentration: float = 45.0  # % SiO2
    alumina_concentration: float = 15.0  # % Al2O3

    # Water availability
    water_activity: float = 0.3  # Very low water activity
    relative_humidity: float = 5.0  # % RH

    # Gravity
    gravity: float = 0.38  # Relative to Earth (38% of Earth's gravity)


class MarsEnvironmentSimulator:
    """
    Comprehensive Mars environment simulator for bacterial stress testing
    """

    def __init__(self, params: Optional[MarsEnvironmentParams] = None):
        self.params = params or MarsEnvironmentParams()
        self.sol_length = 24.6  # Martian day length in hours
        self.year_length = 687  # Martian year length in Earth days

    def generate_diurnal_cycle(
        self, sols: int = 30, resolution: str = "1H"
    ) -> pd.DataFrame:
        """
        Generate realistic Martian diurnal temperature cycles

        Args:
            sols: Number of Martian days to simulate
            resolution: Time resolution ('1H', '30min', etc.)

        Returns:
            DataFrame with time series of environmental parameters
        """
        # Generate time series
        hours_per_sol = 24.6
        total_hours = int(sols * hours_per_sol)

        # Create time index
        time_index = pd.date_range(start=datetime.now(), periods=total_hours, freq="1H")

        # Generate temperature cycles with realistic variations
        temperatures = []
        pressures = []
        uv_levels = []

        for i, time in enumerate(time_index):
            # Diurnal temperature cycle with noise
            hour_of_sol = i % int(hours_per_sol)
            phase = 2 * np.pi * hour_of_sol / hours_per_sol

            # Base temperature cycle
            base_temp = self.params.temp_mean + (
                (self.params.temp_max - self.params.temp_min) / 2
            ) * np.sin(
                phase - np.pi / 2
            )  # Peak at noon

            # Add random variations (±5°C)
            temp_variation = np.random.normal(0, 2.5)
            temperature = base_temp + temp_variation

            # Clamp to realistic bounds
            temperature = np.clip(
                temperature, self.params.temp_min, self.params.temp_max
            )
            temperatures.append(temperature)

            # Atmospheric pressure (relatively constant with slight variations)
            pressure = self.params.pressure_kpa + np.random.normal(0, 0.05)
            pressures.append(pressure)

            # UV radiation (higher during day, zero at night)
            if 6 <= hour_of_sol <= 18:  # Daytime
                uv_base = self.params.uv_intensity * np.sin(
                    np.pi * (hour_of_sol - 6) / 12
                )
                uv_level = uv_base + np.random.normal(0, 0.1)
            else:
                uv_level = 0.0
            uv_levels.append(max(0, uv_level))

        # Create comprehensive environment DataFrame
        env_data = pd.DataFrame(
            {
                "timestamp": time_index,
                "temperature_c": temperatures,
                "pressure_kpa": pressures,
                "uv_intensity": uv_levels,
                "cosmic_radiation": [self.params.cosmic_radiation] * len(time_index),
                "water_activity": [self.params.water_activity] * len(time_index),
                "relative_humidity": [self.params.relative_humidity] * len(time_index),
                "gravity": [self.params.gravity] * len(time_index),
            }
        )

        # Add soil composition (constant for simulation period)
        env_data["perchlorate_concentration"] = self.params.perchlorate_concentration
        env_data["iron_oxide_concentration"] = self.params.iron_oxide_concentration
        env_data["silica_concentration"] = self.params.silica_concentration
        env_data["alumina_concentration"] = self.params.alumina_concentration

        return env_data

    def calculate_stress_factors(self, env_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate bacterial stress factors based on environmental conditions

        Args:
            env_data: Environmental conditions DataFrame

        Returns:
            DataFrame with calculated stress factors
        """
        stress_data = env_data.copy()

        # Temperature stress (optimal range: 20-40°C for B. subtilis)
        optimal_temp = 30.0
        temp_stress = np.abs(stress_data["temperature_c"] - optimal_temp) / 50.0
        stress_data["temperature_stress"] = np.clip(temp_stress, 0, 1)

        # Pressure stress (very low pressure is stressful)
        pressure_stress = 1 - (
            stress_data["pressure_kpa"] / 101.325
        )  # Normalized to Earth pressure
        stress_data["pressure_stress"] = np.clip(pressure_stress, 0, 1)

        # UV radiation stress
        uv_stress = stress_data["uv_intensity"] / 2.0  # Normalized
        stress_data["uv_stress"] = np.clip(uv_stress, 0, 1)

        # Water stress (low water activity is stressful)
        water_stress = 1 - stress_data["water_activity"]
        stress_data["water_stress"] = np.clip(water_stress, 0, 1)

        # Perchlorate stress (toxic compound)
        perchlorate_stress = (
            stress_data["perchlorate_concentration"] / 0.01
        )  # Normalized to 1% threshold
        stress_data["perchlorate_stress"] = np.clip(perchlorate_stress, 0, 1)

        # Combined stress index
        stress_factors = [
            "temperature_stress",
            "pressure_stress",
            "uv_stress",
            "water_stress",
            "perchlorate_stress",
        ]
        stress_data["combined_stress"] = stress_data[stress_factors].mean(axis=1)

        return stress_data

    def simulate_seasonal_variations(self, earth_years: int = 2) -> pd.DataFrame:
        """
        Simulate seasonal variations over multiple Earth years

        Args:
            earth_years: Number of Earth years to simulate

        Returns:
            DataFrame with seasonal environmental variations
        """
        # Convert to Martian years
        mars_years = earth_years * (365.25 / 687)
        sols = int(mars_years * 687)

        # Generate base diurnal cycles
        env_data = self.generate_diurnal_cycle(sols=sols)

        # Add seasonal temperature variations
        days = np.arange(len(env_data))
        seasonal_variation = 10 * np.sin(
            2 * np.pi * days / (687 * 24.6)
        )  # 687 sols per Mars year

        env_data["temperature_c"] += seasonal_variation

        # Add seasonal dust storm effects (reduced UV during storms)
        dust_storm_probability = 0.1  # 10% chance of dust storm per sol
        dust_storms = np.random.random(len(env_data)) < dust_storm_probability

        # Dust storms reduce UV and temperature
        env_data.loc[dust_storms, "uv_intensity"] *= 0.3
        env_data.loc[dust_storms, "temperature_c"] -= 5

        return env_data

    def plot_environmental_conditions(
        self, env_data: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """
        Plot environmental conditions over time

        Args:
            env_data: Environmental data DataFrame
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle("Mars Environmental Conditions Simulation", fontsize=16)

        # Temperature
        axes[0, 0].plot(env_data["timestamp"], env_data["temperature_c"])
        axes[0, 0].set_title("Temperature (°C)")
        axes[0, 0].set_ylabel("Temperature (°C)")
        axes[0, 0].grid(True)

        # Pressure
        axes[0, 1].plot(env_data["timestamp"], env_data["pressure_kpa"])
        axes[0, 1].set_title("Atmospheric Pressure (kPa)")
        axes[0, 1].set_ylabel("Pressure (kPa)")
        axes[0, 1].grid(True)

        # UV Radiation
        axes[1, 0].plot(env_data["timestamp"], env_data["uv_intensity"])
        axes[1, 0].set_title("UV Radiation Intensity")
        axes[1, 0].set_ylabel("UV Intensity (relative)")
        axes[1, 0].grid(True)

        # Water Activity
        axes[1, 1].plot(env_data["timestamp"], env_data["water_activity"])
        axes[1, 1].set_title("Water Activity")
        axes[1, 1].set_ylabel("Water Activity")
        axes[1, 1].grid(True)

        # Combined Stress
        stress_data = self.calculate_stress_factors(env_data)
        axes[2, 0].plot(stress_data["timestamp"], stress_data["combined_stress"])
        axes[2, 0].set_title("Combined Stress Index")
        axes[2, 0].set_ylabel("Stress Index")
        axes[2, 0].grid(True)

        # Perchlorate Concentration
        axes[2, 1].plot(env_data["timestamp"], env_data["perchlorate_concentration"])
        axes[2, 1].set_title("Perchlorate Concentration")
        axes[2, 1].set_ylabel("Concentration (weight %)")
        axes[2, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def export_environment_data(self, env_data: pd.DataFrame, filepath: str) -> None:
        """
        Export environmental data to CSV for further analysis

        Args:
            env_data: Environmental data DataFrame
            filepath: Path to save the CSV file
        """
        env_data.to_csv(filepath, index=False)
        logger.info(f"Environment data exported to {filepath}")

    def get_extreme_conditions(self) -> Dict[str, float]:
        """
        Get extreme Mars conditions for stress testing

        Returns:
            Dictionary of extreme environmental parameters
        """
        return {
            "temperature_c": self.params.temp_min,
            "pressure_kpa": self.params.pressure_kpa,
            "uv_intensity": self.params.uv_intensity * 1.5,  # Enhanced UV
            "water_activity": 0.1,  # Very low water activity
            "perchlorate_concentration": 0.01,  # 1% perchlorate
            "cosmic_radiation": self.params.cosmic_radiation * 2.0,
            "relative_humidity": 1.0,  # Minimal humidity
        }


def main():
    """Main function to demonstrate Mars environment simulation"""

    # Initialize simulator
    simulator = MarsEnvironmentSimulator()

    # Generate 30 sols of environmental data
    logger.info("Generating Mars environmental conditions...")
    env_data = simulator.generate_diurnal_cycle(sols=30)

    # Calculate stress factors
    stress_data = simulator.calculate_stress_factors(env_data)

    # Plot results
    simulator.plot_environmental_conditions(stress_data)

    # Export data
    simulator.export_environment_data(stress_data, "data/mars_environment_30sols.csv")

    # Generate seasonal data
    logger.info("Generating seasonal variations...")
    seasonal_data = simulator.simulate_seasonal_variations(earth_years=1)
    seasonal_stress = simulator.calculate_stress_factors(seasonal_data)

    # Export seasonal data
    simulator.export_environment_data(
        seasonal_stress, "data/mars_environment_seasonal.csv"
    )

    # Print summary statistics
    print("\n=== Mars Environment Summary ===")
    print(
        f"Temperature range: {env_data['temperature_c'].min():.1f}°C to {env_data['temperature_c'].max():.1f}°C"
    )
    print(f"Average pressure: {env_data['pressure_kpa'].mean():.3f} kPa")
    print(f"Average UV intensity: {env_data['uv_intensity'].mean():.3f}")
    print(f"Average combined stress: {stress_data['combined_stress'].mean():.3f}")
    print(
        f"Perchlorate concentration: {env_data['perchlorate_concentration'].iloc[0]:.3f}%"
    )


if __name__ == "__main__":
    main()
