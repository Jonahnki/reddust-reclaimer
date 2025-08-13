"""
Performance Predictor with Ensemble ML Models
Predicts B. subtilis performance under Mars conditions with uncertainty quantification
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import joblib
import logging
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for B. subtilis under Mars conditions"""

    growth_rate: float = 0.0  # Doubling time (hours)
    survival_probability: float = 0.0  # Survival probability
    perchlorate_reduction: float = 0.0  # Reduction efficiency
    metabolic_activity: float = 0.0  # Metabolic activity level
    stress_resistance: float = 0.0  # Overall stress resistance
    remediation_efficiency: float = 0.0  # Overall remediation efficiency


class EnsemblePerformancePredictor:
    """
    Ensemble ML predictor for B. subtilis performance under Mars conditions
    """

    def __init__(self, n_models: int = 5):
        self.n_models = n_models
        self.models = {}
        self.scalers = {}
        self.feature_names = [
            "temperature_c",
            "pressure_kpa",
            "uv_intensity",
            "water_activity",
            "perchlorate_concentration",
            "cosmic_radiation",
            "relative_humidity",
            "growth_rate",
            "perchlorate_reduction_rate",
            "stress_tolerance",
            "cold_shock_proteins",
            "osmotic_stress_proteins",
            "antioxidant_proteins",
            "perchlorate_reductase",
            "glycolysis_efficiency",
            "tca_cycle_efficiency",
            "oxidative_phosphorylation",
            "energy_allocation_growth",
            "energy_allocation_stress",
            "energy_allocation_remediation",
        ]
        self.target_names = [
            "growth_rate_pred",
            "survival_probability",
            "perchlorate_reduction",
            "metabolic_activity",
            "stress_resistance",
            "remediation_efficiency",
        ]

    def generate_training_data(
        self, n_samples: int = 15000
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate comprehensive training data for performance prediction

        Args:
            n_samples: Number of training samples

        Returns:
            Tuple of (features, targets) DataFrames
        """
        np.random.seed(42)

        # Generate environmental conditions
        features = pd.DataFrame(
            {
                "temperature_c": np.random.uniform(-80, 40, n_samples),
                "pressure_kpa": np.random.uniform(0.1, 101.325, n_samples),
                "uv_intensity": np.random.uniform(0, 3, n_samples),
                "water_activity": np.random.uniform(0.1, 1.0, n_samples),
                "perchlorate_concentration": np.random.uniform(0, 0.02, n_samples),
                "cosmic_radiation": np.random.uniform(0, 4, n_samples),
                "relative_humidity": np.random.uniform(0, 100, n_samples),
            }
        )

        # Generate strain characteristics
        strain_features = pd.DataFrame(
            {
                "growth_rate": np.random.uniform(0.1, 1.0, n_samples),
                "perchlorate_reduction_rate": np.random.uniform(0.1, 1.0, n_samples),
                "stress_tolerance": np.random.uniform(0.1, 1.0, n_samples),
                "cold_shock_proteins": np.random.uniform(0.0, 1.0, n_samples),
                "osmotic_stress_proteins": np.random.uniform(0.0, 1.0, n_samples),
                "antioxidant_proteins": np.random.uniform(0.0, 1.0, n_samples),
                "perchlorate_reductase": np.random.uniform(0.0, 1.0, n_samples),
                "glycolysis_efficiency": np.random.uniform(0.3, 1.0, n_samples),
                "tca_cycle_efficiency": np.random.uniform(0.3, 1.0, n_samples),
                "oxidative_phosphorylation": np.random.uniform(0.3, 1.0, n_samples),
                "energy_allocation_growth": np.random.uniform(0.2, 0.5, n_samples),
                "energy_allocation_stress": np.random.uniform(0.2, 0.5, n_samples),
                "energy_allocation_remediation": np.random.uniform(0.2, 0.5, n_samples),
            }
        )

        # Combine features
        all_features = pd.concat([features, strain_features], axis=1)

        # Generate performance targets
        targets = self._calculate_performance_targets(all_features)

        return all_features, targets

    def _calculate_performance_targets(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance targets based on features

        Args:
            features: Combined environmental and strain features

        Returns:
            DataFrame with performance targets
        """
        targets = pd.DataFrame()

        # Extract key features
        temp = features["temperature_c"]
        pressure = features["pressure_kpa"]
        uv = features["uv_intensity"]
        water_activity = features["water_activity"]
        perchlorate = features["perchlorate_concentration"]

        # Strain characteristics
        growth_rate = features["growth_rate"]
        perchlorate_reduction_rate = features["perchlorate_reduction_rate"]
        stress_tolerance = features["stress_tolerance"]

        # Calculate stress factors
        temp_stress = np.abs(temp - 30.0) / 50.0
        pressure_stress = 1 - (pressure / 101.325)
        uv_stress = uv / 2.0
        water_stress = 1 - water_activity
        perchlorate_stress = perchlorate / 0.01

        # 1. Growth Rate Prediction
        growth_pred = (
            growth_rate
            * features["glycolysis_efficiency"]
            * features["energy_allocation_growth"]
            * (1 - temp_stress)
            * (1 - water_stress)
            * (1 - pressure_stress * 0.5)
        )
        targets["growth_rate_pred"] = np.clip(growth_pred, 0.0, 1.0)

        # 2. Survival Probability
        survival_prob = (
            stress_tolerance
            * (
                features["cold_shock_proteins"] * (1 - temp_stress)
                + features["osmotic_stress_proteins"] * (1 - water_stress)
                + features["antioxidant_proteins"] * (1 - uv_stress)
            )
            / 3
        )
        targets["survival_probability"] = np.clip(survival_prob, 0.01, 1.0)

        # 3. Perchlorate Reduction
        perchlorate_reduction = (
            perchlorate_reduction_rate
            * features["perchlorate_reductase"]
            * features["energy_allocation_remediation"]
            * (1 - perchlorate_stress)
            * (1 - uv_stress * 0.3)
        )
        targets["perchlorate_reduction"] = np.clip(perchlorate_reduction, 0.0, 1.0)

        # 4. Metabolic Activity
        metabolic_activity = (
            (
                features["glycolysis_efficiency"] * 0.4
                + features["tca_cycle_efficiency"] * 0.3
                + features["oxidative_phosphorylation"] * 0.3
            )
            * (1 - temp_stress)
            * (1 - water_stress)
        )
        targets["metabolic_activity"] = np.clip(metabolic_activity, 0.0, 1.0)

        # 5. Stress Resistance
        stress_resistance = (
            stress_tolerance
            * (1 - temp_stress)
            * (1 - pressure_stress)
            * (1 - uv_stress)
            * (1 - water_stress)
        )
        targets["stress_resistance"] = np.clip(stress_resistance, 0.0, 1.0)

        # 6. Overall Remediation Efficiency
        remediation_efficiency = (
            targets["perchlorate_reduction"] * 0.6
            + targets["metabolic_activity"] * 0.2
            + targets["stress_resistance"] * 0.2
        )
        targets["remediation_efficiency"] = np.clip(remediation_efficiency, 0.0, 1.0)

        return targets

    def train_ensemble_models(
        self, features: pd.DataFrame, targets: pd.DataFrame
    ) -> None:
        """
        Train ensemble of ML models for each target

        Args:
            features: Input features DataFrame
            targets: Target variables DataFrame
        """
        logger.info("Training ensemble performance prediction models...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers["feature_scaler"] = scaler

        # Train models for each target
        self.models = {}

        for target in self.target_names:
            logger.info(f"Training ensemble for {target}...")

            target_models = []

            # Train multiple models for ensemble
            for i in range(self.n_models):
                # Random Forest
                rf_model = RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42 + i
                )
                rf_model.fit(X_train_scaled, y_train[target])
                target_models.append(("rf", rf_model))

                # Gradient Boosting
                gb_model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=6, random_state=42 + i
                )
                gb_model.fit(X_train_scaled, y_train[target])
                target_models.append(("gb", gb_model))

                # XGBoost
                xgb_model = xgb.XGBRegressor(
                    n_estimators=100, max_depth=6, random_state=42 + i
                )
                xgb_model.fit(X_train_scaled, y_train[target])
                target_models.append(("xgb", xgb_model))

                # Neural Network
                nn_model = MLPRegressor(
                    hidden_layer_sizes=(100, 50), max_iter=500, random_state=42 + i
                )
                nn_model.fit(X_train_scaled, y_train[target])
                target_models.append(("nn", nn_model))

            self.models[target] = target_models

            # Evaluate ensemble performance
            ensemble_pred = self._ensemble_predict(X_test_scaled, target_models)
            r2 = r2_score(y_test[target], ensemble_pred)
            mse = mean_squared_error(y_test[target], ensemble_pred)

            logger.info(f"{target}: R² = {r2:.3f}, MSE = {mse:.4f}")

    def _ensemble_predict(
        self, X: np.ndarray, models: List[Tuple[str, object]]
    ) -> np.ndarray:
        """
        Make ensemble prediction using multiple models

        Args:
            X: Input features
            models: List of (model_name, model) tuples

        Returns:
            Ensemble prediction
        """
        predictions = []

        for _, model in models:
            pred = model.predict(X)
            predictions.append(pred)

        # Return mean prediction
        return np.mean(predictions, axis=0)

    def predict_performance(
        self, features: pd.DataFrame, return_uncertainty: bool = True
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Predict performance with uncertainty quantification

        Args:
            features: Input features DataFrame
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Performance predictions DataFrame and optionally uncertainty estimates
        """
        if not self.models:
            raise ValueError("Models must be trained before making predictions")

        # Scale features
        X_scaled = self.scalers["feature_scaler"].transform(
            features[self.feature_names]
        )

        # Make predictions
        predictions = pd.DataFrame()
        uncertainties = pd.DataFrame()

        for target in self.target_names:
            target_models = self.models[target]

            # Get individual model predictions
            model_predictions = []
            for _, model in target_models:
                pred = model.predict(X_scaled)
                model_predictions.append(pred)

            # Calculate ensemble prediction
            ensemble_pred = np.mean(model_predictions, axis=0)
            predictions[target] = ensemble_pred

            # Calculate uncertainty (standard deviation across models)
            if return_uncertainty:
                uncertainty = np.std(model_predictions, axis=0)
                uncertainties[target] = uncertainty

        # Ensure predictions are within valid ranges
        predictions = predictions.clip(0, 1)

        if return_uncertainty:
            return predictions, uncertainties
        else:
            return predictions

    def calculate_confidence_intervals(
        self,
        predictions: pd.DataFrame,
        uncertainties: pd.DataFrame,
        confidence_level: float = 0.95,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate confidence intervals for predictions

        Args:
            predictions: Predicted values
            uncertainties: Uncertainty estimates
            confidence_level: Confidence level (0.95 for 95% CI)

        Returns:
            Dictionary of confidence intervals for each target
        """
        confidence_intervals = {}

        # Calculate z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        for target in self.target_names:
            mean_pred = predictions[target].values
            std_pred = uncertainties[target].values

            # Calculate confidence intervals
            lower_ci = mean_pred - z_score * std_pred
            upper_ci = mean_pred + z_score * std_pred

            # Clip to valid range
            lower_ci = np.clip(lower_ci, 0, 1)
            upper_ci = np.clip(upper_ci, 0, 1)

            confidence_intervals[target] = (lower_ci, upper_ci)

        return confidence_intervals

    def plot_performance_predictions(
        self,
        features: pd.DataFrame,
        predictions: pd.DataFrame,
        uncertainties: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot performance predictions with uncertainty

        Args:
            features: Input features
            predictions: Performance predictions
            uncertainties: Uncertainty estimates
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "B. subtilis Performance Predictions with Uncertainty", fontsize=16
        )

        # Plot each target
        for i, target in enumerate(self.target_names):
            row = i // 3
            col = i % 3

            # Sort by predictions for better visualization
            sorted_indices = np.argsort(predictions[target])
            sorted_pred = predictions[target].iloc[sorted_indices]

            axes[row, col].plot(sorted_pred, label="Prediction", linewidth=2)

            if uncertainties is not None:
                sorted_uncertainty = uncertainties[target].iloc[sorted_indices]
                axes[row, col].fill_between(
                    range(len(sorted_pred)),
                    sorted_pred - sorted_uncertainty,
                    sorted_pred + sorted_uncertainty,
                    alpha=0.3,
                    label="Uncertainty",
                )

            axes[row, col].set_title(f'{target.replace("_", " ").title()}')
            axes[row, col].set_xlabel("Sample Index")
            axes[row, col].set_ylabel("Predicted Value")
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def save_models(self, filepath: str) -> None:
        """
        Save trained models to disk

        Args:
            filepath: Path to save the models
        """
        model_data = {
            "models": self.models,
            "scalers": self.scalers,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")

    def load_models(self, filepath: str) -> None:
        """
        Load trained models from disk

        Args:
            filepath: Path to load the models from
        """
        model_data = joblib.load(filepath)
        self.models = model_data["models"]
        self.scalers = model_data["scalers"]
        self.feature_names = model_data["feature_names"]
        self.target_names = model_data["target_names"]
        logger.info(f"Models loaded from {filepath}")


def main():
    """Main function to demonstrate performance prediction"""

    # Initialize predictor
    predictor = EnsemblePerformancePredictor(n_models=3)

    # Generate training data
    logger.info("Generating training data...")
    features, targets = predictor.generate_training_data(n_samples=8000)

    # Train models
    predictor.train_ensemble_models(features, targets)

    # Generate test predictions
    test_features = features.sample(n=1000, random_state=42)
    predictions, uncertainties = predictor.predict_performance(
        test_features, return_uncertainty=True
    )

    # Calculate confidence intervals
    confidence_intervals = predictor.calculate_confidence_intervals(
        predictions, uncertainties
    )

    # Plot results
    predictor.plot_performance_predictions(test_features, predictions, uncertainties)

    # Save models
    predictor.save_models("models/performance_predictor_models.pkl")

    # Print summary statistics
    print("\n=== Performance Prediction Summary ===")
    for target in predictor.target_names:
        mean_pred = predictions[target].mean()
        mean_uncertainty = uncertainties[target].mean()
        print(f"{target}: {mean_pred:.3f} ± {mean_uncertainty:.3f}")

    # Print confidence interval coverage
    print("\n=== Confidence Interval Coverage (95%) ===")
    for target, (lower_ci, upper_ci) in confidence_intervals.items():
        coverage = np.mean(
            (lower_ci <= predictions[target].values)
            & (predictions[target].values <= upper_ci)
        )
        print(f"{target}: {coverage:.3f}")


if __name__ == "__main__":
    main()
