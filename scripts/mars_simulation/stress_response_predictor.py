"""
Bacterial Stress Response Predictor
Models B. subtilis responses to Mars environmental stressors
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import joblib
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StressResponseParams:
    """Parameters for B. subtilis stress response modeling"""
    # Cold shock parameters
    cold_shock_threshold: float = 15.0  # °C
    cold_shock_proteins: List[str] = None
    
    # Osmotic stress parameters
    osmotic_stress_threshold: float = 0.8  # Water activity
    compatible_solutes: List[str] = None
    
    # Oxidative stress parameters
    oxidative_stress_threshold: float = 1.0  # UV intensity
    antioxidant_enzymes: List[str] = None
    
    # Perchlorate stress parameters
    perchlorate_toxicity_threshold: float = 0.005  # 0.5% by weight
    perchlorate_reduction_efficiency: float = 0.85
    
    def __post_init__(self):
        if self.cold_shock_proteins is None:
            self.cold_shock_proteins = ['CspB', 'CspC', 'CspD', 'DesK', 'DesR']
        if self.compatible_solutes is None:
            self.compatible_solutes = ['Proline', 'Glycine betaine', 'Ectoine']
        if self.antioxidant_enzymes is None:
            self.antioxidant_enzymes = ['SodA', 'KatA', 'AhpC', 'BdbD']

class StressResponsePredictor:
    """
    Predicts B. subtilis stress responses to Mars environmental conditions
    """
    
    def __init__(self, params: Optional[StressResponseParams] = None):
        self.params = params or StressResponseParams()
        self.models = {}
        self.scalers = {}
        self.feature_names = [
            'temperature_c', 'pressure_kpa', 'uv_intensity', 'water_activity',
            'perchlorate_concentration', 'cosmic_radiation', 'relative_humidity'
        ]
        
    def generate_training_data(self, n_samples: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic training data based on experimental B. subtilis studies
        
        Args:
            n_samples: Number of training samples to generate
            
        Returns:
            Tuple of (features, targets) DataFrames
        """
        np.random.seed(42)  # For reproducible results
        
        # Generate environmental conditions
        features = pd.DataFrame({
            'temperature_c': np.random.uniform(-80, 40, n_samples),
            'pressure_kpa': np.random.uniform(0.1, 101.325, n_samples),
            'uv_intensity': np.random.uniform(0, 3, n_samples),
            'water_activity': np.random.uniform(0.1, 1.0, n_samples),
            'perchlorate_concentration': np.random.uniform(0, 0.02, n_samples),
            'cosmic_radiation': np.random.uniform(0, 4, n_samples),
            'relative_humidity': np.random.uniform(0, 100, n_samples)
        })
        
        # Generate stress response targets based on experimental data
        targets = self._calculate_stress_responses(features)
        
        return features, targets
    
    def _calculate_stress_responses(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate stress responses based on experimental B. subtilis data
        
        Args:
            features: Environmental conditions DataFrame
            
        Returns:
            DataFrame with predicted stress responses
        """
        targets = pd.DataFrame()
        
        # Cold shock response (survival probability)
        optimal_temp = 30.0
        temp_deviation = np.abs(features['temperature_c'] - optimal_temp)
        cold_shock_response = np.exp(-temp_deviation / 20.0)  # Exponential decay
        targets['cold_shock_survival'] = np.clip(cold_shock_response, 0.01, 1.0)
        
        # Osmotic stress response (growth rate)
        osmotic_stress = 1 - features['water_activity']
        osmotic_response = 1 - (osmotic_stress ** 2)  # Quadratic response
        targets['osmotic_stress_growth'] = np.clip(osmotic_response, 0.0, 1.0)
        
        # Oxidative stress response (antioxidant activity)
        oxidative_stress = features['uv_intensity'] + features['cosmic_radiation']
        antioxidant_response = 1 / (1 + oxidative_stress)  # Inverse relationship
        targets['oxidative_stress_response'] = np.clip(antioxidant_response, 0.1, 1.0)
        
        # Perchlorate stress response (reduction efficiency)
        perchlorate_stress = features['perchlorate_concentration'] / 0.01
        perchlorate_response = self.params.perchlorate_reduction_efficiency * np.exp(-perchlorate_stress)
        targets['perchlorate_reduction'] = np.clip(perchlorate_response, 0.0, 1.0)
        
        # Combined stress response (overall survival)
        combined_stress = (
            (1 - targets['cold_shock_survival']) * 0.3 +
            (1 - targets['osmotic_stress_growth']) * 0.2 +
            (1 - targets['oxidative_stress_response']) * 0.3 +
            (1 - targets['perchlorate_reduction']) * 0.2
        )
        targets['overall_survival'] = np.clip(1 - combined_stress, 0.01, 1.0)
        
        # Metabolic activity (growth rate)
        metabolic_activity = (
            targets['cold_shock_survival'] * 0.4 +
            targets['osmotic_stress_growth'] * 0.3 +
            targets['oxidative_stress_response'] * 0.3
        )
        targets['metabolic_activity'] = metabolic_activity
        
        # Stress protein expression levels
        targets['cold_shock_proteins'] = 1 - targets['cold_shock_survival']
        targets['osmotic_stress_proteins'] = 1 - targets['osmotic_stress_growth']
        targets['antioxidant_proteins'] = 1 - targets['oxidative_stress_response']
        
        return targets
    
    def train_models(self, features: pd.DataFrame, targets: pd.DataFrame) -> None:
        """
        Train machine learning models for stress response prediction
        
        Args:
            features: Environmental conditions DataFrame
            targets: Stress response targets DataFrame
        """
        logger.info("Training stress response prediction models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['feature_scaler'] = scaler
        
        # Train models for each target
        target_columns = targets.columns
        self.models = {}
        
        for target in target_columns:
            logger.info(f"Training model for {target}...")
            
            # Use ensemble of models for better prediction
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
            
            # Train and evaluate each model
            best_model = None
            best_score = -np.inf
            
            for name, model in models.items():
                model.fit(X_train_scaled, y_train[target])
                score = model.score(X_test_scaled, y_test[target])
                
                if score > best_score:
                    best_score = score
                    best_model = model
            
            self.models[target] = best_model
            
            # Print model performance
            y_pred = best_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test[target], y_pred)
            r2 = r2_score(y_test[target], y_pred)
            
            logger.info(f"{target}: R² = {r2:.3f}, MSE = {mse:.4f}")
    
    def predict_stress_responses(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict stress responses for given environmental conditions
        
        Args:
            features: Environmental conditions DataFrame
            
        Returns:
            DataFrame with predicted stress responses
        """
        if not self.models:
            raise ValueError("Models must be trained before making predictions")
        
        # Scale features
        X_scaled = self.scalers['feature_scaler'].transform(features[self.feature_names])
        
        # Make predictions
        predictions = pd.DataFrame()
        
        for target, model in self.models.items():
            predictions[target] = model.predict(X_scaled)
        
        # Ensure predictions are within valid ranges
        predictions = predictions.clip(0, 1)
        
        return predictions
    
    def analyze_stress_interactions(self, features: pd.DataFrame, 
                                 predictions: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze interactions between different stress factors
        
        Args:
            features: Environmental conditions
            predictions: Predicted stress responses
            
        Returns:
            Dictionary of stress interaction metrics
        """
        # Calculate correlation matrix
        stress_factors = [
            'cold_shock_survival', 'osmotic_stress_growth', 
            'oxidative_stress_response', 'perchlorate_reduction'
        ]
        
        correlation_matrix = predictions[stress_factors].corr()
        
        # Calculate stress synergy/antagonism
        interactions = {}
        
        # Cold shock + osmotic stress interaction
        cold_osmotic_corr = correlation_matrix.loc['cold_shock_survival', 'osmotic_stress_growth']
        interactions['cold_osmotic_interaction'] = cold_osmotic_corr
        
        # Oxidative + perchlorate stress interaction
        oxidative_perchlorate_corr = correlation_matrix.loc['oxidative_stress_response', 'perchlorate_reduction']
        interactions['oxidative_perchlorate_interaction'] = oxidative_perchlorate_corr
        
        # Overall stress synergy
        stress_synergy = correlation_matrix.mean().mean()
        interactions['overall_stress_synergy'] = stress_synergy
        
        return interactions
    
    def plot_stress_responses(self, features: pd.DataFrame, 
                            predictions: pd.DataFrame,
                            save_path: Optional[str] = None) -> None:
        """
        Plot stress response predictions
        
        Args:
            features: Environmental conditions
            predictions: Predicted stress responses
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('B. subtilis Stress Response Predictions', fontsize=16)
        
        # Temperature vs Cold Shock Survival
        axes[0, 0].scatter(features['temperature_c'], predictions['cold_shock_survival'], alpha=0.6)
        axes[0, 0].set_xlabel('Temperature (°C)')
        axes[0, 0].set_ylabel('Cold Shock Survival')
        axes[0, 0].set_title('Temperature vs Cold Shock Response')
        axes[0, 0].grid(True)
        
        # Water Activity vs Osmotic Stress
        axes[0, 1].scatter(features['water_activity'], predictions['osmotic_stress_growth'], alpha=0.6)
        axes[0, 1].set_xlabel('Water Activity')
        axes[0, 1].set_ylabel('Osmotic Stress Growth')
        axes[0, 1].set_title('Water Activity vs Osmotic Stress')
        axes[0, 1].grid(True)
        
        # UV Intensity vs Oxidative Stress
        axes[0, 2].scatter(features['uv_intensity'], predictions['oxidative_stress_response'], alpha=0.6)
        axes[0, 2].set_xlabel('UV Intensity')
        axes[0, 2].set_ylabel('Oxidative Stress Response')
        axes[0, 2].set_title('UV vs Oxidative Stress')
        axes[0, 2].grid(True)
        
        # Perchlorate vs Reduction Efficiency
        axes[1, 0].scatter(features['perchlorate_concentration'], predictions['perchlorate_reduction'], alpha=0.6)
        axes[1, 0].set_xlabel('Perchlorate Concentration (%)')
        axes[1, 0].set_ylabel('Perchlorate Reduction')
        axes[1, 0].set_title('Perchlorate vs Reduction Efficiency')
        axes[1, 0].grid(True)
        
        # Overall Survival Distribution
        axes[1, 1].hist(predictions['overall_survival'], bins=30, alpha=0.7)
        axes[1, 1].set_xlabel('Overall Survival Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Overall Survival Distribution')
        axes[1, 1].grid(True)
        
        # Metabolic Activity vs Overall Survival
        axes[1, 2].scatter(predictions['metabolic_activity'], predictions['overall_survival'], alpha=0.6)
        axes[1, 2].set_xlabel('Metabolic Activity')
        axes[1, 2].set_ylabel('Overall Survival')
        axes[1, 2].set_title('Metabolic Activity vs Survival')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self, filepath: str) -> None:
        """
        Save trained models to disk
        
        Args:
            filepath: Path to save the models
        """
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'params': self.params
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
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.feature_names = model_data['feature_names']
        self.params = model_data['params']
        logger.info(f"Models loaded from {filepath}")

def main():
    """Main function to demonstrate stress response prediction"""
    
    # Initialize predictor
    predictor = StressResponsePredictor()
    
    # Generate training data
    logger.info("Generating training data...")
    features, targets = predictor.generate_training_data(n_samples=5000)
    
    # Train models
    predictor.train_models(features, targets)
    
    # Generate test predictions
    test_features = features.sample(n=1000, random_state=42)
    predictions = predictor.predict_stress_responses(test_features)
    
    # Analyze stress interactions
    interactions = predictor.analyze_stress_interactions(test_features, predictions)
    
    # Plot results
    predictor.plot_stress_responses(test_features, predictions)
    
    # Save models
    predictor.save_models('models/stress_response_models.pkl')
    
    # Print summary statistics
    print("\n=== Stress Response Summary ===")
    print(f"Average cold shock survival: {predictions['cold_shock_survival'].mean():.3f}")
    print(f"Average osmotic stress growth: {predictions['osmotic_stress_growth'].mean():.3f}")
    print(f"Average oxidative stress response: {predictions['oxidative_stress_response'].mean():.3f}")
    print(f"Average perchlorate reduction: {predictions['perchlorate_reduction'].mean():.3f}")
    print(f"Average overall survival: {predictions['overall_survival'].mean():.3f}")
    print(f"Average metabolic activity: {predictions['metabolic_activity'].mean():.3f}")
    
    print("\n=== Stress Interactions ===")
    for interaction, value in interactions.items():
        print(f"{interaction}: {value:.3f}")

if __name__ == "__main__":
    main()