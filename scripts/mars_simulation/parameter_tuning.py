"""
Parameter Tuning for Optimal Growth and Remediation Conditions
Optimizes B. subtilis performance parameters for Mars deployment
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationTarget:
    """Target for parameter optimization"""
    name: str
    weight: float = 1.0
    min_value: float = 0.0
    max_value: float = 1.0
    target_value: Optional[float] = None

class ParameterTuner:
    """
    Parameter tuning for optimal B. subtilis performance under Mars conditions
    """
    
    def __init__(self):
        self.best_params = {}
        self.optimization_history = []
        self.scalers = {}
        
    def define_parameter_space(self) -> Dict[str, List[float]]:
        """
        Define the parameter space for optimization
        
        Returns:
            Dictionary of parameter ranges
        """
        return {
            # Environmental conditions
            'temperature_c': [-80, 40],
            'pressure_kpa': [0.1, 101.325],
            'water_activity': [0.1, 1.0],
            'perchlorate_concentration': [0.0, 0.02],
            
            # Strain characteristics
            'growth_rate': [0.1, 1.0],
            'perchlorate_reduction_rate': [0.1, 1.0],
            'stress_tolerance': [0.1, 1.0],
            'cold_shock_proteins': [0.0, 1.0],
            'osmotic_stress_proteins': [0.0, 1.0],
            'antioxidant_proteins': [0.0, 1.0],
            'perchlorate_reductase': [0.0, 1.0],
            
            # Metabolic parameters
            'glycolysis_efficiency': [0.3, 1.0],
            'tca_cycle_efficiency': [0.3, 1.0],
            'oxidative_phosphorylation': [0.3, 1.0],
            
            # Energy allocation
            'energy_allocation_growth': [0.2, 0.5],
            'energy_allocation_stress': [0.2, 0.5],
            'energy_allocation_remediation': [0.2, 0.5]
        }
    
    def objective_function(self, params: np.ndarray, 
                          targets: List[OptimizationTarget],
                          environment_conditions: Dict[str, float]) -> float:
        """
        Objective function for optimization
        
        Args:
            params: Parameter values to evaluate
            targets: List of optimization targets
            environment_conditions: Fixed environmental conditions
            
        Returns:
            Objective function value (to be minimized)
        """
        # Convert parameters to dictionary
        param_names = list(self.define_parameter_space().keys())
        param_dict = dict(zip(param_names, params))
        
        # Combine with environment conditions
        all_conditions = {**environment_conditions, **param_dict}
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(all_conditions)
        
        # Calculate objective value
        objective = 0.0
        
        for target in targets:
            if target.name in performance:
                actual_value = performance[target.name]
                
                if target.target_value is not None:
                    # Minimize distance to target
                    error = abs(actual_value - target.target_value)
                else:
                    # Maximize the value (minimize negative)
                    error = -actual_value
                
                objective += target.weight * error
        
        return objective
    
    def _calculate_performance_metrics(self, conditions: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate performance metrics for given conditions
        
        Args:
            conditions: Environmental and strain conditions
            
        Returns:
            Dictionary of performance metrics
        """
        # Extract key parameters
        temp = conditions.get('temperature_c', -63.0)
        pressure = conditions.get('pressure_kpa', 0.6)
        water_activity = conditions.get('water_activity', 0.3)
        perchlorate = conditions.get('perchlorate_concentration', 0.007)
        
        growth_rate = conditions.get('growth_rate', 0.5)
        perchlorate_reduction_rate = conditions.get('perchlorate_reduction_rate', 0.3)
        stress_tolerance = conditions.get('stress_tolerance', 0.5)
        
        # Calculate stress factors
        temp_stress = abs(temp - 30.0) / 50.0
        pressure_stress = 1 - (pressure / 101.325)
        water_stress = 1 - water_activity
        perchlorate_stress = perchlorate / 0.01
        
        # Calculate performance metrics
        performance = {}
        
        # Growth performance
        performance['growth_rate'] = (
            growth_rate * 
            conditions.get('glycolysis_efficiency', 0.5) * 
            conditions.get('energy_allocation_growth', 0.33) * 
            (1 - temp_stress) * 
            (1 - water_stress)
        )
        
        # Survival probability
        performance['survival_probability'] = (
            stress_tolerance * 
            (conditions.get('cold_shock_proteins', 0.5) * (1 - temp_stress) +
             conditions.get('osmotic_stress_proteins', 0.5) * (1 - water_stress)) / 2
        )
        
        # Perchlorate reduction
        performance['perchlorate_reduction'] = (
            perchlorate_reduction_rate * 
            conditions.get('perchlorate_reductase', 0.5) * 
            conditions.get('energy_allocation_remediation', 0.33) * 
            (1 - perchlorate_stress)
        )
        
        # Metabolic activity
        performance['metabolic_activity'] = (
            conditions.get('glycolysis_efficiency', 0.5) * 0.4 +
            conditions.get('tca_cycle_efficiency', 0.5) * 0.3 +
            conditions.get('oxidative_phosphorylation', 0.5) * 0.3
        ) * (1 - temp_stress) * (1 - water_stress)
        
        # Overall remediation efficiency
        performance['remediation_efficiency'] = (
            performance['perchlorate_reduction'] * 0.6 +
            performance['metabolic_activity'] * 0.2 +
            performance['survival_probability'] * 0.2
        )
        
        # Ensure values are in valid range
        for key in performance:
            performance[key] = np.clip(performance[key], 0.0, 1.0)
        
        return performance
    
    def optimize_parameters(self, 
                          targets: List[OptimizationTarget],
                          environment_conditions: Dict[str, float],
                          method: str = 'differential_evolution') -> Dict[str, float]:
        """
        Optimize parameters for given targets
        
        Args:
            targets: List of optimization targets
            environment_conditions: Fixed environmental conditions
            method: Optimization method ('differential_evolution', 'minimize')
            
        Returns:
            Dictionary of optimized parameters
        """
        logger.info(f"Starting parameter optimization using {method}...")
        
        # Define parameter bounds
        param_space = self.define_parameter_space()
        param_names = list(param_space.keys())
        bounds = [param_space[name] for name in param_names]
        
        # Initial guess
        x0 = [np.mean(bound) for bound in bounds]
        
        if method == 'differential_evolution':
            result = differential_evolution(
                lambda x: self.objective_function(x, targets, environment_conditions),
                bounds,
                maxiter=100,
                popsize=15,
                seed=42
            )
        elif method == 'minimize':
            result = minimize(
                lambda x: self.objective_function(x, targets, environment_conditions),
                x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 100}
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Convert result to parameter dictionary
        optimized_params = dict(zip(param_names, result.x))
        
        # Store best parameters
        self.best_params = optimized_params
        
        # Calculate final performance
        all_conditions = {**environment_conditions, **optimized_params}
        final_performance = self._calculate_performance_metrics(all_conditions)
        
        logger.info(f"Optimization complete. Final objective value: {result.fun:.4f}")
        
        return optimized_params, final_performance
    
    def tune_ml_hyperparameters(self, 
                               features: pd.DataFrame, 
                               targets: pd.DataFrame,
                               model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Tune hyperparameters for ML models
        
        Args:
            features: Input features
            targets: Target variables
            model_type: Type of model to tune
            
        Returns:
            Dictionary of best hyperparameters
        """
        logger.info(f"Tuning hyperparameters for {model_type}...")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        
        # Define parameter grids
        if model_type == 'random_forest':
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif model_type == 'neural_network':
            model = MLPRegressor(random_state=42, max_iter=500)
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Perform grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        # Use first target for tuning (can be extended to multi-target)
        first_target = targets.columns[0]
        grid_search.fit(X_scaled, targets[first_target])
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_params_
    
    def multi_objective_optimization(self, 
                                   targets: List[OptimizationTarget],
                                   environment_conditions: Dict[str, float],
                                   n_pareto_points: int = 20) -> List[Dict[str, float]]:
        """
        Perform multi-objective optimization to find Pareto front
        
        Args:
            targets: List of optimization targets
            environment_conditions: Fixed environmental conditions
            n_pareto_points: Number of Pareto optimal points to find
            
        Returns:
            List of Pareto optimal parameter sets
        """
        logger.info("Performing multi-objective optimization...")
        
        pareto_solutions = []
        
        # Generate different weight combinations
        for i in range(n_pareto_points):
            # Vary weights to explore Pareto front
            weights = np.random.dirichlet(np.ones(len(targets)))
            
            # Create weighted targets
            weighted_targets = []
            for j, target in enumerate(targets):
                weighted_target = OptimizationTarget(
                    name=target.name,
                    weight=weights[j],
                    min_value=target.min_value,
                    max_value=target.max_value,
                    target_value=target.target_value
                )
                weighted_targets.append(weighted_target)
            
            # Optimize with current weights
            try:
                optimized_params, performance = self.optimize_parameters(
                    weighted_targets, environment_conditions
                )
                
                pareto_solutions.append({
                    'parameters': optimized_params,
                    'performance': performance,
                    'weights': weights
                })
                
            except Exception as e:
                logger.warning(f"Optimization {i} failed: {e}")
                continue
        
        logger.info(f"Found {len(pareto_solutions)} Pareto optimal solutions")
        return pareto_solutions
    
    def plot_optimization_results(self, 
                                pareto_solutions: List[Dict[str, float]],
                                save_path: Optional[str] = None) -> None:
        """
        Plot optimization results
        
        Args:
            pareto_solutions: List of Pareto optimal solutions
            save_path: Optional path to save the plot
        """
        if not pareto_solutions:
            logger.warning("No Pareto solutions to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multi-Objective Optimization Results', fontsize=16)
        
        # Extract performance metrics
        growth_rates = [sol['performance']['growth_rate'] for sol in pareto_solutions]
        survival_probs = [sol['performance']['survival_probability'] for sol in pareto_solutions]
        remediation_effs = [sol['performance']['remediation_efficiency'] for sol in pareto_solutions]
        metabolic_acts = [sol['performance']['metabolic_activity'] for sol in pareto_solutions]
        
        # Plot Pareto front
        axes[0, 0].scatter(growth_rates, survival_probs, alpha=0.7)
        axes[0, 0].set_xlabel('Growth Rate')
        axes[0, 0].set_ylabel('Survival Probability')
        axes[0, 0].set_title('Growth vs Survival')
        axes[0, 0].grid(True)
        
        axes[0, 1].scatter(remediation_effs, metabolic_acts, alpha=0.7)
        axes[0, 1].set_xlabel('Remediation Efficiency')
        axes[0, 1].set_ylabel('Metabolic Activity')
        axes[0, 1].set_title('Remediation vs Metabolism')
        axes[0, 1].grid(True)
        
        # Plot parameter distributions
        param_names = ['growth_rate', 'perchlorate_reduction_rate', 'stress_tolerance']
        param_values = {name: [] for name in param_names}
        
        for sol in pareto_solutions:
            for name in param_names:
                if name in sol['parameters']:
                    param_values[name].append(sol['parameters'][name])
        
        # Box plot of key parameters
        axes[1, 0].boxplot([param_values[name] for name in param_names], labels=param_names)
        axes[1, 0].set_title('Parameter Distributions')
        axes[1, 0].set_ylabel('Parameter Value')
        axes[1, 0].grid(True)
        
        # Performance summary
        performance_metrics = ['growth_rate', 'survival_probability', 'remediation_efficiency']
        avg_performance = []
        
        for metric in performance_metrics:
            values = [sol['performance'][metric] for sol in pareto_solutions]
            avg_performance.append(np.mean(values))
        
        axes[1, 1].bar(performance_metrics, avg_performance)
        axes[1, 1].set_title('Average Performance Metrics')
        axes[1, 1].set_ylabel('Average Value')
        axes[1, 1].grid(True)
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_optimization_results(self, pareto_solutions: List[Dict[str, float]], 
                                filepath: str) -> None:
        """
        Save optimization results to file
        
        Args:
            pareto_solutions: List of Pareto optimal solutions
            filepath: Path to save results
        """
        import json
        
        # Convert to serializable format
        serializable_solutions = []
        for sol in pareto_solutions:
            serializable_sol = {
                'parameters': sol['parameters'],
                'performance': sol['performance'],
                'weights': sol['weights'].tolist() if hasattr(sol['weights'], 'tolist') else sol['weights']
            }
            serializable_solutions.append(serializable_sol)
        
        results = {
            'pareto_solutions': serializable_solutions,
            'best_params': self.best_params
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results saved to {filepath}")

def main():
    """Main function to demonstrate parameter tuning"""
    
    # Initialize tuner
    tuner = ParameterTuner()
    
    # Define Mars environmental conditions
    mars_conditions = {
        'temperature_c': -63.0,
        'pressure_kpa': 0.6,
        'uv_intensity': 1.5,
        'cosmic_radiation': 2.0,
        'relative_humidity': 5.0
    }
    
    # Define optimization targets
    targets = [
        OptimizationTarget('growth_rate', weight=1.0, target_value=0.8),
        OptimizationTarget('survival_probability', weight=1.0, target_value=0.9),
        OptimizationTarget('remediation_efficiency', weight=1.5, target_value=0.85),
        OptimizationTarget('metabolic_activity', weight=0.8, target_value=0.7)
    ]
    
    # Single objective optimization
    logger.info("Running single objective optimization...")
    optimized_params, performance = tuner.optimize_parameters(
        targets, mars_conditions, method='differential_evolution'
    )
    
    print("\n=== Single Objective Optimization Results ===")
    print("Optimized Parameters:")
    for param, value in optimized_params.items():
        print(f"  {param}: {value:.3f}")
    
    print("\nPerformance Metrics:")
    for metric, value in performance.items():
        print(f"  {metric}: {value:.3f}")
    
    # Multi-objective optimization
    logger.info("Running multi-objective optimization...")
    pareto_solutions = tuner.multi_objective_optimization(
        targets, mars_conditions, n_pareto_points=15
    )
    
    # Plot results
    tuner.plot_optimization_results(pareto_solutions)
    
    # Save results
    tuner.save_optimization_results(pareto_solutions, 'data/parameter_optimization_results.json')
    
    # Print Pareto front summary
    print(f"\n=== Multi-Objective Optimization Results ===")
    print(f"Found {len(pareto_solutions)} Pareto optimal solutions")
    
    if pareto_solutions:
        # Find best solution for each metric
        best_growth = max(pareto_solutions, key=lambda x: x['performance']['growth_rate'])
        best_survival = max(pareto_solutions, key=lambda x: x['performance']['survival_probability'])
        best_remediation = max(pareto_solutions, key=lambda x: x['performance']['remediation_efficiency'])
        
        print(f"\nBest growth rate: {best_growth['performance']['growth_rate']:.3f}")
        print(f"Best survival probability: {best_survival['performance']['survival_probability']:.3f}")
        print(f"Best remediation efficiency: {best_remediation['performance']['remediation_efficiency']:.3f}")

if __name__ == "__main__":
    main()