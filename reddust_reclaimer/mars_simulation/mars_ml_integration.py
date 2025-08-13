"""
Mars ML Integration - Complete Mars Deployment Analysis
Integrates all Mars simulation and ML components for comprehensive deployment analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import logging
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Import our custom modules
from environment_simulator import MarsEnvironmentSimulator, MarsEnvironmentParams
from stress_response_predictor import StressResponsePredictor, StressResponseParams
from strain_optimizer import StrainOptimizer, StrainGenome
from performance_predictor import EnsemblePerformancePredictor
from parameter_tuning import ParameterTuner, OptimizationTarget

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarsMLIntegration:
    """
    Comprehensive Mars deployment analysis integrating all simulation and ML components
    """

    def __init__(self):
        self.environment_simulator = MarsEnvironmentSimulator()
        self.stress_predictor = StressResponsePredictor()
        self.strain_optimizer = StrainOptimizer()
        self.performance_predictor = EnsemblePerformancePredictor()
        self.parameter_tuner = ParameterTuner()

        self.integration_results = {}
        self.risk_assessment = {}
        self.deployment_scenarios = {}

    def run_comprehensive_analysis(
        self, deployment_duration_years: int = 5, confidence_level: float = 0.95
    ) -> Dict[str, any]:
        """
        Run comprehensive Mars deployment analysis

        Args:
            deployment_duration_years: Duration of deployment analysis
            confidence_level: Confidence level for uncertainty quantification

        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting comprehensive Mars deployment analysis...")

        # 1. Generate Mars environmental conditions
        logger.info("Step 1: Generating Mars environmental conditions...")
        env_data = self.environment_simulator.simulate_seasonal_variations(
            earth_years=deployment_duration_years
        )
        stress_data = self.environment_simulator.calculate_stress_factors(env_data)

        # 2. Train stress response models
        logger.info("Step 2: Training stress response prediction models...")
        features, targets = self.stress_predictor.generate_training_data(
            n_samples=10000
        )
        self.stress_predictor.train_models(features, targets)

        # 3. Optimize strain characteristics
        logger.info("Step 3: Optimizing strain characteristics...")
        mars_conditions = {
            "temperature_c": -63.0,
            "pressure_kpa": 0.6,
            "uv_intensity": 1.5,
            "water_activity": 0.3,
            "perchlorate_concentration": 0.007,
            "cosmic_radiation": 2.0,
            "relative_humidity": 5.0,
        }

        best_genome, fitness_history = self.strain_optimizer.optimize(
            environment_conditions=mars_conditions, generations=100
        )

        # 4. Train performance prediction models
        logger.info("Step 4: Training performance prediction models...")
        perf_features, perf_targets = self.performance_predictor.generate_training_data(
            n_samples=12000
        )
        self.performance_predictor.train_ensemble_models(perf_features, perf_targets)

        # 5. Optimize deployment parameters
        logger.info("Step 5: Optimizing deployment parameters...")
        optimization_targets = [
            OptimizationTarget("growth_rate", weight=1.0, target_value=0.8),
            OptimizationTarget("survival_probability", weight=1.0, target_value=0.9),
            OptimizationTarget("remediation_efficiency", weight=1.5, target_value=0.85),
            OptimizationTarget("metabolic_activity", weight=0.8, target_value=0.7),
        ]

        optimized_params, performance = self.parameter_tuner.optimize_parameters(
            optimization_targets, mars_conditions
        )

        # 6. Generate comprehensive predictions
        logger.info("Step 6: Generating comprehensive predictions...")
        # Create test scenarios
        test_scenarios = self._generate_test_scenarios(env_data, best_genome)

        # Predict performance for all scenarios
        predictions, uncertainties = self.performance_predictor.predict_performance(
            test_scenarios, return_uncertainty=True
        )

        # Calculate confidence intervals
        confidence_intervals = (
            self.performance_predictor.calculate_confidence_intervals(
                predictions, uncertainties, confidence_level
            )
        )

        # 7. Risk assessment
        logger.info("Step 7: Performing risk assessment...")
        risk_assessment = self._perform_risk_assessment(
            predictions, uncertainties, confidence_intervals, env_data
        )

        # 8. Generate deployment scenarios
        logger.info("Step 8: Generating deployment scenarios...")
        deployment_scenarios = self._generate_deployment_scenarios(
            best_genome, optimized_params, predictions, risk_assessment
        )

        # Compile results
        self.integration_results = {
            "environment_data": env_data,
            "stress_data": stress_data,
            "best_genome": best_genome.to_dict(),
            "optimized_params": optimized_params,
            "performance_predictions": predictions,
            "uncertainties": uncertainties,
            "confidence_intervals": confidence_intervals,
            "risk_assessment": risk_assessment,
            "deployment_scenarios": deployment_scenarios,
            "analysis_metadata": {
                "deployment_duration_years": deployment_duration_years,
                "confidence_level": confidence_level,
                "analysis_timestamp": datetime.now().isoformat(),
                "model_accuracy": self._calculate_model_accuracy(
                    predictions, uncertainties
                ),
            },
        }

        logger.info("Comprehensive analysis complete!")
        return self.integration_results

    def _generate_test_scenarios(
        self, env_data: pd.DataFrame, best_genome: StrainGenome
    ) -> pd.DataFrame:
        """
        Generate test scenarios combining environmental data with optimized strain
        """
        # Sample environmental conditions
        n_scenarios = min(1000, len(env_data))
        sampled_env = env_data.sample(n=n_scenarios, random_state=42)

        # Create strain characteristics based on optimized genome
        strain_data = pd.DataFrame(
            {
                "growth_rate": [best_genome.growth_rate] * n_scenarios,
                "perchlorate_reduction_rate": [best_genome.perchlorate_reduction_rate]
                * n_scenarios,
                "stress_tolerance": [best_genome.stress_tolerance] * n_scenarios,
                "cold_shock_proteins": [best_genome.cold_shock_proteins] * n_scenarios,
                "osmotic_stress_proteins": [best_genome.osmotic_stress_proteins]
                * n_scenarios,
                "antioxidant_proteins": [best_genome.antioxidant_proteins]
                * n_scenarios,
                "perchlorate_reductase": [best_genome.perchlorate_reductase]
                * n_scenarios,
                "glycolysis_efficiency": [best_genome.glycolysis_efficiency]
                * n_scenarios,
                "tca_cycle_efficiency": [best_genome.tca_cycle_efficiency]
                * n_scenarios,
                "oxidative_phosphorylation": [best_genome.oxidative_phosphorylation]
                * n_scenarios,
                "energy_allocation_growth": [best_genome.energy_allocation_growth]
                * n_scenarios,
                "energy_allocation_stress": [best_genome.energy_allocation_stress]
                * n_scenarios,
                "energy_allocation_remediation": [
                    best_genome.energy_allocation_remediation
                ]
                * n_scenarios,
            }
        )

        # Combine environmental and strain data
        test_scenarios = pd.concat([sampled_env, strain_data], axis=1)

        return test_scenarios

    def _perform_risk_assessment(
        self,
        predictions: pd.DataFrame,
        uncertainties: pd.DataFrame,
        confidence_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]],
        env_data: pd.DataFrame,
    ) -> Dict[str, any]:
        """
        Perform comprehensive risk assessment
        """
        risk_assessment = {}

        # Calculate risk metrics for each performance target
        for target in predictions.columns:
            target_predictions = predictions[target]
            target_uncertainties = uncertainties[target]

            # Risk metrics
            mean_performance = target_predictions.mean()
            std_performance = target_predictions.std()
            cv_performance = (
                std_performance / mean_performance if mean_performance > 0 else 0
            )

            # Failure probability (performance below threshold)
            failure_threshold = 0.5  # 50% performance threshold
            failure_probability = (target_predictions < failure_threshold).mean()

            # High uncertainty risk (uncertainty > 0.2)
            high_uncertainty_risk = (target_uncertainties > 0.2).mean()

            # Worst-case scenario (lower confidence bound)
            if target in confidence_intervals:
                lower_ci, upper_ci = confidence_intervals[target]
                worst_case_performance = lower_ci.mean()
                best_case_performance = upper_ci.mean()
            else:
                worst_case_performance = (
                    target_predictions - target_uncertainties
                ).mean()
                best_case_performance = (
                    target_predictions + target_uncertainties
                ).mean()

            risk_assessment[target] = {
                "mean_performance": mean_performance,
                "std_performance": std_performance,
                "coefficient_of_variation": cv_performance,
                "failure_probability": failure_probability,
                "high_uncertainty_risk": high_uncertainty_risk,
                "worst_case_performance": worst_case_performance,
                "best_case_performance": best_case_performance,
                "risk_level": self._calculate_risk_level(
                    failure_probability, cv_performance
                ),
            }

        # Overall risk assessment
        overall_failure_prob = np.mean(
            [
                risk_assessment[target]["failure_probability"]
                for target in predictions.columns
            ]
        )
        overall_uncertainty_risk = np.mean(
            [
                risk_assessment[target]["high_uncertainty_risk"]
                for target in predictions.columns
            ]
        )

        risk_assessment["overall"] = {
            "overall_failure_probability": overall_failure_prob,
            "overall_uncertainty_risk": overall_uncertainty_risk,
            "overall_risk_level": self._calculate_risk_level(overall_failure_prob, 0.5),
        }

        return risk_assessment

    def _calculate_risk_level(self, failure_probability: float, cv: float) -> str:
        """
        Calculate risk level based on failure probability and coefficient of variation
        """
        if failure_probability > 0.3 or cv > 0.5:
            return "HIGH"
        elif failure_probability > 0.1 or cv > 0.3:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_deployment_scenarios(
        self,
        best_genome: StrainGenome,
        optimized_params: Dict[str, float],
        predictions: pd.DataFrame,
        risk_assessment: Dict[str, any],
    ) -> Dict[str, any]:
        """
        Generate deployment scenarios with recommendations
        """
        scenarios = {}

        # Scenario 1: Conservative deployment (low risk)
        conservative_threshold = 0.7
        conservative_mask = (
            (predictions["survival_probability"] > conservative_threshold)
            & (predictions["remediation_efficiency"] > conservative_threshold)
            & (predictions["growth_rate_pred"] > conservative_threshold)
        )

        if conservative_mask.any():
            conservative_scenarios = predictions[conservative_mask]
            scenarios["conservative"] = {
                "description": "Conservative deployment with high performance guarantees",
                "n_scenarios": len(conservative_scenarios),
                "mean_performance": conservative_scenarios.mean().to_dict(),
                "recommendation": "Proceed with deployment under controlled conditions",
                "risk_level": "LOW",
            }

        # Scenario 2: Aggressive deployment (higher risk, higher reward)
        aggressive_threshold = 0.5
        aggressive_mask = (
            predictions["remediation_efficiency"] > aggressive_threshold
        ) & (predictions["metabolic_activity"] > aggressive_threshold)

        if aggressive_mask.any():
            aggressive_scenarios = predictions[aggressive_mask]
            scenarios["aggressive"] = {
                "description": "Aggressive deployment maximizing remediation efficiency",
                "n_scenarios": len(aggressive_scenarios),
                "mean_performance": aggressive_scenarios.mean().to_dict(),
                "recommendation": "Proceed with monitoring and contingency plans",
                "risk_level": "MEDIUM",
            }

        # Scenario 3: Experimental deployment (research focus)
        experimental_mask = (predictions["perchlorate_reduction"] > 0.6) & (
            predictions["stress_resistance"] > 0.5
        )

        if experimental_mask.any():
            experimental_scenarios = predictions[experimental_mask]
            scenarios["experimental"] = {
                "description": "Experimental deployment for research and optimization",
                "n_scenarios": len(experimental_scenarios),
                "mean_performance": experimental_scenarios.mean().to_dict(),
                "recommendation": "Proceed with extensive monitoring and data collection",
                "risk_level": "MEDIUM-HIGH",
            }

        return scenarios

    def _calculate_model_accuracy(
        self, predictions: pd.DataFrame, uncertainties: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate model accuracy metrics
        """
        accuracy_metrics = {}

        for target in predictions.columns:
            # Calculate coefficient of variation as accuracy metric
            mean_pred = predictions[target].mean()
            mean_uncertainty = uncertainties[target].mean()

            if mean_pred > 0:
                accuracy = 1 - (mean_uncertainty / mean_pred)
                accuracy_metrics[target] = max(0, accuracy)
            else:
                accuracy_metrics[target] = 0.0

        # Overall accuracy
        accuracy_metrics["overall"] = np.mean(list(accuracy_metrics.values()))

        return accuracy_metrics

    def plot_comprehensive_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive analysis results
        """
        if not self.integration_results:
            logger.warning("No integration results to plot. Run analysis first.")
            return

        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle("Comprehensive Mars Deployment Analysis Results", fontsize=16)

        predictions = self.integration_results["performance_predictions"]
        uncertainties = self.integration_results["uncertainties"]
        risk_assessment = self.integration_results["risk_assessment"]

        # 1. Performance predictions with uncertainty
        for i, target in enumerate(predictions.columns[:3]):
            row = i
            col = 0

            sorted_indices = np.argsort(predictions[target])
            sorted_pred = predictions[target].iloc[sorted_indices]
            sorted_uncertainty = uncertainties[target].iloc[sorted_indices]

            axes[row, col].plot(sorted_pred, label="Prediction", linewidth=2)
            axes[row, col].fill_between(
                range(len(sorted_pred)),
                sorted_pred - sorted_uncertainty,
                sorted_pred + sorted_uncertainty,
                alpha=0.3,
                label="Uncertainty",
            )
            axes[row, col].set_title(f'{target.replace("_", " ").title()}')
            axes[row, col].set_xlabel("Scenario Index")
            axes[row, col].set_ylabel("Predicted Value")
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend()

        # 2. Risk assessment summary
        risk_targets = list(risk_assessment.keys())[:3]
        risk_metrics = ["failure_probability", "high_uncertainty_risk"]

        for i, metric in enumerate(risk_metrics):
            values = [
                risk_assessment[target][metric]
                for target in risk_targets
                if target != "overall"
            ]
            axes[0, 1].bar(
                risk_targets, values, alpha=0.7, label=metric.replace("_", " ")
            )

        axes[0, 1].set_title("Risk Assessment Summary")
        axes[0, 1].set_ylabel("Risk Probability")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Performance correlation matrix
        correlation_matrix = predictions.corr()
        im = axes[0, 2].imshow(correlation_matrix, cmap="coolwarm", aspect="auto")
        axes[0, 2].set_title("Performance Correlation Matrix")
        axes[0, 2].set_xticks(range(len(correlation_matrix.columns)))
        axes[0, 2].set_yticks(range(len(correlation_matrix.columns)))
        axes[0, 2].set_xticklabels(correlation_matrix.columns, rotation=45)
        axes[0, 2].set_yticklabels(correlation_matrix.columns)
        plt.colorbar(im, ax=axes[0, 2])

        # 4. Uncertainty distribution
        uncertainty_data = [uncertainties[col] for col in uncertainties.columns]
        axes[1, 1].boxplot(uncertainty_data, labels=uncertainties.columns)
        axes[1, 1].set_title("Uncertainty Distribution")
        axes[1, 1].set_ylabel("Uncertainty")
        axes[1, 1].grid(True, alpha=0.3)
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

        # 5. Performance vs Uncertainty scatter
        for i, target in enumerate(predictions.columns[:2]):
            row = 1 + i
            col = 2

            axes[row, col].scatter(
                predictions[target], uncertainties[target], alpha=0.6
            )
            axes[row, col].set_xlabel(f'{target.replace("_", " ").title()}')
            axes[row, col].set_ylabel("Uncertainty")
            axes[row, col].set_title(
                f'{target.replace("_", " ").title()} vs Uncertainty'
            )
            axes[row, col].grid(True, alpha=0.3)

        # 6. Deployment scenario summary
        scenarios = self.integration_results["deployment_scenarios"]
        if scenarios:
            scenario_names = list(scenarios.keys())
            scenario_performances = []

            for scenario_name in scenario_names:
                scenario = scenarios[scenario_name]
                mean_perf = scenario["mean_performance"]["remediation_efficiency"]
                scenario_performances.append(mean_perf)

            axes[2, 0].bar(scenario_names, scenario_performances)
            axes[2, 0].set_title("Deployment Scenario Performance")
            axes[2, 0].set_ylabel("Remediation Efficiency")
            axes[2, 0].grid(True, alpha=0.3)

        # 7. Model accuracy summary
        accuracy_metrics = self.integration_results["analysis_metadata"][
            "model_accuracy"
        ]
        accuracy_targets = [k for k in accuracy_metrics.keys() if k != "overall"]
        accuracy_values = [accuracy_metrics[k] for k in accuracy_targets]

        axes[2, 1].bar(accuracy_targets, accuracy_values)
        axes[2, 1].set_title("Model Accuracy by Target")
        axes[2, 1].set_ylabel("Accuracy")
        axes[2, 1].grid(True, alpha=0.3)
        plt.setp(axes[2, 1].xaxis.get_majorticklabels(), rotation=45)

        # 8. Overall summary statistics
        summary_stats = {
            "Mean Performance": predictions.mean().mean(),
            "Mean Uncertainty": uncertainties.mean().mean(),
            "Overall Risk Level": risk_assessment["overall"]["overall_risk_level"],
            "Model Accuracy": accuracy_metrics["overall"],
        }

        axes[2, 2].text(0.1, 0.8, "Summary Statistics:", fontsize=12, fontweight="bold")
        y_pos = 0.7
        for stat, value in summary_stats.items():
            if isinstance(value, float):
                text = f"{stat}: {value:.3f}"
            else:
                text = f"{stat}: {value}"
            axes[2, 2].text(0.1, y_pos, text, fontsize=10)
            y_pos -= 0.1

        axes[2, 2].set_xlim(0, 1)
        axes[2, 2].set_ylim(0, 1)
        axes[2, 2].set_title("Analysis Summary")
        axes[2, 2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def save_integration_results(self, filepath: str) -> None:
        """
        Save comprehensive integration results
        """
        # Convert to serializable format
        serializable_results = {}

        for key, value in self.integration_results.items():
            if isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict()
            elif hasattr(value, "to_dict"):
                serializable_results[key] = value.to_dict()
            else:
                serializable_results[key] = value

        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Integration results saved to {filepath}")

    def generate_deployment_report(self, filepath: str) -> None:
        """
        Generate comprehensive deployment report
        """
        if not self.integration_results:
            logger.warning("No integration results available. Run analysis first.")
            return

        report = []
        report.append("=" * 80)
        report.append("MARS DEPLOYMENT ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)

        accuracy = self.integration_results["analysis_metadata"]["model_accuracy"][
            "overall"
        ]
        risk_level = self.integration_results["risk_assessment"]["overall"][
            "overall_risk_level"
        ]

        report.append(f"Model Accuracy: {accuracy:.1%}")
        report.append(f"Overall Risk Level: {risk_level}")
        report.append(
            f"Analysis Confidence: {self.integration_results['analysis_metadata']['confidence_level']:.1%}"
        )
        report.append("")

        # Performance Summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 40)

        predictions = self.integration_results["performance_predictions"]
        for target in predictions.columns:
            mean_perf = predictions[target].mean()
            std_perf = predictions[target].std()
            report.append(
                f"{target.replace('_', ' ').title()}: {mean_perf:.3f} ± {std_perf:.3f}"
            )
        report.append("")

        # Risk Assessment
        report.append("RISK ASSESSMENT")
        report.append("-" * 40)

        risk_assessment = self.integration_results["risk_assessment"]
        for target, risk_data in risk_assessment.items():
            if target != "overall":
                report.append(f"{target.replace('_', ' ').title()}:")
                report.append(f"  Risk Level: {risk_data['risk_level']}")
                report.append(
                    f"  Failure Probability: {risk_data['failure_probability']:.1%}"
                )
                report.append(
                    f"  High Uncertainty Risk: {risk_data['high_uncertainty_risk']:.1%}"
                )
                report.append("")

        # Deployment Recommendations
        report.append("DEPLOYMENT RECOMMENDATIONS")
        report.append("-" * 40)

        scenarios = self.integration_results["deployment_scenarios"]
        for scenario_name, scenario_data in scenarios.items():
            report.append(f"{scenario_name.upper()} SCENARIO:")
            report.append(f"  Description: {scenario_data['description']}")
            report.append(f"  Risk Level: {scenario_data['risk_level']}")
            report.append(f"  Recommendation: {scenario_data['recommendation']}")
            report.append(
                f"  Number of Viable Scenarios: {scenario_data['n_scenarios']}"
            )
            report.append("")

        # Write report to file
        with open(filepath, "w") as f:
            f.write("\n".join(report))

        logger.info(f"Deployment report saved to {filepath}")


def main():
    """Main function to demonstrate comprehensive Mars ML integration"""

    # Initialize integration
    integration = MarsMLIntegration()

    # Run comprehensive analysis
    results = integration.run_comprehensive_analysis(
        deployment_duration_years=3, confidence_level=0.95
    )

    # Plot results
    integration.plot_comprehensive_results()

    # Save results
    integration.save_integration_results("data/mars_ml_integration_results.json")

    # Generate deployment report
    integration.generate_deployment_report("data/mars_deployment_report.txt")

    # Print key findings
    print("\n" + "=" * 80)
    print("MARS DEPLOYMENT ANALYSIS - KEY FINDINGS")
    print("=" * 80)

    accuracy = results["analysis_metadata"]["model_accuracy"]["overall"]
    risk_level = results["risk_assessment"]["overall"]["overall_risk_level"]

    print(f"Model Accuracy: {accuracy:.1%}")
    print(f"Overall Risk Level: {risk_level}")
    print(
        f"Analysis Confidence: {results['analysis_metadata']['confidence_level']:.1%}"
    )

    print("\nPerformance Metrics (Mean ± Std):")
    predictions = results["performance_predictions"]
    for target in predictions.columns:
        mean_perf = predictions[target].mean()
        std_perf = predictions[target].std()
        print(f"  {target.replace('_', ' ').title()}: {mean_perf:.3f} ± {std_perf:.3f}")

    print(f"\nDeployment Scenarios Available: {len(results['deployment_scenarios'])}")
    for scenario_name, scenario_data in results["deployment_scenarios"].items():
        print(f"  {scenario_name.title()}: {scenario_data['risk_level']} risk")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
