#!/usr/bin/env python3
"""
Simple flux balance analysis for Mars resource utilization

This module implements flux balance analysis (FBA) for metabolic networks
optimized for Mars atmospheric processing and resource utilization.
"""

import argparse
import logging
import sys
from typing import Dict, List, Tuple, Optional
import json

try:
    import numpy as np
    from scipy.optimize import linprog
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Install with: pip install numpy scipy matplotlib")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MarsMetabolicNetwork:
    """
    Simplified metabolic network model for Mars atmospheric processing

    Models key pathways for:
    - CO2 fixation from Mars atmosphere
    - Water utilization from subsurface ice
    - Biomass production under extreme conditions
    """

    def __init__(self):
        """Initialize Mars metabolic network"""
        # Define metabolites
        self.metabolites = [
            "CO2_ext",  # External CO2 (Mars atmosphere)
            "H2O_ext",  # External H2O (subsurface ice)
            "CO2_int",  # Internal CO2
            "H2O_int",  # Internal H2O
            "G3P",  # Glyceraldehyde-3-phosphate
            "PYR",  # Pyruvate
            "ATP",  # Adenosine triphosphate
            "NADH",  # Reduced nicotinamide adenine dinucleotide
            "BIOMASS",  # Biomass precursors
        ]

        # Define reactions with stoichiometry
        # Format: [metabolite_index: coefficient, ...]
        self.reactions = {
            "CO2_uptake": {0: -1, 2: 1},  # CO2_ext -> CO2_int
            "H2O_uptake": {1: -1, 3: 1},  # H2O_ext -> H2O_int
            "CO2_fixation": {2: -3, 3: -1, 4: 1, 6: -2},  # 3 CO2 + H2O + 2 ATP -> G3P
            "glycolysis": {4: -2, 5: 1, 6: 2, 7: 1},  # 2 G3P -> PYR + 2 ATP + NADH
            "respiration": {5: -1, 6: 15, 7: -3},  # PYR -> 15 ATP + 3 NADH (simplified)
            "biomass_synthesis": {
                4: -1,
                5: -2,
                6: -10,
                8: 1,
            },  # G3P + 2 PYR + 10 ATP -> BIOMASS
            "ATP_maintenance": {6: -1},  # ATP maintenance cost
        }

        # Reaction bounds [lower, upper] (mmol/gDW/h)
        self.reaction_bounds = {
            "CO2_uptake": [0, 20],  # Mars atmosphere is 96% CO2
            "H2O_uptake": [0, 10],  # Limited water availability
            "CO2_fixation": [0, 15],  # Calvin cycle capacity
            "glycolysis": [0, 20],  # Glycolytic capacity
            "respiration": [0, 10],  # Respiratory capacity
            "biomass_synthesis": [0, 5],  # Growth rate limit
            "ATP_maintenance": [1, 2],  # Maintenance energy requirement
        }

        # Build stoichiometric matrix
        self.build_stoichiometric_matrix()

        logger.info(
            f"Initialized Mars metabolic network: {len(self.metabolites)} metabolites, {len(self.reactions)} reactions"
        )

    def build_stoichiometric_matrix(self):
        """Build the stoichiometric matrix S for FBA"""
        n_metabolites = len(self.metabolites)
        n_reactions = len(self.reactions)

        self.S = np.zeros((n_metabolites, n_reactions))
        self.reaction_names = list(self.reactions.keys())

        for j, (reaction_name, stoich) in enumerate(self.reactions.items()):
            for metabolite_idx, coefficient in stoich.items():
                self.S[metabolite_idx, j] = coefficient

        logger.debug(f"Built stoichiometric matrix: {self.S.shape}")

    def mars_metabolic_flux_analysis(
        self, objective: str = "biomass_synthesis"
    ) -> dict:
        """
        Perform flux balance analysis for Mars atmospheric processing

        Args:
            objective: Reaction to optimize (default: biomass_synthesis)

        Returns:
            Dictionary containing FBA results and analysis
        """
        logger.info(f"Running FBA with objective: {objective}")

        if objective not in self.reaction_names:
            raise ValueError(f"Unknown objective: {objective}")

        # Set up linear programming problem
        # Maximize objective reaction (minimize negative)
        objective_idx = self.reaction_names.index(objective)
        c = np.zeros(len(self.reaction_names))
        c[objective_idx] = -1  # Minimize negative = maximize

        # Bounds for reactions
        bounds = []
        for reaction_name in self.reaction_names:
            lower, upper = self.reaction_bounds[reaction_name]
            bounds.append((lower, upper))

        # Equality constraints: S * v = 0 (steady state)
        A_eq = self.S
        b_eq = np.zeros(len(self.metabolites))

        try:
            # Solve linear programming problem
            result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

            if not result.success:
                raise RuntimeError(f"FBA optimization failed: {result.message}")

            # Parse results
            flux_values = result.x
            objective_value = -result.fun  # Convert back from minimization

            # Create results dictionary
            results = {
                "objective": objective,
                "objective_value": round(objective_value, 4),
                "success": True,
                "fluxes": {},
                "mars_efficiency_metrics": {},
            }

            # Store flux values
            for i, reaction_name in enumerate(self.reaction_names):
                results["fluxes"][reaction_name] = round(flux_values[i], 4)

            # Calculate Mars-specific efficiency metrics
            co2_uptake_rate = flux_values[self.reaction_names.index("CO2_uptake")]
            biomass_rate = flux_values[self.reaction_names.index("biomass_synthesis")]

            results["mars_efficiency_metrics"] = {
                "co2_fixation_efficiency": round(
                    biomass_rate / max(co2_uptake_rate, 0.001), 4
                ),
                "water_usage_efficiency": round(
                    biomass_rate
                    / max(flux_values[self.reaction_names.index("H2O_uptake")], 0.001),
                    4,
                ),
                "energy_efficiency": round(
                    biomass_rate
                    / max(
                        flux_values[self.reaction_names.index("ATP_maintenance")], 0.001
                    ),
                    4,
                ),
            }

            logger.info(
                f"FBA completed successfully: objective value = {objective_value:.4f}"
            )
            return results

        except Exception as e:
            logger.error(f"FBA failed: {e}")
            return {"objective": objective, "success": False, "error": str(e)}

    def print_flux_analysis(self, results: dict) -> None:
        """Print formatted flux analysis results"""
        print(f"\n🔬 Mars Metabolic Flux Analysis")
        print("=" * 50)

        if not results["success"]:
            print(f"❌ Analysis failed: {results['error']}")
            return

        print(f"Objective: {results['objective']}")
        print(f"Optimal value: {results['objective_value']:.4f} mmol/gDW/h")

        print(f"\n📈 Reaction Fluxes:")
        print("-" * 50)
        for reaction, flux in results["fluxes"].items():
            status = "→" if flux > 0.001 else "○"
            print(f"{status} {reaction:20s}: {flux:8.4f} mmol/gDW/h")

        print(f"\n🌍 Mars Efficiency Metrics:")
        print("-" * 50)
        metrics = results["mars_efficiency_metrics"]
        print(f"CO2 Fixation Efficiency:  {metrics['co2_fixation_efficiency']:.4f}")
        print(f"Water Usage Efficiency:   {metrics['water_usage_efficiency']:.4f}")
        print(f"Energy Efficiency:        {metrics['energy_efficiency']:.4f}")

    def plot_flux_distribution(
        self, results: dict, save_path: Optional[str] = None
    ) -> None:
        """Create visualization of flux distribution"""
        if not results["success"]:
            logger.warning("Cannot plot failed FBA results")
            return

        # Extract active fluxes (> 0.001)
        active_fluxes = {k: v for k, v in results["fluxes"].items() if abs(v) > 0.001}

        if not active_fluxes:
            logger.warning("No significant fluxes to plot")
            return

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))

        reactions = list(active_fluxes.keys())
        fluxes = list(active_fluxes.values())
        colors = ["red" if f < 0 else "green" for f in fluxes]

        bars = ax.bar(reactions, fluxes, color=colors, alpha=0.7)

        ax.set_ylabel("Flux (mmol/gDW/h)")
        ax.set_title("Mars Metabolic Flux Distribution")
        ax.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, flux in zip(bars, fluxes):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{flux:.3f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Flux plot saved to: {save_path}")
        else:
            plt.show()


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Mars metabolic flux balance analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python metabolic_flux.py
  python metabolic_flux.py --objective CO2_fixation
  python metabolic_flux.py --plot --output flux_plot.png
  python metabolic_flux.py --save-results results.json
        """,
    )

    parser.add_argument(
        "--objective",
        default="biomass_synthesis",
        choices=["biomass_synthesis", "CO2_fixation", "ATP_maintenance"],
        help="Optimization objective (default: biomass_synthesis)",
    )
    parser.add_argument(
        "--model", help="Path to SBML model file (optional, uses built-in model)"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate flux distribution plot"
    )
    parser.add_argument("--output", help="Output file for plot or results")
    parser.add_argument("--save-results", help="Save FBA results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize metabolic network
        if args.model:
            logger.info(f"Loading model from: {args.model}")
            # TODO: Implement SBML model loading
            network = MarsMetabolicNetwork()
        else:
            network = MarsMetabolicNetwork()

        # Run flux balance analysis
        results = network.mars_metabolic_flux_analysis(objective=args.objective)

        # Display results
        network.print_flux_analysis(results)

        # Generate plot if requested
        if args.plot and results["success"]:
            plot_path = args.output if args.output else "mars_flux_distribution.png"
            network.plot_flux_distribution(results, save_path=plot_path)

        # Save results to JSON if requested
        if args.save_results and results["success"]:
            with open(args.save_results, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.save_results}")

        print("\n✅ Metabolic flux analysis completed successfully!")

    except Exception as e:
        logger.error(f"Metabolic flux analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
