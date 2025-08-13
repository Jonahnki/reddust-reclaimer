#!/usr/bin/env python3
"""
Base Model Loader for RedDust Reclaimer Project
==============================================

This module loads and prepares the B. subtilis genome-scale metabolic model
for perchlorate detoxification pathway integration and Mars condition modeling.

Author: RedDust Reclaimer AI
Date: 2024
License: MIT
"""

import cobra
import cobra.test
import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlretrieve
import tempfile
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BSubtilisModelLoader:
    """
    Loader and manager for B. subtilis genome-scale metabolic models.

    Supports multiple model versions and provides utilities for model
    validation, analysis, and Mars-specific modifications.
    """

    def __init__(self, model_dir: str = "../../models/metabolic/"):
        """
        Initialize the model loader.

        Args:
            model_dir: Directory to store metabolic models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.current_model = None
        self.model_info = {}

        # Available B. subtilis models
        self.available_models = {
            "iYO844": {
                "description": "B. subtilis 168 genome-scale model (844 genes)",
                "url": "https://github.com/opencobra/m_model_collection/raw/main/sbml/iYO844.xml",
                "file": "iYO844.xml",
                "genes": 844,
                "reactions": 1020,
                "metabolites": 1001,
                "reference": "Oh et al. (2007) BMC Systems Biology",
            },
            "iBsu1103": {
                "description": "Updated B. subtilis model (1103 genes)",
                "url": "https://github.com/SBRG/bigg_models/raw/master/models/iBsu1103.xml",
                "file": "iBsu1103.xml",
                "genes": 1103,
                "reactions": 1437,
                "metabolites": 1138,
                "reference": "Henry et al. (2009) Nature Biotechnology",
            },
        }

        # Mars-specific constraints
        self.mars_constraints = {
            "temperature": 4,  # Celsius - Mars surface temperature
            "pressure": 0.006,  # atm - Mars atmospheric pressure
            "oxygen_partial_pressure": 0.0013,  # Very low O2
            "co2_partial_pressure": 0.0057,  # High CO2
            "water_activity": 0.03,  # Very low water activity
            "perchlorate_concentration": 0.5,  # % w/w in soil
            "radiation_dose": 0.67,  # mSv/day
        }

    def download_model(self, model_name: str) -> Path:
        """
        Download a B. subtilis model if not already present.

        Args:
            model_name: Name of the model to download

        Returns:
            Path to the downloaded model file
        """
        if model_name not in self.available_models:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(self.available_models.keys())}"
            )

        model_info = self.available_models[model_name]
        model_path = self.model_dir / model_info["file"]

        if model_path.exists():
            logger.info(f"Model {model_name} already exists at {model_path}")
            return model_path

        logger.info(f"Downloading {model_name} from {model_info['url']}")

        try:
            urlretrieve(model_info["url"], model_path)
            logger.info(f"Successfully downloaded {model_name}")
            return model_path

        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            # Fallback: create a basic model structure if download fails
            return self._create_fallback_model(model_name)

    def _create_fallback_model(self, model_name: str) -> Path:
        """Create a basic fallback model for testing."""

        logger.warning(f"Creating fallback model for {model_name}")

        # Create a minimal B. subtilis model
        model = cobra.Model(f"{model_name}_fallback")

        # Add basic compartments
        model.compartments = {"c": "cytosol", "e": "extracellular", "p": "periplasm"}

        # Add basic metabolites
        metabolites = {
            "glc__D_e": "D-glucose (extracellular)",
            "glc__D_c": "D-glucose (cytosol)",
            "atp_c": "ATP (cytosol)",
            "adp_c": "ADP (cytosol)",
            "pi_c": "phosphate (cytosol)",
            "h2o_c": "water (cytosol)",
            "o2_e": "oxygen (extracellular)",
            "co2_e": "carbon dioxide (extracellular)",
            "nh4_e": "ammonium (extracellular)",
            "pi_e": "phosphate (extracellular)",
            "h_e": "proton (extracellular)",
            "h_c": "proton (cytosol)",
            "biomass_c": "biomass (cytosol)",
        }

        for met_id, name in metabolites.items():
            compartment = met_id.split("_")[-1]
            met = cobra.Metabolite(met_id, name=name, compartment=compartment)
            model.add_metabolites([met])

        # Add basic reactions
        # Glucose uptake
        glc_uptake = cobra.Reaction("GLCpts")
        glc_uptake.name = "Glucose transport via PTS"
        glc_uptake.add_metabolites(
            {model.metabolites.glc__D_e: -1, model.metabolites.glc__D_c: 1}
        )
        glc_uptake.bounds = (0, 10)

        # ATP synthesis
        atp_synth = cobra.Reaction("ATPS")
        atp_synth.name = "ATP synthase"
        atp_synth.add_metabolites(
            {
                model.metabolites.adp_c: -1,
                model.metabolites.pi_c: -1,
                model.metabolites.atp_c: 1,
                model.metabolites.h2o_c: 1,
            }
        )

        # Biomass reaction
        biomass = cobra.Reaction("BIOMASS")
        biomass.name = "Biomass synthesis"
        biomass.add_metabolites(
            {
                model.metabolites.glc__D_c: -1,
                model.metabolites.atp_c: -10,
                model.metabolites.nh4_e: -1,
                model.metabolites.pi_e: -1,
                model.metabolites.biomass_c: 1,
                model.metabolites.adp_c: 10,
                model.metabolites.pi_c: 10,
            }
        )
        biomass.bounds = (0, 1000)

        # Exchange reactions
        exchanges = ["glc__D_e", "o2_e", "co2_e", "nh4_e", "pi_e", "h_e"]
        for met_id in exchanges:
            ex_rxn = cobra.Reaction(f"EX_{met_id}")
            ex_rxn.name = f"{model.metabolites.get_by_id(met_id).name} exchange"
            ex_rxn.add_metabolites({model.metabolites.get_by_id(met_id): -1})
            ex_rxn.bounds = (-1000, 1000)
            model.add_reactions([ex_rxn])

        model.add_reactions([glc_uptake, atp_synth, biomass])
        model.objective = "BIOMASS"

        # Save fallback model
        fallback_path = self.model_dir / f"{model_name}_fallback.xml"
        cobra.io.write_sbml_model(model, fallback_path)

        return fallback_path

    def load_model(
        self, model_name: str = "iYO844", force_download: bool = False
    ) -> cobra.Model:
        """
        Load a B. subtilis metabolic model.

        Args:
            model_name: Name of the model to load
            force_download: Whether to force re-download

        Returns:
            COBRApy model object
        """
        logger.info(f"Loading B. subtilis model: {model_name}")

        # Download if needed
        if (
            force_download
            or not (self.model_dir / self.available_models[model_name]["file"]).exists()
        ):
            model_path = self.download_model(model_name)
        else:
            model_path = self.model_dir / self.available_models[model_name]["file"]

        try:
            # Load the model
            model = cobra.io.read_sbml_model(str(model_path))

            # Validate the model
            self._validate_model(model)

            # Store model info
            self.current_model = model
            self.model_info = {
                "name": model_name,
                "file_path": model_path,
                "genes": len(model.genes),
                "reactions": len(model.reactions),
                "metabolites": len(model.metabolites),
                "load_time": pd.Timestamp.now(),
            }

            logger.info(f"Successfully loaded {model_name}:")
            logger.info(f"  Genes: {len(model.genes)}")
            logger.info(f"  Reactions: {len(model.reactions)}")
            logger.info(f"  Metabolites: {len(model.metabolites)}")

            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.info("Attempting to create fallback model...")
            fallback_path = self._create_fallback_model(model_name)
            return cobra.io.read_sbml_model(str(fallback_path))

    def _validate_model(self, model: cobra.Model) -> None:
        """Validate the loaded model."""

        logger.info("Validating model...")

        # Check basic properties
        if len(model.reactions) == 0:
            raise ValueError("Model has no reactions")

        if len(model.metabolites) == 0:
            raise ValueError("Model has no metabolites")

        # Check for objective function
        if not model.objective:
            logger.warning("Model has no objective function")

        # Test basic growth
        try:
            solution = model.optimize()
            if solution.status != "optimal":
                logger.warning(f"Model optimization failed: {solution.status}")
            else:
                logger.info(
                    f"Model validation successful, growth rate: {solution.objective_value:.3f}"
                )

        except Exception as e:
            logger.warning(f"Model validation failed: {e}")

    def analyze_model(self, model: cobra.Model = None) -> Dict:
        """
        Perform comprehensive analysis of the metabolic model.

        Args:
            model: Model to analyze (uses current_model if None)

        Returns:
            Dictionary containing analysis results
        """
        if model is None:
            model = self.current_model

        if model is None:
            raise ValueError("No model loaded")

        logger.info("Performing model analysis...")

        analysis = {}

        # Basic statistics
        analysis["basic_stats"] = {
            "genes": len(model.genes),
            "reactions": len(model.reactions),
            "metabolites": len(model.metabolites),
            "compartments": len(model.compartments),
        }

        # Compartment analysis
        analysis["compartments"] = dict(model.compartments)

        # Reaction analysis
        analysis["reaction_stats"] = {
            "reversible": len([r for r in model.reactions if r.reversibility]),
            "irreversible": len([r for r in model.reactions if not r.reversibility]),
            "exchange": len(model.exchanges),
            "transport": len(
                [
                    r
                    for r in model.reactions
                    if len(set([m.compartment for m in r.metabolites])) > 1
                ]
            ),
            "with_genes": len([r for r in model.reactions if r.genes]),
        }

        # Gene analysis
        analysis["gene_stats"] = {
            "total": len(model.genes),
            "essential": len(self._find_essential_genes(model)),
            "orphaned": len([g for g in model.genes if len(g.reactions) == 0]),
        }

        # Growth analysis
        analysis["growth_analysis"] = self._analyze_growth_conditions(model)

        # Pathway analysis
        analysis["pathway_coverage"] = self._analyze_pathway_coverage(model)

        return analysis

    def _find_essential_genes(self, model: cobra.Model) -> List[str]:
        """Find essential genes using single gene deletion."""

        essential_genes = []

        try:
            # Perform single gene deletion analysis
            deletion_results = cobra.flux_analysis.single_gene_deletion(model)

            # Genes are essential if their deletion significantly reduces growth
            growth_threshold = 0.1 * model.optimize().objective_value

            for gene_id, growth in deletion_results["growth"].items():
                if growth < growth_threshold:
                    essential_genes.append(gene_id)

        except Exception as e:
            logger.warning(f"Essential gene analysis failed: {e}")

        return essential_genes

    def _analyze_growth_conditions(self, model: cobra.Model) -> Dict:
        """Analyze growth under different conditions."""

        growth_analysis = {}

        try:
            # Standard conditions
            standard_growth = model.optimize().objective_value
            growth_analysis["standard"] = standard_growth

            # Anaerobic conditions
            with model:
                if "EX_o2_e" in model.reactions:
                    model.reactions.EX_o2_e.lower_bound = 0
                    anaerobic_growth = model.optimize().objective_value
                    growth_analysis["anaerobic"] = anaerobic_growth
                else:
                    growth_analysis["anaerobic"] = None

            # Minimal media
            with model:
                # Set all exchanges to 0 except essential ones
                for rxn in model.exchanges:
                    if rxn.id not in [
                        "EX_glc__D_e",
                        "EX_pi_e",
                        "EX_nh4_e",
                        "EX_so4_e",
                        "EX_mg2_e",
                        "EX_fe3_e",
                    ]:
                        rxn.lower_bound = 0

                minimal_growth = model.optimize().objective_value
                growth_analysis["minimal_media"] = minimal_growth

        except Exception as e:
            logger.warning(f"Growth condition analysis failed: {e}")
            growth_analysis = {"error": str(e)}

        return growth_analysis

    def _analyze_pathway_coverage(self, model: cobra.Model) -> Dict:
        """Analyze metabolic pathway coverage."""

        pathway_coverage = {}

        # Define key metabolic pathways
        key_pathways = {
            "glycolysis": ["GAPD", "PGK", "PGM", "ENO", "PYK"],
            "tca_cycle": [
                "CS",
                "ACONT",
                "ICDH",
                "AKGDH",
                "SUCOAS",
                "SUCD",
                "FUM",
                "MDH",
            ],
            "pentose_phosphate": ["G6PDH2r", "PGL", "GND", "RPI", "RPE"],
            "amino_acid_synthesis": ["ASPTA", "ASPK", "ASAD", "HSDy", "HSK", "THRS"],
            "fatty_acid_synthesis": ["ACCOAC", "HACD1", "ECOAH1", "ACACT1r"],
            "nucleotide_synthesis": ["ADSL1r", "ADSL2r", "IMPC", "IMPD"],
        }

        for pathway, reactions in key_pathways.items():
            found_reactions = []
            for rxn_pattern in reactions:
                matching_rxns = [r.id for r in model.reactions if rxn_pattern in r.id]
                found_reactions.extend(matching_rxns)

            pathway_coverage[pathway] = {
                "total_expected": len(reactions),
                "found": len(found_reactions),
                "coverage": len(found_reactions) / len(reactions) if reactions else 0,
                "found_reactions": found_reactions,
            }

        return pathway_coverage

    def apply_mars_constraints(self, model: cobra.Model = None) -> cobra.Model:
        """
        Apply Mars-specific environmental constraints to the model.

        Args:
            model: Model to constrain (uses current_model if None)

        Returns:
            Model with Mars constraints applied
        """
        if model is None:
            model = self.current_model.copy()
        else:
            model = model.copy()

        logger.info("Applying Mars environmental constraints...")

        # Low oxygen environment
        if "EX_o2_e" in model.reactions:
            # Reduce oxygen availability significantly
            original_o2 = model.reactions.EX_o2_e.lower_bound
            model.reactions.EX_o2_e.lower_bound = max(
                -1, original_o2 * 0.001
            )  # 0.1% of Earth oxygen
            logger.info(
                f"Reduced O2 availability: {original_o2} -> {model.reactions.EX_o2_e.lower_bound}"
            )

        # High CO2 environment
        if "EX_co2_e" in model.reactions:
            model.reactions.EX_co2_e.lower_bound = -1000  # Abundant CO2
            logger.info("Set high CO2 availability")

        # Low water activity
        if "EX_h2o_e" in model.reactions:
            # Reduce water availability
            original_h2o = model.reactions.EX_h2o_e.lower_bound
            model.reactions.EX_h2o_e.lower_bound = max(
                -10, original_h2o * 0.1
            )  # 10% of normal water
            logger.info(
                f"Reduced H2O availability: {original_h2o} -> {model.reactions.EX_h2o_e.lower_bound}"
            )

        # Cold temperature effects (simulated by reduced enzymatic efficiency)
        # Apply 50% reduction to all reaction fluxes to simulate cold
        for reaction in model.reactions:
            if not reaction.id.startswith("EX_"):  # Don't constrain exchanges
                if reaction.upper_bound > 0:
                    reaction.upper_bound *= 0.5
                if reaction.lower_bound < 0:
                    reaction.lower_bound *= 0.5

        logger.info("Applied cold temperature constraints (50% flux reduction)")

        # Add perchlorate availability
        self._add_perchlorate_exchanges(model)

        return model

    def _add_perchlorate_exchanges(self, model: cobra.Model) -> None:
        """Add perchlorate and chlorite exchange reactions."""

        # Add perchlorate metabolite if not present
        if "clo4_e" not in model.metabolites:
            clo4_e = cobra.Metabolite("clo4_e", name="perchlorate", compartment="e")
            model.add_metabolites([clo4_e])

            # Add perchlorate exchange
            ex_clo4 = cobra.Reaction("EX_clo4_e")
            ex_clo4.name = "Perchlorate exchange"
            ex_clo4.add_metabolites({clo4_e: -1})
            ex_clo4.bounds = (-10, 0)  # Available in environment
            model.add_reactions([ex_clo4])

            logger.info("Added perchlorate exchange reaction")

        # Add chlorite metabolite if not present
        if "clo2_e" not in model.metabolites:
            clo2_e = cobra.Metabolite("clo2_e", name="chlorite", compartment="e")
            model.add_metabolites([clo2_e])

            # Add chlorite exchange
            ex_clo2 = cobra.Reaction("EX_clo2_e")
            ex_clo2.name = "Chlorite exchange"
            ex_clo2.add_metabolites({clo2_e: -1})
            ex_clo2.bounds = (-1000, 1000)  # Can be consumed or produced
            model.add_reactions([ex_clo2])

            logger.info("Added chlorite exchange reaction")

    def save_model_analysis(self, analysis: Dict, filename: str = None) -> Path:
        """Save model analysis results."""

        if filename is None:
            filename = (
                f"model_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        output_path = self.model_dir / filename

        # Convert any numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Recursively convert numpy types
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert_numpy(obj)

        analysis_clean = recursive_convert(analysis)

        with open(output_path, "w") as f:
            json.dump(analysis_clean, f, indent=2, default=str)

        logger.info(f"Saved model analysis to {output_path}")
        return output_path

    def visualize_model_stats(self, analysis: Dict) -> None:
        """Create visualizations of model statistics."""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("B. subtilis Model Analysis", fontsize=16)

        # Basic statistics
        ax1 = axes[0, 0]
        basic_stats = analysis["basic_stats"]
        ax1.bar(basic_stats.keys(), basic_stats.values())
        ax1.set_title("Model Components")
        ax1.set_ylabel("Count")

        # Reaction types
        ax2 = axes[0, 1]
        reaction_stats = analysis["reaction_stats"]
        ax2.pie(
            reaction_stats.values(), labels=reaction_stats.keys(), autopct="%1.1f%%"
        )
        ax2.set_title("Reaction Types")

        # Pathway coverage
        ax3 = axes[1, 0]
        if "pathway_coverage" in analysis:
            pathways = list(analysis["pathway_coverage"].keys())
            coverages = [analysis["pathway_coverage"][p]["coverage"] for p in pathways]
            ax3.barh(pathways, coverages)
            ax3.set_title("Pathway Coverage")
            ax3.set_xlabel("Coverage Fraction")
            ax3.set_xlim(0, 1)

        # Growth conditions
        ax4 = axes[1, 1]
        if "growth_analysis" in analysis and isinstance(
            analysis["growth_analysis"], dict
        ):
            growth_data = {
                k: v
                for k, v in analysis["growth_analysis"].items()
                if v is not None and isinstance(v, (int, float))
            }
            if growth_data:
                ax4.bar(growth_data.keys(), growth_data.values())
                ax4.set_title("Growth Under Different Conditions")
                ax4.set_ylabel("Growth Rate")
                ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save plot
        plot_path = self.model_dir / "model_analysis_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved analysis plots to {plot_path}")

        plt.show()


def main():
    """Main function for model loading and analysis."""

    logger.info("Starting B. subtilis model loading and analysis")

    # Initialize loader
    loader = BSubtilisModelLoader()

    # Load model
    try:
        model = loader.load_model("iYO844")
    except:
        logger.warning("Failed to load iYO844, trying iBsu1103...")
        try:
            model = loader.load_model("iBsu1103")
        except:
            logger.error("Failed to load any model, creating fallback...")
            model = loader.load_model("iYO844")  # Will create fallback

    # Analyze model
    analysis = loader.analyze_model(model)

    # Save analysis
    analysis_file = loader.save_model_analysis(analysis)

    # Create visualizations
    loader.visualize_model_stats(analysis)

    # Apply Mars constraints
    mars_model = loader.apply_mars_constraints(model)

    # Analyze Mars-constrained model
    mars_analysis = loader.analyze_model(mars_model)
    mars_analysis_file = loader.save_model_analysis(
        mars_analysis, "mars_model_analysis.json"
    )

    # Save Mars model
    mars_model_path = loader.model_dir / "bsubtilis_mars_constrained.xml"
    cobra.io.write_sbml_model(mars_model, mars_model_path)

    logger.info("Model loading and analysis completed successfully!")
    logger.info(f"Standard analysis: {analysis_file}")
    logger.info(f"Mars analysis: {mars_analysis_file}")
    logger.info(f"Mars model: {mars_model_path}")


if __name__ == "__main__":
    main()
