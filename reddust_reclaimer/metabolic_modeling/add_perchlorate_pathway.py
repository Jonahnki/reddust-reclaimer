#!/usr/bin/env python3
"""
Perchlorate Pathway Integration for RedDust Reclaimer Project
===========================================================

This module integrates the perchlorate detoxification pathway (pcrA/pcrB/cld)
into the B. subtilis metabolic model for Mars environment simulation.

Author: RedDust Reclaimer AI
Date: 2024
License: MIT
"""

import cobra
import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from load_base_model import BSubtilisModelLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PerchloratePathwayIntegrator:
    """
    Integrates perchlorate detoxification pathway into B. subtilis metabolic models.

    Implements the complete pathway:
    ClO4- + 2H+ + 2e- -> ClO2- + H2O (perchlorate reductase, pcrA/pcrB)
    ClO2- -> Cl- + O2 (chlorite dismutase, cld)
    """

    def __init__(self, model_loader: BSubtilisModelLoader = None):
        """
        Initialize the pathway integrator.

        Args:
            model_loader: BSubtilisModelLoader instance
        """
        if model_loader is None:
            self.loader = BSubtilisModelLoader()
        else:
            self.loader = model_loader

        # Perchlorate pathway reaction definitions
        self.pathway_reactions = {
            "PCR": {
                "name": "Perchlorate reductase",
                "equation": "clo4_c + 2 h_c + 2 fadh2_c -> clo2_c + h2o_c + 2 fad_c",
                "genes": ["pcrA", "pcrB"],
                "enzyme_complex": "perchlorate reductase (pcrA/pcrB)",
                "ec_number": "EC 1.97.1.6",
                "cofactors": ["molybdenum", "fad", "heme"],
                "bounds": (0, 1000),
                "mars_optimized": True,
            },
            "CLD": {
                "name": "Chlorite dismutase",
                "equation": "clo2_c -> cl_c + 0.5 o2_c",
                "genes": ["cld"],
                "enzyme_complex": "chlorite dismutase",
                "ec_number": "EC 1.13.11.49",
                "cofactors": ["heme"],
                "bounds": (0, 1000),
                "mars_optimized": True,
            },
            "CLO4_TRANSPORT": {
                "name": "Perchlorate transport",
                "equation": "clo4_e -> clo4_c",
                "genes": ["perchlorate_transporter"],  # Hypothetical transporter
                "enzyme_complex": "perchlorate transporter",
                "bounds": (0, 100),
                "mars_optimized": True,
            },
            "CLO2_TRANSPORT": {
                "name": "Chlorite transport",
                "equation": "clo2_c -> clo2_e",
                "genes": ["chlorite_transporter"],  # Hypothetical transporter
                "enzyme_complex": "chlorite transporter",
                "bounds": (-1000, 1000),
                "mars_optimized": True,
            },
            "CL_TRANSPORT": {
                "name": "Chloride transport",
                "equation": "cl_c -> cl_e",
                "genes": ["chloride_transporter"],  # Existing chloride transporter
                "enzyme_complex": "chloride transporter",
                "bounds": (-1000, 1000),
                "mars_optimized": False,
            },
        }

        # Metabolite definitions
        self.pathway_metabolites = {
            "clo4_c": {
                "name": "perchlorate",
                "formula": "ClO4",
                "charge": -1,
                "compartment": "c",
            },
            "clo4_e": {
                "name": "perchlorate",
                "formula": "ClO4",
                "charge": -1,
                "compartment": "e",
            },
            "clo2_c": {
                "name": "chlorite",
                "formula": "ClO2",
                "charge": -1,
                "compartment": "c",
            },
            "clo2_e": {
                "name": "chlorite",
                "formula": "ClO2",
                "charge": -1,
                "compartment": "e",
            },
            "cl_c": {
                "name": "chloride",
                "formula": "Cl",
                "charge": -1,
                "compartment": "c",
            },
            "cl_e": {
                "name": "chloride",
                "formula": "Cl",
                "charge": -1,
                "compartment": "e",
            },
        }

        # Gene definitions for the pathway
        self.pathway_genes = {
            "pcrA": {
                "name": "perchlorate reductase subunit A",
                "locus_tag": "BSU_pcrA",
                "protein_name": "perchlorate reductase alpha subunit",
                "function": "perchlorate reduction, molybdenum-containing subunit",
                "mars_optimized": True,
            },
            "pcrB": {
                "name": "perchlorate reductase subunit B",
                "locus_tag": "BSU_pcrB",
                "protein_name": "perchlorate reductase beta subunit",
                "function": "electron transport for perchlorate reduction",
                "mars_optimized": True,
            },
            "cld": {
                "name": "chlorite dismutase",
                "locus_tag": "BSU_cld",
                "protein_name": "chlorite dismutase",
                "function": "chlorite dismutation to chloride and oxygen",
                "mars_optimized": True,
            },
        }

    def integrate_pathway(self, model: cobra.Model) -> cobra.Model:
        """
        Integrate the complete perchlorate detoxification pathway into the model.

        Args:
            model: B. subtilis metabolic model

        Returns:
            Model with integrated pathway
        """
        logger.info("Integrating perchlorate detoxification pathway...")

        # Work with a copy to avoid modifying the original
        integrated_model = model.copy()

        # Add genes
        self._add_pathway_genes(integrated_model)

        # Add metabolites
        self._add_pathway_metabolites(integrated_model)

        # Add reactions
        self._add_pathway_reactions(integrated_model)

        # Add exchange reactions
        self._add_exchange_reactions(integrated_model)

        # Set up electron transport coupling
        self._setup_electron_transport(integrated_model)

        # Validate the integrated pathway
        self._validate_pathway_integration(integrated_model)

        logger.info("Pathway integration completed successfully")
        return integrated_model

    def _add_pathway_genes(self, model: cobra.Model) -> None:
        """Add pathway genes to the model."""

        logger.info("Adding pathway genes...")

        for gene_id, gene_info in self.pathway_genes.items():
            if gene_id not in model.genes:
                gene = cobra.Gene(gene_id)
                gene.name = gene_info["name"]
                gene.annotation = {
                    "locus_tag": gene_info["locus_tag"],
                    "protein_name": gene_info["protein_name"],
                    "function": gene_info["function"],
                    "mars_optimized": gene_info["mars_optimized"],
                }
                model.genes.add(gene)
                logger.info(f"Added gene: {gene_id}")

    def _add_pathway_metabolites(self, model: cobra.Model) -> None:
        """Add pathway metabolites to the model."""

        logger.info("Adding pathway metabolites...")

        for met_id, met_info in self.pathway_metabolites.items():
            if met_id not in model.metabolites:
                metabolite = cobra.Metabolite(
                    id=met_id,
                    name=met_info["name"],
                    formula=met_info["formula"],
                    charge=met_info["charge"],
                    compartment=met_info["compartment"],
                )
                model.add_metabolites([metabolite])
                logger.info(f"Added metabolite: {met_id}")

    def _add_pathway_reactions(self, model: cobra.Model) -> None:
        """Add pathway reactions to the model."""

        logger.info("Adding pathway reactions...")

        for rxn_id, rxn_info in self.pathway_reactions.items():
            if rxn_id not in model.reactions:
                reaction = self._create_reaction(rxn_id, rxn_info, model)
                model.add_reactions([reaction])
                logger.info(f"Added reaction: {rxn_id} - {rxn_info['name']}")

    def _create_reaction(
        self, rxn_id: str, rxn_info: Dict, model: cobra.Model
    ) -> cobra.Reaction:
        """Create a reaction object from reaction information."""

        reaction = cobra.Reaction(rxn_id)
        reaction.name = rxn_info["name"]
        reaction.bounds = rxn_info["bounds"]

        # Parse the equation and add metabolites
        equation = rxn_info["equation"]
        metabolites_dict = self._parse_reaction_equation(equation, model)
        reaction.add_metabolites(metabolites_dict)

        # Add gene associations
        if "genes" in rxn_info and rxn_info["genes"]:
            gene_reaction_rule = " and ".join(rxn_info["genes"])
            reaction.gene_reaction_rule = gene_reaction_rule

        # Add annotations
        reaction.annotation = {
            "enzyme_complex": rxn_info.get("enzyme_complex", ""),
            "ec_number": rxn_info.get("ec_number", ""),
            "cofactors": rxn_info.get("cofactors", []),
            "mars_optimized": rxn_info.get("mars_optimized", False),
        }

        return reaction

    def _parse_reaction_equation(self, equation: str, model: cobra.Model) -> Dict:
        """Parse a reaction equation string into metabolite coefficients."""

        # Split into reactants and products
        if "->" not in equation:
            raise ValueError(f"Invalid reaction equation: {equation}")

        reactants_str, products_str = equation.split("->")

        metabolites_dict = {}

        # Process reactants (negative coefficients)
        for term in reactants_str.split("+"):
            term = term.strip()
            if term:
                coeff, met_id = self._parse_metabolite_term(term)
                if met_id in model.metabolites:
                    metabolites_dict[model.metabolites.get_by_id(met_id)] = -coeff
                else:
                    logger.warning(f"Metabolite {met_id} not found in model")

        # Process products (positive coefficients)
        for term in products_str.split("+"):
            term = term.strip()
            if term:
                coeff, met_id = self._parse_metabolite_term(term)
                if met_id in model.metabolites:
                    metabolites_dict[model.metabolites.get_by_id(met_id)] = coeff
                else:
                    logger.warning(f"Metabolite {met_id} not found in model")

        return metabolites_dict

    def _parse_metabolite_term(self, term: str) -> Tuple[float, str]:
        """Parse a single metabolite term to extract coefficient and ID."""

        term = term.strip()

        # Check if there's a coefficient
        parts = term.split(" ")
        if len(parts) == 1:
            # No explicit coefficient, assume 1
            return 1.0, parts[0]
        elif len(parts) == 2:
            # Coefficient and metabolite ID
            try:
                coeff = float(parts[0])
                return coeff, parts[1]
            except ValueError:
                # First part is not a number, assume coefficient is 1
                return 1.0, term
        else:
            # More complex term, take the last part as metabolite ID
            return 1.0, parts[-1]

    def _add_exchange_reactions(self, model: cobra.Model) -> None:
        """Add exchange reactions for pathway metabolites."""

        logger.info("Adding exchange reactions...")

        exchange_metabolites = ["clo4_e", "clo2_e", "cl_e"]

        for met_id in exchange_metabolites:
            ex_rxn_id = f"EX_{met_id}"

            if ex_rxn_id not in model.reactions and met_id in model.metabolites:
                ex_reaction = cobra.Reaction(ex_rxn_id)
                ex_reaction.name = (
                    f"{model.metabolites.get_by_id(met_id).name} exchange"
                )
                ex_reaction.add_metabolites({model.metabolites.get_by_id(met_id): -1})

                # Set appropriate bounds
                if met_id == "clo4_e":
                    ex_reaction.bounds = (-100, 0)  # Can only be consumed
                elif met_id == "clo2_e":
                    ex_reaction.bounds = (-1000, 1000)  # Can be consumed or produced
                else:  # cl_e
                    ex_reaction.bounds = (-1000, 1000)  # Can be consumed or produced

                model.add_reactions([ex_reaction])
                logger.info(f"Added exchange reaction: {ex_rxn_id}")

    def _setup_electron_transport(self, model: cobra.Model) -> None:
        """Set up electron transport coupling for perchlorate reductase."""

        logger.info("Setting up electron transport coupling...")

        # The perchlorate reductase requires electrons, typically from FADH2
        # We need to ensure proper electron donor availability

        # Check if FADH2/FAD are available in the model
        fadh2_present = any("fadh2" in met.id for met in model.metabolites)
        fad_present = any("fad" in met.id for met in model.metabolites)

        if not fadh2_present or not fad_present:
            logger.warning(
                "FAD/FADH2 not found in model, adding basic electron transport"
            )
            self._add_basic_electron_transport(model)

        # Ensure proper coupling with respiratory chain
        self._couple_with_respiratory_chain(model)

    def _add_basic_electron_transport(self, model: cobra.Model) -> None:
        """Add basic electron transport components if missing."""

        # Add FAD/FADH2 if not present
        if "fad_c" not in model.metabolites:
            fad = cobra.Metabolite(
                "fad_c", name="FAD", formula="C27H33N9O15P2", compartment="c"
            )
            model.add_metabolites([fad])

        if "fadh2_c" not in model.metabolites:
            fadh2 = cobra.Metabolite(
                "fadh2_c", name="FADH2", formula="C27H35N9O15P2", compartment="c"
            )
            model.add_metabolites([fadh2])

        # Add FADH2 regeneration reaction (simplified)
        if "FADH2_REGEN" not in model.reactions:
            fadh2_regen = cobra.Reaction("FADH2_REGEN")
            fadh2_regen.name = "FADH2 regeneration (simplified)"
            fadh2_regen.add_metabolites(
                {
                    model.metabolites.fad_c: -1,
                    model.metabolites.nadh_c: (
                        -1 if "nadh_c" in model.metabolites else 0
                    ),
                    model.metabolites.fadh2_c: 1,
                    model.metabolites.nad_c: 1 if "nad_c" in model.metabolites else 0,
                }
            )
            fadh2_regen.bounds = (0, 1000)
            model.add_reactions([fadh2_regen])

    def _couple_with_respiratory_chain(self, model: cobra.Model) -> None:
        """Couple perchlorate reduction with the respiratory electron transport chain."""

        # Find respiratory chain reactions
        respiratory_reactions = [
            rxn
            for rxn in model.reactions
            if any(
                term in rxn.name.lower()
                for term in [
                    "cytochrome",
                    "ubiquinone",
                    "nadh dehydrogenase",
                    "respiratory",
                ]
            )
        ]

        if respiratory_reactions:
            logger.info(
                f"Found {len(respiratory_reactions)} respiratory chain reactions"
            )

            # Add coupling reaction to link perchlorate reduction with ATP synthesis
            if "PCR_ATP_COUPLING" not in model.reactions:
                coupling_rxn = cobra.Reaction("PCR_ATP_COUPLING")
                coupling_rxn.name = "Perchlorate reduction ATP coupling"

                # Simplified coupling: 2 electrons from perchlorate reduction -> ATP
                coupling_metabolites = {}
                if (
                    "adp_c" in model.metabolites
                    and "atp_c" in model.metabolites
                    and "pi_c" in model.metabolites
                ):
                    coupling_metabolites = {
                        model.metabolites.adp_c: -1,
                        model.metabolites.pi_c: -1,
                        model.metabolites.atp_c: 1,
                        model.metabolites.h2o_c: (
                            1 if "h2o_c" in model.metabolites else 0
                        ),
                    }
                    coupling_rxn.add_metabolites(coupling_metabolites)
                    coupling_rxn.bounds = (0, 1000)
                    model.add_reactions([coupling_rxn])
                    logger.info("Added ATP coupling for perchlorate reduction")

    def _validate_pathway_integration(self, model: cobra.Model) -> None:
        """Validate the integrated pathway."""

        logger.info("Validating pathway integration...")

        # Check that all pathway reactions are present
        missing_reactions = []
        for rxn_id in self.pathway_reactions.keys():
            if rxn_id not in model.reactions:
                missing_reactions.append(rxn_id)

        if missing_reactions:
            logger.warning(f"Missing reactions: {missing_reactions}")

        # Check that all pathway metabolites are present
        missing_metabolites = []
        for met_id in self.pathway_metabolites.keys():
            if met_id not in model.metabolites:
                missing_metabolites.append(met_id)

        if missing_metabolites:
            logger.warning(f"Missing metabolites: {missing_metabolites}")

        # Test pathway functionality
        try:
            # Set up a test scenario with perchlorate consumption
            with model:
                # Enable perchlorate uptake
                if "EX_clo4_e" in model.reactions:
                    model.reactions.EX_clo4_e.lower_bound = -10

                # Optimize and check if perchlorate can be consumed
                solution = model.optimize()

                if solution.status == "optimal":
                    # Check flux through perchlorate reactions
                    pcr_flux = solution.fluxes.get("PCR", 0)
                    cld_flux = solution.fluxes.get("CLD", 0)

                    if pcr_flux > 0 or cld_flux > 0:
                        logger.info(
                            "Pathway validation successful - detoxification reactions active"
                        )
                    else:
                        logger.warning(
                            "Pathway validation warning - no flux through detoxification reactions"
                        )
                else:
                    logger.warning(
                        f"Pathway validation failed - optimization status: {solution.status}"
                    )

        except Exception as e:
            logger.warning(f"Pathway validation error: {e}")

        # Check mass balance
        self._check_mass_balance(model)

    def _check_mass_balance(self, model: cobra.Model) -> None:
        """Check mass balance of pathway reactions."""

        logger.info("Checking mass balance...")

        pathway_reaction_ids = ["PCR", "CLD"]

        for rxn_id in pathway_reaction_ids:
            if rxn_id in model.reactions:
                reaction = model.reactions.get_by_id(rxn_id)

                # Check if reaction is mass balanced
                try:
                    # This will raise an exception if not mass balanced
                    elements = {}
                    for metabolite, coefficient in reaction.metabolites.items():
                        if hasattr(metabolite, "formula") and metabolite.formula:
                            for element, count in metabolite.elements.items():
                                if element in elements:
                                    elements[element] += coefficient * count
                                else:
                                    elements[element] = coefficient * count

                    # Check if all elements balance to zero
                    unbalanced = {
                        elem: count
                        for elem, count in elements.items()
                        if abs(count) > 1e-6
                    }

                    if unbalanced:
                        logger.warning(
                            f"Reaction {rxn_id} is not mass balanced: {unbalanced}"
                        )
                    else:
                        logger.info(f"Reaction {rxn_id} is mass balanced")

                except Exception as e:
                    logger.warning(f"Could not check mass balance for {rxn_id}: {e}")

    def analyze_pathway_performance(self, model: cobra.Model) -> Dict:
        """Analyze the performance of the integrated pathway."""

        logger.info("Analyzing pathway performance...")

        analysis = {}

        # Test different perchlorate concentrations
        clo4_concentrations = [0, 1, 5, 10, 20, 50]  # mmol/gDW/h
        results = []

        for clo4_conc in clo4_concentrations:
            with model:
                # Set perchlorate availability
                if "EX_clo4_e" in model.reactions:
                    model.reactions.EX_clo4_e.lower_bound = -clo4_conc

                try:
                    solution = model.optimize()

                    if solution.status == "optimal":
                        result = {
                            "clo4_concentration": clo4_conc,
                            "growth_rate": solution.objective_value,
                            "pcr_flux": solution.fluxes.get("PCR", 0),
                            "cld_flux": solution.fluxes.get("CLD", 0),
                            "clo4_uptake": solution.fluxes.get("EX_clo4_e", 0),
                            "cl_production": solution.fluxes.get("EX_cl_e", 0),
                        }
                    else:
                        result = {
                            "clo4_concentration": clo4_conc,
                            "growth_rate": 0,
                            "pcr_flux": 0,
                            "cld_flux": 0,
                            "clo4_uptake": 0,
                            "cl_production": 0,
                            "status": solution.status,
                        }

                    results.append(result)

                except Exception as e:
                    logger.warning(
                        f"Analysis failed for ClO4 concentration {clo4_conc}: {e}"
                    )
                    results.append({"clo4_concentration": clo4_conc, "error": str(e)})

        analysis["concentration_response"] = results

        # Calculate pathway efficiency
        analysis["pathway_efficiency"] = self._calculate_pathway_efficiency(results)

        # Test Mars conditions
        analysis["mars_performance"] = self._test_mars_conditions(model)

        return analysis

    def _calculate_pathway_efficiency(self, results: List[Dict]) -> Dict:
        """Calculate pathway efficiency metrics."""

        efficiency = {}

        # Filter successful results
        successful_results = [
            r for r in results if "error" not in r and r.get("growth_rate", 0) > 0
        ]

        if successful_results:
            # Maximum detoxification rate
            max_detox_rate = max(r["pcr_flux"] for r in successful_results)
            efficiency["max_detoxification_rate"] = max_detox_rate

            # Efficiency at different concentrations
            efficiency["efficiency_by_concentration"] = {
                r["clo4_concentration"]: r["pcr_flux"] / max(r["clo4_concentration"], 1)
                for r in successful_results
                if r["clo4_concentration"] > 0
            }

            # Growth impact
            growth_without_clo4 = next(
                (
                    r["growth_rate"]
                    for r in successful_results
                    if r["clo4_concentration"] == 0
                ),
                0,
            )
            efficiency["growth_impact"] = {
                r["clo4_concentration"]: (
                    r["growth_rate"] / max(growth_without_clo4, 1e-6)
                )
                for r in successful_results
            }

        return efficiency

    def _test_mars_conditions(self, model: cobra.Model) -> Dict:
        """Test pathway performance under Mars conditions."""

        logger.info("Testing pathway under Mars conditions...")

        mars_results = {}

        # Apply Mars constraints
        mars_model = self.loader.apply_mars_constraints(model)

        try:
            # Test with moderate perchlorate load
            with mars_model:
                if "EX_clo4_e" in mars_model.reactions:
                    mars_model.reactions.EX_clo4_e.lower_bound = -10  # 10 mmol/gDW/h

                solution = mars_model.optimize()

                if solution.status == "optimal":
                    mars_results = {
                        "status": "optimal",
                        "growth_rate": solution.objective_value,
                        "pcr_flux": solution.fluxes.get("PCR", 0),
                        "cld_flux": solution.fluxes.get("CLD", 0),
                        "detoxification_efficiency": solution.fluxes.get("PCR", 0)
                        / 10.0,  # fraction of available ClO4 processed
                        "survival_feasible": solution.objective_value
                        > 0.01,  # Minimal growth threshold
                    }
                else:
                    mars_results = {
                        "status": solution.status,
                        "survival_feasible": False,
                    }

        except Exception as e:
            mars_results = {"error": str(e), "survival_feasible": False}

        return mars_results

    def visualize_pathway_performance(self, analysis: Dict, output_dir: Path) -> None:
        """Create visualizations of pathway performance."""

        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract concentration response data
        if "concentration_response" in analysis:
            results = analysis["concentration_response"]
            successful_results = [r for r in results if "error" not in r]

            if successful_results:
                df = pd.DataFrame(successful_results)

                # Create subplots
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(
                    "Perchlorate Detoxification Pathway Performance", fontsize=16
                )

                # Growth rate vs perchlorate concentration
                ax1 = axes[0, 0]
                ax1.plot(df["clo4_concentration"], df["growth_rate"], "b-o")
                ax1.set_xlabel("Perchlorate Concentration (mmol/gDW/h)")
                ax1.set_ylabel("Growth Rate (1/h)")
                ax1.set_title("Growth vs Perchlorate Load")
                ax1.grid(True, alpha=0.3)

                # Detoxification flux vs concentration
                ax2 = axes[0, 1]
                ax2.plot(
                    df["clo4_concentration"], df["pcr_flux"], "r-o", label="PCR flux"
                )
                ax2.plot(
                    df["clo4_concentration"], df["cld_flux"], "g-o", label="CLD flux"
                )
                ax2.set_xlabel("Perchlorate Concentration (mmol/gDW/h)")
                ax2.set_ylabel("Flux (mmol/gDW/h)")
                ax2.set_title("Detoxification Enzyme Activity")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Chloride production
                ax3 = axes[1, 0]
                ax3.plot(df["clo4_concentration"], -df["cl_production"], "m-o")
                ax3.set_xlabel("Perchlorate Concentration (mmol/gDW/h)")
                ax3.set_ylabel("Chloride Production (mmol/gDW/h)")
                ax3.set_title("Chloride Output")
                ax3.grid(True, alpha=0.3)

                # Efficiency
                ax4 = axes[1, 1]
                efficiency_data = [
                    r["pcr_flux"] / max(r["clo4_concentration"], 1)
                    for r in successful_results
                    if r["clo4_concentration"] > 0
                ]
                conc_data = [
                    r["clo4_concentration"]
                    for r in successful_results
                    if r["clo4_concentration"] > 0
                ]
                if efficiency_data:
                    ax4.plot(conc_data, efficiency_data, "c-o")
                    ax4.set_xlabel("Perchlorate Concentration (mmol/gDW/h)")
                    ax4.set_ylabel("Detoxification Efficiency")
                    ax4.set_title("Pathway Efficiency")
                    ax4.grid(True, alpha=0.3)

                plt.tight_layout()

                # Save plot
                plot_path = output_dir / "pathway_performance.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                logger.info(f"Saved performance plots to {plot_path}")

                plt.show()

    def save_integrated_model(
        self, model: cobra.Model, filename: str, output_dir: Path
    ) -> Path:
        """Save the integrated model."""

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        # Save as SBML
        cobra.io.write_sbml_model(model, output_path)

        logger.info(f"Saved integrated model to {output_path}")
        return output_path


def main():
    """Main function for pathway integration."""

    logger.info("Starting perchlorate pathway integration")

    # Initialize components
    loader = BSubtilisModelLoader()
    integrator = PerchloratePathwayIntegrator(loader)

    # Load base model
    try:
        base_model = loader.load_model("iYO844")
    except:
        logger.warning("Failed to load iYO844, creating fallback model...")
        base_model = loader.load_model("iYO844")  # Will create fallback

    # Integrate pathway
    integrated_model = integrator.integrate_pathway(base_model)

    # Analyze pathway performance
    analysis = integrator.analyze_pathway_performance(integrated_model)

    # Save results
    output_dir = Path("../../models/metabolic/")

    # Save integrated model
    model_path = integrator.save_integrated_model(
        integrated_model, "bsubtilis_with_perchlorate_pathway.xml", output_dir
    )

    # Save analysis results
    analysis_path = output_dir / "pathway_performance_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # Create visualizations
    integrator.visualize_pathway_performance(analysis, output_dir)

    logger.info("Perchlorate pathway integration completed successfully!")
    logger.info(f"Integrated model: {model_path}")
    logger.info(f"Analysis results: {analysis_path}")


if __name__ == "__main__":
    main()
