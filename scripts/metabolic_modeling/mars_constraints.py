#!/usr/bin/env python3
"""
Mars Environmental Constraints for RedDust Reclaimer Project
==========================================================

This module implements detailed Mars environmental constraints for metabolic modeling,
including temperature, pressure, radiation, and chemical composition effects.

Author: RedDust Reclaimer AI
Date: 2024
License: MIT
"""

import cobra
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarsSeason(Enum):
    """Mars seasonal variations."""
    SPRING = "spring"
    SUMMER = "summer" 
    AUTUMN = "autumn"
    WINTER = "winter"

class MarsLocation(Enum):
    """Mars geographical locations with different conditions."""
    EQUATORIAL = "equatorial"
    POLAR = "polar"
    UNDERGROUND = "underground"
    POLAR_ICE = "polar_ice"

@dataclass
class MarsEnvironmentalConditions:
    """Mars environmental conditions data structure."""
    
    # Temperature conditions
    temperature_celsius: float = 4  # Average surface temperature
    temperature_range: Tuple[float, float] = (-80, 20)  # Min/max temperatures
    
    # Atmospheric conditions
    atmospheric_pressure: float = 0.006  # atm (0.6% of Earth)
    co2_partial_pressure: float = 0.0057  # atm (95% of atmosphere)
    o2_partial_pressure: float = 0.0013  # atm (0.13% of atmosphere)
    n2_partial_pressure: float = 0.00015  # atm (2.7% of atmosphere)
    ar_partial_pressure: float = 0.00012  # atm (1.9% of atmosphere)
    
    # Water availability
    water_activity: float = 0.03  # Very low water activity
    relative_humidity: float = 0.001  # 0.1% RH
    
    # Chemical composition
    perchlorate_concentration: float = 0.5  # % w/w in regolith
    chloride_concentration: float = 0.1  # % w/w in regolith
    sulfate_concentration: float = 0.2  # % w/w in regolith
    iron_oxide_concentration: float = 10.0  # % w/w (rust content)
    
    # Radiation environment
    cosmic_radiation: float = 0.67  # mSv/day (240 mSv/year)
    uv_radiation: float = 100  # Times higher than Earth
    
    # Seasonal/location modifiers
    season: MarsSeason = MarsSeason.SUMMER
    location: MarsLocation = MarsLocation.EQUATORIAL
    
    # Soil properties
    ph: float = 8.5  # Alkaline soil
    salinity: float = 5.0  # g/L equivalent NaCl
    
    # Day/night cycle effects
    sol_duration: float = 24.6  # Mars day length in hours
    day_night_temp_variation: float = 100  # °C temperature swing

class MarsConstraintEngine:
    """
    Advanced Mars environmental constraint engine for metabolic modeling.
    
    Implements realistic Mars conditions with seasonal, geographical, and
    temporal variations for accurate survival predictions.
    """
    
    def __init__(self):
        """Initialize the Mars constraint engine."""
        
        # Define base Mars conditions
        self.base_conditions = MarsEnvironmentalConditions()
        
        # Enzymatic efficiency multipliers under Mars conditions
        self.enzyme_efficiency_factors = {
            'cold_temperature': 0.1,  # 10% efficiency at low temperatures
            'low_pressure': 0.8,      # 80% efficiency at low pressure
            'high_radiation': 0.9,    # 90% efficiency with radiation damage
            'low_water_activity': 0.7, # 70% efficiency with low water
            'alkaline_ph': 0.85       # 85% efficiency at high pH
        }
        
        # Metabolic pathway sensitivities
        self.pathway_sensitivities = {
            'glycolysis': {
                'temperature_coefficient': 2.5,
                'pressure_sensitive': False,
                'radiation_sensitive': True
            },
            'tca_cycle': {
                'temperature_coefficient': 3.0,
                'pressure_sensitive': True,
                'radiation_sensitive': True
            },
            'electron_transport': {
                'temperature_coefficient': 4.0,
                'pressure_sensitive': True,
                'radiation_sensitive': True
            },
            'perchlorate_detox': {
                'temperature_coefficient': 2.0,  # Cold-adapted enzymes
                'pressure_sensitive': False,
                'radiation_sensitive': False  # Robust pathway
            }
        }
        
        # Stress response thresholds
        self.stress_thresholds = {
            'cold_shock': -10,  # °C
            'heat_shock': 35,   # °C
            'osmotic_stress': 3.0,  # g/L NaCl equivalent
            'oxidative_stress': 50,  # Relative units
            'radiation_damage': 0.5   # mSv/day threshold
        }
    
    def apply_mars_constraints(self, model: cobra.Model, 
                             conditions: MarsEnvironmentalConditions = None,
                             severity: str = 'realistic') -> cobra.Model:
        """
        Apply comprehensive Mars environmental constraints to a metabolic model.
        
        Args:
            model: Metabolic model to constrain
            conditions: Specific Mars conditions (uses base if None)
            severity: Constraint severity ('mild', 'realistic', 'extreme')
            
        Returns:
            Mars-constrained model
        """
        
        if conditions is None:
            conditions = self.base_conditions
        
        logger.info(f"Applying Mars constraints with {severity} severity")
        logger.info(f"Location: {conditions.location.value}, Season: {conditions.season.value}")
        
        # Work with model copy
        mars_model = model.copy()
        
        # Apply temperature constraints
        self._apply_temperature_constraints(mars_model, conditions, severity)
        
        # Apply atmospheric constraints
        self._apply_atmospheric_constraints(mars_model, conditions, severity)
        
        # Apply water availability constraints
        self._apply_water_constraints(mars_model, conditions, severity)
        
        # Apply radiation constraints
        self._apply_radiation_constraints(mars_model, conditions, severity)
        
        # Apply chemical environment constraints
        self._apply_chemical_constraints(mars_model, conditions, severity)
        
        # Apply stress response modifications
        self._apply_stress_responses(mars_model, conditions, severity)
        
        # Add Mars-specific exchange reactions
        self._add_mars_exchanges(mars_model, conditions)
        
        # Validate constraints
        self._validate_mars_constraints(mars_model, conditions)
        
        logger.info("Mars constraints applied successfully")
        return mars_model
    
    def _apply_temperature_constraints(self, model: cobra.Model, 
                                     conditions: MarsEnvironmentalConditions,
                                     severity: str) -> None:
        """Apply temperature-based constraints to enzymatic reactions."""
        
        logger.info(f"Applying temperature constraints ({conditions.temperature_celsius}°C)")
        
        # Calculate temperature factor using Arrhenius equation approximation
        earth_temp = 37  # °C, optimal for mesophiles
        mars_temp = conditions.temperature_celsius
        
        # Temperature factor calculation
        activation_energy = 50000  # J/mol, typical for enzymes
        R = 8.314  # J/mol/K, gas constant
        
        temp_factor = np.exp(-activation_energy/R * (1/(mars_temp + 273.15) - 1/(earth_temp + 273.15)))
        
        # Apply severity modification
        severity_factors = {'mild': 0.8, 'realistic': 0.5, 'extreme': 0.2}
        base_factor = severity_factors.get(severity, 0.5)
        
        # Cold adaptation factor for Mars-optimized enzymes
        mars_adapted_factor = 2.0  # Mars-adapted enzymes are twice as efficient in cold
        
        # Apply to all intracellular reactions
        for reaction in model.reactions:
            if not reaction.id.startswith('EX_'):  # Skip exchange reactions
                
                # Check if reaction is Mars-adapted
                is_mars_adapted = reaction.annotation.get('mars_optimized', False)
                
                if is_mars_adapted:
                    efficiency = min(1.0, temp_factor * mars_adapted_factor)
                else:
                    efficiency = temp_factor * base_factor
                
                # Apply efficiency to reaction bounds
                if reaction.upper_bound > 0:
                    reaction.upper_bound *= efficiency
                if reaction.lower_bound < 0:
                    reaction.lower_bound *= efficiency
        
        logger.info(f"Applied temperature factor: {temp_factor:.3f}, efficiency: {efficiency:.3f}")
    
    def _apply_atmospheric_constraints(self, model: cobra.Model,
                                     conditions: MarsEnvironmentalConditions,
                                     severity: str) -> None:
        """Apply atmospheric composition constraints."""
        
        logger.info("Applying atmospheric constraints")
        
        # Oxygen availability (very limited)
        if 'EX_o2_e' in model.reactions:
            o2_availability = conditions.o2_partial_pressure / 0.21  # Fraction of Earth O2
            model.reactions.EX_o2_e.lower_bound *= o2_availability
            logger.info(f"Reduced O2 availability to {o2_availability:.4f} of Earth levels")
        
        # CO2 availability (abundant)
        if 'EX_co2_e' in model.reactions:
            model.reactions.EX_co2_e.lower_bound = -1000  # Unlimited CO2
            logger.info("Set unlimited CO2 availability")
        
        # Nitrogen limitations
        if 'EX_nh4_e' in model.reactions:
            n2_factor = conditions.n2_partial_pressure / 0.78  # Fraction of Earth N2
            model.reactions.EX_nh4_e.lower_bound *= n2_factor * 0.1  # Limited fixation
            logger.info(f"Reduced nitrogen availability by factor {n2_factor:.4f}")
        
        # Pressure effects on gas solubility
        pressure_factor = conditions.atmospheric_pressure / 1.0  # atm
        
        gas_exchanges = ['EX_o2_e', 'EX_co2_e', 'EX_h2s_e', 'EX_h2_e']
        for ex_id in gas_exchanges:
            if ex_id in model.reactions and ex_id != 'EX_co2_e':  # Don't limit CO2
                reaction = model.reactions.get_by_id(ex_id)
                if reaction.lower_bound < 0:  # Gas uptake
                    reaction.lower_bound *= pressure_factor
    
    def _apply_water_constraints(self, model: cobra.Model,
                               conditions: MarsEnvironmentalConditions,
                               severity: str) -> None:
        """Apply water availability constraints."""
        
        logger.info(f"Applying water constraints (aw = {conditions.water_activity})")
        
        # Water availability factor
        water_factor = conditions.water_activity  # 0.03 for Mars vs 0.99 for Earth
        
        # Apply to water exchange
        if 'EX_h2o_e' in model.reactions:
            original_bound = model.reactions.EX_h2o_e.lower_bound
            model.reactions.EX_h2o_e.lower_bound = max(-10, original_bound * water_factor)
            logger.info(f"Reduced water availability: {original_bound} -> {model.reactions.EX_h2o_e.lower_bound}")
        
        # Osmotic stress affects membrane transport
        if conditions.salinity > self.stress_thresholds['osmotic_stress']:
            osmotic_factor = 1.0 / (1.0 + 0.1 * conditions.salinity)
            
            # Reduce efficiency of all transport reactions
            transport_reactions = [rxn for rxn in model.reactions 
                                 if len(set(met.compartment for met in rxn.metabolites)) > 1]
            
            for reaction in transport_reactions:
                if reaction.upper_bound > 0:
                    reaction.upper_bound *= osmotic_factor
                if reaction.lower_bound < 0:
                    reaction.lower_bound *= osmotic_factor
            
            logger.info(f"Applied osmotic stress factor: {osmotic_factor:.3f}")
    
    def _apply_radiation_constraints(self, model: cobra.Model,
                                   conditions: MarsEnvironmentalConditions,
                                   severity: str) -> None:
        """Apply radiation damage constraints."""
        
        logger.info(f"Applying radiation constraints ({conditions.cosmic_radiation} mSv/day)")
        
        # Radiation damage factor
        radiation_damage = min(1.0, conditions.cosmic_radiation / self.stress_thresholds['radiation_damage'])
        damage_factor = 1.0 - 0.1 * radiation_damage  # Up to 10% efficiency loss
        
        # DNA repair cost (increases maintenance)
        if 'ATPM' in model.reactions:  # ATP maintenance
            original_bound = model.reactions.ATPM.lower_bound
            repair_cost = radiation_damage * 0.5  # Additional ATP cost for DNA repair
            model.reactions.ATPM.lower_bound = original_bound - repair_cost
            logger.info(f"Increased ATP maintenance for DNA repair: {repair_cost:.3f}")
        
        # Protein damage affects all enzymatic reactions
        for reaction in model.reactions:
            if reaction.genes and not reaction.id.startswith('EX_'):
                if reaction.upper_bound > 0:
                    reaction.upper_bound *= damage_factor
                if reaction.lower_bound < 0:
                    reaction.lower_bound *= damage_factor
        
        logger.info(f"Applied radiation damage factor: {damage_factor:.3f}")
    
    def _apply_chemical_constraints(self, model: cobra.Model,
                                  conditions: MarsEnvironmentalConditions,
                                  severity: str) -> None:
        """Apply Mars-specific chemical environment constraints."""
        
        logger.info("Applying chemical environment constraints")
        
        # pH effects on enzymatic activity
        optimal_ph = 7.0
        ph_deviation = abs(conditions.ph - optimal_ph)
        ph_factor = max(0.3, 1.0 - 0.1 * ph_deviation)  # 10% loss per pH unit
        
        # Apply pH effects to all enzymatic reactions
        for reaction in model.reactions:
            if reaction.genes and not reaction.id.startswith('EX_'):
                if reaction.upper_bound > 0:
                    reaction.upper_bound *= ph_factor
                if reaction.lower_bound < 0:
                    reaction.lower_bound *= ph_factor
        
        logger.info(f"Applied pH factor ({conditions.ph}): {ph_factor:.3f}")
        
        # Add perchlorate availability
        if 'EX_clo4_e' in model.reactions:
            perchlorate_availability = -conditions.perchlorate_concentration * 20  # mmol/gDW/h
            model.reactions.EX_clo4_e.lower_bound = perchlorate_availability
            logger.info(f"Set perchlorate availability: {perchlorate_availability}")
        
        # Add sulfate availability (common in Mars soil)
        if 'EX_so4_e' in model.reactions:
            sulfate_availability = -conditions.sulfate_concentration * 10
            model.reactions.EX_so4_e.lower_bound = sulfate_availability
            logger.info(f"Set sulfate availability: {sulfate_availability}")
        
        # Iron availability (abundant as Fe2O3)
        if 'EX_fe3_e' in model.reactions:
            iron_availability = -conditions.iron_oxide_concentration
            model.reactions.EX_fe3_e.lower_bound = iron_availability
            logger.info(f"Set iron availability: {iron_availability}")
    
    def _apply_stress_responses(self, model: cobra.Model,
                              conditions: MarsEnvironmentalConditions,
                              severity: str) -> None:
        """Apply stress response metabolic modifications."""
        
        logger.info("Applying stress response modifications")
        
        stress_factors = {}
        
        # Cold stress
        if conditions.temperature_celsius < self.stress_thresholds['cold_shock']:
            stress_factors['cold_shock'] = True
            logger.info("Cold shock stress detected")
        
        # Osmotic stress
        if conditions.salinity > self.stress_thresholds['osmotic_stress']:
            stress_factors['osmotic_stress'] = True
            logger.info("Osmotic stress detected")
        
        # Radiation stress
        if conditions.cosmic_radiation > self.stress_thresholds['radiation_damage']:
            stress_factors['radiation_stress'] = True
            logger.info("Radiation stress detected")
        
        # Add stress response reactions if needed
        if stress_factors:
            self._add_stress_response_reactions(model, stress_factors)
    
    def _add_stress_response_reactions(self, model: cobra.Model, stress_factors: Dict) -> None:
        """Add stress response reactions to the model."""
        
        # Cold shock protein synthesis
        if stress_factors.get('cold_shock'):
            if 'COLD_SHOCK_RESPONSE' not in model.reactions:
                cold_response = cobra.Reaction('COLD_SHOCK_RESPONSE')
                cold_response.name = 'Cold shock protein synthesis'
                
                # Requires ATP and amino acids
                if 'atp_c' in model.metabolites and 'adp_c' in model.metabolites:
                    cold_response.add_metabolites({
                        model.metabolites.atp_c: -5,
                        model.metabolites.adp_c: 5,
                        model.metabolites.pi_c: 5 if 'pi_c' in model.metabolites else 0
                    })
                    cold_response.bounds = (0, 10)
                    model.add_reactions([cold_response])
                    logger.info("Added cold shock response reaction")
        
        # Osmotic stress response (trehalose synthesis)
        if stress_factors.get('osmotic_stress'):
            if 'OSMOTIC_RESPONSE' not in model.reactions:
                osmotic_response = cobra.Reaction('OSMOTIC_RESPONSE')
                osmotic_response.name = 'Osmotic stress response'
                
                # Simplified trehalose synthesis
                if 'glc__D_c' in model.metabolites and 'atp_c' in model.metabolites:
                    osmotic_response.add_metabolites({
                        model.metabolites.glc__D_c: -2,
                        model.metabolites.atp_c: -2,
                        model.metabolites.adp_c: 2 if 'adp_c' in model.metabolites else 0
                    })
                    osmotic_response.bounds = (0, 5)
                    model.add_reactions([osmotic_response])
                    logger.info("Added osmotic stress response reaction")
    
    def _add_mars_exchanges(self, model: cobra.Model, 
                          conditions: MarsEnvironmentalConditions) -> None:
        """Add Mars-specific exchange reactions."""
        
        logger.info("Adding Mars-specific exchange reactions")
        
        # Mars-specific compounds
        mars_compounds = {
            'perchlorate': ('clo4_e', -conditions.perchlorate_concentration * 20),
            'chlorite': ('clo2_e', (-1000, 1000)),  # Can be produced or consumed
            'chloride': ('cl_e', (-1000, 1000)),
            'sulfate': ('so4_e', -conditions.sulfate_concentration * 10),
            'iron_oxide': ('fe2o3_e', -conditions.iron_oxide_concentration),
            'regolith_dust': ('dust_e', -100)  # Abundant
        }
        
        for compound, (met_id, bounds) in mars_compounds.items():
            ex_id = f'EX_{met_id}'
            
            if ex_id not in model.reactions and met_id in model.metabolites:
                ex_reaction = cobra.Reaction(ex_id)
                ex_reaction.name = f'{compound} exchange (Mars environment)'
                ex_reaction.add_metabolites({model.metabolites.get_by_id(met_id): -1})
                
                if isinstance(bounds, tuple):
                    ex_reaction.bounds = bounds
                else:
                    ex_reaction.bounds = (bounds, 0)
                
                model.add_reactions([ex_reaction])
                logger.info(f"Added Mars exchange reaction: {ex_id}")
    
    def _validate_mars_constraints(self, model: cobra.Model,
                                 conditions: MarsEnvironmentalConditions) -> None:
        """Validate applied Mars constraints."""
        
        logger.info("Validating Mars constraints...")
        
        try:
            solution = model.optimize()
            
            if solution.status == 'optimal':
                growth_rate = solution.objective_value
                
                if growth_rate > 0:
                    logger.info(f"Mars constraints validation successful - growth rate: {growth_rate:.6f}")
                    
                    # Check if perchlorate detoxification is active
                    if 'PCR' in model.reactions:
                        pcr_flux = solution.fluxes.get('PCR', 0)
                        if pcr_flux > 0:
                            logger.info(f"Perchlorate detoxification active - flux: {pcr_flux:.6f}")
                        else:
                            logger.warning("Perchlorate detoxification inactive under Mars conditions")
                else:
                    logger.warning("No growth possible under applied Mars conditions")
            else:
                logger.warning(f"Mars constraints validation failed - status: {solution.status}")
        
        except Exception as e:
            logger.error(f"Mars constraints validation error: {e}")
    
    def generate_seasonal_conditions(self, base_location: MarsLocation = MarsLocation.EQUATORIAL) -> Dict[MarsSeason, MarsEnvironmentalConditions]:
        """Generate seasonal variations of Mars conditions."""
        
        seasonal_conditions = {}
        
        base_temp = self.base_conditions.temperature_celsius
        
        seasonal_modifiers = {
            MarsSeason.SPRING: {'temp_delta': 5, 'pressure_factor': 1.1, 'dust_factor': 1.2},
            MarsSeason.SUMMER: {'temp_delta': 10, 'pressure_factor': 1.2, 'dust_factor': 0.8},
            MarsSeason.AUTUMN: {'temp_delta': 0, 'pressure_factor': 1.0, 'dust_factor': 1.5},
            MarsSeason.WINTER: {'temp_delta': -15, 'pressure_factor': 0.8, 'dust_factor': 2.0}
        }
        
        for season, modifiers in seasonal_modifiers.items():
            conditions = MarsEnvironmentalConditions(
                temperature_celsius=base_temp + modifiers['temp_delta'],
                atmospheric_pressure=self.base_conditions.atmospheric_pressure * modifiers['pressure_factor'],
                season=season,
                location=base_location
            )
            seasonal_conditions[season] = conditions
        
        return seasonal_conditions
    
    def analyze_survival_envelope(self, model: cobra.Model) -> Dict:
        """Analyze survival envelope under various Mars conditions."""
        
        logger.info("Analyzing survival envelope under Mars conditions")
        
        survival_data = []
        
        # Test temperature range
        temperatures = np.linspace(-40, 25, 14)  # °C
        
        # Test different severities
        severities = ['mild', 'realistic', 'extreme']
        
        # Test different locations
        locations = [MarsLocation.EQUATORIAL, MarsLocation.POLAR, MarsLocation.UNDERGROUND]
        
        for temp in temperatures:
            for severity in severities:
                for location in locations:
                    
                    # Create test conditions
                    test_conditions = MarsEnvironmentalConditions(
                        temperature_celsius=temp,
                        location=location
                    )
                    
                    try:
                        # Apply constraints
                        test_model = self.apply_mars_constraints(model, test_conditions, severity)
                        
                        # Test growth
                        solution = test_model.optimize()
                        
                        survival_data.append({
                            'temperature': temp,
                            'severity': severity,
                            'location': location.value,
                            'growth_rate': solution.objective_value if solution.status == 'optimal' else 0,
                            'status': solution.status,
                            'survivable': solution.objective_value > 0.001 if solution.status == 'optimal' else False
                        })
                    
                    except Exception as e:
                        survival_data.append({
                            'temperature': temp,
                            'severity': severity,
                            'location': location.value,
                            'growth_rate': 0,
                            'status': 'error',
                            'survivable': False,
                            'error': str(e)
                        })
        
        return {'survival_envelope': survival_data}
    
    def visualize_survival_envelope(self, analysis: Dict, output_dir: Path) -> None:
        """Visualize survival envelope analysis."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(analysis['survival_envelope'])
        
        # Create survival heatmap
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('B. subtilis Survival Envelope on Mars', fontsize=16)
        
        severities = ['mild', 'realistic', 'extreme']
        
        for i, severity in enumerate(severities):
            ax = axes[i]
            
            # Filter data for this severity
            severity_data = df[df['severity'] == severity]
            
            # Create pivot table for heatmap
            pivot_data = severity_data.pivot_table(
                values='survivable', 
                index='temperature', 
                columns='location', 
                aggfunc='mean'
            )
            
            # Create heatmap
            sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', 
                       cbar_kws={'label': 'Survival Probability'},
                       ax=ax)
            ax.set_title(f'{severity.title()} Constraints')
            ax.set_xlabel('Location')
            ax.set_ylabel('Temperature (°C)')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / 'mars_survival_envelope.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved survival envelope plot to {plot_path}")
        
        plt.show()


def main():
    """Main function for testing Mars constraints."""
    
    logger.info("Testing Mars environmental constraints")
    
    # This would normally import a model, but for testing we'll create a simple one
    # In practice, this would use the loaded B. subtilis model
    
    # Initialize constraint engine
    constraint_engine = MarsConstraintEngine()
    
    # Generate seasonal conditions
    seasonal_conditions = constraint_engine.generate_seasonal_conditions()
    
    logger.info("Generated seasonal conditions:")
    for season, conditions in seasonal_conditions.items():
        logger.info(f"{season.value}: T={conditions.temperature_celsius}°C, P={conditions.atmospheric_pressure:.4f} atm")
    
    # Save constraint configurations
    output_dir = Path("../../models/metabolic/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Mars conditions data
    conditions_data = {
        'base_conditions': {
            'temperature_celsius': constraint_engine.base_conditions.temperature_celsius,
            'atmospheric_pressure': constraint_engine.base_conditions.atmospheric_pressure,
            'perchlorate_concentration': constraint_engine.base_conditions.perchlorate_concentration,
            'water_activity': constraint_engine.base_conditions.water_activity,
            'cosmic_radiation': constraint_engine.base_conditions.cosmic_radiation
        },
        'seasonal_variations': {
            season.value: {
                'temperature_celsius': conditions.temperature_celsius,
                'atmospheric_pressure': conditions.atmospheric_pressure
            }
            for season, conditions in seasonal_conditions.items()
        },
        'constraint_parameters': {
            'enzyme_efficiency_factors': constraint_engine.enzyme_efficiency_factors,
            'stress_thresholds': constraint_engine.stress_thresholds
        }
    }
    
    conditions_file = output_dir / 'mars_environmental_conditions.json'
    with open(conditions_file, 'w') as f:
        json.dump(conditions_data, f, indent=2)
    
    logger.info(f"Saved Mars conditions to {conditions_file}")
    logger.info("Mars constraints module ready for use")


if __name__ == "__main__":
    main()