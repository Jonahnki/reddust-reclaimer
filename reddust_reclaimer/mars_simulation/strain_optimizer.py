"""
Strain Optimizer using Genetic Algorithms
Optimizes B. subtilis performance for Mars environmental conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging
import json
from copy import deepcopy
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StrainGenome:
    """Represents a B. subtilis strain genome for optimization"""

    # Metabolic parameters
    growth_rate: float = 0.5  # Doubling time (hours)
    perchlorate_reduction_rate: float = 0.3  # Reduction efficiency
    stress_tolerance: float = 0.5  # Overall stress resistance

    # Protein expression levels (0-1)
    cold_shock_proteins: float = 0.5
    osmotic_stress_proteins: float = 0.5
    antioxidant_proteins: float = 0.5
    perchlorate_reductase: float = 0.5

    # Metabolic pathway efficiencies
    glycolysis_efficiency: float = 0.5
    tca_cycle_efficiency: float = 0.5
    oxidative_phosphorylation: float = 0.5

    # Resource allocation
    energy_allocation_growth: float = 0.33
    energy_allocation_stress: float = 0.33
    energy_allocation_remediation: float = 0.34

    def mutate(self, mutation_rate: float = 0.1) -> "StrainGenome":
        """Create a mutated copy of the genome"""
        mutated = deepcopy(self)

        for field in mutated.__dataclass_fields__:
            if random.random() < mutation_rate:
                current_value = getattr(mutated, field)

                # Add Gaussian noise
                noise = np.random.normal(0, 0.1)
                new_value = current_value + noise

                # Clamp to valid range [0, 1]
                new_value = np.clip(new_value, 0.0, 1.0)
                setattr(mutated, field, new_value)

        return mutated

    def crossover(self, other: "StrainGenome") -> Tuple["StrainGenome", "StrainGenome"]:
        """Perform crossover between two genomes"""
        child1 = deepcopy(self)
        child2 = deepcopy(other)

        # Random crossover points
        fields = list(self.__dataclass_fields__.keys())
        crossover_point = random.randint(1, len(fields) - 1)

        for i, field in enumerate(fields):
            if i < crossover_point:
                # Swap values
                temp = getattr(child1, field)
                setattr(child1, field, getattr(child2, field))
                setattr(child2, field, temp)

        return child1, child2

    def to_dict(self) -> Dict[str, float]:
        """Convert genome to dictionary"""
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "StrainGenome":
        """Create genome from dictionary"""
        return cls(**data)


class StrainOptimizer:
    """
    Genetic algorithm optimizer for B. subtilis strain performance
    """

    def __init__(
        self,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_size: int = 10,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.population = []
        self.fitness_history = []
        self.best_genome = None
        self.best_fitness = -np.inf

    def initialize_population(self) -> None:
        """Initialize random population of strain genomes"""
        self.population = []

        for _ in range(self.population_size):
            genome = StrainGenome(
                growth_rate=np.random.uniform(0.1, 1.0),
                perchlorate_reduction_rate=np.random.uniform(0.1, 1.0),
                stress_tolerance=np.random.uniform(0.1, 1.0),
                cold_shock_proteins=np.random.uniform(0.0, 1.0),
                osmotic_stress_proteins=np.random.uniform(0.0, 1.0),
                antioxidant_proteins=np.random.uniform(0.0, 1.0),
                perchlorate_reductase=np.random.uniform(0.0, 1.0),
                glycolysis_efficiency=np.random.uniform(0.3, 1.0),
                tca_cycle_efficiency=np.random.uniform(0.3, 1.0),
                oxidative_phosphorylation=np.random.uniform(0.3, 1.0),
                energy_allocation_growth=np.random.uniform(0.2, 0.5),
                energy_allocation_stress=np.random.uniform(0.2, 0.5),
                energy_allocation_remediation=np.random.uniform(0.2, 0.5),
            )
            self.population.append(genome)

    def calculate_fitness(
        self, genome: StrainGenome, environment_conditions: Dict[str, float]
    ) -> float:
        """
        Calculate fitness score for a strain genome under given conditions

        Args:
            genome: Strain genome to evaluate
            environment_conditions: Mars environmental conditions

        Returns:
            Fitness score (higher is better)
        """
        # Extract environmental factors
        temperature = environment_conditions.get("temperature_c", -63.0)
        pressure = environment_conditions.get("pressure_kpa", 0.6)
        uv_intensity = environment_conditions.get("uv_intensity", 1.5)
        water_activity = environment_conditions.get("water_activity", 0.3)
        perchlorate_concentration = environment_conditions.get(
            "perchlorate_concentration", 0.007
        )

        # Calculate stress factors
        temp_stress = abs(temperature - 30.0) / 50.0  # Optimal temp: 30°C
        pressure_stress = 1 - (pressure / 101.325)
        uv_stress = uv_intensity / 2.0
        water_stress = 1 - water_activity
        perchlorate_stress = perchlorate_concentration / 0.01

        # Calculate performance components
        # 1. Growth performance
        growth_performance = (
            genome.growth_rate
            * genome.glycolysis_efficiency
            * genome.energy_allocation_growth
            * (1 - temp_stress)
            * (1 - water_stress)
        )

        # 2. Stress tolerance
        stress_tolerance = (
            genome.stress_tolerance
            * (
                genome.cold_shock_proteins * (1 - temp_stress)
                + genome.osmotic_stress_proteins * (1 - water_stress)
                + genome.antioxidant_proteins * (1 - uv_stress)
            )
            / 3
        )

        # 3. Perchlorate remediation
        remediation_performance = (
            genome.perchlorate_reduction_rate
            * genome.perchlorate_reductase
            * genome.energy_allocation_remediation
            * (1 - perchlorate_stress)
        )

        # 4. Metabolic efficiency
        metabolic_efficiency = (
            genome.glycolysis_efficiency * 0.4
            + genome.tca_cycle_efficiency * 0.3
            + genome.oxidative_phosphorylation * 0.3
        )

        # 5. Energy balance constraint
        energy_balance = 1.0 - abs(
            genome.energy_allocation_growth
            + genome.energy_allocation_stress
            + genome.energy_allocation_remediation
            - 1.0
        )

        # Combined fitness score
        fitness = (
            growth_performance * 0.25
            + stress_tolerance * 0.25
            + remediation_performance * 0.25
            + metabolic_efficiency * 0.15
            + energy_balance * 0.10
        )

        return max(0.0, fitness)

    def evaluate_population(
        self, environment_conditions: Dict[str, float]
    ) -> List[Tuple[StrainGenome, float]]:
        """
        Evaluate fitness for entire population

        Args:
            environment_conditions: Mars environmental conditions

        Returns:
            List of (genome, fitness) tuples sorted by fitness
        """
        fitness_scores = []

        for genome in self.population:
            fitness = self.calculate_fitness(genome, environment_conditions)
            fitness_scores.append((genome, fitness))

        # Sort by fitness (descending)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        return fitness_scores

    def select_parents(
        self, fitness_scores: List[Tuple[StrainGenome, float]]
    ) -> List[StrainGenome]:
        """
        Select parents for next generation using tournament selection

        Args:
            fitness_scores: List of (genome, fitness) tuples

        Returns:
            List of selected parent genomes
        """
        parents = []

        # Keep elite individuals
        elite = [genome for genome, _ in fitness_scores[: self.elite_size]]
        parents.extend(elite)

        # Tournament selection for remaining parents
        tournament_size = 5

        while len(parents) < self.population_size:
            # Select tournament participants
            tournament = random.sample(fitness_scores, tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            parents.append(winner[0])

        return parents

    def create_next_generation(self, parents: List[StrainGenome]) -> None:
        """
        Create next generation through crossover and mutation

        Args:
            parents: List of parent genomes
        """
        new_population = []

        # Keep elite individuals
        new_population.extend(parents[: self.elite_size])

        # Create offspring through crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(parents) >= 2:
                # Crossover
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = parent1.crossover(parent2)

                # Mutate children
                child1 = child1.mutate(self.mutation_rate)
                child2 = child2.mutate(self.mutation_rate)

                new_population.extend([child1, child2])
            else:
                # Mutation only
                parent = random.choice(parents)
                child = parent.mutate(self.mutation_rate)
                new_population.append(child)

        # Ensure population size
        self.population = new_population[: self.population_size]

    def optimize(
        self, environment_conditions: Dict[str, float], generations: int = 100
    ) -> Tuple[StrainGenome, List[float]]:
        """
        Run genetic algorithm optimization

        Args:
            environment_conditions: Mars environmental conditions
            generations: Number of generations to run

        Returns:
            Tuple of (best genome, fitness history)
        """
        logger.info(f"Starting optimization for {generations} generations...")

        # Initialize population
        self.initialize_population()

        # Track best fitness
        best_fitness_history = []

        for generation in range(generations):
            # Evaluate population
            fitness_scores = self.evaluate_population(environment_conditions)

            # Track best fitness
            best_fitness = fitness_scores[0][1]
            best_fitness_history.append(best_fitness)

            # Update best genome
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_genome = fitness_scores[0][0]

            # Log progress
            if generation % 10 == 0:
                avg_fitness = np.mean([f for _, f in fitness_scores])
                logger.info(
                    f"Generation {generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}"
                )

            # Select parents
            parents = self.select_parents(fitness_scores)

            # Create next generation
            self.create_next_generation(parents)

        logger.info(f"Optimization complete. Best fitness: {self.best_fitness:.4f}")
        return self.best_genome, best_fitness_history

    def plot_optimization_progress(
        self, fitness_history: List[float], save_path: Optional[str] = None
    ) -> None:
        """
        Plot optimization progress

        Args:
            fitness_history: List of best fitness scores per generation
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 8))

        # Plot fitness history
        plt.subplot(2, 2, 1)
        plt.plot(fitness_history)
        plt.title("Optimization Progress")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness Score")
        plt.grid(True)

        # Plot best genome characteristics
        if self.best_genome:
            genome_dict = self.best_genome.to_dict()

            # Metabolic parameters
            plt.subplot(2, 2, 2)
            metabolic_params = [
                "growth_rate",
                "perchlorate_reduction_rate",
                "stress_tolerance",
            ]
            values = [genome_dict[param] for param in metabolic_params]
            plt.bar(metabolic_params, values)
            plt.title("Best Strain - Metabolic Parameters")
            plt.ylabel("Value")
            plt.xticks(rotation=45)

            # Protein expression
            plt.subplot(2, 2, 3)
            protein_params = [
                "cold_shock_proteins",
                "osmotic_stress_proteins",
                "antioxidant_proteins",
                "perchlorate_reductase",
            ]
            values = [genome_dict[param] for param in protein_params]
            plt.bar(protein_params, values)
            plt.title("Best Strain - Protein Expression")
            plt.ylabel("Expression Level")
            plt.xticks(rotation=45)

            # Energy allocation
            plt.subplot(2, 2, 4)
            energy_params = [
                "energy_allocation_growth",
                "energy_allocation_stress",
                "energy_allocation_remediation",
            ]
            values = [genome_dict[param] for param in energy_params]
            plt.pie(values, labels=energy_params, autopct="%1.1f%%")
            plt.title("Best Strain - Energy Allocation")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def save_optimization_results(self, filepath: str) -> None:
        """
        Save optimization results to file

        Args:
            filepath: Path to save results
        """
        if self.best_genome:
            results = {
                "best_genome": self.best_genome.to_dict(),
                "best_fitness": self.best_fitness,
                "fitness_history": self.fitness_history,
            }

            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Optimization results saved to {filepath}")


def main():
    """Main function to demonstrate strain optimization"""

    # Define Mars environmental conditions
    mars_conditions = {
        "temperature_c": -63.0,
        "pressure_kpa": 0.6,
        "uv_intensity": 1.5,
        "water_activity": 0.3,
        "perchlorate_concentration": 0.007,
        "cosmic_radiation": 2.0,
        "relative_humidity": 5.0,
    }

    # Initialize optimizer
    optimizer = StrainOptimizer(
        population_size=100, mutation_rate=0.1, crossover_rate=0.8, elite_size=10
    )

    # Run optimization
    best_genome, fitness_history = optimizer.optimize(
        environment_conditions=mars_conditions, generations=100
    )

    # Store fitness history
    optimizer.fitness_history = fitness_history

    # Plot results
    optimizer.plot_optimization_progress(fitness_history)

    # Save results
    optimizer.save_optimization_results("data/optimization_results.json")

    # Print best strain characteristics
    print("\n=== Optimized Strain Characteristics ===")
    best_dict = best_genome.to_dict()
    for param, value in best_dict.items():
        print(f"{param}: {value:.3f}")

    print(f"\nBest Fitness Score: {optimizer.best_fitness:.4f}")


if __name__ == "__main__":
    main()
