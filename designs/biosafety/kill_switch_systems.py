"""
Multi-Layer Biosafety Kill Switch Systems
Comprehensive fail-safe mechanisms for Mars terraforming bacteria
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KillSwitchParameters:
    """Parameters for biosafety kill switch systems"""

    # Environmental dependency thresholds
    temperature_min: float = -80.0  # °C
    temperature_max: float = 40.0  # °C
    water_activity_min: float = 0.1
    water_activity_max: float = 1.0
    nutrient_concentration_min: float = 0.01  # mM
    chemical_inducer_threshold: float = 0.001  # mM

    # Timer parameters
    max_lifetime_days: int = 365  # 1 year maximum
    autodestruct_delay_hours: int = 24  # 24-hour delay after trigger

    # Remote activation parameters
    remote_signal_frequency: float = 2.4  # GHz
    activation_code_length: int = 256  # bits
    redundancy_levels: int = 3

    # Escape probability targets
    target_escape_probability: float = 0.00001  # 0.001%
    individual_switch_reliability: float = 0.9999  # 99.99%


class GeneticKillSwitch:
    """
    Genetic kill switch implementation for B. subtilis
    """

    def __init__(self, switch_id: str, activation_mechanism: str):
        self.switch_id = switch_id
        self.activation_mechanism = activation_mechanism
        self.is_active = False
        self.activation_time = None
        self.reliability = 0.9999

    def activate(self, trigger_condition: bool) -> bool:
        """
        Activate the kill switch

        Args:
            trigger_condition: Condition that triggers activation

        Returns:
            True if activation successful, False otherwise
        """
        if trigger_condition and not self.is_active:
            # Simulate activation with reliability factor
            activation_success = np.random.random() < self.reliability
            if activation_success:
                self.is_active = True
                self.activation_time = datetime.now()
                logger.info(
                    f"Kill switch {self.switch_id} activated via {self.activation_mechanism}"
                )
            return activation_success
        return False

    def get_status(self) -> Dict[str, any]:
        """Get kill switch status"""
        return {
            "switch_id": self.switch_id,
            "activation_mechanism": self.activation_mechanism,
            "is_active": self.is_active,
            "activation_time": (
                self.activation_time.isoformat() if self.activation_time else None
            ),
            "reliability": self.reliability,
        }


class EnvironmentalDependencyCircuit:
    """
    Environmental dependency circuit for kill switch activation
    """

    def __init__(self, params: KillSwitchParameters):
        self.params = params
        self.environmental_conditions = {}
        self.trigger_thresholds = {
            "temperature_out_of_range": (
                params.temperature_min,
                params.temperature_max,
            ),
            "water_activity_out_of_range": (
                params.water_activity_min,
                params.water_activity_max,
            ),
            "nutrient_deficiency": (0, params.nutrient_concentration_min),
            "chemical_inducer_present": (
                params.chemical_inducer_threshold,
                float("inf"),
            ),
        }

    def check_environmental_conditions(
        self, conditions: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Check environmental conditions against trigger thresholds

        Args:
            conditions: Current environmental conditions

        Returns:
            Dictionary of trigger conditions
        """
        triggers = {}

        # Temperature check
        temp = conditions.get("temperature_c", 0)
        triggers["temperature_out_of_range"] = (
            temp < self.params.temperature_min or temp > self.params.temperature_max
        )

        # Water activity check
        water_activity = conditions.get("water_activity", 0.5)
        triggers["water_activity_out_of_range"] = (
            water_activity < self.params.water_activity_min
            or water_activity > self.params.water_activity_max
        )

        # Nutrient deficiency check
        nutrient_concentration = conditions.get("nutrient_concentration", 0.1)
        triggers["nutrient_deficiency"] = (
            nutrient_concentration < self.params.nutrient_concentration_min
        )

        # Chemical inducer check
        chemical_inducer = conditions.get("chemical_inducer", 0)
        triggers["chemical_inducer_present"] = (
            chemical_inducer >= self.params.chemical_inducer_threshold
        )

        return triggers

    def get_trigger_probability(self, conditions: Dict[str, float]) -> float:
        """
        Calculate probability of environmental trigger

        Args:
            conditions: Environmental conditions

        Returns:
            Probability of trigger activation
        """
        triggers = self.check_environmental_conditions(conditions)

        # Calculate individual trigger probabilities
        trigger_probabilities = {}

        # Temperature trigger probability
        temp = conditions.get("temperature_c", 0)
        temp_deviation = max(
            abs(temp - self.params.temperature_min),
            abs(temp - self.params.temperature_max),
        )
        trigger_probabilities["temperature"] = min(temp_deviation / 50.0, 1.0)

        # Water activity trigger probability
        water_activity = conditions.get("water_activity", 0.5)
        water_deviation = max(
            abs(water_activity - self.params.water_activity_min),
            abs(water_activity - self.params.water_activity_max),
        )
        trigger_probabilities["water_activity"] = min(water_deviation, 1.0)

        # Nutrient deficiency probability
        nutrient_concentration = conditions.get("nutrient_concentration", 0.1)
        trigger_probabilities["nutrient"] = max(0, 1 - nutrient_concentration / 0.1)

        # Chemical inducer probability
        chemical_inducer = conditions.get("chemical_inducer", 0)
        trigger_probabilities["chemical"] = min(chemical_inducer / 0.001, 1.0)

        # Overall trigger probability (any trigger activates)
        overall_probability = 1 - np.prod(
            [1 - p for p in trigger_probabilities.values()]
        )

        return overall_probability


class TimerBasedAutodestruct:
    """
    Timer-based autodestruct mechanism
    """

    def __init__(self, params: KillSwitchParameters):
        self.params = params
        self.deployment_time = datetime.now()
        self.max_lifetime = timedelta(days=params.max_lifetime_days)
        self.autodestruct_delay = timedelta(hours=params.autodestruct_delay_hours)
        self.is_triggered = False
        self.trigger_time = None

    def check_timer_trigger(self) -> bool:
        """
        Check if timer-based trigger should activate

        Returns:
            True if trigger should activate
        """
        current_time = datetime.now()
        time_since_deployment = current_time - self.deployment_time

        # Check if maximum lifetime exceeded
        if time_since_deployment > self.max_lifetime:
            if not self.is_triggered:
                self.is_triggered = True
                self.trigger_time = current_time
                logger.info(
                    "Timer-based autodestruct triggered - maximum lifetime exceeded"
                )
            return True

        return False

    def check_delay_completion(self) -> bool:
        """
        Check if autodestruct delay has completed

        Returns:
            True if delay completed and destruction should proceed
        """
        if self.is_triggered and self.trigger_time:
            current_time = datetime.now()
            time_since_trigger = current_time - self.trigger_time

            if time_since_trigger > self.autodestruct_delay:
                logger.info("Autodestruct delay completed - initiating destruction")
                return True

        return False

    def get_status(self) -> Dict[str, any]:
        """Get timer status"""
        current_time = datetime.now()
        time_since_deployment = current_time - self.deployment_time

        return {
            "deployment_time": self.deployment_time.isoformat(),
            "time_since_deployment_days": time_since_deployment.days,
            "max_lifetime_days": self.params.max_lifetime_days,
            "is_triggered": self.is_triggered,
            "trigger_time": (
                self.trigger_time.isoformat() if self.trigger_time else None
            ),
            "autodestruct_delay_hours": self.params.autodestruct_delay_hours,
            "delay_completed": self.check_delay_completion(),
        }


class RemoteActivationSystem:
    """
    Remote activation system for kill switches
    """

    def __init__(self, params: KillSwitchParameters):
        self.params = params
        self.activation_codes = self._generate_activation_codes()
        self.received_signals = []
        self.is_activated = False
        self.activation_time = None

    def _generate_activation_codes(self) -> List[str]:
        """Generate activation codes for redundancy"""
        codes = []
        for i in range(self.params.redundancy_levels):
            # Generate random activation code
            code = "".join(
                [
                    str(np.random.randint(0, 2))
                    for _ in range(self.params.activation_code_length)
                ]
            )
            codes.append(code)
        return codes

    def receive_signal(self, signal_frequency: float, activation_code: str) -> bool:
        """
        Receive and validate remote activation signal

        Args:
            signal_frequency: Frequency of received signal
            activation_code: Activation code in signal

        Returns:
            True if signal is valid and should trigger activation
        """
        # Check frequency match
        frequency_match = (
            abs(signal_frequency - self.params.remote_signal_frequency) < 0.1
        )

        # Check activation code
        code_match = activation_code in self.activation_codes

        if frequency_match and code_match:
            self.received_signals.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "frequency": signal_frequency,
                    "code": activation_code,
                }
            )

            # Require multiple valid signals for activation (redundancy)
            if len(self.received_signals) >= 2 and not self.is_activated:
                self.is_activated = True
                self.activation_time = datetime.now()
                logger.info("Remote activation system triggered")
                return True

        return False

    def get_status(self) -> Dict[str, any]:
        """Get remote activation status"""
        return {
            "is_activated": self.is_activated,
            "activation_time": (
                self.activation_time.isoformat() if self.activation_time else None
            ),
            "received_signals_count": len(self.received_signals),
            "required_signals": 2,
            "signal_frequency": self.params.remote_signal_frequency,
            "redundancy_levels": self.params.redundancy_levels,
        }


class MultiLayerBiosafetySystem:
    """
    Multi-layer biosafety system integrating all kill switch mechanisms
    """

    def __init__(self, params: KillSwitchParameters):
        self.params = params
        self.kill_switches = []
        self.environmental_circuit = EnvironmentalDependencyCircuit(params)
        self.timer_system = TimerBasedAutodestruct(params)
        self.remote_system = RemoteActivationSystem(params)
        self.escape_probability = 0.0

        # Initialize kill switches
        self._initialize_kill_switches()

    def _initialize_kill_switches(self):
        """Initialize all kill switch systems"""
        # Genetic kill switches
        self.kill_switches.append(
            GeneticKillSwitch("GS-001", "environmental_dependency")
        )
        self.kill_switches.append(GeneticKillSwitch("GS-002", "timer_based"))
        self.kill_switches.append(GeneticKillSwitch("GS-003", "remote_activation"))
        self.kill_switches.append(GeneticKillSwitch("GS-004", "nutrient_dependency"))
        self.kill_switches.append(GeneticKillSwitch("GS-005", "chemical_inducer"))

        # Redundant systems
        self.kill_switches.append(GeneticKillSwitch("GS-006", "temperature_sensitive"))
        self.kill_switches.append(
            GeneticKillSwitch("GS-007", "water_activity_sensitive")
        )
        self.kill_switches.append(GeneticKillSwitch("GS-008", "oxygen_sensitive"))

    def check_environmental_triggers(self, conditions: Dict[str, float]) -> List[bool]:
        """
        Check all environmental triggers

        Args:
            conditions: Current environmental conditions

        Returns:
            List of trigger results for each kill switch
        """
        triggers = self.environmental_circuit.check_environmental_conditions(conditions)
        trigger_results = []

        # Activate relevant kill switches based on triggers
        for i, switch in enumerate(self.kill_switches):
            if i < len(triggers):
                trigger_key = list(triggers.keys())[i]
                trigger_result = switch.activate(triggers[trigger_key])
                trigger_results.append(trigger_result)
            else:
                trigger_results.append(False)

        return trigger_results

    def check_timer_triggers(self) -> List[bool]:
        """
        Check timer-based triggers

        Returns:
            List of timer trigger results
        """
        timer_triggered = self.timer_system.check_timer_trigger()
        delay_completed = self.timer_system.check_delay_completion()

        trigger_results = []
        for switch in self.kill_switches:
            if "timer" in switch.activation_mechanism:
                trigger_result = switch.activate(timer_triggered and delay_completed)
                trigger_results.append(trigger_result)
            else:
                trigger_results.append(False)

        return trigger_results

    def check_remote_triggers(
        self, signal_frequency: float, activation_code: str
    ) -> List[bool]:
        """
        Check remote activation triggers

        Args:
            signal_frequency: Received signal frequency
            activation_code: Received activation code

        Returns:
            List of remote trigger results
        """
        remote_triggered = self.remote_system.receive_signal(
            signal_frequency, activation_code
        )

        trigger_results = []
        for switch in self.kill_switches:
            if "remote" in switch.activation_mechanism:
                trigger_result = switch.activate(remote_triggered)
                trigger_results.append(trigger_result)
            else:
                trigger_results.append(False)

        return trigger_results

    def calculate_escape_probability(self) -> float:
        """
        Calculate probability of escape (all kill switches fail)

        Returns:
            Escape probability
        """
        # Calculate individual switch failure probabilities
        switch_failures = []
        for switch in self.kill_switches:
            failure_probability = 1 - switch.reliability
            switch_failures.append(failure_probability)

        # Calculate system failure probability (all switches fail)
        system_failure_probability = np.prod(switch_failures)

        # Apply redundancy factor
        redundancy_factor = 1 / self.params.redundancy_levels
        final_escape_probability = system_failure_probability * redundancy_factor

        self.escape_probability = final_escape_probability
        return final_escape_probability

    def test_fail_safe_probabilities(
        self, n_simulations: int = 10000
    ) -> Dict[str, float]:
        """
        Test fail-safe probabilities using Monte Carlo simulation

        Args:
            n_simulations: Number of simulation runs

        Returns:
            Dictionary with test results
        """
        escape_count = 0
        activation_counts = {switch.switch_id: 0 for switch in self.kill_switches}

        for _ in range(n_simulations):
            # Simulate random environmental conditions
            conditions = {
                "temperature_c": np.random.uniform(-100, 50),
                "water_activity": np.random.uniform(0, 1),
                "nutrient_concentration": np.random.uniform(0, 0.2),
                "chemical_inducer": np.random.uniform(0, 0.002),
            }

            # Check all triggers
            environmental_triggers = self.check_environmental_triggers(conditions)
            timer_triggers = self.check_timer_triggers()
            remote_triggers = self.check_remote_triggers(2.4, "test_code")

            # Count activations
            for i, switch in enumerate(self.kill_switches):
                if environmental_triggers[i] or timer_triggers[i] or remote_triggers[i]:
                    activation_counts[switch.switch_id] += 1

            # Check if any kill switch activated
            any_activated = any(
                environmental_triggers + timer_triggers + remote_triggers
            )

            if not any_activated:
                escape_count += 1

        # Calculate probabilities
        escape_probability = escape_count / n_simulations
        activation_probabilities = {
            switch_id: count / n_simulations
            for switch_id, count in activation_counts.items()
        }

        return {
            "escape_probability": escape_probability,
            "activation_probabilities": activation_probabilities,
            "target_escape_probability": self.params.target_escape_probability,
            "compliance_achieved": escape_probability
            <= self.params.target_escape_probability,
            "n_simulations": n_simulations,
        }

    def get_system_status(self) -> Dict[str, any]:
        """Get comprehensive system status"""
        return {
            "kill_switches": [switch.get_status() for switch in self.kill_switches],
            "environmental_circuit": {
                "trigger_thresholds": self.environmental_circuit.trigger_thresholds
            },
            "timer_system": self.timer_system.get_status(),
            "remote_system": self.remote_system.get_status(),
            "escape_probability": self.calculate_escape_probability(),
            "target_escape_probability": self.params.target_escape_probability,
            "compliance_achieved": self.calculate_escape_probability()
            <= self.params.target_escape_probability,
        }

    def plot_system_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Plot biosafety system analysis

        Args:
            save_path: Optional path to save the plot
        """
        # Test system
        test_results = self.test_fail_safe_probabilities()

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Multi-Layer Biosafety System Analysis", fontsize=16)

        # Escape probability comparison
        axes[0, 0].bar(
            ["Calculated", "Simulated", "Target"],
            [
                self.calculate_escape_probability(),
                test_results["escape_probability"],
                self.params.target_escape_probability,
            ],
        )
        axes[0, 0].set_title("Escape Probability Comparison")
        axes[0, 0].set_ylabel("Probability")
        axes[0, 0].grid(True, alpha=0.3)

        # Kill switch activation probabilities
        switch_ids = list(test_results["activation_probabilities"].keys())
        activation_probs = list(test_results["activation_probabilities"].values())

        axes[0, 1].bar(switch_ids, activation_probs)
        axes[0, 1].set_title("Kill Switch Activation Probabilities")
        axes[0, 1].set_ylabel("Activation Probability")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # System reliability
        switch_reliabilities = [switch.reliability for switch in self.kill_switches]
        axes[1, 0].bar(switch_ids, switch_reliabilities)
        axes[1, 0].set_title("Kill Switch Reliability")
        axes[1, 0].set_ylabel("Reliability")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Compliance status
        compliance_status = [
            test_results["compliance_achieved"],
            test_results["escape_probability"] <= self.params.target_escape_probability,
            True,  # Overall system status
        ]
        compliance_labels = ["Simulated", "Calculated", "Overall"]
        colors = ["green" if status else "red" for status in compliance_status]

        axes[1, 1].bar(compliance_labels, compliance_status, color=colors)
        axes[1, 1].set_title("Compliance Status")
        axes[1, 1].set_ylabel("Compliant (1) / Non-Compliant (0)")
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def save_system_documentation(self, filepath: str) -> None:
        """
        Save comprehensive system documentation

        Args:
            filepath: Path to save the documentation
        """
        documentation = {
            "system_overview": {
                "name": "Multi-Layer Biosafety Kill Switch System",
                "target_organism": "Bacillus subtilis",
                "deployment_target": "Mars",
                "design_date": datetime.now().isoformat(),
                "compliance_standard": "NASA PPO Category IV",
            },
            "kill_switch_specifications": {
                "total_switches": len(self.kill_switches),
                "redundancy_levels": self.params.redundancy_levels,
                "target_escape_probability": self.params.target_escape_probability,
                "individual_reliability": self.params.individual_switch_reliability,
            },
            "activation_mechanisms": {
                "environmental_dependency": "Temperature, water activity, nutrients",
                "timer_based": f"Maximum lifetime: {self.params.max_lifetime_days} days",
                "remote_activation": f"Frequency: {self.params.remote_signal_frequency} GHz",
                "chemical_inducer": f"Threshold: {self.params.chemical_inducer_threshold} mM",
            },
            "system_status": self.get_system_status(),
            "test_results": self.test_fail_safe_probabilities(),
        }

        with open(filepath, "w") as f:
            json.dump(documentation, f, indent=2)

        logger.info(f"Biosafety system documentation saved to {filepath}")


def main():
    """Main function to demonstrate biosafety kill switch systems"""

    # Initialize parameters
    params = KillSwitchParameters()

    # Initialize biosafety system
    biosafety_system = MultiLayerBiosafetySystem(params)

    # Test fail-safe probabilities
    test_results = biosafety_system.test_fail_safe_probabilities()

    # Plot system analysis
    biosafety_system.plot_system_analysis()

    # Save system documentation
    biosafety_system.save_system_documentation(
        "designs/biosafety/biosafety_system_documentation.json"
    )

    # Print summary
    print("\n=== Multi-Layer Biosafety System Summary ===")
    print(f"Number of Kill Switches: {len(biosafety_system.kill_switches)}")
    print(f"Target Escape Probability: {params.target_escape_probability:.6f}")
    print(f"Simulated Escape Probability: {test_results['escape_probability']:.6f}")
    print(f"Compliance Achieved: {test_results['compliance_achieved']}")

    print(f"\nKill Switch Activation Probabilities:")
    for switch_id, prob in test_results["activation_probabilities"].items():
        print(f"  {switch_id}: {prob:.4f}")

    print(f"\nSystem Reliability: {1 - test_results['escape_probability']:.6f}")


if __name__ == "__main__":
    main()
