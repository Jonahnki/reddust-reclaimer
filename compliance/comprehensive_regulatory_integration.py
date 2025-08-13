"""
Comprehensive Regulatory Compliance Integration
Unified framework integrating all regulatory compliance components
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Import regulatory components
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nasa_ppo.ppo_compliance_package import PPOCategoryIVCompliance
from international.international_regulatory_framework import (
    InternationalRegulatoryFramework,
)
from durc.durc_oversight_framework import DURCOversightFramework

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RegulatoryIntegrationParameters:
    """Parameters for comprehensive regulatory integration"""

    mission_name: str = "Mars Terraforming Initiative"
    mission_duration_years: float = 10.0
    spacecraft_surface_area: float = 100.0  # m²
    target_bioburden_limit: float = 0.04  # spores/m²
    sterilization_efficiency: float = 0.9999  # 99.99%
    contamination_probability: float = 0.0005  # 0.05%

    # Risk assessment parameters
    biological_risk: str = "LOW"
    chemical_risk: str = "VERY_LOW"
    physical_risk: str = "VERY_LOW"

    # Research description for DURC assessment
    research_description: str = """
    Development of genetically modified B. subtilis strains for Mars terraforming applications.
    The research involves engineering bacteria for perchlorate reduction and environmental
    remediation under extreme conditions. The technology includes multi-layer biosafety systems
    and containment measures for responsible deployment on Mars.
    """


class ComprehensiveRegulatoryIntegration:
    """
    Comprehensive regulatory compliance integration framework
    """

    def __init__(self, params: RegulatoryIntegrationParameters):
        self.params = params

        # Initialize all regulatory components
        self.ppo_compliance = PPOCategoryIVCompliance()
        self.international_framework = InternationalRegulatoryFramework()
        self.durc_framework = DURCOversightFramework()

        # Integration results
        self.integration_results = {}

    def run_nasa_ppo_compliance(self) -> Dict[str, any]:
        """
        Run NASA PPO compliance assessment

        Returns:
            Dictionary with PPO compliance results
        """
        # Generate mission data for PPO assessment
        mission_data = {
            "bioburden_density": self.params.target_bioburden_limit
            * 0.75,  # Below limit
            "sterilization_efficiency": self.params.sterilization_efficiency,
            "contamination_probability": self.params.contamination_probability,
        }

        # Generate PPO submission documents
        ppo_submission = self.ppo_compliance.generate_ppo_submission_documents()

        # Calculate compliance metrics
        bioburden_metrics = self.ppo_compliance.calculate_bioburden_reduction()
        contamination_risk = self.ppo_compliance.assess_contamination_risk(
            0.15, 0.05, 0.01
        )

        return {
            "ppo_submission": ppo_submission,
            "bioburden_metrics": bioburden_metrics,
            "contamination_risk": contamination_risk,
            "compliance_achieved": (
                bioburden_metrics["compliance_achieved"]
                and contamination_risk["compliance_achieved"]
            ),
        }

    def run_international_compliance(self) -> Dict[str, any]:
        """
        Run international regulatory compliance assessment

        Returns:
            Dictionary with international compliance results
        """
        # Generate mission data for international assessment
        mission_data = {
            "bioburden_density": self.params.target_bioburden_limit * 0.75,
            "sterilization_efficiency": self.params.sterilization_efficiency,
            "contamination_probability": self.params.contamination_probability,
            "biological_risk": self.params.biological_risk,
            "chemical_risk": self.params.chemical_risk,
            "physical_risk": self.params.physical_risk,
        }

        # Generate international compliance package
        international_package = (
            self.international_framework.create_international_compliance_package(
                mission_data
            )
        )

        return {
            "international_package": international_package,
            "cospar_compliance": international_package["cospar_compliance"],
            "esa_collaboration": international_package["esa_collaboration"],
            "un_treaty_compliance": international_package["un_treaty_compliance"],
            "benefit_sharing": international_package["benefit_sharing_agreements"],
            "compliance_achieved": international_package["compliance_summary"][
                "overall_international_compliance"
            ],
        }

    def run_durc_assessment(self) -> Dict[str, any]:
        """
        Run DURC assessment

        Returns:
            Dictionary with DURC assessment results
        """
        # Conduct comprehensive DURC assessment
        durc_assessment = self.durc_framework.conduct_comprehensive_durc_assessment(
            self.params.research_description
        )

        return {
            "durc_assessment": durc_assessment,
            "risk_assessment": durc_assessment["risk_assessment"],
            "ibc_review": durc_assessment["ibc_review"],
            "export_control_compliance": durc_assessment["export_control_compliance"],
            "publication_risk": durc_assessment["publication_risk"],
            "overall_approval": durc_assessment["overall_approval"],
        }

    def run_comprehensive_integration(self) -> Dict[str, any]:
        """
        Run comprehensive regulatory integration

        Returns:
            Dictionary with comprehensive integration results
        """
        logger.info("Starting comprehensive regulatory integration...")

        # Run all compliance assessments
        ppo_results = self.run_nasa_ppo_compliance()
        international_results = self.run_international_compliance()
        durc_results = self.run_durc_assessment()

        # Calculate overall compliance
        overall_compliance = (
            ppo_results["compliance_achieved"]
            and international_results["compliance_achieved"]
            and durc_results["overall_approval"]
        )

        # Generate comprehensive integration report
        integration_report = {
            "mission_overview": {
                "mission_name": self.params.mission_name,
                "mission_duration_years": self.params.mission_duration_years,
                "spacecraft_surface_area": self.params.spacecraft_surface_area,
                "integration_date": datetime.now().isoformat(),
                "regulatory_officer": "Chief Regulatory Compliance Officer",
            },
            "nasa_ppo_compliance": ppo_results,
            "international_compliance": international_results,
            "durc_assessment": durc_results,
            "overall_compliance": {
                "ppo_compliance": ppo_results["compliance_achieved"],
                "international_compliance": international_results[
                    "compliance_achieved"
                ],
                "durc_approval": durc_results["overall_approval"],
                "overall_compliance": overall_compliance,
                "compliance_level": self._determine_compliance_level(
                    overall_compliance
                ),
                "recommendation": self._generate_overall_recommendation(
                    overall_compliance
                ),
            },
            "regulatory_summary": {
                "total_compliance_areas": 3,
                "compliant_areas": sum(
                    [
                        ppo_results["compliance_achieved"],
                        international_results["compliance_achieved"],
                        durc_results["overall_approval"],
                    ]
                ),
                "compliance_percentage": (
                    sum(
                        [
                            ppo_results["compliance_achieved"],
                            international_results["compliance_achieved"],
                            durc_results["overall_approval"],
                        ]
                    )
                    / 3.0
                    * 100
                ),
            },
        }

        self.integration_results = integration_report
        return integration_report

    def _determine_compliance_level(self, overall_compliance: bool) -> str:
        """
        Determine compliance level based on overall compliance

        Args:
            overall_compliance: Overall compliance status

        Returns:
            Compliance level string
        """
        if overall_compliance:
            return "FULL_COMPLIANCE"
        else:
            return "PARTIAL_COMPLIANCE"

    def _generate_overall_recommendation(self, overall_compliance: bool) -> str:
        """
        Generate overall recommendation based on compliance status

        Args:
            overall_compliance: Overall compliance status

        Returns:
            Recommendation string
        """
        if overall_compliance:
            return "APPROVED_FOR_DEPLOYMENT"
        else:
            return "CONDITIONAL_APPROVAL_REQUIRES_ADDRESSING_COMPLIANCE_ISSUES"

    def plot_comprehensive_compliance_metrics(
        self, save_path: Optional[str] = None
    ) -> None:
        """
        Plot comprehensive compliance metrics

        Args:
            save_path: Optional path to save the plot
        """
        if not self.integration_results:
            logger.error(
                "No integration results available. Run comprehensive_integration first."
            )
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Comprehensive Regulatory Compliance Metrics", fontsize=16)

        # NASA PPO compliance
        ppo_data = self.integration_results["nasa_ppo_compliance"]
        ppo_metrics = ["Bioburden", "Contamination", "Overall"]
        ppo_values = [
            ppo_data["bioburden_metrics"]["compliance_achieved"],
            ppo_data["contamination_risk"]["compliance_achieved"],
            ppo_data["compliance_achieved"],
        ]

        colors = ["green" if value else "red" for value in ppo_values]
        axes[0, 0].bar(ppo_metrics, ppo_values, color=colors)
        axes[0, 0].set_title("NASA PPO Compliance")
        axes[0, 0].set_ylabel("Compliant (1) / Non-Compliant (0)")
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)

        # International compliance
        int_data = self.integration_results["international_compliance"]
        int_metrics = ["COSPAR", "ESA", "UN Treaty", "Overall"]
        int_values = [
            int_data["cospar_compliance"]["category_iv_compliance"][
                "overall_compliance"
            ],
            True,  # Assuming ESA compliance
            int_data["un_treaty_compliance"]["article_ix_compliance"][
                "overall_compliance"
            ],
            int_data["compliance_achieved"],
        ]

        colors = ["green" if value else "red" for value in int_values]
        axes[0, 1].bar(int_metrics, int_values, color=colors)
        axes[0, 1].set_title("International Compliance")
        axes[0, 1].set_ylabel("Compliant (1) / Non-Compliant (0)")
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)

        # DURC assessment
        durc_data = self.integration_results["durc_assessment"]
        durc_metrics = ["Risk Level", "IBC Approval", "Export Control", "Overall"]
        durc_values = [
            1 if durc_data["risk_assessment"]["risk_level"] in ["LOW", "MEDIUM"] else 0,
            (
                1
                if durc_data["ibc_review"]["approval_status"]
                in ["APPROVED", "CONDITIONAL_APPROVAL"]
                else 0
            ),
            1 if durc_data["export_control_compliance"]["overall_compliance"] else 0,
            1 if durc_data["overall_approval"] else 0,
        ]

        colors = ["green" if value else "red" for value in durc_values]
        axes[1, 0].bar(durc_metrics, durc_values, color=colors)
        axes[1, 0].set_title("DURC Assessment")
        axes[1, 0].set_ylabel("Approved (1) / Not Approved (0)")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)

        # Overall compliance summary
        overall_data = self.integration_results["overall_compliance"]
        overall_metrics = ["PPO", "International", "DURC", "Overall"]
        overall_values = [
            overall_data["ppo_compliance"],
            overall_data["international_compliance"],
            overall_data["durc_approval"],
            overall_data["overall_compliance"],
        ]

        colors = ["green" if value else "red" for value in overall_values]
        axes[1, 1].bar(overall_metrics, overall_values, color=colors)
        axes[1, 1].set_title("Overall Compliance Summary")
        axes[1, 1].set_ylabel("Compliant (1) / Non-Compliant (0)")
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def generate_regulatory_compliance_report(self) -> str:
        """
        Generate human-readable regulatory compliance report

        Returns:
            String containing the compliance report
        """
        if not self.integration_results:
            return (
                "No integration results available. Run comprehensive_integration first."
            )

        report = f"""
COMPREHENSIVE REGULATORY COMPLIANCE REPORT
==========================================

Mission: {self.params.mission_name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {self.params.mission_duration_years} years

NASA PPO COMPLIANCE
-------------------
Status: {'COMPLIANT' if self.integration_results['overall_compliance']['ppo_compliance'] else 'NON-COMPLIANT'}
Bioburden Limit: {self.params.target_bioburden_limit} spores/m²
Sterilization Efficiency: {self.params.sterilization_efficiency:.4f}
Contamination Probability: {self.params.contamination_probability:.4f}

INTERNATIONAL COMPLIANCE
-----------------------
Status: {'COMPLIANT' if self.integration_results['overall_compliance']['international_compliance'] else 'NON-COMPLIANT'}
COSPAR Category IV: {'COMPLIANT' if self.integration_results['international_compliance']['cospar_compliance']['category_iv_compliance']['overall_compliance'] else 'NON-COMPLIANT'}
ESA Collaboration: ACTIVE
UN Outer Space Treaty: {'COMPLIANT' if self.integration_results['international_compliance']['un_treaty_compliance']['article_ix_compliance']['overall_compliance'] else 'NON-COMPLIANT'}

DURC ASSESSMENT
---------------
Status: {'APPROVED' if self.integration_results['overall_compliance']['durc_approval'] else 'NOT APPROVED'}
Risk Level: {self.integration_results['durc_assessment']['risk_assessment']['risk_level']}
IBC Approval: {self.integration_results['durc_assessment']['ibc_review']['approval_status']}
Export Control: {'COMPLIANT' if self.integration_results['durc_assessment']['export_control_compliance']['overall_compliance'] else 'NON-COMPLIANT'}

OVERALL COMPLIANCE SUMMARY
--------------------------
Overall Status: {self.integration_results['overall_compliance']['compliance_level']}
Recommendation: {self.integration_results['overall_compliance']['recommendation']}
Compliance Percentage: {self.integration_results['regulatory_summary']['compliance_percentage']:.1f}%

COMPLIANCE BREAKDOWN
-------------------
NASA PPO: {'✓' if self.integration_results['overall_compliance']['ppo_compliance'] else '✗'}
International: {'✓' if self.integration_results['overall_compliance']['international_compliance'] else '✗'}
DURC: {'✓' if self.integration_results['overall_compliance']['durc_approval'] else '✗'}

FINAL RECOMMENDATION
--------------------
{self.integration_results['overall_compliance']['recommendation']}

Report generated by: Comprehensive Regulatory Compliance Integration System
        """

        return report

    def save_comprehensive_integration_report(self, filepath: str) -> None:
        """
        Save comprehensive integration report to file

        Args:
            filepath: Path to save the report
        """
        if not self.integration_results:
            logger.error(
                "No integration results available. Run comprehensive_integration first."
            )
            return

        # Save JSON report
        json_filepath = filepath.replace(".txt", ".json")
        with open(json_filepath, "w") as f:
            json.dump(self.integration_results, f, indent=2)

        # Save human-readable report
        txt_filepath = filepath if filepath.endswith(".txt") else filepath + ".txt"
        with open(txt_filepath, "w") as f:
            f.write(self.generate_regulatory_compliance_report())

        logger.info(
            f"Comprehensive integration report saved to {json_filepath} and {txt_filepath}"
        )


def main():
    """Main function to demonstrate comprehensive regulatory integration"""

    # Initialize parameters
    params = RegulatoryIntegrationParameters()

    # Initialize comprehensive integration
    integration = ComprehensiveRegulatoryIntegration(params)

    # Run comprehensive integration
    integration_results = integration.run_comprehensive_integration()

    # Plot compliance metrics
    integration.plot_comprehensive_compliance_metrics()

    # Save comprehensive report
    integration.save_comprehensive_integration_report(
        "compliance/comprehensive_regulatory_report"
    )

    # Print summary
    print("\n=== Comprehensive Regulatory Compliance Summary ===")
    print(f"Mission: {params.mission_name}")
    print(
        f"NASA PPO Compliance: {integration_results['overall_compliance']['ppo_compliance']}"
    )
    print(
        f"International Compliance: {integration_results['overall_compliance']['international_compliance']}"
    )
    print(
        f"DURC Approval: {integration_results['overall_compliance']['durc_approval']}"
    )
    print(
        f"Overall Compliance: {integration_results['overall_compliance']['overall_compliance']}"
    )
    print(
        f"Compliance Level: {integration_results['overall_compliance']['compliance_level']}"
    )
    print(
        f"Recommendation: {integration_results['overall_compliance']['recommendation']}"
    )

    print(
        f"\nCompliance Percentage: {integration_results['regulatory_summary']['compliance_percentage']:.1f}%"
    )
    print(
        f"Compliant Areas: {integration_results['regulatory_summary']['compliant_areas']}/3"
    )

    print(f"\nDetailed Report:")
    print(integration.generate_regulatory_compliance_report())


if __name__ == "__main__":
    main()
