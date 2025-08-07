"""
Dual-Use Research Oversight (DURC) Framework
Comprehensive oversight for responsible research and technology transfer
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DURCRiskFactors:
    """Risk factors for dual-use research assessment"""
    # Technology risk factors
    technology_readiness_level: int = 7  # TRL 1-9
    complexity_level: int = 3  # 1-5 scale
    accessibility_level: int = 2  # 1-5 scale (1=highly restricted)
    
    # Application risk factors
    dual_use_potential: float = 0.3  # 0-1 scale
    weaponization_risk: float = 0.1  # 0-1 scale
    proliferation_risk: float = 0.2  # 0-1 scale
    
    # Mitigation factors
    containment_effectiveness: float = 0.9  # 0-1 scale
    oversight_mechanisms: float = 0.8  # 0-1 scale
    responsible_governance: float = 0.9  # 0-1 scale

@dataclass
class ExportControlRegulations:
    """Export control compliance parameters"""
    # ITAR (International Traffic in Arms Regulations)
    itar_controlled: bool = False
    itar_category: str = "N/A"
    itar_exemption: str = "N/A"
    
    # EAR (Export Administration Regulations)
    ear_controlled: bool = True
    ear_category: str = "EAR99"
    ear_license_required: bool = False
    
    # Wassenaar Arrangement
    wassenaar_controlled: bool = False
    wassenaar_category: str = "N/A"
    
    # Additional controls
    technology_transfer_restrictions: List[str] = None
    end_user_verification: bool = True
    destination_controls: List[str] = None
    
    def __post_init__(self):
        if self.technology_transfer_restrictions is None:
            self.technology_transfer_restrictions = [
                'Restricted to authorized research institutions',
                'No transfer to embargoed countries',
                'End-use verification required',
                'Technology transfer agreements mandatory'
            ]
        
        if self.destination_controls is None:
            self.destination_controls = [
                'United States',
                'European Union',
                'Canada',
                'Australia',
                'Japan',
                'South Korea'
            ]

class DURCRiskAssessment:
    """
    Comprehensive dual-use risk assessment framework
    """
    
    def __init__(self, risk_factors: DURCRiskFactors):
        self.risk_factors = risk_factors
        self.assessment_data = {}
        
    def assess_technology_risk(self) -> Dict[str, float]:
        """
        Assess technology-specific risk factors
        
        Returns:
            Dictionary with technology risk assessment
        """
        # Technology readiness risk (lower TRL = higher risk)
        trl_risk = (10 - self.risk_factors.technology_readiness_level) / 9.0
        
        # Complexity risk (higher complexity = higher risk)
        complexity_risk = (self.risk_factors.complexity_level - 1) / 4.0
        
        # Accessibility risk (lower accessibility = lower risk)
        accessibility_risk = (self.risk_factors.accessibility_level - 1) / 4.0
        
        # Combined technology risk
        technology_risk = (trl_risk + complexity_risk + accessibility_risk) / 3.0
        
        return {
            'trl_risk': trl_risk,
            'complexity_risk': complexity_risk,
            'accessibility_risk': accessibility_risk,
            'technology_risk': technology_risk
        }
    
    def assess_application_risk(self) -> Dict[str, float]:
        """
        Assess application-specific risk factors
        
        Returns:
            Dictionary with application risk assessment
        """
        # Dual-use potential risk
        dual_use_risk = self.risk_factors.dual_use_potential
        
        # Weaponization risk
        weaponization_risk = self.risk_factors.weaponization_risk
        
        # Proliferation risk
        proliferation_risk = self.risk_factors.proliferation_risk
        
        # Combined application risk
        application_risk = (dual_use_risk + weaponization_risk + proliferation_risk) / 3.0
        
        return {
            'dual_use_risk': dual_use_risk,
            'weaponization_risk': weaponization_risk,
            'proliferation_risk': proliferation_risk,
            'application_risk': application_risk
        }
    
    def assess_mitigation_effectiveness(self) -> Dict[str, float]:
        """
        Assess effectiveness of mitigation measures
        
        Returns:
            Dictionary with mitigation effectiveness assessment
        """
        # Containment effectiveness
        containment_effectiveness = self.risk_factors.containment_effectiveness
        
        # Oversight mechanisms
        oversight_effectiveness = self.risk_factors.oversight_mechanisms
        
        # Responsible governance
        governance_effectiveness = self.risk_factors.responsible_governance
        
        # Combined mitigation effectiveness
        mitigation_effectiveness = (containment_effectiveness + oversight_effectiveness + governance_effectiveness) / 3.0
        
        return {
            'containment_effectiveness': containment_effectiveness,
            'oversight_effectiveness': oversight_effectiveness,
            'governance_effectiveness': governance_effectiveness,
            'mitigation_effectiveness': mitigation_effectiveness
        }
    
    def calculate_overall_risk(self) -> Dict[str, any]:
        """
        Calculate overall DURC risk assessment
        
        Returns:
            Dictionary with overall risk assessment
        """
        # Get individual risk assessments
        technology_risk = self.assess_technology_risk()
        application_risk = self.assess_application_risk()
        mitigation_effectiveness = self.assess_mitigation_effectiveness()
        
        # Calculate overall risk
        overall_risk = (
            technology_risk['technology_risk'] * 0.3 +
            application_risk['application_risk'] * 0.5 +
            (1 - mitigation_effectiveness['mitigation_effectiveness']) * 0.2
        )
        
        # Risk categorization
        if overall_risk < 0.2:
            risk_level = "LOW"
            recommendation = "Proceed with standard oversight"
        elif overall_risk < 0.5:
            risk_level = "MEDIUM"
            recommendation = "Enhanced oversight required"
        elif overall_risk < 0.8:
            risk_level = "HIGH"
            recommendation = "Strict oversight and controls required"
        else:
            risk_level = "VERY_HIGH"
            recommendation = "Research may require restriction or prohibition"
        
        return {
            'technology_risk': technology_risk,
            'application_risk': application_risk,
            'mitigation_effectiveness': mitigation_effectiveness,
            'overall_risk': overall_risk,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'assessment_date': datetime.now().isoformat()
        }

class InstitutionalBiosafetyCommittee:
    """
    Institutional Biosafety Committee (IBC) review framework
    """
    
    def __init__(self):
        self.committee_members = [
            'Biosafety Officer',
            'Microbiologist',
            'Environmental Scientist',
            'Legal Counsel',
            'Public Health Representative',
            'Community Representative',
            'Research Ethics Specialist'
        ]
        self.review_criteria = [
            'Risk assessment adequacy',
            'Containment measures',
            'Emergency response procedures',
            'Training requirements',
            'Monitoring protocols',
            'Reporting requirements'
        ]
    
    def conduct_ibc_review(self, risk_assessment: Dict[str, any]) -> Dict[str, any]:
        """
        Conduct IBC review of research proposal
        
        Args:
            risk_assessment: DURC risk assessment results
            
        Returns:
            Dictionary with IBC review results
        """
        # Review criteria assessment
        review_scores = {}
        for criterion in self.review_criteria:
            # Simulate review scores (0-1 scale)
            review_scores[criterion] = np.random.uniform(0.7, 1.0)
        
        # Overall review score
        overall_score = np.mean(list(review_scores.values()))
        
        # Approval decision
        if overall_score >= 0.8 and risk_assessment['risk_level'] in ['LOW', 'MEDIUM']:
            approval_status = 'APPROVED'
            conditions = ['Standard monitoring required']
        elif overall_score >= 0.7 and risk_assessment['risk_level'] == 'HIGH':
            approval_status = 'CONDITIONAL_APPROVAL'
            conditions = [
                'Enhanced monitoring required',
                'Additional containment measures',
                'Regular reporting to IBC'
            ]
        else:
            approval_status = 'NOT_APPROVED'
            conditions = ['Risk level too high for current protocols']
        
        return {
            'review_scores': review_scores,
            'overall_score': overall_score,
            'approval_status': approval_status,
            'conditions': conditions,
            'committee_members': self.committee_members,
            'review_date': datetime.now().isoformat()
        }

class ExportControlCompliance:
    """
    Export control compliance framework
    """
    
    def __init__(self, regulations: ExportControlRegulations):
        self.regulations = regulations
        self.compliance_data = {}
    
    def assess_itar_compliance(self) -> Dict[str, any]:
        """
        Assess ITAR compliance
        
        Returns:
            Dictionary with ITAR compliance assessment
        """
        itar_compliance = {
            'itar_controlled': self.regulations.itar_controlled,
            'itar_category': self.regulations.itar_category,
            'itar_exemption': self.regulations.itar_exemption,
            'compliance_required': self.regulations.itar_controlled,
            'license_required': self.regulations.itar_controlled,
            'registration_required': self.regulations.itar_controlled
        }
        
        return itar_compliance
    
    def assess_ear_compliance(self) -> Dict[str, any]:
        """
        Assess EAR compliance
        
        Returns:
            Dictionary with EAR compliance assessment
        """
        ear_compliance = {
            'ear_controlled': self.regulations.ear_controlled,
            'ear_category': self.regulations.ear_category,
            'ear_license_required': self.regulations.ear_license_required,
            'technology_transfer_restrictions': self.regulations.technology_transfer_restrictions,
            'end_user_verification': self.regulations.end_user_verification,
            'destination_controls': self.regulations.destination_controls
        }
        
        return ear_compliance
    
    def assess_wassenaar_compliance(self) -> Dict[str, any]:
        """
        Assess Wassenaar Arrangement compliance
        
        Returns:
            Dictionary with Wassenaar compliance assessment
        """
        wassenaar_compliance = {
            'wassenaar_controlled': self.regulations.wassenaar_controlled,
            'wassenaar_category': self.regulations.wassenaar_category,
            'dual_use_controls': self.regulations.wassenaar_controlled,
            'export_license_required': self.regulations.wassenaar_controlled
        }
        
        return wassenaar_compliance
    
    def generate_export_control_report(self) -> Dict[str, any]:
        """
        Generate comprehensive export control compliance report
        
        Returns:
            Dictionary with export control compliance report
        """
        itar_compliance = self.assess_itar_compliance()
        ear_compliance = self.assess_ear_compliance()
        wassenaar_compliance = self.assess_wassenaar_compliance()
        
        # Overall compliance assessment
        overall_compliance = (
            not itar_compliance['compliance_required'] and
            ear_compliance['ear_controlled'] and
            not wassenaar_compliance['wassenaar_controlled']
        )
        
        return {
            'itar_compliance': itar_compliance,
            'ear_compliance': ear_compliance,
            'wassenaar_compliance': wassenaar_compliance,
            'overall_compliance': overall_compliance,
            'recommendations': [
                'Maintain EAR99 classification',
                'Implement technology transfer controls',
                'Conduct regular compliance audits',
                'Update export control procedures as needed'
            ],
            'report_date': datetime.now().isoformat()
        }

class ResponsiblePublicationProtocol:
    """
    Responsible publication and technology transfer protocols
    """
    
    def __init__(self):
        self.publication_guidelines = [
            'Assess dual-use potential before publication',
            'Consider security implications of research',
            'Implement appropriate access controls',
            'Include responsible research statements',
            'Provide context for potential applications'
        ]
        
        self.technology_transfer_protocols = [
            'End-user verification required',
            'Technology transfer agreements mandatory',
            'Restricted to authorized institutions',
            'Regular compliance monitoring',
            'Reporting requirements for transfers'
        ]
    
    def assess_publication_risk(self, research_description: str) -> Dict[str, any]:
        """
        Assess publication risk for research
        
        Args:
            research_description: Description of research to be published
            
        Returns:
            Dictionary with publication risk assessment
        """
        # Simulate risk assessment based on research description
        risk_keywords = [
            'pathogen', 'toxin', 'weapon', 'military', 'defense',
            'bioweapon', 'synthesis', 'modification', 'enhancement'
        ]
        
        risk_score = 0.0
        for keyword in risk_keywords:
            if keyword.lower() in research_description.lower():
                risk_score += 0.2
        
        risk_score = min(risk_score, 1.0)
        
        if risk_score < 0.3:
            publication_status = 'APPROVED'
            restrictions = 'Standard publication protocols'
        elif risk_score < 0.7:
            publication_status = 'CONDITIONAL_APPROVAL'
            restrictions = 'Security review required, access controls'
        else:
            publication_status = 'RESTRICTED'
            restrictions = 'Limited distribution, security clearance required'
        
        return {
            'risk_score': risk_score,
            'publication_status': publication_status,
            'restrictions': restrictions,
            'guidelines_applied': self.publication_guidelines
        }
    
    def generate_technology_transfer_protocol(self, recipient_institution: str) -> Dict[str, any]:
        """
        Generate technology transfer protocol
        
        Args:
            recipient_institution: Name of recipient institution
            
        Returns:
            Dictionary with technology transfer protocol
        """
        # Verify recipient institution
        authorized_institutions = [
            'NASA', 'ESA', 'Academic Research Institutions',
            'Government Research Laboratories', 'International Space Agencies'
        ]
        
        is_authorized = any(inst.lower() in recipient_institution.lower() 
                          for inst in authorized_institutions)
        
        if is_authorized:
            transfer_status = 'APPROVED'
            conditions = self.technology_transfer_protocols
        else:
            transfer_status = 'DENIED'
            conditions = ['Recipient not authorized for technology transfer']
        
        return {
            'recipient_institution': recipient_institution,
            'transfer_status': transfer_status,
            'conditions': conditions,
            'verification_required': True,
            'monitoring_required': True,
            'reporting_frequency': 'Quarterly'
        }

class DURCOversightFramework:
    """
    Comprehensive DURC oversight framework
    """
    
    def __init__(self):
        self.risk_factors = DURCRiskFactors()
        self.risk_assessor = DURCRiskAssessment(self.risk_factors)
        self.ibc = InstitutionalBiosafetyCommittee()
        self.export_controls = ExportControlCompliance(ExportControlRegulations())
        self.publication_protocols = ResponsiblePublicationProtocol()
        
    def conduct_comprehensive_durc_assessment(self, research_description: str) -> Dict[str, any]:
        """
        Conduct comprehensive DURC assessment
        
        Args:
            research_description: Description of research to be assessed
            
        Returns:
            Dictionary with comprehensive DURC assessment
        """
        # Risk assessment
        risk_assessment = self.risk_assessor.calculate_overall_risk()
        
        # IBC review
        ibc_review = self.ibc.conduct_ibc_review(risk_assessment)
        
        # Export control assessment
        export_control_report = self.export_controls.generate_export_control_report()
        
        # Publication risk assessment
        publication_risk = self.publication_protocols.assess_publication_risk(research_description)
        
        # Technology transfer protocol
        tech_transfer_protocol = self.publication_protocols.generate_technology_transfer_protocol(
            'International Research Consortium'
        )
        
        # Overall assessment
        overall_approval = (
            risk_assessment['risk_level'] in ['LOW', 'MEDIUM'] and
            ibc_review['approval_status'] in ['APPROVED', 'CONDITIONAL_APPROVAL'] and
            export_control_report['overall_compliance'] and
            publication_risk['publication_status'] != 'RESTRICTED'
        )
        
        return {
            'risk_assessment': risk_assessment,
            'ibc_review': ibc_review,
            'export_control_compliance': export_control_report,
            'publication_risk': publication_risk,
            'technology_transfer_protocol': tech_transfer_protocol,
            'overall_approval': overall_approval,
            'recommendations': self._generate_recommendations(
                risk_assessment, ibc_review, export_control_report, publication_risk
            ),
            'assessment_date': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, risk_assessment: Dict[str, any],
                                ibc_review: Dict[str, any],
                                export_control_report: Dict[str, any],
                                publication_risk: Dict[str, any]) -> List[str]:
        """
        Generate recommendations based on assessment results
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Risk-based recommendations
        if risk_assessment['risk_level'] in ['HIGH', 'VERY_HIGH']:
            recommendations.append('Implement enhanced oversight and monitoring')
            recommendations.append('Consider additional containment measures')
        
        # IBC-based recommendations
        if ibc_review['approval_status'] == 'CONDITIONAL_APPROVAL':
            recommendations.extend(ibc_review['conditions'])
        
        # Export control recommendations
        if not export_control_report['overall_compliance']:
            recommendations.append('Review and update export control procedures')
        
        # Publication recommendations
        if publication_risk['publication_status'] == 'RESTRICTED':
            recommendations.append('Implement strict publication controls')
        
        return recommendations
    
    def plot_durc_assessment_metrics(self, assessment_data: Dict[str, any],
                                   save_path: Optional[str] = None) -> None:
        """
        Plot DURC assessment metrics
        
        Args:
            assessment_data: Comprehensive assessment data
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DURC Assessment Metrics', fontsize=16)
        
        # Risk assessment breakdown
        risk_data = assessment_data['risk_assessment']
        risk_components = ['Technology', 'Application', 'Mitigation', 'Overall']
        risk_values = [
            risk_data['technology_risk']['technology_risk'],
            risk_data['application_risk']['application_risk'],
            1 - risk_data['mitigation_effectiveness']['mitigation_effectiveness'],
            risk_data['overall_risk']
        ]
        
        colors = ['red' if v > 0.5 else 'orange' if v > 0.3 else 'green' for v in risk_values]
        axes[0, 0].bar(risk_components, risk_values, color=colors)
        axes[0, 0].set_title('Risk Assessment Breakdown')
        axes[0, 0].set_ylabel('Risk Score')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # IBC review scores
        ibc_data = assessment_data['ibc_review']
        ibc_criteria = list(ibc_data['review_scores'].keys())
        ibc_scores = list(ibc_data['review_scores'].values())
        
        axes[0, 1].bar(range(len(ibc_criteria)), ibc_scores)
        axes[0, 1].set_title('IBC Review Scores')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(range(len(ibc_criteria)))
        axes[0, 1].set_xticklabels(ibc_criteria, rotation=45, ha='right')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Export control compliance
        export_data = assessment_data['export_control_compliance']
        export_areas = ['ITAR', 'EAR', 'Wassenaar', 'Overall']
        export_status = [
            not export_data['itar_compliance']['compliance_required'],
            export_data['ear_compliance']['ear_controlled'],
            not export_data['wassenaar_compliance']['wassenaar_controlled'],
            export_data['overall_compliance']
        ]
        
        colors = ['green' if status else 'red' for status in export_status]
        axes[1, 0].bar(export_areas, export_status, color=colors)
        axes[1, 0].set_title('Export Control Compliance')
        axes[1, 0].set_ylabel('Compliant (1) / Non-Compliant (0)')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Publication risk
        pub_data = assessment_data['publication_risk']
        pub_metrics = ['Risk Score', 'Approval Status']
        pub_values = [pub_data['risk_score'], 1 if pub_data['publication_status'] == 'APPROVED' else 0.5]
        
        colors = ['green' if v > 0.8 else 'orange' if v > 0.5 else 'red' for v in pub_values]
        axes[1, 1].bar(pub_metrics, pub_values, color=colors)
        axes[1, 1].set_title('Publication Risk Assessment')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_durc_assessment_report(self, assessment_data: Dict[str, any], 
                                  filepath: str) -> None:
        """
        Save DURC assessment report to file
        
        Args:
            assessment_data: Comprehensive assessment data
            filepath: Path to save the report
        """
        with open(filepath, 'w') as f:
            json.dump(assessment_data, f, indent=2)
        
        logger.info(f"DURC assessment report saved to {filepath}")

def main():
    """Main function to demonstrate DURC oversight framework"""
    
    # Initialize DURC framework
    durc_framework = DURCOversightFramework()
    
    # Sample research description
    research_description = """
    Development of genetically modified B. subtilis strains for Mars terraforming applications.
    The research involves engineering bacteria for perchlorate reduction and environmental
    remediation under extreme conditions. The technology includes biosafety systems and
    containment measures for responsible deployment.
    """
    
    # Conduct comprehensive assessment
    assessment_data = durc_framework.conduct_comprehensive_durc_assessment(research_description)
    
    # Plot assessment metrics
    durc_framework.plot_durc_assessment_metrics(assessment_data)
    
    # Save assessment report
    durc_framework.save_durc_assessment_report(
        assessment_data, 
        'compliance/durc/durc_assessment_report.json'
    )
    
    # Print summary
    print("\n=== DURC Assessment Summary ===")
    print(f"Risk Level: {assessment_data['risk_assessment']['risk_level']}")
    print(f"IBC Approval: {assessment_data['ibc_review']['approval_status']}")
    print(f"Export Control Compliance: {assessment_data['export_control_compliance']['overall_compliance']}")
    print(f"Publication Status: {assessment_data['publication_risk']['publication_status']}")
    print(f"Overall Approval: {assessment_data['overall_approval']}")
    
    print(f"\nRecommendations:")
    for recommendation in assessment_data['recommendations']:
        print(f"  - {recommendation}")

if __name__ == "__main__":
    main()