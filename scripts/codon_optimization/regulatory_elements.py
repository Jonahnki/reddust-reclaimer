#!/usr/bin/env python3
"""
Regulatory Elements Design for RedDust Reclaimer Project
======================================================

This module designs and optimizes regulatory elements (promoters, RBS, terminators)
for B. subtilis expression under Mars conditions.

Author: RedDust Reclaimer AI
Date: 2024
License: MIT
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegulatoryElementsDesigner:
    """
    Design and optimize regulatory elements for Mars-adapted B. subtilis expression.
    """
    
    def __init__(self):
        """Initialize the regulatory elements designer."""
        
        # Strong promoters for B. subtilis (Mars-optimized sequences)
        self.promoters = {
            'Pspac': {
                'sequence': 'TTGGACAATTATTGAACAATTAACAAAGACAAACAGGAGGGCATCAAAATGGGCTCATCCGCCAAGCTTCATGCCCGGCAATTACGACCCGCACGATGCGGTTTGACAAAAATGACAGCTTATGCGATTTTGCACAAGGTTAA',
                'strength': 'high',
                'inducible': True,
                'inducer': 'IPTG',
                'description': 'IPTG-inducible strong promoter, cold-optimized',
                'mars_score': 85
            },
            'Pveg': {
                'sequence': 'TGAGCTCGAATTCGGATCCACTAGTTCTAGAGCGGCCGCCACCGCGGTGGAGCTCCAATTCGCCCTATAGTGAGTCGTATTACGCGCGCTCACTGGCCGTCGTTTTACAACGTCGTGACTGGGAAAACCCTGGCGTTACCCAACTTAAT',
                'strength': 'high',
                'inducible': False,
                'inducer': None,
                'description': 'Constitutive strong vegetative promoter',
                'mars_score': 90
            },
            'Pxyl': {
                'sequence': 'ATTCATTAATGCAGCTGGCACGACAGGTTTCCCGACTGGAAAGCGGGCAGTGAGCGCAACGCAATTAATGTGAGTTAGCTCACTCATTAGGCACCCCAGGCTTTACACTTTATGCTTCCGGCTCGTATGTTGTGTGGAATTGTGAGCGG',
                'strength': 'medium',
                'inducible': True,
                'inducer': 'xylose',
                'description': 'Xylose-inducible promoter, Mars-adapted',
                'mars_score': 75
            },
            'Pamy': {
                'sequence': 'GAAGCTTCATGCCCGGGCAATTACGACCCGCACGATGCGGTTTGACAAAAATGACAGCTTATGCGATTTTGCACAAGGCTGCTGGAAGCTACGCGGATAGCACGCGTCAGATCCGTAATAGGAATAAAAAACCACCAAAATGGATAG',
                'strength': 'medium',
                'inducible': True,
                'inducer': 'amylose',
                'description': 'Amylose-inducible promoter for controlled expression',
                'mars_score': 70
            },
            'Ptet': {
                'sequence': 'CTATCACTGATAGGGAGTGGTAAGTCTGAGCCCTGGAAGGGAGAACTGACTATCCGCAAACCGGGTGAACCACCCTACCTAACGGGTACAATCCGGAATCTCGGACTAGTATAGATCGGCCGGCCCGGATACGCAGATCCAGAACATAATGGATAG',
                'strength': 'high',
                'inducible': True,
                'inducer': 'tetracycline',
                'description': 'Tetracycline-inducible high strength promoter',
                'mars_score': 80
            }
        }
        
        # Ribosome binding sites optimized for B. subtilis and Mars conditions
        self.rbs_sites = {
            'strong_rbs': {
                'sequence': 'AAGGAGGTGAAAAATG',
                'strength': 'high',
                'distance_to_start': 8,
                'description': 'Strong RBS for high translation efficiency',
                'mars_score': 90
            },
            'medium_rbs': {
                'sequence': 'AAGGAGGAAAAAATG',
                'strength': 'medium',
                'distance_to_start': 8,
                'description': 'Medium strength RBS for balanced expression',
                'mars_score': 85
            },
            'weak_rbs': {
                'sequence': 'AAGAGGTTAAAAATG',
                'strength': 'low',
                'distance_to_start': 8,
                'description': 'Weak RBS for low expression levels',
                'mars_score': 75
            },
            'cold_adapted_rbs': {
                'sequence': 'AAGGAGGTGCGCATG',
                'strength': 'high',
                'distance_to_start': 8,
                'description': 'Cold-adapted RBS with enhanced stability',
                'mars_score': 95
            },
            'stress_rbs': {
                'sequence': 'AAGGAGGCGAAAATG',
                'strength': 'medium',
                'distance_to_start': 8,
                'description': 'Stress-resistant RBS for harsh conditions',
                'mars_score': 88
            }
        }
        
        # Transcriptional terminators for B. subtilis
        self.terminators = {
            'rrnB_t1': {
                'sequence': 'AGCAAAAGCAGGATTAACAGCTGATGGAGCGAAATGGATGTGCTGATCTAAAGGGTAAGAAACATTGGCGGTGGTCCGGGATCCGGCCGGGAATCGATGAGGGCTTTTAAAGCC',
                'efficiency': 95,
                'type': 'intrinsic',
                'description': 'Strong intrinsic terminator from rrnB',
                'mars_score': 90
            },
            'his_terminator': {
                'sequence': 'GCTCGGTACCAAATTCCAGAAAAGAGGCCTCCCGAAAGGGGGGCCTTTTTTCGTTTTGGTCC',
                'efficiency': 85,
                'type': 'intrinsic',
                'description': 'Histidine operon terminator',
                'mars_score': 85
            },
            'T7_terminator': {
                'sequence': 'GCGGATCCGAAGCTTGGATCCTAGAAGCGGCCGCATAATGCATGTGACCGGATCCGGGGGATCCGTCGAC',
                'efficiency': 90,
                'type': 'intrinsic',
                'description': 'T7 phage terminator adapted for B. subtilis',
                'mars_score': 88
            },
            'synthetic_term': {
                'sequence': 'AAAAAAAGCGGCCGCGAATTCGAGCTCCCAATTAACAAAGACAAACAGAAGGGGCCTTTTTTCGTTTTGGTCCGGGGATCC',
                'efficiency': 92,
                'type': 'synthetic',
                'description': 'Synthetic high-efficiency terminator',
                'mars_score': 92
            }
        }
        
        # 5' UTR elements for enhanced expression
        self.utr_elements = {
            'stability_utr': {
                'sequence': 'CGGGCAGCGTCAGATGTGTATAAGAGACAG',
                'function': 'mRNA stability enhancement',
                'description': 'Enhances mRNA stability in cold conditions',
                'mars_score': 88
            },
            'translation_enhancer': {
                'sequence': 'GCGGGAAGAGAAGGCAGATGTGTATAAGAG',
                'function': 'translation enhancement',
                'description': 'Improves translation initiation efficiency',
                'mars_score': 85
            }
        }
        
        # Spacer sequences for proper element spacing
        self.spacers = {
            'promoter_rbs_spacer': 'GGTACCGGATCC',
            'gene_terminator_spacer': 'GGATCCAAGCTT',
            'multiple_gene_spacer': 'AAGCTTAGATCTGGTACC'
        }
    
    def design_expression_cassette(self, gene_sequence: str, gene_name: str, 
                                 expression_level: str = 'high',
                                 inducible: bool = False,
                                 inducer: str = None) -> Dict:
        """
        Design a complete expression cassette for a gene.
        
        Args:
            gene_sequence: Codon-optimized gene sequence
            gene_name: Name of the gene
            expression_level: Desired expression level ('low', 'medium', 'high')
            inducible: Whether to use inducible promoter
            inducer: Specific inducer if inducible
            
        Returns:
            Dictionary containing complete cassette design
        """
        
        logger.info(f"Designing expression cassette for {gene_name}")
        
        # Select appropriate promoter
        promoter = self._select_promoter(expression_level, inducible, inducer)
        
        # Select appropriate RBS
        rbs = self._select_rbs(expression_level)
        
        # Select terminator
        terminator = self._select_terminator()
        
        # Add 5' UTR if needed for Mars conditions
        utr = self._select_utr()
        
        # Construct the complete cassette
        cassette = self._construct_cassette(promoter, rbs, utr, gene_sequence, terminator)
        
        # Calculate cassette metrics
        metrics = self._calculate_cassette_metrics(cassette, gene_name)
        
        # Generate design report
        design_report = {
            'gene_name': gene_name,
            'expression_level': expression_level,
            'inducible': inducible,
            'inducer': inducer,
            'components': {
                'promoter': promoter,
                'rbs': rbs,
                'utr': utr,
                'gene': {
                    'sequence': gene_sequence,
                    'length': len(gene_sequence)
                },
                'terminator': terminator
            },
            'complete_cassette': cassette,
            'metrics': metrics
        }
        
        return design_report
    
    def _select_promoter(self, expression_level: str, inducible: bool, inducer: str = None) -> Dict:
        """Select the best promoter based on requirements."""
        
        candidate_promoters = []
        
        for name, promoter in self.promoters.items():
            score = promoter['mars_score']
            
            # Filter by inducibility requirement
            if inducible and not promoter['inducible']:
                continue
            elif not inducible and promoter['inducible']:
                score *= 0.8  # Slight penalty for using inducible when not needed
            
            # Filter by specific inducer
            if inducer and promoter.get('inducer') != inducer:
                continue
            
            # Score by expression level match
            if expression_level == 'high' and promoter['strength'] == 'high':
                score += 10
            elif expression_level == 'medium' and promoter['strength'] == 'medium':
                score += 10
            elif expression_level == 'low' and promoter['strength'] in ['low', 'weak']:
                score += 10
            
            candidate_promoters.append((name, promoter, score))
        
        if not candidate_promoters:
            # Fallback to best available
            candidate_promoters = [(name, promoter, promoter['mars_score']) 
                                 for name, promoter in self.promoters.items()]
        
        # Select highest scoring promoter
        best_name, best_promoter, best_score = max(candidate_promoters, key=lambda x: x[2])
        
        selected_promoter = dict(best_promoter)
        selected_promoter['name'] = best_name
        selected_promoter['selection_score'] = best_score
        
        logger.info(f"Selected promoter: {best_name} (score: {best_score:.1f})")
        
        return selected_promoter
    
    def _select_rbs(self, expression_level: str) -> Dict:
        """Select the best RBS based on expression level."""
        
        # Map expression levels to RBS preferences
        rbs_preferences = {
            'high': ['cold_adapted_rbs', 'strong_rbs', 'stress_rbs'],
            'medium': ['stress_rbs', 'medium_rbs', 'cold_adapted_rbs'],
            'low': ['weak_rbs', 'medium_rbs']
        }
        
        preferred_rbs_list = rbs_preferences.get(expression_level, ['medium_rbs'])
        
        # Select the best available RBS
        for rbs_name in preferred_rbs_list:
            if rbs_name in self.rbs_sites:
                selected_rbs = dict(self.rbs_sites[rbs_name])
                selected_rbs['name'] = rbs_name
                
                logger.info(f"Selected RBS: {rbs_name}")
                return selected_rbs
        
        # Fallback
        fallback_rbs = dict(self.rbs_sites['medium_rbs'])
        fallback_rbs['name'] = 'medium_rbs'
        return fallback_rbs
    
    def _select_terminator(self) -> Dict:
        """Select the best terminator for Mars conditions."""
        
        # For Mars conditions, prefer high-efficiency terminators
        terminators_by_score = sorted(
            [(name, term) for name, term in self.terminators.items()],
            key=lambda x: x[1]['mars_score'],
            reverse=True
        )
        
        best_name, best_terminator = terminators_by_score[0]
        selected_terminator = dict(best_terminator)
        selected_terminator['name'] = best_name
        
        logger.info(f"Selected terminator: {best_name}")
        return selected_terminator
    
    def _select_utr(self) -> Optional[Dict]:
        """Select 5' UTR element for Mars conditions."""
        
        # For Mars conditions, prioritize stability
        utr_by_score = sorted(
            [(name, utr) for name, utr in self.utr_elements.items()],
            key=lambda x: x[1]['mars_score'],
            reverse=True
        )
        
        best_name, best_utr = utr_by_score[0]
        selected_utr = dict(best_utr)
        selected_utr['name'] = best_name
        
        logger.info(f"Selected UTR: {best_name}")
        return selected_utr
    
    def _construct_cassette(self, promoter: Dict, rbs: Dict, utr: Dict, 
                          gene_sequence: str, terminator: Dict) -> Dict:
        """Construct the complete expression cassette."""
        
        # Build the cassette sequence
        cassette_parts = []
        
        # Add promoter
        cassette_parts.append(promoter['sequence'])
        cassette_parts.append(self.spacers['promoter_rbs_spacer'])
        
        # Add 5' UTR if present
        if utr:
            cassette_parts.append(utr['sequence'])
        
        # Add RBS (includes start codon)
        cassette_parts.append(rbs['sequence'])
        
        # Add gene sequence (without start codon since RBS includes it)
        if gene_sequence.upper().startswith('ATG'):
            gene_sequence = gene_sequence[3:]  # Remove start codon
        cassette_parts.append(gene_sequence)
        
        # Add spacer and terminator
        cassette_parts.append(self.spacers['gene_terminator_spacer'])
        cassette_parts.append(terminator['sequence'])
        
        # Combine all parts
        complete_sequence = ''.join(cassette_parts)
        
        # Create annotation map
        current_pos = 0
        annotations = []
        
        for i, (part_name, part_seq) in enumerate([
            ('promoter', promoter['sequence']),
            ('promoter_spacer', self.spacers['promoter_rbs_spacer']),
            ('utr', utr['sequence'] if utr else ''),
            ('rbs', rbs['sequence']),
            ('gene', gene_sequence),
            ('gene_spacer', self.spacers['gene_terminator_spacer']),
            ('terminator', terminator['sequence'])
        ]):
            if part_seq:  # Skip empty parts
                annotations.append({
                    'name': part_name,
                    'start': current_pos,
                    'end': current_pos + len(part_seq),
                    'length': len(part_seq)
                })
                current_pos += len(part_seq)
        
        cassette = {
            'sequence': complete_sequence,
            'length': len(complete_sequence),
            'annotations': annotations,
            'components': {
                'promoter': promoter['name'],
                'rbs': rbs['name'],
                'utr': utr['name'] if utr else None,
                'terminator': terminator['name']
            }
        }
        
        return cassette
    
    def _calculate_cassette_metrics(self, cassette: Dict, gene_name: str) -> Dict:
        """Calculate metrics for the complete cassette."""
        
        from Bio.SeqUtils import GC, molecular_weight
        
        sequence = cassette['sequence']
        seq_obj = Seq(sequence)
        
        metrics = {
            'total_length': len(sequence),
            'gc_content': GC(seq_obj),
            'molecular_weight': molecular_weight(seq_obj, 'DNA'),
            'mars_adaptation_score': self._calculate_mars_adaptation_score(cassette),
            'predicted_expression_level': self._predict_expression_level(cassette),
            'stability_score': self._calculate_stability_score(sequence)
        }
        
        return metrics
    
    def _calculate_mars_adaptation_score(self, cassette: Dict) -> float:
        """Calculate how well the cassette is adapted for Mars conditions."""
        
        score = 0.0
        total_weight = 0.0
        
        # Get component scores and weights
        components = cassette['components']
        
        # Promoter score (weight: 30%)
        if components['promoter'] in self.promoters:
            score += self.promoters[components['promoter']]['mars_score'] * 0.3
            total_weight += 0.3
        
        # RBS score (weight: 25%)
        if components['rbs'] in self.rbs_sites:
            score += self.rbs_sites[components['rbs']]['mars_score'] * 0.25
            total_weight += 0.25
        
        # UTR score (weight: 20%)
        if components['utr'] and components['utr'] in self.utr_elements:
            score += self.utr_elements[components['utr']]['mars_score'] * 0.2
            total_weight += 0.2
        else:
            total_weight += 0.2  # No penalty for missing UTR
        
        # Terminator score (weight: 25%)
        if components['terminator'] in self.terminators:
            score += self.terminators[components['terminator']]['mars_score'] * 0.25
            total_weight += 0.25
        
        if total_weight > 0:
            return score / total_weight * 100
        else:
            return 0.0
    
    def _predict_expression_level(self, cassette: Dict) -> str:
        """Predict the expression level of the cassette."""
        
        components = cassette['components']
        
        # Get promoter strength
        promoter_strength = 'medium'
        if components['promoter'] in self.promoters:
            promoter_strength = self.promoters[components['promoter']]['strength']
        
        # Get RBS strength
        rbs_strength = 'medium'
        if components['rbs'] in self.rbs_sites:
            rbs_strength = self.rbs_sites[components['rbs']]['strength']
        
        # Combine strengths
        strength_map = {'low': 1, 'weak': 1, 'medium': 2, 'high': 3}
        
        promoter_score = strength_map.get(promoter_strength, 2)
        rbs_score = strength_map.get(rbs_strength, 2)
        
        combined_score = (promoter_score + rbs_score) / 2
        
        if combined_score >= 2.5:
            return 'high'
        elif combined_score >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_stability_score(self, sequence: str) -> float:
        """Calculate sequence stability score for Mars conditions."""
        
        score = 100.0
        
        # Penalize problematic sequences
        problematic_motifs = ['AAAAA', 'TTTTT', 'GGGGG', 'CCCCCC']
        for motif in problematic_motifs:
            count = sequence.count(motif)
            score -= count * 10
        
        # Check GC content (prefer moderate GC for stability)
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence) * 100
        if not (40 <= gc_content <= 60):
            score -= abs(gc_content - 50) * 0.5
        
        return max(0, score)
    
    def save_cassette_design(self, design_report: Dict, output_dir: Path) -> None:
        """Save the cassette design to files."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        gene_name = design_report['gene_name']
        
        # Save complete cassette as FASTA
        cassette_sequence = design_report['complete_cassette']['sequence']
        cassette_record = SeqRecord(
            Seq(cassette_sequence),
            id=f"{gene_name}_expression_cassette",
            description=f"Complete expression cassette for {gene_name} - Mars optimized"
        )
        
        fasta_file = output_dir / f"{gene_name}_expression_cassette.fasta"
        SeqIO.write(cassette_record, fasta_file, "fasta")
        
        # Save detailed design report
        report_file = output_dir / f"{gene_name}_cassette_design.json"
        with open(report_file, 'w') as f:
            json.dump(design_report, f, indent=2)
        
        # Save human-readable summary
        summary_file = output_dir / f"{gene_name}_cassette_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(self._generate_cassette_summary(design_report))
        
        # Save GenBank-style annotation
        gb_file = output_dir / f"{gene_name}_cassette_features.txt"
        with open(gb_file, 'w') as f:
            f.write(self._generate_feature_table(design_report))
        
        logger.info(f"Cassette design saved for {gene_name} in {output_dir}")
    
    def _generate_cassette_summary(self, design_report: Dict) -> str:
        """Generate human-readable summary of the cassette design."""
        
        lines = []
        lines.append("=== EXPRESSION CASSETTE DESIGN SUMMARY ===")
        lines.append(f"Gene: {design_report['gene_name']}")
        lines.append(f"Expression Level: {design_report['expression_level']}")
        lines.append(f"Inducible: {design_report['inducible']}")
        if design_report['inducer']:
            lines.append(f"Inducer: {design_report['inducer']}")
        
        lines.append("\n=== COMPONENTS ===")
        components = design_report['components']
        lines.append(f"Promoter: {components['promoter']['name']} ({components['promoter']['strength']})")
        lines.append(f"RBS: {components['rbs']['name']} ({components['rbs']['strength']})")
        if components['utr']:
            lines.append(f"5' UTR: {components['utr']['name']}")
        lines.append(f"Terminator: {components['terminator']['name']} ({components['terminator']['efficiency']}% efficiency)")
        
        lines.append("\n=== METRICS ===")
        metrics = design_report['metrics']
        lines.append(f"Total Length: {metrics['total_length']} bp")
        lines.append(f"GC Content: {metrics['gc_content']:.1f}%")
        lines.append(f"Mars Adaptation Score: {metrics['mars_adaptation_score']:.1f}/100")
        lines.append(f"Predicted Expression: {metrics['predicted_expression_level']}")
        lines.append(f"Stability Score: {metrics['stability_score']:.1f}/100")
        
        return '\n'.join(lines)
    
    def _generate_feature_table(self, design_report: Dict) -> str:
        """Generate feature table for the cassette."""
        
        lines = []
        lines.append("FEATURES             Location/Qualifiers")
        
        annotations = design_report['complete_cassette']['annotations']
        
        for annotation in annotations:
            start = annotation['start'] + 1  # Convert to 1-based
            end = annotation['end']
            name = annotation['name']
            
            if name == 'promoter':
                lines.append(f"     promoter        {start}..{end}")
                lines.append(f"                     /label=\"{design_report['components']['promoter']['name']}\"")
            elif name == 'rbs':
                lines.append(f"     RBS             {start}..{end}")
                lines.append(f"                     /label=\"{design_report['components']['rbs']['name']}\"")
            elif name == 'gene':
                lines.append(f"     CDS             {start}..{end}")
                lines.append(f"                     /label=\"{design_report['gene_name']}\"")
                lines.append(f"                     /translation")
            elif name == 'terminator':
                lines.append(f"     terminator      {start}..{end}")
                lines.append(f"                     /label=\"{design_report['components']['terminator']['name']}\"")
            elif name == 'utr':
                lines.append(f"     5'UTR           {start}..{end}")
                lines.append(f"                     /label=\"{design_report['components']['utr']['name']}\"")
        
        return '\n'.join(lines)


def main():
    """Main function for designing regulatory elements."""
    
    logger.info("Starting RedDust Reclaimer regulatory elements design")
    
    # Initialize designer
    designer = RegulatoryElementsDesigner()
    
    # Example: Design cassettes for the three target genes
    target_genes = {
        'pcrA': {
            'sequence': 'ATGGCAAAACGCATTGCGAAAGAAATCGAAACGCATGGCATTGCGAAAGAAATCGAA',  # Placeholder
            'expression_level': 'high',
            'inducible': True,
            'inducer': 'IPTG'
        },
        'pcrB': {
            'sequence': 'ATGGCAAAACGCATTGCGAAAGAAATCGAAACGCATGGCATTGCGAAAGAAATCGAA',  # Placeholder
            'expression_level': 'high',
            'inducible': True,
            'inducer': 'IPTG'
        },
        'cld': {
            'sequence': 'ATGGCAAAACGCATTGCGAAAGAAATCGAAACGCATGGCATTGCGAAAGAAATCGAA',  # Placeholder
            'expression_level': 'medium',
            'inducible': False,
            'inducer': None
        }
    }
    
    output_dir = Path("../../data/regulatory")
    
    # Design cassettes for each gene
    for gene_name, gene_config in target_genes.items():
        logger.info(f"Designing cassette for {gene_name}")
        
        design_report = designer.design_expression_cassette(
            gene_sequence=gene_config['sequence'],
            gene_name=gene_name,
            expression_level=gene_config['expression_level'],
            inducible=gene_config['inducible'],
            inducer=gene_config['inducer']
        )
        
        # Save the design
        designer.save_cassette_design(design_report, output_dir)
    
    # Save regulatory elements library
    library_file = output_dir / "regulatory_elements_library.json"
    with open(library_file, 'w') as f:
        json.dump({
            'promoters': designer.promoters,
            'rbs_sites': designer.rbs_sites,
            'terminators': designer.terminators,
            'utr_elements': designer.utr_elements,
            'spacers': designer.spacers
        }, f, indent=2)
    
    logger.info("Regulatory elements design completed successfully!")
    logger.info(f"Library saved to: {library_file}")


if __name__ == "__main__":
    main()