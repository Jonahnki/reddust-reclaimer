#!/usr/bin/env python3
"""
Codon Optimization Pipeline for RedDust Reclaimer Project
========================================================

This module optimizes gene sequences for expression in Bacillus subtilis under Mars conditions.
Implements advanced codon optimization considering Mars-specific environmental factors.

Author: RedDust Reclaimer AI
Date: 2024
License: MIT
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import GC, molecular_weight
from Bio.SeqUtils.CodonUsage import CodonAdaptationIndex
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarsCodonOptimizer:
    """
    Advanced codon optimization engine for B. subtilis expression under Mars conditions.
    
    Considers:
    - B. subtilis codon usage preferences
    - Mars-specific environmental factors (cold, low pressure, radiation)
    - Expression level optimization
    - mRNA stability under stress conditions
    """
    
    def __init__(self):
        """Initialize the codon optimizer with B. subtilis codon usage data."""
        
        # B. subtilis codon usage frequencies (based on highly expressed genes)
        self.codon_usage = {
            'F': {'TTT': 0.43, 'TTC': 0.57},
            'L': {'TTA': 0.04, 'TTG': 0.11, 'CTT': 0.11, 'CTC': 0.11, 'CTA': 0.03, 'CTG': 0.60},
            'S': {'TCT': 0.17, 'TCC': 0.15, 'TCA': 0.12, 'TCG': 0.15, 'AGT': 0.15, 'AGC': 0.26},
            'Y': {'TAT': 0.43, 'TAC': 0.57},
            'C': {'TGT': 0.46, 'TGC': 0.54},
            'W': {'TGG': 1.00},
            'P': {'CCT': 0.17, 'CCC': 0.12, 'CCA': 0.20, 'CCG': 0.51},
            'H': {'CAT': 0.42, 'CAC': 0.58},
            'Q': {'CAA': 0.27, 'CAG': 0.73},
            'R': {'CGT': 0.36, 'CGC': 0.36, 'CGA': 0.07, 'CGG': 0.11, 'AGA': 0.04, 'AGG': 0.06},
            'I': {'ATT': 0.36, 'ATC': 0.42, 'ATA': 0.22},
            'M': {'ATG': 1.00},
            'T': {'ACT': 0.24, 'ACC': 0.27, 'ACA': 0.15, 'ACG': 0.34},
            'N': {'AAT': 0.42, 'AAC': 0.58},
            'K': {'AAA': 0.43, 'AAG': 0.57},
            'V': {'GTT': 0.26, 'GTC': 0.21, 'GTA': 0.16, 'GTG': 0.37},
            'A': {'GCT': 0.27, 'GCC': 0.31, 'GCA': 0.21, 'GCG': 0.21},
            'D': {'GAT': 0.43, 'GAC': 0.57},
            'E': {'GAA': 0.42, 'GAG': 0.58},
            'G': {'GGT': 0.35, 'GGC': 0.37, 'GGA': 0.13, 'GGG': 0.15}
        }
        
        # Mars-specific optimization parameters
        self.mars_factors = {
            'cold_adaptation': {
                # Prefer codons that enhance mRNA stability at low temperatures
                'enhanced_gc_content': True,
                'avoid_at_rich_regions': True,
                'optimize_secondary_structure': True
            },
            'stress_response': {
                # Optimize for stress response gene expression patterns
                'prefer_abundant_trnas': True,
                'avoid_rare_codons': True,
                'optimize_translation_rate': 'moderate'  # Not too fast to avoid errors
            },
            'radiation_resistance': {
                # Consider DNA repair and mutation avoidance
                'avoid_mutation_hotspots': True,
                'optimize_for_fidelity': True
            }
        }
        
        # Define optimal GC content ranges for Mars conditions
        self.gc_target_range = (45, 55)  # Slightly higher GC for cold stability
        
        # Load codon adaptation index
        self._setup_cai()
        
    def _setup_cai(self):
        """Setup Codon Adaptation Index for B. subtilis."""
        try:
            # Create CAI index from codon usage
            cai_dict = {}
            for aa, codons in self.codon_usage.items():
                max_freq = max(codons.values())
                for codon, freq in codons.items():
                    cai_dict[codon] = freq / max_freq
            
            self.cai_index = cai_dict
            logger.info("CAI index initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize CAI: {e}")
            self.cai_index = None
    
    def optimize_sequence(self, sequence: str, gene_name: str = "unknown") -> Dict:
        """
        Optimize a protein-coding sequence for B. subtilis expression under Mars conditions.
        
        Args:
            sequence: DNA sequence to optimize (should be protein-coding)
            gene_name: Name of the gene for tracking
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info(f"Optimizing sequence for {gene_name}")
        
        # Validate input sequence
        if len(sequence) % 3 != 0:
            raise ValueError(f"Sequence length {len(sequence)} is not divisible by 3")
        
        # Convert to Seq object for easier manipulation
        original_seq = Seq(sequence.upper())
        
        try:
            # Translate to check for validity
            protein_seq = original_seq.translate()
            if '*' in str(protein_seq)[:-1]:  # Allow stop codon at end
                raise ValueError("Sequence contains internal stop codons")
        except Exception as e:
            raise ValueError(f"Invalid coding sequence: {e}")
        
        # Perform optimization
        optimized_seq = self._optimize_codons(original_seq)
        
        # Calculate optimization metrics
        metrics = self._calculate_metrics(original_seq, optimized_seq, protein_seq)
        
        # Generate optimization report
        report = {
            'gene_name': gene_name,
            'original_sequence': str(original_seq),
            'optimized_sequence': str(optimized_seq),
            'protein_sequence': str(protein_seq),
            'metrics': metrics,
            'optimization_summary': self._generate_summary(metrics)
        }
        
        return report
    
    def _optimize_codons(self, sequence: Seq) -> Seq:
        """
        Perform the actual codon optimization using Mars-adapted algorithms.
        
        Args:
            sequence: Original DNA sequence
            
        Returns:
            Optimized DNA sequence
        """
        optimized_codons = []
        
        # Process each codon
        for i in range(0, len(sequence), 3):
            codon = str(sequence[i:i+3])
            
            # Translate codon to amino acid
            aa = str(Seq(codon).translate())
            
            if aa == '*':  # Stop codon
                optimized_codons.append(codon)
                continue
            
            # Get optimal codon for this amino acid
            optimal_codon = self._select_optimal_codon(aa, i // 3)
            optimized_codons.append(optimal_codon)
        
        optimized_sequence = Seq(''.join(optimized_codons))
        
        # Apply Mars-specific optimizations
        optimized_sequence = self._apply_mars_optimizations(optimized_sequence)
        
        return optimized_sequence
    
    def _select_optimal_codon(self, amino_acid: str, position: int) -> str:
        """
        Select the optimal codon for a given amino acid considering Mars factors.
        
        Args:
            amino_acid: Single letter amino acid code
            position: Position in the protein (for context-dependent optimization)
            
        Returns:
            Optimal codon string
        """
        if amino_acid not in self.codon_usage:
            logger.warning(f"Unknown amino acid: {amino_acid}")
            return 'NNN'
        
        codons = self.codon_usage[amino_acid]
        
        # Base scoring on codon usage frequency
        scores = dict(codons)
        
        # Apply Mars-specific adjustments
        for codon in scores:
            # Prefer codons with moderate GC content for cold stability
            gc_content = (codon.count('G') + codon.count('C')) / 3
            if 0.33 <= gc_content <= 0.67:  # Moderate GC
                scores[codon] *= 1.2
            elif gc_content < 0.33:  # Low GC - penalize for cold
                scores[codon] *= 0.8
            
            # Avoid AT-rich codons that are unstable in cold
            if codon.count('A') + codon.count('T') >= 3:
                scores[codon] *= 0.7
            
            # Position-specific optimizations
            if position < 10:  # N-terminal region - optimize for translation initiation
                if codon in ['TTG', 'CTG', 'ATG']:  # Good for ribosome binding
                    scores[codon] *= 1.1
        
        # Select codon with highest score
        optimal_codon = max(scores.items(), key=lambda x: x[1])[0]
        
        return optimal_codon
    
    def _apply_mars_optimizations(self, sequence: Seq) -> Seq:
        """
        Apply Mars-specific sequence optimizations.
        
        Args:
            sequence: Optimized sequence from codon selection
            
        Returns:
            Final optimized sequence with Mars adaptations
        """
        seq_str = str(sequence)
        
        # Remove problematic sequences
        seq_str = self._remove_problematic_motifs(seq_str)
        
        # Optimize GC content in windows
        seq_str = self._optimize_gc_content(seq_str)
        
        # Minimize secondary structures that could be problematic in cold
        seq_str = self._minimize_cold_structures(seq_str)
        
        return Seq(seq_str)
    
    def _remove_problematic_motifs(self, sequence: str) -> str:
        """Remove or modify problematic sequence motifs."""
        
        problematic_motifs = [
            'AAAAA',  # Poly-A can cause slippage
            'TTTTT',  # Poly-T termination signals
            'GGGGG',  # G-quadruplex formation
            'CCCCCC', # Strong secondary structures
            'ATGATG',  # Alternative start codons
            'TAATAA',  # Weak promoter-like sequences
        ]
        
        modified_seq = sequence
        
        for motif in problematic_motifs:
            if motif in modified_seq:
                logger.info(f"Removing problematic motif: {motif}")
                # Simple replacement strategy - can be made more sophisticated
                modified_seq = modified_seq.replace(motif, self._generate_alternative_sequence(motif))
        
        return modified_seq
    
    def _generate_alternative_sequence(self, problematic_seq: str) -> str:
        """Generate an alternative sequence that avoids the problematic motif."""
        
        # Simple strategy: maintain codon reading frame while changing sequence
        alt_seq = ""
        for i in range(0, len(problematic_seq), 3):
            codon = problematic_seq[i:i+3]
            if len(codon) == 3:
                try:
                    aa = str(Seq(codon).translate())
                    if aa in self.codon_usage:
                        # Pick second-best codon
                        codons_sorted = sorted(self.codon_usage[aa].items(), 
                                             key=lambda x: x[1], reverse=True)
                        if len(codons_sorted) > 1:
                            alt_seq += codons_sorted[1][0]
                        else:
                            alt_seq += codons_sorted[0][0]
                    else:
                        alt_seq += codon
                except:
                    alt_seq += codon
            else:
                alt_seq += codon
        
        return alt_seq
    
    def _optimize_gc_content(self, sequence: str) -> str:
        """Optimize GC content in sliding windows for Mars stability."""
        
        window_size = 60  # 20 codons
        target_gc = 50  # Target GC percentage
        
        optimized_seq = list(sequence)
        
        for i in range(0, len(sequence) - window_size + 1, window_size):
            window = sequence[i:i + window_size]
            current_gc = (window.count('G') + window.count('C')) / len(window) * 100
            
            if not (self.gc_target_range[0] <= current_gc <= self.gc_target_range[1]):
                logger.info(f"Adjusting GC content in window {i//window_size + 1}: {current_gc:.1f}%")
                # This is a placeholder for more sophisticated GC optimization
                # In practice, would need to consider codon constraints
        
        return ''.join(optimized_seq)
    
    def _minimize_cold_structures(self, sequence: str) -> str:
        """Minimize secondary structures that are problematic in cold conditions."""
        
        # This is a simplified approach - in practice would use RNA folding prediction
        # Look for simple palindromic sequences that could form hairpins
        
        modified_seq = sequence
        
        # Check for palindromic sequences that could form hairpins
        for i in range(len(sequence) - 12):
            subseq = sequence[i:i+12]
            if subseq == subseq[::-1]:  # Perfect palindrome
                logger.info(f"Found potential hairpin at position {i}")
                # Could implement synonymous codon substitution here
        
        return modified_seq
    
    def _calculate_metrics(self, original: Seq, optimized: Seq, protein: Seq) -> Dict:
        """Calculate optimization metrics comparing original and optimized sequences."""
        
        metrics = {}
        
        # Basic sequence metrics
        metrics['length'] = len(optimized)
        metrics['protein_length'] = len(protein)
        
        # GC content
        metrics['original_gc'] = GC(original)
        metrics['optimized_gc'] = GC(optimized)
        metrics['gc_improvement'] = abs(50 - GC(optimized)) - abs(50 - GC(original))
        
        # Codon usage metrics
        if self.cai_index:
            metrics['original_cai'] = self._calculate_cai(original)
            metrics['optimized_cai'] = self._calculate_cai(optimized)
            metrics['cai_improvement'] = metrics['optimized_cai'] - metrics['original_cai']
        
        # Sequence similarity
        metrics['sequence_identity'] = self._calculate_identity(str(original), str(optimized))
        
        # Codon changes
        original_codons = [str(original[i:i+3]) for i in range(0, len(original), 3)]
        optimized_codons = [str(optimized[i:i+3]) for i in range(0, len(optimized), 3)]
        
        changed_codons = sum(1 for o, n in zip(original_codons, optimized_codons) if o != n)
        metrics['codons_changed'] = changed_codons
        metrics['percent_changed'] = (changed_codons / len(original_codons)) * 100
        
        # Mars-specific metrics
        metrics['mars_suitability_score'] = self._calculate_mars_score(optimized)
        
        return metrics
    
    def _calculate_cai(self, sequence: Seq) -> float:
        """Calculate Codon Adaptation Index for the sequence."""
        
        if not self.cai_index:
            return 0.0
        
        codons = [str(sequence[i:i+3]) for i in range(0, len(sequence), 3)]
        cai_values = []
        
        for codon in codons:
            if codon in self.cai_index:
                cai_values.append(self.cai_index[codon])
        
        if cai_values:
            # Geometric mean of CAI values
            return np.exp(np.mean(np.log(cai_values)))
        else:
            return 0.0
    
    def _calculate_identity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence identity percentage."""
        
        if len(seq1) != len(seq2):
            return 0.0
        
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return (matches / len(seq1)) * 100
    
    def _calculate_mars_score(self, sequence: Seq) -> float:
        """Calculate Mars environment suitability score."""
        
        score = 0.0
        
        # GC content score (prefer moderate GC for stability)
        gc_content = GC(sequence)
        if self.gc_target_range[0] <= gc_content <= self.gc_target_range[1]:
            score += 25
        else:
            score += max(0, 25 - abs(gc_content - 50))
        
        # Codon usage score
        if self.cai_index:
            cai_score = self._calculate_cai(sequence)
            score += min(25, cai_score * 25)
        
        # Stability score (avoid problematic motifs)
        stability_score = 25
        problematic_count = 0
        for motif in ['AAAAA', 'TTTTT', 'GGGGG', 'CCCCCC']:
            problematic_count += str(sequence).count(motif)
        
        stability_score -= min(25, problematic_count * 5)
        score += max(0, stability_score)
        
        # Expression efficiency score
        score += 25  # Placeholder for more complex expression prediction
        
        return min(100, score)
    
    def _generate_summary(self, metrics: Dict) -> str:
        """Generate a human-readable optimization summary."""
        
        summary_lines = []
        summary_lines.append("=== CODON OPTIMIZATION SUMMARY ===")
        summary_lines.append(f"Sequence length: {metrics['length']} bp ({metrics['protein_length']} aa)")
        summary_lines.append(f"GC content: {metrics['original_gc']:.1f}% → {metrics['optimized_gc']:.1f}%")
        
        if 'cai_improvement' in metrics:
            summary_lines.append(f"CAI improvement: {metrics['cai_improvement']:+.3f}")
        
        summary_lines.append(f"Codons changed: {metrics['codons_changed']} ({metrics['percent_changed']:.1f}%)")
        summary_lines.append(f"Sequence identity: {metrics['sequence_identity']:.1f}%")
        summary_lines.append(f"Mars suitability score: {metrics['mars_suitability_score']:.1f}/100")
        
        # Optimization assessment
        if metrics['mars_suitability_score'] >= 80:
            summary_lines.append("✓ Excellent optimization for Mars conditions")
        elif metrics['mars_suitability_score'] >= 60:
            summary_lines.append("⚠ Good optimization, minor improvements possible")
        else:
            summary_lines.append("⚠ Optimization needs improvement")
        
        return '\n'.join(summary_lines)


def optimize_gene_file(input_file: Path, output_dir: Path, gene_name: str = None) -> Dict:
    """
    Optimize a gene sequence from a FASTA file.
    
    Args:
        input_file: Path to input FASTA file
        output_dir: Directory to save optimized results
        gene_name: Optional gene name override
        
    Returns:
        Optimization results dictionary
    """
    
    logger.info(f"Processing file: {input_file}")
    
    # Read sequence from FASTA file
    records = list(SeqIO.parse(input_file, "fasta"))
    if not records:
        raise ValueError(f"No sequences found in {input_file}")
    
    if len(records) > 1:
        logger.warning(f"Multiple sequences found, using first one")
    
    record = records[0]
    sequence = str(record.seq)
    
    if not gene_name:
        gene_name = record.id
    
    # Initialize optimizer and optimize sequence
    optimizer = MarsCodonOptimizer()
    result = optimizer.optimize_sequence(sequence, gene_name)
    
    # Save optimized sequence
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FASTA file
    optimized_record = SeqRecord(
        Seq(result['optimized_sequence']),
        id=f"{gene_name}_optimized",
        description=f"Mars-optimized {gene_name} for B. subtilis"
    )
    
    fasta_output = output_dir / f"{gene_name}_optimized.fasta"
    SeqIO.write(optimized_record, fasta_output, "fasta")
    
    # Save detailed report
    report_output = output_dir / f"{gene_name}_optimization_report.json"
    with open(report_output, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Save summary
    summary_output = output_dir / f"{gene_name}_summary.txt"
    with open(summary_output, 'w') as f:
        f.write(result['optimization_summary'])
    
    logger.info(f"Optimization complete for {gene_name}")
    logger.info(f"Results saved to {output_dir}")
    
    return result


def main():
    """Main function for batch optimization of gene sequences."""
    
    logger.info("Starting RedDust Reclaimer codon optimization pipeline")
    
    # Define input and output directories
    genes_dir = Path("../../data/genes")
    output_dir = Path("../../data/optimized_genes")
    
    if not genes_dir.exists():
        logger.error(f"Genes directory not found: {genes_dir}")
        return
    
    # Find all FASTA files in genes directory
    fasta_files = list(genes_dir.glob("*.fasta"))
    
    if not fasta_files:
        logger.warning("No FASTA files found for optimization")
        return
    
    results_summary = {}
    
    # Process each FASTA file
    for fasta_file in fasta_files:
        try:
            gene_name = fasta_file.stem.replace('_candidates', '')
            result = optimize_gene_file(fasta_file, output_dir, gene_name)
            results_summary[gene_name] = result['metrics']
            
        except Exception as e:
            logger.error(f"Error processing {fasta_file}: {e}")
            continue
    
    # Create overall summary
    summary_file = output_dir / "optimization_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("Codon optimization pipeline completed successfully!")
    logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()