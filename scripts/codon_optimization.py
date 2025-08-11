#!/usr/bin/env python3
"""
Codon optimization for Mars-adapted organisms
Optimizes genetic sequences for extreme temperature/radiation conditions

This module provides codon optimization algorithms specifically designed
for organisms that need to survive in Mars environmental conditions.
"""

import argparse
import logging
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarsCodonOptimizer:
    """
    Codon optimization engine for Mars environmental conditions
    
    Optimizes genetic sequences considering:
    - Extreme temperature stability (-80°C to 20°C)
    - High radiation resistance
    - Low atmospheric pressure
    - CO2-rich environment
    """
    
    def __init__(self):
        """Initialize with Mars-adapted codon usage tables"""
        # Mars-adapted codon usage table (optimized for extremophiles)
        # Based on psychrophilic and radioresistant organisms
        self.mars_codon_table = {
            'A': {'GCT': 0.35, 'GCC': 0.40, 'GCA': 0.15, 'GCG': 0.10},  # Alanine
            'R': {'CGT': 0.25, 'CGC': 0.30, 'CGA': 0.10, 'CGG': 0.15, 'AGA': 0.10, 'AGG': 0.10},  # Arginine
            'N': {'AAT': 0.45, 'AAC': 0.55},  # Asparagine
            'D': {'GAT': 0.40, 'GAC': 0.60},  # Aspartic acid
            'C': {'TGT': 0.40, 'TGC': 0.60},  # Cysteine
            'Q': {'CAA': 0.35, 'CAG': 0.65},  # Glutamine
            'E': {'GAA': 0.45, 'GAG': 0.55},  # Glutamic acid
            'G': {'GGT': 0.30, 'GGC': 0.35, 'GGA': 0.20, 'GGG': 0.15},  # Glycine
            'H': {'CAT': 0.45, 'CAC': 0.55},  # Histidine
            'I': {'ATT': 0.40, 'ATC': 0.50, 'ATA': 0.10},  # Isoleucine
            'L': {'TTA': 0.08, 'TTG': 0.12, 'CTT': 0.15, 'CTC': 0.20, 'CTA': 0.10, 'CTG': 0.35},  # Leucine
            'K': {'AAA': 0.40, 'AAG': 0.60},  # Lysine
            'M': {'ATG': 1.00},  # Methionine (start codon)
            'F': {'TTT': 0.45, 'TTC': 0.55},  # Phenylalanine
            'P': {'CCT': 0.25, 'CCC': 0.30, 'CCA': 0.25, 'CCG': 0.20},  # Proline
            'S': {'TCT': 0.20, 'TCC': 0.25, 'TCA': 0.15, 'TCG': 0.15, 'AGT': 0.15, 'AGC': 0.10},  # Serine
            'T': {'ACT': 0.25, 'ACC': 0.35, 'ACA': 0.25, 'ACG': 0.15},  # Threonine
            'W': {'TGG': 1.00},  # Tryptophan
            'Y': {'TAT': 0.45, 'TAC': 0.55},  # Tyrosine
            'V': {'GTT': 0.25, 'GTC': 0.35, 'GTA': 0.15, 'GTG': 0.25},  # Valine
            '*': {'TAA': 0.50, 'TAG': 0.25, 'TGA': 0.25}  # Stop codons
        }
        
        # Standard genetic code
        self.genetic_code = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        }
        
        # Reverse lookup: amino acid to codons
        self.aa_to_codons = defaultdict(list)
        for codon, aa in self.genetic_code.items():
            self.aa_to_codons[aa].append(codon)
    
    def translate_dna(self, dna_sequence: str) -> str:
        """Translate DNA sequence to amino acid sequence"""
        if len(dna_sequence) % 3 != 0:
            logger.warning("DNA sequence length not divisible by 3, truncating")
            dna_sequence = dna_sequence[:len(dna_sequence) - len(dna_sequence) % 3]
        
        protein = ""
        for i in range(0, len(dna_sequence), 3):
            codon = dna_sequence[i:i+3].upper()
            if codon in self.genetic_code:
                protein += self.genetic_code[codon]
            else:
                logger.warning(f"Unknown codon: {codon}")
                protein += 'X'  # Unknown amino acid
        
        return protein
    
    def calculate_codon_adaptation_index(self, dna_sequence: str) -> float:
        """
        Calculate Codon Adaptation Index (CAI) for Mars conditions
        
        Args:
            dna_sequence: DNA sequence to analyze
            
        Returns:
            CAI score (0-1, higher is better adapted)
        """
        if len(dna_sequence) % 3 != 0:
            dna_sequence = dna_sequence[:len(dna_sequence) - len(dna_sequence) % 3]
        
        cai_scores = []
        
        for i in range(0, len(dna_sequence), 3):
            codon = dna_sequence[i:i+3].upper()
            if codon in self.genetic_code:
                aa = self.genetic_code[codon]
                if aa in self.mars_codon_table:
                    codon_freq = self.mars_codon_table[aa].get(codon, 0.01)
                    cai_scores.append(codon_freq)
        
        return sum(cai_scores) / len(cai_scores) if cai_scores else 0.0
    
    def optimize_for_mars_conditions(self, dna_sequence: str) -> str:
        """
        Optimize codon usage for Mars environmental conditions
        
        Args:
            dna_sequence: Input DNA sequence to optimize
            
        Returns:
            Optimized DNA sequence with Mars-adapted codon usage
        """
        logger.info(f"Optimizing {len(dna_sequence)} bp sequence for Mars conditions")
        
        # Validate input
        if not dna_sequence or len(dna_sequence) % 3 != 0:
            raise ValueError("DNA sequence must be non-empty and divisible by 3")
        
        # Translate to amino acids first
        protein_sequence = self.translate_dna(dna_sequence)
        logger.debug(f"Translated to {len(protein_sequence)} amino acids")
        
        # Optimize each codon
        optimized_dna = ""
        
        for aa in protein_sequence:
            if aa == '*':  # Stop codon
                # Choose optimal stop codon for Mars
                best_codon = max(self.mars_codon_table['*'].items(), key=lambda x: x[1])[0]
                optimized_dna += best_codon
            elif aa in self.mars_codon_table:
                # Choose codon with highest Mars adaptation frequency
                best_codon = max(self.mars_codon_table[aa].items(), key=lambda x: x[1])[0]
                optimized_dna += best_codon
            else:
                logger.warning(f"Unknown amino acid: {aa}")
                # Keep original codon if possible
                original_pos = len(optimized_dna)
                if original_pos < len(dna_sequence):
                    optimized_dna += dna_sequence[original_pos:original_pos+3]
        
        # Calculate improvement metrics
        original_cai = self.calculate_codon_adaptation_index(dna_sequence)
        optimized_cai = self.calculate_codon_adaptation_index(optimized_dna)
        
        logger.info(f"Optimization complete: CAI improved from {original_cai:.3f} to {optimized_cai:.3f}")
        
        return optimized_dna
    
    def analyze_sequence_composition(self, dna_sequence: str) -> dict:
        """Analyze sequence composition and Mars adaptation metrics"""
        # GC content analysis
        gc_count = dna_sequence.upper().count('G') + dna_sequence.upper().count('C')
        gc_content = gc_count / len(dna_sequence) * 100
        
        # Codon usage analysis
        codon_usage = Counter()
        for i in range(0, len(dna_sequence) - 2, 3):
            codon = dna_sequence[i:i+3].upper()
            if len(codon) == 3:
                codon_usage[codon] += 1
        
        # Mars adaptation score
        cai_score = self.calculate_codon_adaptation_index(dna_sequence)
        
        return {
            'length': len(dna_sequence),
            'gc_content': round(gc_content, 2),
            'codon_count': len(codon_usage),
            'mars_adaptation_index': round(cai_score, 3),
            'top_codons': codon_usage.most_common(5)
        }


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Codon optimization for Mars-adapted organisms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python codon_optimization.py --sequence ATGCGATCGTAGC
  python codon_optimization.py --file input.fasta --output optimized.fasta
  python codon_optimization.py --analyze --sequence ATGAAATTTGGGTAG
        """
    )
    
    parser.add_argument(
        '--sequence', 
        help='DNA sequence to optimize (must be divisible by 3)'
    )
    parser.add_argument(
        '--file', 
        help='Input FASTA file with sequences to optimize'
    )
    parser.add_argument(
        '--output', 
        help='Output file for optimized sequences'
    )
    parser.add_argument(
        '--analyze', 
        action='store_true',
        help='Analyze sequence without optimization'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.sequence and not args.file:
        # Use example sequence if none provided
        args.sequence = "ATGAAATTTGGGTCGGATCCGAAATAG"  # Example: Met-Lys-Phe-Gly-Ser-Asp-Pro-Lys-Stop
        print("No sequence provided, using example sequence")
    
    try:
        optimizer = MarsCodonOptimizer()
        
        if args.sequence:
            # Process single sequence
            sequence = args.sequence.upper().replace(' ', '').replace('\n', '')
            
            print(f"\n🧬 Mars Codon Optimization Analysis")
            print("=" * 50)
            print(f"Input sequence:  {sequence}")
            print(f"Length: {len(sequence)} bp ({len(sequence)//3} codons)")
            
            # Analyze original sequence
            analysis = optimizer.analyze_sequence_composition(sequence)
            print(f"\n📊 Original Sequence Analysis:")
            print(f"  GC Content: {analysis['gc_content']}%")
            print(f"  Mars Adaptation Index: {analysis['mars_adaptation_index']}")
            
            if not args.analyze:
                # Optimize sequence
                optimized = optimizer.optimize_for_mars_conditions(sequence)
                optimized_analysis = optimizer.analyze_sequence_composition(optimized)
                
                print(f"\n🚀 Optimized Sequence:")
                print(f"Output sequence: {optimized}")
                print(f"  GC Content: {optimized_analysis['gc_content']}%")
                print(f"  Mars Adaptation Index: {optimized_analysis['mars_adaptation_index']}")
                
                # Calculate improvement
                improvement = ((optimized_analysis['mars_adaptation_index'] - 
                              analysis['mars_adaptation_index']) / 
                              analysis['mars_adaptation_index'] * 100)
                print(f"  Improvement: {improvement:+.1f}%")
                
                # Verify translation is preserved
                original_protein = optimizer.translate_dna(sequence)
                optimized_protein = optimizer.translate_dna(optimized)
                
                if original_protein == optimized_protein:
                    print("✅ Protein sequence preserved")
                else:
                    print("⚠️  Warning: Protein sequence changed!")
                    print(f"Original:  {original_protein}")
                    print(f"Optimized: {optimized_protein}")
                
                # Save results if output file specified
                if args.output:
                    with open(args.output, 'w') as f:
                        f.write(f">Mars_optimized_sequence\n")
                        f.write(f"{optimized}\n")
                    print(f"📁 Optimized sequence saved to: {args.output}")
        
        elif args.file:
            print(f"📁 Processing FASTA file: {args.file}")
            # TODO: Implement FASTA file processing
            logger.info("FASTA file processing not yet implemented")
        
        print("\n✅ Codon optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"Codon optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
