"""
RedDust Reclaimer Scripts Package

This package contains example workflows and tools for Mars terraforming
computational biology research.

Modules:
- dock_example: Protein-ligand docking simulations for Mars conditions
- codon_optimization: Genetic sequence optimization for extremophiles
- metabolic_flux: Flux balance analysis for Mars resource utilization
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main classes for easy access
try:
    from .dock_example import MarsEnzymeDocking
    from .codon_optimization import MarsCodonOptimizer
    from .metabolic_flux import MarsMetabolicNetwork
    
    __all__ = [
        'MarsEnzymeDocking',
        'MarsCodonOptimizer', 
        'MarsMetabolicNetwork'
    ]
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Some dependencies missing: {e}")
    __all__ = []
