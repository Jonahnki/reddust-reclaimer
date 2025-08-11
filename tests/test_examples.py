"""Tests for example scripts to ensure they run without errors"""
import pytest
import sys
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

def test_dock_example_imports():
    """Test that dock_example can be imported"""
    try:
        import dock_example
        assert hasattr(dock_example, 'MarsEnzymeDocking')
        assert hasattr(dock_example, 'main')
        
        # Test class instantiation
        docker = dock_example.MarsEnzymeDocking()
        assert docker.temperature == 233.15
        assert docker.pressure == 0.006
        
    except ImportError as e:
        pytest.skip(f"Missing dependencies for docking: {e}")

def test_codon_optimization_imports():
    """Test that codon_optimization can be imported"""
    import codon_optimization
    assert hasattr(codon_optimization, 'MarsCodonOptimizer')
    assert hasattr(codon_optimization, 'main')
    
    # Test class instantiation
    optimizer = codon_optimization.MarsCodonOptimizer()
    assert hasattr(optimizer, 'mars_codon_table')
    assert hasattr(optimizer, 'optimize_for_mars_conditions')

def test_metabolic_flux_imports():
    """Test that metabolic_flux can be imported"""
    try:
        import metabolic_flux
        assert hasattr(metabolic_flux, 'MarsMetabolicNetwork')
        assert hasattr(metabolic_flux, 'main')
        
        # Test class instantiation
        network = metabolic_flux.MarsMetabolicNetwork()
        assert hasattr(network, 'metabolites')
        assert hasattr(network, 'reactions')
        
    except ImportError as e:
        pytest.skip(f"Missing dependencies for flux analysis: {e}")

def test_codon_optimization_basic_functionality():
    """Test basic codon optimization functionality"""
    import codon_optimization
    
    optimizer = codon_optimization.MarsCodonOptimizer()
    
    # Test with simple sequence
    test_sequence = "ATGAAATTTGGGTAG"  # Met-Lys-Phe-Gly-Stop
    
    # Test translation
    protein = optimizer.translate_dna(test_sequence)
    assert protein == "MKFG*"
    
    # Test optimization
    optimized = optimizer.optimize_for_mars_conditions(test_sequence)
    assert len(optimized) == len(test_sequence)
    assert len(optimized) % 3 == 0
    
    # Verify protein sequence is preserved
    optimized_protein = optimizer.translate_dna(optimized)
    assert protein == optimized_protein

def test_metabolic_flux_basic_functionality():
    """Test basic metabolic flux analysis functionality"""
    try:
        import metabolic_flux
        
        network = metabolic_flux.MarsMetabolicNetwork()
        
        # Test FBA execution
        results = network.mars_metabolic_flux_analysis()
        
        assert 'objective' in results
        assert 'success' in results
        assert 'fluxes' in results
        
        if results['success']:
            assert 'objective_value' in results
            assert 'mars_efficiency_metrics' in results
            assert len(results['fluxes']) == len(network.reactions)
        
    except ImportError:
        pytest.skip("Missing dependencies for flux analysis")

def test_docking_basic_functionality():
    """Test basic docking functionality"""
    try:
        import dock_example
        
        docker = dock_example.MarsEnzymeDocking()
        
        # Test ligand generation
        ligands = docker.generate_mars_ligands()
        assert len(ligands) > 0
        
        # Test binding affinity calculation
        if ligands:
            affinity = docker.calculate_mars_binding_affinity(ligands[0])
            assert isinstance(affinity, float)
        
    except ImportError:
        pytest.skip("Missing dependencies for docking")

def test_sequence_validation():
    """Test sequence validation and error handling"""
    import codon_optimization
    
    optimizer = codon_optimization.MarsCodonOptimizer()
    
    # Test invalid sequence length
    with pytest.raises(ValueError):
        optimizer.optimize_for_mars_conditions("ATGC")  # Not divisible by 3
    
    # Test empty sequence
    with pytest.raises(ValueError):
        optimizer.optimize_for_mars_conditions("")
    
    # Test valid sequence
    valid_seq = "ATGAAATAG"  # Met-Lys-Stop
    result = optimizer.optimize_for_mars_conditions(valid_seq)
    assert len(result) == len(valid_seq)

@pytest.mark.parametrize("sequence,expected_length", [
    ("ATGAAATAG", 9),
    ("ATGCGATCGTAGC", 12),
    ("ATGAAATTTGGGTCGGATCCGAAATAG", 27)
])
def test_codon_optimization_lengths(sequence, expected_length):
    """Test codon optimization preserves sequence length"""
    import codon_optimization
    
    optimizer = codon_optimization.MarsCodonOptimizer()
    optimized = optimizer.optimize_for_mars_conditions(sequence)
    assert len(optimized) == expected_length
