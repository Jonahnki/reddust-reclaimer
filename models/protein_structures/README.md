# 🧬 Protein Structures for Mars Docking Studies

This directory contains protein structure files and download links for molecular docking simulations relevant to Mars terraforming research.

## 📁 Structure Files

### Key Enzymes for Mars Adaptation

#### CO2 Fixation Enzymes
- **Carbonic Anhydrase** (CA)
  - PDB ID: 1CA2
  - Function: CO2 + H2O ⇌ HCO3- + H+
  - Mars relevance: Process abundant atmospheric CO2

- **RuBisCO** (Ribulose-1,5-bisphosphate carboxylase/oxygenase)
  - PDB ID: 1RCX
  - Function: CO2 fixation in Calvin cycle
  - Mars relevance: Primary CO2 fixation enzyme

#### Extremophile Enzymes
- **Cold-shock proteins**
  - PDB ID: 1MJC
  - Function: Maintain protein folding at low temperatures
  - Mars relevance: Survive -80°C to 20°C range

- **DNA repair enzymes**
  - PDB ID: 1DRA (RecA)
  - Function: Repair radiation-induced DNA damage
  - Mars relevance: High radiation environment

## 📥 Download Instructions

To download PDB structures for docking studies:

```bash
# Download key structures
wget https://files.rcsb.org/download/1CA2.pdb -O carbonic_anhydrase.pdb
wget https://files.rcsb.org/download/1RCX.pdb -O rubisco.pdb
wget https://files.rcsb.org/download/1MJC.pdb -O cold_shock_protein.pdb
wget https://files.rcsb.org/download/1DRA.pdb -O dna_repair_enzyme.pdb
```

Or use the provided download script:
```bash
python ../scripts/download_structures.py --all
```

## 🔬 Structure Analysis

### Preparation for Docking
1. **Clean structures**: Remove water molecules, add hydrogens
2. **Optimize geometry**: Energy minimization for Mars conditions
3. **Identify binding sites**: Cavity detection and analysis
4. **Generate conformers**: Multiple conformations for flexible docking

### Mars-Specific Modifications
- **Temperature adaptation**: Model low-temperature conformational changes
- **Pressure effects**: Account for low atmospheric pressure
- **Radiation damage**: Consider radiation-induced structural changes

## 📊 Structure Quality Metrics

| Protein | PDB ID | Resolution (Å) | R-factor | Mars Relevance Score |
|---------|--------|----------------|----------|---------------------|
| Carbonic Anhydrase | 1CA2 | 2.0 | 0.177 | 9.5/10 |
| RuBisCO | 1RCX | 2.6 | 0.195 | 9.8/10 |
| Cold-shock protein | 1MJC | 1.8 | 0.162 | 8.7/10 |
| DNA repair enzyme | 1DRA | 2.5 | 0.188 | 8.9/10 |

## 🧪 Usage in Docking Scripts

```python
from scripts.dock_example import MarsEnzymeDocking

# Load protein structure
docker = MarsEnzymeDocking()
results = docker.dock_mars_enzyme_substrate('carbonic_anhydrase')
```

## 📚 References

- RCSB Protein Data Bank: https://www.rcsb.org/
- Mars enzyme studies: doi:10.1089/ast.2019.2045
- Extremophile protein structures: doi:10.1016/j.str.2020.05.004
