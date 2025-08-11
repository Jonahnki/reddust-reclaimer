# 🧬 RedDust Reclaimer

A computational biology toolkit for Mars terraforming research that combines molecular docking, metabolic modeling, and genetic engineering workflows.

## 🚀 Quickstart (5 minutes)

### Prerequisites
- Python 3.9+ 
- Git

### Quick Setup
```bash
git clone https://github.com/Jonahnki/reddust-reclaimer.git
cd reddust-reclaimer
pip install -r requirements.txt

# Run example workflows
python scripts/dock_example.py
python scripts/codon_optimization.py --sequence ATGCGATCGTAGC
python scripts/metabolic_flux.py --model models/mars_microbe_core.xml
```

### Docker Option (Recommended)
```bash
docker pull jonahnki/reddust-reclaimer:latest
docker run -it --rm -v $(pwd):/workspace reddust-reclaimer python scripts/dock_example.py
```

### Verify Installation
```bash
python -c "import scripts.dock_example; print('✅ Installation successful')"
```

## 🔬 Features

### Molecular Docking
- **Mars-adapted enzyme docking**: Simulate protein-ligand interactions under Mars environmental conditions
- **Atmospheric processing**: Focus on CO2 fixation and extremophile metabolic pathways
- **Environmental factors**: Account for low temperature, high radiation, and low pressure

### Codon Optimization
- **Extremophile adaptation**: Optimize genetic sequences for Mars environmental conditions
- **Temperature stability**: Codon usage optimized for -80°C to 20°C range
- **Radiation resistance**: Enhanced genetic stability under high radiation

### Metabolic Modeling
- **Flux balance analysis**: Analyze metabolic fluxes for Mars resource utilization
- **CO2 fixation pathways**: Model carbon dioxide processing from Mars atmosphere
- **Resource efficiency**: Optimize water and energy usage under scarcity

## 📊 Example Usage

### Molecular Docking
```python
from scripts.dock_example import MarsEnzymeDocking

# Initialize docking for Mars conditions
docker = MarsEnzymeDocking(temperature=233.15, pressure=0.006)

# Run docking simulation
results = docker.dock_mars_enzyme_substrate('carbonic_anhydrase')
docker.print_docking_summary(results)
```

### Codon Optimization
```python
from scripts.codon_optimization import MarsCodonOptimizer

# Optimize sequence for Mars conditions
optimizer = MarsCodonOptimizer()
optimized_seq = optimizer.optimize_for_mars_conditions("ATGAAATTTGGGTAG")
print(f"Optimized: {optimized_seq}")
```

### Metabolic Flux Analysis
```python
from scripts.metabolic_flux import MarsMetabolicNetwork

# Analyze metabolic fluxes
network = MarsMetabolicNetwork()
results = network.mars_metabolic_flux_analysis('biomass_synthesis')
network.print_flux_analysis(results)
```

## 🗺️ Roadmap

### Phase 1: Foundation (v0.1-0.3) ✅
- [x] Core docking and metabolic modeling workflows
- [x] Example scripts and interactive demos  
- [x] CI/CD pipeline and documentation

### Phase 2: Advanced Features (v0.4-0.6) 🚧
- [ ] Machine learning models for protein design
- [ ] Multi-scale simulation integration
- [ ] Web-based analysis dashboard
- [ ] API for external tool integration

### Phase 3: Ecosystem (v0.7-1.0) 📋
- [ ] Plugin architecture for custom workflows
- [ ] Cloud deployment templates (AWS/GCP)
- [ ] Integration with major bioinformatics databases
- [ ] Educational curriculum and tutorials

[View detailed roadmap →](https://github.com/Jonahnki/reddust-reclaimer/projects/1)

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=scripts
```

Test individual components:
```bash
# Test docking workflow
python scripts/dock_example.py --verbose

# Test codon optimization
python scripts/codon_optimization.py --sequence ATGAAATTTGGGTAG --analyze

# Test metabolic flux analysis
python scripts/metabolic_flux.py --plot
```

## 📁 Repository Structure

```
reddust-reclaimer/
├── scripts/                    # Example workflows and tools
│   ├── dock_example.py        # Protein-ligand docking demo
│   ├── codon_optimization.py  # Genetic sequence optimization
│   └── metabolic_flux.py      # Flux balance analysis
├── models/                     # Biological models and data
│   ├── mars_microbe_core.xml  # SBML metabolic model
│   ├── protein_structures/    # Sample PDB files
│   └── compound_library.sdf   # Mars-relevant compounds
├── data/                       # Sample datasets
├── notebooks/                  # Interactive Jupyter demos
├── tests/                      # Test suite
└── .github/                    # CI/CD and community templates
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

### Development Setup
```bash
git clone https://github.com/Jonahnki/reddust-reclaimer.git
cd reddust-reclaimer
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/

# Format code
black scripts/ tests/
flake8 scripts/ tests/
```

## 📖 Documentation

- [API Documentation](https://jonahnki.github.io/reddust-reclaimer/)
- [Tutorial Notebooks](notebooks/)
- [Model Documentation](models/README.md)

## 📄 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{reddust_reclaimer,
  title={RedDust Reclaimer: Computational Biology Toolkit for Mars Terraforming},
  author={Your Name},
  year={2024},
  url={https://github.com/Jonahnki/reddust-reclaimer},
  version={0.1.0}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- Mars atmospheric data from NASA JPL
- Extremophile genomic data from NCBI
- Metabolic modeling frameworks: COBRApy, SBML
- Molecular docking tools: RDKit, AutoDock Vina

## 🔗 Related Projects

- [Mars Sample Return Mission](https://mars.nasa.gov/msr/)
- [Extremophile Database](http://www.extremophiles.org/)
- [Astrobiology Roadmap](https://astrobiology.nasa.gov/)

---

**🚀 Ready to contribute to Mars terraforming research? Get started with the quickstart guide above!**
