# 🧬 RedDust Reclaimer

![CI (Micromamba Matrix)](https://github.com/Jonahnki/reddust-reclaimer/actions/workflows/ci-conda.yml/badge.svg)  
![Docs](https://img.shields.io/badge/docs-Sphinx-blue)  
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)  
![Python](https://img.shields.io/badge/Python-3.8--3.11-blue)  

A computational biology toolkit for Mars terraforming research that combines molecular docking, metabolic modeling, and genetic engineering workflows.

---

## 🚀 Quickstart (5 minutes)

### Prerequisites
- Python 3.8–3.11
- Git
- Optional: Micromamba/Conda for reproducible environments
- Optional: Docker

### Quick Setup (Micromamba/Conda recommended)
```bash
git clone https://github.com/Jonahnki/reddust-reclaimer.git
cd reddust-reclaimer

# Create environment with micromamba (recommended)
# If you use conda, replace `micromamba` with `conda`
micromamba create -y -f environment.yml
micromamba activate reddust-reclaimer

# Install dev extras
python -m pip install -e . pytest-xdist

# Run example workflows
python scripts/dock_example.py --help
python scripts/codon_optimization.py --sequence ATGCGATCGTAGC --analyze
python scripts/metabolic_flux.py --model models/mars_microbe_core.xml
````

### Alternative: Pip-only setup

```bash
pip install -r requirements.txt
python -m pip install -e .
```

### Docker Option

```bash
# Pull published image (if available)
docker pull jonahnki/reddust-reclaimer:latest || true

# Or build locally
docker build -t reddust-reclaimer:latest .

# Run a workflow inside the container
docker run -it --rm -v $(pwd):/workspace -w /workspace reddust-reclaimer:latest \
  python scripts/dock_example.py --help
```

### Verify Installation

```bash
python -c "import scripts.dock_example; print('✅ Installation successful')"
```

---

## 🔬 Features

### Molecular Docking

* Mars-adapted enzyme docking simulating protein-ligand interactions under Mars environmental conditions
* Atmospheric processing with focus on CO2 fixation and extremophile metabolic pathways
* Environmental factors accounted for: low temperature, high radiation, low pressure

### Codon Optimization

* Genetic sequence optimization tailored for Mars extremophile adaptation
* Codon usage stability for temperature range from -80°C to 20°C
* Enhanced radiation resistance for genetic stability

### Metabolic Modeling

* Flux balance analysis (FBA) targeting Mars resource utilization
* Modeling of CO2 fixation pathways using Mars atmosphere data
* Optimization of water and energy use under resource scarcity

---

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

---

## 🗺️ Roadmap

### Phase 1: Foundation (v0.1-0.3) ✅

* [x] Core docking and metabolic modeling workflows
* [x] Example scripts and interactive demos
* [x] CI/CD pipeline and documentation

### Phase 2: Advanced Features (v0.4-0.6) 🚧

* [ ] Machine learning models for protein design
* [ ] Multi-scale simulation integration
* [ ] Web-based analysis dashboard
* [ ] API for external tool integration

### Phase 3: Ecosystem (v0.7-1.0) 📋

* [ ] Plugin architecture for custom workflows
* [ ] Cloud deployment templates (AWS/GCP)
* [ ] Integration with major bioinformatics databases
* [ ] Educational curriculum and tutorials

[View detailed roadmap →](https://github.com/Jonahnki/reddust-reclaimer/projects/1)

---

## 🧪 Testing

Run the full test suite:

```bash
pytest tests/ -v --cov=scripts
```

Test individual components:

```bash
# Docking workflow
python scripts/dock_example.py --verbose

# Codon optimization
python scripts/codon_optimization.py --sequence ATGAAATTTGGGTAG --analyze

# Metabolic flux analysis
python scripts/metabolic_flux.py --plot
```

---

## 📁 Repository Structure

```
reddust-reclaimer/
├── scripts/                    # Example workflows and tools
│   ├── dock_example.py         # Protein-ligand docking demo
│   ├── codon_optimization.py   # Genetic sequence optimization
│   └── metabolic_flux.py       # Flux balance analysis
├── models/                     # Biological models and data
│   ├── mars_microbe_core.xml   # SBML metabolic model
│   ├── protein_structures/     # Sample PDB files
│   └── compound_library.sdf    # Mars-relevant compounds
├── data/                       # Sample datasets
├── notebooks/                  # Interactive Jupyter demos
├── tests/                      # Test suite
└── .github/                    # CI/CD and community templates
```

---

## 🤝 Contributing

Contributions are welcome! Please review the [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

### Development Setup

```bash
git clone https://github.com/Jonahnki/reddust-reclaimer.git
cd reddust-reclaimer
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/

# Code formatting and linting
black scripts/ tests/
flake8 scripts/ tests/
```

### Pre-commit hooks (recommended)

```bash
pre-commit install          # Enable git hooks
pre-commit run --all-files  # Run hooks on all files
```

---

## 🛠️ Makefile shortcuts

Common tasks:

```bash
make install       # Install dependencies and package
make test          # Run tests with coverage
make format        # Apply black formatting
make format-check  # Check formatting compliance
make lint          # Run flake8 + mypy
make examples      # Run example scripts
make docs          # Build Sphinx documentation
```

---

## 🧵 Nextflow pipeline

A minimal Nextflow workflow is provided at `workflows/nextflow/main.nf`.

Run with conda environment:

```bash
nextflow run workflows/nextflow/main.nf \
  --sequence ATGAAATTTGGGTAG \
  --model models/mars_microbe_core.xml \
  --outdir results/nextflow \
  --threads 2
```

Parameters:

* `--sequence` Codon optimization input sequence (default: example sequence)
* `--model` SBML metabolic model (default: `models/mars_microbe_core.xml`)
* `--outdir` Output directory (default: `results/nextflow`)
* `--threads` Number of CPU threads (default: 2)
* `--dry_run` Use `--dry_run true` to test pipeline without heavy computation

Default Nextflow config (`workflows/nextflow/nextflow.config`) enables conda and sensible defaults. Docker profile available (`-profile docker`) if image built/published.

---

## 📖 Documentation

* [API Documentation](https://jonahnki.github.io/reddust-reclaimer/)
* [Tutorial Notebooks](notebooks/)
* [Model Documentation](models/README.md)

---

## 📄 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{reddust_reclaimer,
  title={RedDust Reclaimer: Computational Biology Toolkit for Mars Terraforming},
  author={John Adedeji},
  year={2025},
  url={https://github.com/Jonahnki/reddust-reclaimer},
  version={0.1.0}
}
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments

* Mars atmospheric data from NASA JPL
* Extremophile genomic data from NCBI
* Metabolic modeling frameworks: COBRApy, SBML
* Molecular docking tools: RDKit, AutoDock Vina

---

## 🔗 Related Projects

* [Mars Sample Return Mission](https://mars.nasa.gov/msr/)
* [Extremophile Database](http://www.extremophiles.org/)
* [Astrobiology Roadmap](https://astrobiology.nasa.gov/)

---

## 🧰 CI & Artifacts

* CI powered by GitHub Actions using Micromamba and Python 3.8–3.11 matrix (`.github/workflows/ci-conda.yml`)
* Linting (flake8/black), typing (mypy), tests (pytest + coverage), notebook execution, security scans (pip-audit, bandit), Docker image builds
* Artifacts include coverage reports, executed notebooks, processed data, documentation builds, results, and logs
* Legacy pip-based CI (`.github/workflows/ci.yml`) for manual runs only

---

## 🧯 Troubleshooting

* **RDKit installation issues:**

  * On Linux, if `rdkit-pypi` fails, update pip and setuptools: `python -m pip install --upgrade pip setuptools wheel`
  * Prefer conda-forge package: `conda install -c conda-forge rdkit` and remove pip `rdkit-pypi` dependency

* **libSBML errors (ImportError: libsbml not found):**

  * Install via conda: `conda install -c conda-forge python-libsbml` (included in environment.yml)
  * For pip-only, ensure system libs installed; consider switching to conda environment if issues persist

* **macOS M1/M2/M3 or ARM runners:**

  * Use conda-forge packages; some pip wheels may be unavailable on ARM
  * Docker fallback recommended

* **Pre-commit fails on notebooks:**

  * nbQA runs Black and Flake8 over notebooks
  * To skip files, add excludes to `.pre-commit-config.yaml` or commit with `-n` to bypass temporarily

---

**🚀 Ready to contribute to Mars terraforming research? Start with the Quickstart guide above!**

```
```
