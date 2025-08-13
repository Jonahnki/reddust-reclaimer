![RedDust Reclaimer Banner](assets/banner_1.webp)


# 🧬 RedDust Reclaimer

[![Micromamba CI Status](https://img.shields.io/github/actions/workflow/status/Jonahnki/reddust-reclaimer/ci-conda.yml?branch=main&label=Micromamba%20CI&style=flat-square&color=2ea44f)](https://github.com/Jonahnki/reddust-reclaimer/actions/workflows/ci-conda.yml)
![Docs](https://img.shields.io/badge/docs-Sphinx-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.8--3.11-blue?style=flat-square)

A computational biology toolkit for Mars terraforming research, combining molecular docking, metabolic modeling, and genetic engineering workflows tailored for Mars environmental conditions.

---

## 🚀 Quickstart (5 minutes)

### Prerequisites
- Python 3.8–3.11
- Git
- Optional: Micromamba/Conda (recommended for reproducible environments)
- Optional: Docker

### Quick Setup (Micromamba/Conda recommended)
```bash
git clone https://github.com/Jonahnki/reddust-reclaimer.git
cd reddust-reclaimer

# Create environment (replace micromamba with conda if needed)
micromamba create -y -f environment.yml
micromamba activate reddust-reclaimer

# Install package in editable mode + dev extras
python -m pip install -e . pytest-xdist

# Run example workflows
python -m reddust_reclaimer.dock_example --help
python -m reddust_reclaimer.codon_optimization --sequence ATGCGATCGTAGC --analyze
python -m reddust_reclaimer.metabolic_flux --model models/mars_microbe_core.xml
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
docker run -it --rm -v "$(pwd)":/workspace -w /workspace reddust-reclaimer:latest \
  python -m reddust_reclaimer.dock_example --help
```

### Verify Installation

```bash
python -c "import reddust_reclaimer.dock_example as d; print('✅ Installation successful')"
```

---

## 🔬 Features

### Molecular Docking

* Mars-adapted enzyme docking simulating protein–ligand interactions under Mars conditions
* Atmospheric processing with emphasis on CO2 fixation and extremophile pathways
* Accounts for low temperature, high radiation, and low pressure

### Codon Optimization

* Sequence optimization tailored for Mars extremophile adaptation
* Codon usage stability across −80°C to 20°C
* Emphasis on radiation-resilient encoding choices

### Metabolic Modeling

* Flux balance analysis (FBA) for efficient Mars resource utilization
* Modeling of CO2 fixation pathways informed by Mars atmospheric data
* Optimization for water and energy use under scarcity

---

## 📊 Example Usage

### Molecular Docking

```python
from reddust_reclaimer.dock_example import MarsEnzymeDocking

# Initialize for Mars conditions (T in Kelvin, P in bar)
docker = MarsEnzymeDocking(temperature=233.15, pressure=0.006)

# Run docking simulation
results = docker.dock_mars_enzyme_substrate("carbonic_anhydrase")
docker.print_docking_summary(results)
```

### Codon Optimization

```python
from reddust_reclaimer.codon_optimization import MarsCodonOptimizer

optimizer = MarsCodonOptimizer()
optimized_seq = optimizer.optimize_for_mars_conditions("ATGAAATTTGGGTAG")
print(f"Optimized: {optimized_seq}")
```

### Metabolic Flux Analysis

```python
from reddust_reclaimer.metabolic_flux import MarsMetabolicNetwork

network = MarsMetabolicNetwork()
results = network.mars_metabolic_flux_analysis("biomass_synthesis")
network.print_flux_analysis(results)
```

---

## 🗺️ Roadmap

### Phase 1: Foundation (v0.1–0.3) ✅

* Core docking and metabolic modeling workflows
* Example scripts and interactive demos
* CI/CD pipeline and documentation

### Phase 2: Advanced Features (v0.4–0.6) 🚧

* Machine learning models for protein design
* Multi-scale simulation integration
* Web-based analysis dashboard
* API for external tool integration

### Phase 3: Ecosystem (v0.7–1.0) 📋

* Plugin architecture for custom workflows
* Cloud deployment templates (AWS/GCP)
* Integration with major bioinformatics databases
* Educational curriculum and tutorials

[View detailed roadmap →](https://github.com/Jonahnki/reddust-reclaimer/projects/1)

---

## 🧪 Testing

Run the full test suite:

```bash
pytest tests/ -v --cov=reddust_reclaimer
```

Test individual components:

```bash
# Docking workflow
python -m reddust_reclaimer.dock_example --verbose

# Codon optimization
python -m reddust_reclaimer.codon_optimization --sequence ATGAAATTTGGGTAG --analyze

# Metabolic flux analysis
python -m reddust_reclaimer.metabolic_flux --plot
```

---

## 📁 Repository Structure

```
reddust-reclaimer/
├── reddust_reclaimer/          # Package: workflows and modules
│   ├── dock_example.py         # Protein–ligand docking demo
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
black reddust_reclaimer/ tests/
flake8 reddust_reclaimer/ tests/
```

### Pre-commit hooks (recommended)

```bash
pre-commit install          # Enable git hooks
pre-commit run --all-files  # Run hooks on all files
```

---

## 🛠️ Makefile Shortcuts

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

## 🧵 Nextflow Pipeline

A minimal Nextflow workflow is provided at `workflows/nextflow/main.nf`. Run with conda environment:

```bash
nextflow run workflows/nextflow/main.nf \
  --sequence ATGAAATTTGGGTAG \
  --model models/mars_microbe_core.xml \
  --outdir results/nextflow \
  --threads 2
```

Parameters:

* `--sequence` Codon optimization input sequence (default: example sequence)
* `--model` SBML metabolic model (default: models/mars\_microbe\_core.xml)
* `--outdir` Output directory (default: results/nextflow)
* `--threads` Number of CPU threads (default: 2)
* `--dry_run` Use `--dry_run true` to test pipeline without heavy computation

The default Nextflow config (`workflows/nextflow/nextflow.config`) enables conda and sensible defaults. A Docker profile is available (`-profile docker`) if an image is built/published.

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
  title   = {RedDust Reclaimer: Computational Biology Toolkit for Mars Terraforming},
  author  = {John Adedeji},
  year    = {2025},
  url     = {https://github.com/Jonahnki/reddust-reclaimer},
  version = {0.1.0}
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

* CI powered by GitHub Actions using Micromamba and Python 3.8–3.11 matrix (`.github/workflows/ci-conda.yml`) — **all tests passing ✅**
* Linting (flake8/black), typing (mypy), tests (pytest + coverage), notebook execution, security scans (pip-audit, bandit), Docker image builds
* Artifacts include coverage reports, executed notebooks, processed data, documentation builds, results, and logs
* Legacy pip-based CI (`.github/workflows/ci.yml`) is retained for manual runs only

---

## 🧯 Troubleshooting

* **RDKit installation issues:**

  * On Linux, if `rdkit-pypi` fails, update pip/setuptools/wheel:
    `python -m pip install --upgrade pip setuptools wheel`
  * Prefer conda-forge package:
    `conda install -c conda-forge rdkit` and remove pip `rdkit-pypi` dependency

* **libSBML errors (ImportError: libsbml not found):**

  * Install via conda: `conda install -c conda-forge python-libsbml` (included in `environment.yml`)
  * For pip-only, ensure system libs installed; consider switching to a conda environment

* **macOS M1/M2/M3 or ARM runners:**

  * Use conda-forge packages; some pip wheels may be unavailable on ARM
  * Docker fallback recommended

* **Pre-commit fails on notebooks:**

  * `nbQA` runs Black and Flake8 over notebooks
  * To skip files, add excludes to `.pre-commit-config.yaml` or commit with `-n` to bypass temporarily
