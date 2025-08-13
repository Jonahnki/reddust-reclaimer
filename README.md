# 🧬 RedDust Reclaimer

![CI (Micromamba Matrix)](https://github.com/Jonahnki/reddust-reclaimer/actions/workflows/ci-conda.yml/badge.svg)
![Docs](https://img.shields.io/badge/docs-Sphinx-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.8--3.11-blue)

A computational biology toolkit for Mars terraforming research that combines molecular docking, metabolic modeling, and genetic engineering workflows.

## 🚀 Quickstart (5 minutes)

### Prerequisites
- Python 3.8–3.11
- Git
- Optional: Micromamba/Conda for reproducible envs
- Optional: Docker

### Quick Setup (Micromamba/Conda recommended)
```bash
git clone https://github.com/Jonahnki/reddust-reclaimer.git
cd reddust-reclaimer

# Create environment with micromamba (recommended)
# If you have conda instead, replace `micromamba` with `conda`
micromamba create -y -f environment.yml
micromamba activate reddust-reclaimer

# Dev extras
python -m pip install -e . pytest-xdist

# Run example workflows
python scripts/dock_example.py --help
python scripts/codon_optimization.py --sequence ATGCGATCGTAGC --analyze
python scripts/metabolic_flux.py --model models/mars_microbe_core.xml
```

### Alternative: Pip-only setup
```bash
pip install -r requirements.txt
python -m pip install -e .
```

### Docker Option
```bash
# If an image is published
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

### Pre-commit hooks (recommended)
```bash
pre-commit install          # enable git hooks
pre-commit run --all-files  # run hooks on entire repo
```

## 🛠️ Makefile shortcuts

Common tasks are available via the `Makefile`:
```bash
make install       # install deps and package
make test          # run tests with coverage
make format        # apply black
make format-check  # verify formatting
make lint          # flake8 + mypy
make examples      # run example scripts
make docs          # build Sphinx docs
```

## 🧵 Nextflow pipeline

We provide a minimal Nextflow workflow at `workflows/nextflow/main.nf`.

Run with the conda environment:
```bash
nextflow run workflows/nextflow/main.nf \
  --sequence ATGAAATTTGGGTAG \
  --model models/mars_microbe_core.xml \
  --outdir results/nextflow \
  --threads 2
```

Parameters:
- `--sequence` Codon optimization input sequence (default: example sequence)
- `--model` SBML model for FBA (default: `models/mars_microbe_core.xml`)
- `--outdir` Output directory (default: `results/nextflow`)
- `--threads` CPU threads per process (default: 2)
- `--dry_run` Use `--dry_run true` to exercise the pipeline without heavy computation

Note: A default Nextflow config at `workflows/nextflow/nextflow.config` enables conda and sets sensible defaults. You can also run with Docker using profile `-profile docker` if you have a built/published image.

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

---

## 🧰 CI & Artifacts

- CI is powered by GitHub Actions using Micromamba with a Python 3.8–3.11 matrix (`.github/workflows/ci-conda.yml`).
- Linting (flake8/black), typing (mypy), tests (pytest + coverage), notebook execution, security scans (pip-audit, bandit), and Docker image build are included.
- Artifacts uploaded from CI include:
  - `coverage.xml`, executed notebooks under `build/notebooks-executed/`
  - `data/processed/**`, `docs/_build/**`
  - `results/**` and `logs/**` (collected from the runner even if `.gitignore` excludes them)

Note: An older pip-based CI file (`.github/workflows/ci.yml`) is retained for manual runs only and does not trigger automatically.

---

## 🧯 Troubleshooting

- **RDKit installation issues**
  - If `rdkit-pypi` fails on Linux, ensure a recent pip and manylinux support: `python -m pip install --upgrade pip setuptools wheel`.
  - If using conda, prefer conda-forge build: `conda install -c conda-forge rdkit` and remove `rdkit-pypi` from pip deps.

- **libSBML errors (ImportError: libsbml not found)**
  - Use conda package: `conda install -c conda-forge python-libsbml` (already included in `environment.yml`).
  - For pip-only, ensure system libs are present; if issues persist, consider switching to the conda environment.

- **M1/M2/M3 macOS or ARM runners**
  - Prefer conda-forge packages; some pip wheels may be unavailable for ARM. Use Docker as a fallback.

- **Pre-commit fails on notebooks**
  - nbQA runs Black/Flake8 over notebooks. To skip a file: add to `.pre-commit-config.yaml` exclude or commit with `-n` to bypass hooks temporarily.
