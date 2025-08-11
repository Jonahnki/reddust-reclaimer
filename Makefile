# RedDust Reclaimer Makefile
# Convenience commands for development and testing

.PHONY: help install test lint format clean docker-build docker-run docs

help:  ## Show this help message
	@echo "🧬 RedDust Reclaimer Development Commands"
	@echo "=========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies and package
	pip install -r requirements.txt
	pip install -e .

test:  ## Run test suite
	pytest tests/ -v --cov=scripts --cov-report=term-missing

lint:  ## Run linting checks
	flake8 scripts/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	mypy scripts/ --ignore-missing-imports

format:  ## Format code with black
	black scripts/ tests/

format-check:  ## Check code formatting
	black --check scripts/ tests/

examples:  ## Run example scripts
	@echo "🧬 Testing Mars docking example..."
	python scripts/dock_example.py --help
	@echo "🧬 Testing codon optimization..."
	python scripts/codon_optimization.py --sequence ATGAAATTTGGGTAG --analyze
	@echo "🧬 Testing metabolic flux analysis..."
	python scripts/metabolic_flux.py --help

clean:  ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -f docking_results_*.txt
	rm -f *.png

docker-build:  ## Build Docker image
	docker build -t reddust-reclaimer:latest .

docker-run:  ## Run Docker container
	docker run -it --rm -v $(PWD):/workspace reddust-reclaimer:latest

docker-jupyter:  ## Run Jupyter in Docker
	docker-compose up jupyter

docs:  ## Build documentation
	cd docs && make html

notebook:  ## Run Jupyter notebook server
	jupyter lab notebooks/

all-checks: format-check lint test examples  ## Run all quality checks

ci-local: all-checks  ## Simulate CI pipeline locally
	@echo "✅ All CI checks passed locally!"
