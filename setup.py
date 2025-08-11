#!/usr/bin/env python3
"""Setup script for reddust-reclaimer package"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="reddust-reclaimer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Computational biology toolkit for Mars terraforming research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jonahnki/reddust-reclaimer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "flake8>=3.9.0",
            "black>=21.6.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "nbsphinx>=0.8.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
            "ipywidgets>=7.6.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "mars-dock=scripts.dock_example:main",
            "mars-codon=scripts.codon_optimization:main",
            "mars-flux=scripts.metabolic_flux:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.xml", "*.sdf", "*.json", "*.md"],
    },
    project_urls={
        "Bug Reports": "https://github.com/Jonahnki/reddust-reclaimer/issues",
        "Source": "https://github.com/Jonahnki/reddust-reclaimer",
        "Documentation": "https://jonahnki.github.io/reddust-reclaimer/",
    },
)
