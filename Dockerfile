# Multi-stage Dockerfile for reddust-reclaimer
FROM continuumio/miniconda3:latest as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Activate environment
SHELL ["conda", "run", "-n", "reddust-reclaimer", "/bin/bash", "-c"]

# Development stage
FROM base as development

# Copy source code
COPY . .

# Install package in development mode
RUN conda run -n reddust-reclaimer pip install -e .

# Set environment
ENV CONDA_DEFAULT_ENV=reddust-reclaimer
ENV PATH=/opt/conda/envs/reddust-reclaimer/bin:$PATH

# Expose Jupyter port
EXPOSE 8888

# Default command for development
CMD ["conda", "run", "-n", "reddust-reclaimer", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base as production

# Copy only necessary files
COPY scripts/ scripts/
COPY models/ models/
COPY data/ data/
COPY requirements.txt .
COPY setup.py .

# Install package
RUN conda run -n reddust-reclaimer pip install .

# Set environment
ENV CONDA_DEFAULT_ENV=reddust-reclaimer
ENV PATH=/opt/conda/envs/reddust-reclaimer/bin:$PATH

# Create non-root user
RUN useradd -m -u 1000 marsuser
USER marsuser

# Default command for production
CMD ["conda", "run", "-n", "reddust-reclaimer", "python", "scripts/dock_example.py", "--help"]
