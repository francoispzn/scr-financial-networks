# SCR-Financial-Networks

A Python framework for analyzing financial networks using Spectral Coarse-Graining (SCG) and Agent-Based Modeling (ABM).

## Overview

This project implements a hybrid approach combining Spectral Coarse-Graining (SCG) with Agent-Based Modeling (ABM) to analyze interbank contagion dynamics among European banks. The framework enables the study of systemic risk, contagion pathways, and financial stability by preserving both macro-level network effects and micro-level bank behaviors.

The methodology is based on the theoretical framework described in "Spectral Coarse-Graining to Financial Networks" and addresses the research question posed in the "Contingency BW 9-08 x10-15" preliminary report:

*"How can we effectively integrate spectral coarse-graining (SCG) with an agent-based model (ABM) enhanced by machine learning and large language models (LLMs) to dynamically simulate interbank contagion during crisis periods (e.g., the Black Week of 2008), while ensuring that both macro-level network effects and micro-level bank behaviors are accurately captured and calibrated?"*

## Features

- **Data Collection**: Interfaces with public financial data sources (EBA, ECB, FRED, etc.)
- **Network Construction**: Tools for building and analyzing financial networks
- **Spectral Analysis**: Implementation of spectral decomposition and coarse-graining techniques
- **Agent-Based Modeling**: Simulation of bank behaviors during stress scenarios
- **Variational Autoencoder**: Feature extraction and dimensionality reduction for financial data
- **Machine Learning Integration**: LSTM models for market sentiment prediction
- **Validation Tools**: Methods for testing and validating model performance

## Installation

### Using Conda

```bash
# Clone the repository
git clone https://github.com/francoispzn/scr-financial-networks.git
cd scr-financial-networks

# Create and activate the conda environment
conda env create -f environment.yml
conda activate scr-financial

# Install the package in development mode
pip install -e .
```

### Using pip

```bash
pip install scr-financial-networks
```

## Quick Start

```python
from scr_financial.data.collectors.ecb_collector import ECBCollector
from scr_financial.data.collectors.eba_collector import EBACollector
from scr_financial.abm.simulation import BankingSystemSimulation

# Generate sample data
ecb = ECBCollector()
eba = EBACollector()
ciss_df = ecb.collect_ciss_data("2020-01-01", "2021-12-31")
bank_df = eba.collect_transparency_data("2020-01-01", "2021-12-31")
print(f"CISS data: {len(ciss_df)} rows")
print(f"Bank data: {len(bank_df)} rows")
```

## Development Setup

```
git clone https://github.com/francoispzn/scr-financial-networks
cd scr-financial-networks
pip install -e ".[dev]"
pytest tests/
```

## Project Structure

```
scr-financial-networks/
├── LICENSE
├── README.md
├── pyproject.toml
├── setup.py
├── environment.yml
├── .gitignore
├── docs/
│   ├── conf.py
│   ├── index.rst
│   ├── api/
│   │   ├── data_preprocessing.rst
│   │   ├── network_construction.rst
│   │   ├── spectral_analysis.rst
│   │   ├── coarse_graining.rst
│   │   └── abm.rst
│   └── examples/
│       ├── black_week_simulation.rst
│       ├── COVID19_simulation.rst
│       └── network_visualization.rst
├── scr_financial/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessor.py
│   │   └── utils.py
│   ├── network/
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── spectral.py
│   │   └── coarse_graining.py
│   ├── abm/
│   │   ├── __init__.py
│   │   ├── bank_agent.py
│   │   ├── simulation.py
│   │   └── decision_models.py
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py
│       └── validation.py
├── tests/
│   ├── __init__.py
│   ├── test_data_preprocessing.py
│   ├── test_network_construction.py
│   ├── test_spectral_analysis.py
│   ├── test_coarse_graining.py
│   └── test_abm.py
└── examples/
    ├── black_week_simulation.py
    └── network_visualization.py
```

## Key Components

### Spectral Coarse-Graining (SCG)

The SCG module implements techniques for simplifying complex financial networks while preserving their essential structural and dynamic properties. Key features include:

- Spectral decomposition of the graph Laplacian
- Identification of characteristic scales via spectral gaps
- Coarse-graining based on dominant eigenmodes
- Rescaling to preserve diffusion dynamics

### Agent-Based Model (ABM)

The ABM module simulates the behavior of financial institutions during stress scenarios. Features include:

- Bank agent implementation with decision-making capabilities
- Interbank lending simulation
- Stress propagation mechanics
- Liquidity and solvency assessment

### Variational Autoencoder (VAE)

The VAE module provides dimensionality reduction and feature extraction for financial data:

- Encoder-decoder architecture for latent representation
- Training utilities for financial time series
- Anomaly detection capabilities
- Integration with the ABM for enhanced agent decision-making

## Documentation

For detailed documentation, please check the source folder with the HTML documentation files.

## Examples

Example scripts and notebooks can be found in the `examples/` directory:

- `black_week_simulation.py`: Simulates the 2008 financial crisis "Black Week"
- `network_visualization.py`: Visualizes financial networks and their coarse-grained representations
- `vae_financial_features.py`: Demonstrates feature extraction from financial data using VAE

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Acknowledgments

- This project builds on the theoretical framework described in "Spectral Coarse-Graining to Financial Networks" by Francois Petizon
- The implementation is inspired by the "Contingency BW 9-08 x10-15" preliminary report
- We acknowledge the use of publicly available financial indicators from EBA, ECB, and other sources

## Contact

For any questions or feedback, please contact me:
- Francois Petizon - francois.petizon@kcl.ac.uk
- Department of Informatics, King's College London
