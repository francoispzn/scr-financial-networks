# SCR-Financial-Networks

A Python framework for analyzing systemic risk in financial networks using **Spectral Coarse-Graining (SCG)**, **Agent-Based Modeling (ABM)**, and **Graph Neural Networks (GNN)**.

[![Documentation](https://img.shields.io/badge/docs-gh--pages-blue)](https://francoispzn.github.io/scr-financial-networks/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

## Overview

This project implements a hybrid approach combining Spectral Coarse-Graining with Agent-Based Modeling and temporal Graph Neural Networks to analyze interbank contagion dynamics among European banks. The framework proves that SCG-based spectral risk indicators outperform traditional Basel/SREP stress metrics for early-warning detection of systemic crises.

The methodology is based on the theoretical framework described in *"Spectral Coarse-Graining to Financial Networks"* and addresses the research question posed in the *"Contingency BW 9-08 x10-15"* preliminary report.

## Key Features

### Spectral Coarse-Graining
- Spectral decomposition of the graph Laplacian with eigenvalue analysis
- Identification of characteristic scales via spectral gaps
- Coarse-graining that preserves diffusion dynamics
- Reconstruction accuracy measurement (R² and RMSE over time)
- SCG risk score: `risk = 1 - λ₂/ρ` as a spectral early-warning indicator

### Temporal Graph Neural Network
- **Architecture**: multi-layer GCNConv → BatchNorm → global_mean_pool → LSTM → FC
- **Targets**: predicts algebraic connectivity (λ₂), spectral gap, and spectral radius
- **Data**: trained on real daily market data from 10 European banks via yfinance
- **Node features**: volatility, mean return, log price, beta proxy, momentum
- **Edge weights**: rolling Pearson correlation of daily returns (threshold > 0.3)
- Batched graph encoding via PyTorch Geometric `Batch.from_data_list()` for 10-50x speedup

### Agent-Based Model
- Individual bank agents with balance sheet dynamics (CET1, LCR, leverage)
- Interbank lending simulation with contagion mechanics
- Configurable stress scenarios (liquidity crunch, credit shock, sovereign crisis)
- Adjacency matrix export for downstream spectral analysis

### Risk Metrics
- **Delta-CoVaR**: bank-level contribution to systemic risk
- **Marginal Expected Shortfall (MES)**: expected loss during market stress
- **SCG-vs-Basel comparison**: spectral risk scores benchmarked against Basel stress counts
- Historical backtesting against 2008 GFC, 2011 European debt crisis, 2020 COVID

### Real-Time Data Pipeline
- Daily stock prices for 10 European banks via yfinance (DBK.DE, BNP.PA, SAN.MC, UCG.MI, INGA.AS, NDA-FI.HE, UBSG.SW, BARC.L, HSBA.L, ACA.PA)
- ECB Statistical Data Warehouse integration (CISS, sovereign spreads)
- Rolling correlation → adjacency matrices → graph snapshots → spectral targets
- ~1000+ daily snapshots from 3 years of market data

### Interactive Dashboard
- **5-page Dash application** with dark-themed UI
- **Network page**: interactive interbank exposure graph with health indicators
- **Simulate page**: run ABM stress scenarios in real-time
- **Spectral page**: eigenvalue spectrum, diffusion distances, coarse-graining visualization
- **Evolve page**: train GNN from the UI with full hyperparameter controls, live progress banner (epoch, loss, RAM, ETA), and result charts
- **AI/Data page**: Cerebras LLM-powered risk analysis, API data fetching, GNN dataset export

## Installation

```bash
git clone https://github.com/francoispzn/scr-financial-networks.git
cd scr-financial-networks

# Using Conda (recommended)
conda env create -f environment.yml
conda activate systemic_risk
pip install -e .

# Or using pip
pip install -e ".[dev]"
```

### Requirements

Core: `numpy`, `scipy`, `pandas`, `networkx`, `torch`, `torch-geometric` (PyG 2.6+), `dash`, `plotly`, `yfinance`, `requests`

## Quick Start

### Launch the Dashboard

```bash
python dashboard/app.py
# Open http://localhost:8050
```

### Train a GNN from the UI

1. Navigate to **Evolve** → select **Real Market (yfinance)** → click **Load Data**
2. Adjust hyperparameters (hidden dim, GCN/LSTM layers, epochs, learning rate, dropout)
3. Click **Train GNN** — watch live progress with loss, RAM usage, and ETA
4. Review training loss curve, test scatter plot, and spectral evolution predictions

### Programmatic Usage

```python
from scr_financial.abm.simulation import BankingSystemSimulation
from scr_financial.network.spectral import compute_laplacian, eigendecomposition
from scr_financial.network.coarse_graining import SpectralCoarseGraining
from scr_financial.ml.gnn_predictor import GNNPredictor

# Run ABM simulation
sim = BankingSystemSimulation()
sim.setup_banks()
sim.run_simulation(100)

# Spectral analysis
adj = sim.get_adjacency_matrix()
L = compute_laplacian(adj, normalized=True)
eigenvalues, eigenvectors = eigendecomposition(L)

# Coarse-grain the network
scg = SpectralCoarseGraining.from_adjacency(adj, list(sim.banks.keys()))
scg.coarse_grain()
accuracy = scg.compute_reconstruction_accuracy(time_steps=15)

# Train GNN on market data
from dashboard.data_api import build_daily_graph_snapshots
snapshots = build_daily_graph_snapshots(lookback_years=3)
predictor = GNNPredictor(seq_len=10, hidden_dim=64, num_gcn_layers=3)
predictor.train(snapshots, epochs=200, lr=3e-3)
predictions = predictor.predict(snapshots, steps=30)
```

## Project Structure

```
scr-financial-networks/
├── scr_financial/
│   ├── abm/                    # Agent-Based Modeling
│   │   ├── bank_agent.py           # Bank agent with balance sheet
│   │   ├── simulation.py           # Multi-bank simulation engine
│   │   └── decision_models.py      # Lending/borrowing logic
│   ├── network/                # Network analysis
│   │   ├── builder.py              # Interbank graph construction
│   │   ├── spectral.py             # Laplacian, eigendecomposition
│   │   └── coarse_graining.py      # Spectral Coarse-Graining
│   ├── ml/                     # Machine Learning
│   │   ├── gnn_predictor.py        # Temporal GCN+LSTM model
│   │   ├── scg_predictor.py        # SCG feature predictor
│   │   ├── predictors.py           # Base predictor classes
│   │   └── sentiment_model.py      # Market sentiment analysis
│   ├── risk/                   # Systemic risk metrics
│   │   └── metrics.py              # Delta-CoVaR, MES
│   ├── backtesting/            # Historical validation
│   │   └── engine.py               # Crisis backtesting engine
│   ├── data/                   # Data collection (EBA, ECB, market)
│   └── vae/                    # Variational AutoEncoder
├── dashboard/
│   ├── app.py                  # Main Dash application (5 pages)
│   ├── api.py                  # FastAPI REST endpoints
│   ├── data_loader.py          # Bank tickers + yfinance/ECB
│   ├── data_api.py             # Daily graph snapshot pipeline
│   ├── prediction.py           # GNN training orchestration
│   ├── gnn_export.py           # PyG dataset export
│   ├── simulation_state.py     # Shared state management
│   ├── llm.py                  # Cerebras LLM integration
│   └── demo_data.py            # Shock scenarios + defaults
├── tests/                      # Test suite
├── examples/                   # Example scripts
└── docs/                       # Sphinx documentation source
```

## Documentation

Full API documentation is available at **[francoispzn.github.io/scr-financial-networks](https://francoispzn.github.io/scr-financial-networks/)**.

Covers: installation, quickstart, API reference for all modules, spectral coarse-graining theory, and GNN architecture details.

## GNN Architecture

```
Input: sequence of T daily graph snapshots (10 banks, 5 node features)
    ↓
[GCNConv → BatchNorm → ReLU → Dropout] × L layers
    ↓
global_mean_pool → graph-level embedding per snapshot
    ↓
LSTM (M layers) over T time steps
    ↓
FC → [λ₂, spectral_gap, spectral_radius]
```

Default configuration: hidden_dim=64, 3 GCN layers, 2 LSTM layers, dropout=0.1, cosine annealing LR, gradient clipping (max norm 1.0).

## Acknowledgments

- Theoretical framework: *"Spectral Coarse-Graining to Financial Networks"* by Francois Petizon
- Preliminary report: *"Contingency BW 9-08 x10-15"*
- Data sources: EBA Transparency Exercise, ECB Statistical Data Warehouse, yfinance

## Contact

**Francois Petizon** — francois.petizon@kcl.ac.uk
Department of Informatics, King's College London
