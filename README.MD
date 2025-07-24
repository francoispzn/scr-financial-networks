# SCR-Financial-Networks

**Spectral Coarse-Graining for Systemic Risk Analysis in Financial Networks** — a Python framework combining Spectral Coarse-Graining (SCG), temporal Graph Attention Networks (GAT-LSTM), and multi-channel Agent-Based stress testing for interbank contagion analysis.

> **[Dissertation (PDF)](paper/dissertation.pdf)** · **[LaTeX source](paper/)**

## Overview

This project adapts spectral coarse-graining — a graph renormalization procedure developed by Schmidt, Caccioli & Aste for general complex networks — to systemic risk analysis in financial systems. A daily interbank network is built from stock-return correlations of European banks (2021–2024), and three methodological components are integrated on top of it:

1. **Spectral Coarse-Graining (SCG)** with an Erdős–Rényi null-model gap test for statistically grounded dimensionality reduction;
2. **A temporal Graph Attention Network (GAT-LSTM)** that predicts the evolution of spectral risk indicators (λ₂, spectral gap, spectral radius) from sequences of daily graph snapshots;
3. **A multi-channel Agent-Based Model (ABM)** simulating contagion through credit losses, funding-liquidity stress, and fire-sale channels.

The full pipeline — data collection → network construction → spectral analysis → coarse-graining → GNN prediction → ABM stress testing → backtesting — is exposed both as a Python package (`scr_financial`) and through an interactive Dash dashboard.

## Key Findings

Evaluated across ~1,100 daily graph snapshots with walk-forward cross-validation (full details and caveats in the [dissertation](paper/dissertation.pdf)):

- **SCG works as advertised on dynamics**: diffusion reconstruction is essentially perfect (R² ≈ 1.0) across all tested synthetic topologies (block models, Erdős–Rényi, scale-free, star).
- **Evaluation methodology matters more than architecture**: the GAT-LSTM reaches R² = 0.38 on a single train/test split but a mean R² of −0.42 ± 0.57 under walk-forward CV at N=10 banks — a strong regime-dependence warning for temporal GNNs on small financial networks.
- **Classification is easier than regression**: binary stress-regime (high-volatility) classification achieves walk-forward AUC ≈ 0.93–0.98 across classifiers in the experiment suite; ensemble stacking on graph features beats the persistence baseline for λ₂ prediction (R² 0.517 vs 0.489 under 5-fold walk-forward).
- **The SCG risk score `1 − λ₂/ρ` anti-correlates with market volatility** (r ≈ −0.63): it captures network fragmentation rather than aggregate stress, so spectral indicators need volatility conditioning before operational use.
- **Multi-channel contagion amplifies**: the 3-channel ABM produces ~10× more defaults than a credit-only ABM under severe liquidity stress.
- **Random matrix theory caution**: at N=10, 9 of 10 correlation eigenvalues are consistent with noise — spectral analysis of small networks must be interpreted carefully (larger universes `eu_50` / `global_100` are provided).

## Features

### Spectral Coarse-Graining (`scr_financial.network`)
- Laplacian spectral decomposition, spectral-gap identification with ER null-model test
- Coarse-graining that preserves diffusion dynamics, with reconstruction accuracy (R²/RMSE) tracking
- Multi-scale SCG, RMT denoising (Marchenko–Pastur), principled threshold selection (MDL, percolation, CV spectral stability)
- Alternative network constructions: threshold, MST, PMFG, partial correlation
- Change-point detection (CUSUM, binary segmentation) and volatility-conditioned spectral indicators

### Machine Learning (`scr_financial.ml`)
- **GAT-LSTM predictor**: `GATConv` layers → `global_mean_pool` → LSTM → FC, predicting [λ₂, spectral gap, ρ]
- GNN encoder variants (GCN, GIN, Graph Transformer) and a GAT-based binary stress classifier
- Walk-forward cross-validation, statistical baselines (persistence, AR, gradient boosting), Diebold–Mariano tests
- Uncertainty quantification: quantile loss, MC dropout, conformal prediction
- Feature/architecture ablation suites and benchmarking utilities

### Agent-Based Model (`scr_financial.abm`)
- Bank agents with balance-sheet dynamics (CET1, LCR, leverage) and configurable decision models
- Three contagion channels: credit losses, funding liquidity, fire sales — with Monte Carlo simulation
- Asset dynamics via Ornstein–Uhlenbeck, jump-diffusion, or regime-switching processes
- Calibration against empirical targets, Sobol sensitivity analysis, comparison with Montagna–Kok and Poledna frameworks

### Risk Metrics & Backtesting (`scr_financial.risk`, `scr_financial.backtesting`)
- Delta-CoVaR, MES, VaR and system-level risk aggregation
- SCG-vs-Basel comparison via a rolling backtesting engine (quarterly EBA snapshots, next-quarter stress prediction)
- Predefined crisis-window definitions (2008 GFC, 2011 euro sovereign debt crisis, 2020 COVID, and more) in `scr_financial/config/crisis_periods.py`

### Data Pipeline (`scr_financial.data`, `dashboard`)
- Daily prices via yfinance for configurable bank universes: `eu_10` (default), `eu_50`, `global_100` (`scr_financial/config/bank_universes.yaml`)
- ECB Statistical Data Warehouse integration (sovereign yields, FX rates) with local caching
- Rolling correlation → adjacency matrices → daily graph snapshots → spectral targets (~1,100 snapshots over 2021–2024)

### Interactive Dashboard (`dashboard`)
- 5-page Dash application (dark UI): **Network**, **Simulate**, **Spectral**, **Evolve** (train the GNN from the UI with live progress), **AI/Data** (LLM-assisted analysis, data fetching, PyG dataset export)
- FastAPI REST backend (`dashboard/api.py`)

## Installation

```bash
git clone https://github.com/francoispzn/scr-financial-networks.git
cd scr-financial-networks

# Using Conda (recommended)
conda env create -f environment.yml
conda activate scg-financial
pip install -e .

# Or using pip (dashboard extra pulls in dash/plotly/fastapi)
pip install -e ".[dashboard]"
```

Core requirements: `numpy`, `scipy`, `pandas`, `networkx`, `torch`, `scikit-learn`, `requests`; the dashboard extra adds `dash`, `plotly`, `fastapi`; `torch-geometric` and `yfinance` come with the conda environment (or install them manually when using pip).

## Quick Start

### Launch the dashboard

```bash
python dashboard/app.py
# open http://localhost:8050
```

### Run the experiments

```bash
python experiments/run_all.py          # quick verification (~5 min)
python experiments/run_publication.py  # full publication suite, exp01–exp13 (~30–60 min)
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for seeds, data sources, and compute requirements.

### Programmatic usage

```python
from scr_financial.abm.simulation import BankingSystemSimulation
from scr_financial.network.spectral import compute_laplacian, eigendecomposition
from scr_financial.network.coarse_graining import SpectralCoarseGraining
from scr_financial.ml.gnn_predictor import GNNPredictor

# Run ABM simulation (inputs from the real data pipeline)
from dashboard.data_loader import load_simulation_inputs
bank_data, network_data, system_indicators = load_simulation_inputs()
sim = BankingSystemSimulation(bank_data, network_data, system_indicators)
sim.run_simulation(100)

# Spectral analysis
adj = sim.get_adjacency_matrix()
L = compute_laplacian(adj, normalized=True)
eigenvalues, eigenvectors = eigendecomposition(L)

# Coarse-grain the network
scg = SpectralCoarseGraining.from_adjacency(adj, list(sim.banks.keys()))
scg.coarse_grain()
accuracy = scg.compute_reconstruction_accuracy(time_steps=15)

# Train the GAT-LSTM on real market data
from dashboard.data_api import build_daily_graph_snapshots
snapshots = build_daily_graph_snapshots(lookback_years=3)
predictor = GNNPredictor(seq_len=10, hidden_dim=64)
predictor.train(snapshots, epochs=200, lr=3e-3)
predictions = predictor.predict(snapshots, steps=30)
```

## Project Structure

```
scr-financial-networks/
├── paper/                      # Dissertation PDF + LaTeX sources + figures
├── scr_financial/
│   ├── network/                # Graph construction & spectral methods
│   │   ├── builder.py              # Interbank graph construction
│   │   ├── spectral.py             # Laplacian, eigendecomposition, spectral gap
│   │   ├── coarse_graining.py      # Spectral Coarse-Graining (SCG)
│   │   ├── multiscale_scg.py       # Multi-scale SCG
│   │   ├── rmt.py                  # Random matrix theory denoising
│   │   ├── threshold.py            # Principled threshold selection
│   │   ├── alternative_builders.py # MST / PMFG / partial correlation
│   │   ├── change_point.py         # CUSUM, binary segmentation
│   │   └── conditional_spectral.py # Volatility-conditioned indicators
│   ├── ml/                     # Machine learning
│   │   ├── gnn_predictor.py        # Temporal GAT-LSTM predictor
│   │   ├── gnn_variants.py         # GCN / GIN / Graph Transformer encoders
│   │   ├── gnn_classifier.py       # GAT stress-regime classifier
│   │   ├── baselines.py            # Persistence / AR / GB baselines
│   │   ├── walk_forward.py         # Walk-forward cross-validation
│   │   ├── uncertainty.py          # Quantile, MC dropout, conformal
│   │   └── ablation.py             # Ablation suites
│   ├── abm/                    # Agent-based modeling
│   │   ├── bank_agent.py           # Bank agent with balance sheet
│   │   ├── simulation.py           # Multi-bank simulation engine
│   │   ├── dynamics.py             # OU / jump-diffusion / regime-switching
│   │   ├── calibration.py          # Parameter calibration
│   │   ├── sensitivity.py          # Sobol sensitivity analysis
│   │   └── literature_comparison.py# Montagna-Kok, Poledna frameworks
│   ├── risk/                   # Delta-CoVaR, MES, SCG-vs-Basel
│   ├── backtesting/            # Rolling backtesting engine (EBA snapshots)
│   ├── stats/                  # Bootstrap, hypothesis tests
│   ├── data/                   # EBA / ECB / market collectors, FX conversion
│   ├── config/                 # Bank universes (eu_10, eu_50, global_100)
│   └── vae/                    # Variational autoencoder (exploratory)
├── dashboard/                  # Dash UI + FastAPI backend
├── experiments/                # Publication experiment suite (exp01–exp13) + figures
├── tests/                      # Test suite
└── docs/                       # Sphinx documentation source
```

## GNN Architecture

```
Input: sequence of T daily graph snapshots (node features: volatility, mean return, log price, beta proxy, momentum)
    ↓
[GATConv (multi-head) → BatchNorm → ELU → Dropout] × L layers
    ↓
global_mean_pool → graph-level embedding per snapshot
    ↓
LSTM (M layers) over T time steps
    ↓
FC → [λ₂, spectral_gap, spectral_radius]
```

## Citation

If you use this software, please cite it via [CITATION.cff](CITATION.cff), and see the full write-up:

> Petizon, F. *Spectral Coarse-Graining for Systemic Risk Analysis in Financial Networks: A Graph Attention Network Framework with Multi-Channel Agent-Based Stress Testing.* Department of Informatics, King's College London. ([PDF](paper/dissertation.pdf))

## Acknowledgments

- SCG methodology: Schmidt, Caccioli & Aste, *Spectral coarse graining and rescaling for preserving structural and dynamical properties in graphs*
- Supervision: Dr. Teresa Yu Bi and Prof. Peter McBurney, King's College London
- Data: ECB Statistical Data Warehouse, Yahoo Finance, EBA Transparency Exercise

## Contact

**Francois Petizon** — francois.petizon@kcl.ac.uk
Department of Informatics, King's College London
