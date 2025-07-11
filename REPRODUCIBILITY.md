# Reproducibility Guide

## Environment

```bash
conda env create -f environment.yml
conda activate systemic_risk
pip install -e ".[dashboard]"
```

## Running Experiments

### Quick verification (10-bank, ~5 minutes)
```bash
python experiments/run_all.py
```

### Full publication suite (~30-60 minutes)
```bash
python experiments/run_publication.py
```

### Specific experiment sets
```bash
python experiments/run_comprehensive.py   # SCG + GAT sweep + ABM stress + backtesting
python experiments/run_additional.py      # Attention analysis + per-target + threshold sensitivity
```

## Data Sources

- **Stock prices**: Yahoo Finance via `yfinance` (free, no API key required)
- **Sovereign yields**: ECB Statistical Data Warehouse (free, no authentication)
- **FX rates**: ECB SDW (free, no authentication)

All data is fetched at runtime. For offline reproducibility, cached snapshots are stored in `experiments/results/`.

## Random Seeds

All experiments use explicit random seeds (default: 42). Multi-seed experiments iterate seeds as `seed * 42` for `seed in range(n_seeds)`.

## Bank Universes

Three pre-configured universes in `scr_financial/config/bank_universes.yaml`:
- `eu_10`: 10 European banks (default, matches original experiments)
- `eu_50`: 50 European banks across 15 countries
- `global_100`: ~95 banks across 18 currencies worldwide

## Computational Requirements

| Experiment | Time (M1 Mac) | Peak Memory |
|-----------|---------------|-------------|
| SCG validation (5 topologies) | ~30s | ~100 MB |
| GAT sweep (6 configs × 5 seeds) | ~3 min | ~500 MB |
| ABM stress (7 scenarios × 100 runs) | ~15s | ~200 MB |
| Backtesting (4 thresholds × 4 horizons) | ~30s | ~200 MB |
| Full publication suite | ~30-60 min | ~1 GB |
