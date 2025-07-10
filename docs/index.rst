SCR-Financial-Networks Documentation
=====================================

A Python framework for analyzing financial networks using **Spectral Coarse-Graining (SCG)**, **Agent-Based Modeling (ABM)**, and **Graph Neural Networks (GNN)**.

Overview
--------

This project implements a hybrid approach combining Spectral Coarse-Graining with Agent-Based Modeling and temporal Graph Neural Networks to analyze interbank contagion dynamics among European banks.

Key capabilities:

- **Spectral Coarse-Graining**: reduce complex interbank networks while preserving diffusion dynamics
- **Agent-Based Simulation**: model individual bank behaviors under stress scenarios
- **Temporal GNN**: predict spectral properties from graph-structured market data
- **Risk Metrics**: Delta-CoVaR, MES, and SCG-based systemic risk scores
- **Interactive Dashboard**: Dash UI with live GNN training and SCG-vs-Basel comparison
- **Real Market Data**: yfinance stock data and ECB macro indicators

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/abm
   api/network
   api/spectral
   api/ml
   api/risk
   api/data
   api/dashboard

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/spectral_coarse_graining
   theory/gnn_architecture

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/black_week_simulation
   examples/network_visualization


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
