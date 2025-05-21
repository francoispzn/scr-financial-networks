"""
SCR-Financial-Networks: Spectral Coarse-Graining for Financial Networks.

This package provides tools for analyzing financial networks using
spectral coarse-graining and agent-based modeling techniques.
"""

from .network.builder import FinancialNetworkBuilder
from .network.coarse_graining import SpectralCoarseGraining
from .abm.simulation import BankingSystemSimulation
from .vae import FinancialVAE
from .data.preprocessor import DataPreprocessor

__version__ = "0.1.0"

__all__ = [
    "FinancialNetworkBuilder",
    "SpectralCoarseGraining",
    "BankingSystemSimulation",
    "FinancialVAE",
    "DataPreprocessor",
]
