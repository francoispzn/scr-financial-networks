"""
Network module for financial network analysis.

This module provides tools for constructing, analyzing, and coarse-graining
financial networks based on interbank exposures and other financial data.
"""

from .builder import FinancialNetworkBuilder
from .spectral import find_spectral_gap, compute_diffusion_modes
from .coarse_graining import SpectralCoarseGraining

__all__ = [
    "FinancialNetworkBuilder",
    "find_spectral_gap",
    "compute_diffusion_modes",
    "SpectralCoarseGraining",
]
