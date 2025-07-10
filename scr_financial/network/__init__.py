"""
Network module for financial network analysis.

This module provides tools for constructing, analyzing, and coarse-graining
financial networks based on interbank exposures and other financial data.
"""

from .builder import FinancialNetworkBuilder
from .coarse_graining import SpectralCoarseGraining
from .spectral import (
    compute_laplacian,
    eigendecomposition,
    find_spectral_gap,
    compute_diffusion_modes,
    analyze_spectral_properties,
    compute_spectral_embedding,
    compute_diffusion_distance,
)

__all__ = [
    "FinancialNetworkBuilder",
    "SpectralCoarseGraining",
    "compute_laplacian",
    "eigendecomposition",
    "find_spectral_gap",
    "compute_diffusion_modes",
    "analyze_spectral_properties",
    "compute_spectral_embedding",
    "compute_diffusion_distance",
]
