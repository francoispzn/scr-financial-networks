"""
SCR-Financial-Networks: Spectral Coarse-Graining for Financial Networks.

This package provides tools for analyzing financial networks using
spectral coarse-graining and agent-based modeling techniques.
"""

from .network.builder import FinancialNetworkBuilder
from .network.coarse_graining import SpectralCoarseGraining
from .abm.simulation import BankingSystemSimulation
from .abm.bank_agent import BankAgent
from .abm.decision_models import DefaultDecisionModel, StressDecisionModel, LearningDecisionModel
from .data.preprocessor import DataPreprocessor

try:
    from .vae import FinancialVAE
except ImportError:
    FinancialVAE = None  # type: ignore[assignment,misc]

__version__ = "0.1.0"

__all__ = [
    "DataPreprocessor",
    "FinancialNetworkBuilder",
    "SpectralCoarseGraining",
    "BankAgent",
    "BankingSystemSimulation",
    "DefaultDecisionModel",
    "StressDecisionModel",
    "LearningDecisionModel",
    "FinancialVAE",
]
