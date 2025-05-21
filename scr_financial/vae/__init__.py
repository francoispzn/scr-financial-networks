"""
Variational Autoencoder (VAE) module for financial network analysis.

This module provides implementations of VAE architectures for extracting
latent representations from financial data, which can be used for dimensionality
reduction, anomaly detection, and feature extraction.
"""

from .model import FinancialVAE
from .training import train_vae, evaluate_vae

__all__ = ["FinancialVAE", "train_vae", "evaluate_vae"]
