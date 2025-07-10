"""
Complete Variational Autoencoder model for financial data.

This module combines the encoder and decoder to create a complete VAE model
for dimensionality reduction and feature extraction from financial data.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder

logger = logging.getLogger(__name__)


class FinancialVAE(nn.Module):
    """
    Variational Autoencoder for financial data.

    This model combines an encoder and decoder to create a VAE that can be
    used for dimensionality reduction, anomaly detection, and feature
    extraction from financial time series data.

    Attributes:
        latent_dim: Dimensionality of the latent space.
        encoder: Encoder network.
        decoder: Decoder network.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
    ) -> None:
        """Initialize the VAE model.

        Args:
            input_dim: Dimensionality of the input features.
            hidden_dims: Dimensions of the hidden layers.
            latent_dim: Dimensionality of the latent space.

        Raises:
            ValueError: If any dimension argument is not positive.
        """
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")

        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick to sample from latent distribution.

        Args:
            mu: Mean of the latent distribution.
            logvar: Log variance of the latent distribution.

        Returns:
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the VAE.

        Args:
            x: Input data tensor of shape [batch_size, input_dim].

        Returns:
            Dictionary containing:
            - ``'reconstruction'``: Reconstructed input.
            - ``'mu'``: Mean of the latent distribution.
            - ``'logvar'``: Log variance of the latent distribution.
            - ``'z'``: Sampled latent vector.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)

        return {
            "reconstruction": reconstruction,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kld_weight: float = 0.005,
    ) -> Dict[str, torch.Tensor]:
        """Compute the VAE loss function.

        The loss consists of a reconstruction term (MSE) and a KL divergence
        term.

        Args:
            recon_x: Reconstructed input.
            x: Original input.
            mu: Mean of the latent distribution.
            logvar: Log variance of the latent distribution.
            kld_weight: Weight for the KL divergence term.

        Returns:
            Dictionary containing:
            - ``'loss'``: Total loss.
            - ``'reconstruction_loss'``: MSE reconstruction loss.
            - ``'kld_loss'``: KL divergence loss.
        """
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_weight * kld_loss

        return {
            "loss": loss,
            "reconstruction_loss": recon_loss,
            "kld_loss": kld_loss,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input data to the latent space.

        Args:
            x: Input data tensor.

        Returns:
            Latent representation (mean vector).
        """
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to the original feature space.

        Args:
            z: Latent representation.

        Returns:
            Reconstructed data.
        """
        return self.decoder(z)

    def sample(self, num_samples: int, device: str = "cpu") -> torch.Tensor:
        """Generate samples from the latent space.

        Args:
            num_samples: Number of samples to generate.
            device: Device to generate samples on.

        Returns:
            Generated samples.
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
