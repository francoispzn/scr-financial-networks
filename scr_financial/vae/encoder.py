"""
Encoder implementation for the Variational Autoencoder.

This module defines the encoder architecture that maps input financial data
to a latent distribution (mean and log variance).
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """
    Encoder network for the Variational Autoencoder.

    The encoder maps input financial data to a latent distribution
    parameterized by mean and log variance vectors.

    Attributes:
        encoder: Sequential encoder network.
        fc_mu: Linear layer producing the mean vector.
        fc_logvar: Linear layer producing the log-variance vector.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
    ) -> None:
        """Initialize the encoder network.

        Args:
            input_dim: Dimensionality of the input features.
            hidden_dims: Dimensions of the hidden layers.
            latent_dim: Dimensionality of the latent space.

        Raises:
            ValueError: If any of the dimension arguments are not positive.
        """
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {input_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")
        for i, d in enumerate(hidden_dims):
            if d <= 0:
                raise ValueError(
                    f"hidden_dims[{i}] must be > 0, got {d}"
                )

        modules: List[nn.Module] = []

        modules.append(nn.Linear(input_dim, hidden_dims[0]))
        modules.append(nn.BatchNorm1d(hidden_dims[0]))
        modules.append(nn.ReLU())

        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            modules.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            modules.append(nn.ReLU())

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the encoder.

        Args:
            x: Input data tensor of shape [batch_size, input_dim].

        Returns:
            Tuple (mu, logvar) where *mu* is the mean and *logvar* is the
            log-variance of the latent distribution.
        """
        result = self.encoder(x)
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        return mu, logvar
