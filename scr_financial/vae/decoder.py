"""
Decoder implementation for the Variational Autoencoder.

This module defines the decoder architecture that maps latent representations
back to the original feature space.
"""

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Decoder(nn.Module):
    """
    Decoder network for the Variational Autoencoder.

    The decoder maps latent representations back to the original feature space.

    Attributes:
        decoder: Sequential decoder network.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
    ) -> None:
        """Initialize the decoder network.

        Args:
            latent_dim: Dimensionality of the latent space.
            hidden_dims: Dimensions of the hidden layers (in reverse order
                from encoder).
            output_dim: Dimensionality of the output features.

        Raises:
            ValueError: If any of the dimension arguments are not positive.
        """
        super().__init__()

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {output_dim}")
        for i, d in enumerate(hidden_dims):
            if d <= 0:
                raise ValueError(
                    f"hidden_dims[{i}] must be > 0, got {d}"
                )

        # Reverse hidden dimensions for decoder
        hidden_dims = hidden_dims[::-1]

        modules: List[nn.Module] = []

        modules.append(nn.Linear(latent_dim, hidden_dims[0]))
        modules.append(nn.BatchNorm1d(hidden_dims[0]))
        modules.append(nn.ReLU())

        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            modules.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            modules.append(nn.ReLU())

        modules.append(nn.Linear(hidden_dims[-1], output_dim))

        self.decoder = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            z: Latent representation tensor of shape [batch_size, latent_dim].

        Returns:
            Reconstructed data tensor of shape [batch_size, output_dim].
        """
        return self.decoder(z)
