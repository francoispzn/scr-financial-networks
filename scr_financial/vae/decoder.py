"""
Decoder implementation for the Variational Autoencoder.

This module defines the decoder architecture that maps latent representations
back to the original feature space.
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder network for the Variational Autoencoder.
    
    The decoder maps latent representations back to the original feature space.
    
    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space
    hidden_dims : list of int
        Dimensions of the hidden layers (in reverse order from encoder)
    output_dim : int
        Dimensionality of the output features
    """
    
    def __init__(self, latent_dim, hidden_dims, output_dim):
        """Initialize the decoder network."""
        super(Decoder, self).__init__()
        
        # Reverse hidden dimensions for decoder
        hidden_dims = hidden_dims[::-1]
        
        # Build decoder layers
        modules = []
        
        # Input layer from latent space
        modules.append(nn.Linear(latent_dim, hidden_dims[0]))
        modules.append(nn.BatchNorm1d(hidden_dims[0]))
        modules.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            modules.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            modules.append(nn.ReLU())
            
        # Output layer
        modules.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, z):
        """
        Forward pass through the decoder.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent representation tensor of shape [batch_size, latent_dim]
            
        Returns
        -------
        torch.Tensor
            Reconstructed data tensor of shape [batch_size, output_dim]
        """
        return self.decoder(z)
