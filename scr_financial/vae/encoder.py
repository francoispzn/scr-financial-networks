"""
Encoder implementation for the Variational Autoencoder.

This module defines the encoder architecture that maps input financial data
to a latent distribution (mean and log variance).
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder network for the Variational Autoencoder.
    
    The encoder maps input financial data to a latent distribution
    parameterized by mean and log variance vectors.
    
    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features
    hidden_dims : list of int
        Dimensions of the hidden layers
    latent_dim : int
        Dimensionality of the latent space
    """
    
    def __init__(self, input_dim, hidden_dims, latent_dim):
        """Initialize the encoder network."""
        super(Encoder, self).__init__()
        
        # Build encoder layers
        modules = []
        
        # Input layer
        modules.append(nn.Linear(input_dim, hidden_dims[0]))
        modules.append(nn.BatchNorm1d(hidden_dims[0]))
        modules.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            modules.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            modules.append(nn.ReLU())
            
        self.encoder = nn.Sequential(*modules)
        
        # Mean and variance layers
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape [batch_size, input_dim]
            
        Returns
        -------
        mu : torch.Tensor
            Mean of the latent distribution
        logvar : torch.Tensor
            Log variance of the latent distribution
        """
        result = self.encoder(x)
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        
        return mu, logvar
