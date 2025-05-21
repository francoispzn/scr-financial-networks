"""
Complete Variational Autoencoder model for financial data.

This module combines the encoder and decoder to create a complete VAE model
for dimensionality reduction and feature extraction from financial data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder


class FinancialVAE(nn.Module):
    """
    Variational Autoencoder for financial data.
    
    This model combines an encoder and decoder to create a VAE that can be used
    for dimensionality reduction, anomaly detection, and feature extraction
    from financial time series data.
    
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
        """Initialize the VAE model."""
        super(FinancialVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Initialize encoder and decoder
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from latent distribution.
        
        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent distribution
        logvar : torch.Tensor
            Log variance of the latent distribution
            
        Returns
        -------
        torch.Tensor
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor of shape [batch_size, input_dim]
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'reconstruction': Reconstructed input
            - 'mu': Mean of the latent distribution
            - 'logvar': Log variance of the latent distribution
            - 'z': Sampled latent vector
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def loss_function(self, recon_x, x, mu, logvar, kld_weight=0.005):
        """
        Compute the VAE loss function.
        
        The loss consists of a reconstruction term (MSE) and a KL divergence term.
        
        Parameters
        ----------
        recon_x : torch.Tensor
            Reconstructed input
        x : torch.Tensor
            Original input
        mu : torch.Tensor
            Mean of the latent distribution
        logvar : torch.Tensor
            Log variance of the latent distribution
        kld_weight : float, optional
            Weight for the KL divergence term, by default 0.005
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'loss': Total loss
            - 'reconstruction_loss': MSE reconstruction loss
            - 'kld_loss': KL divergence loss
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + kld_weight * kld_loss
        
        return {
            'loss': loss,
            'reconstruction_loss': recon_loss,
            'kld_loss': kld_loss
        }
    
    def encode(self, x):
        """
        Encode input data to the latent space.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data tensor
            
        Returns
        -------
        torch.Tensor
            Latent representation (mean vector)
        """
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z):
        """
        Decode latent representation to the original feature space.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent representation
            
        Returns
        -------
        torch.Tensor
            Reconstructed data
        """
        return self.decoder(z)
    
    def sample(self, num_samples, device='cpu'):
        """
        Generate samples from the latent space.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        device : str, optional
            Device to generate samples on, by default 'cpu'
            
        Returns
        -------
        torch.Tensor
            Generated samples
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples
