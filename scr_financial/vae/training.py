"""
Training utilities for the Variational Autoencoder.

This module provides functions for training and evaluating VAE models
on financial data.
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def train_vae(model, data, batch_size=64, epochs=100, learning_rate=1e-3, 
              kld_weight=0.005, device='cpu', verbose=True):
    """
    Train a VAE model on financial data.
    
    Parameters
    ----------
    model : FinancialVAE
        The VAE model to train
    data : torch.Tensor or numpy.ndarray
        Training data of shape [n_samples, n_features]
    batch_size : int, optional
        Batch size for training, by default 64
    epochs : int, optional
        Number of training epochs, by default 100
    learning_rate : float, optional
        Learning rate for the optimizer, by default 1e-3
    kld_weight : float, optional
        Weight for the KL divergence term in the loss, by default 0.005
    device : str, optional
        Device to train on ('cpu' or 'cuda'), by default 'cpu'
    verbose : bool, optional
        Whether to display progress bars, by default True
        
    Returns
    -------
    dict
        Dictionary containing training history:
        - 'total_loss': List of total loss values
        - 'reconstruction_loss': List of reconstruction loss values
        - 'kld_loss': List of KL divergence loss values
    """
    # Convert data to torch tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.FloatTensor(data)
    
    # Create dataset and dataloader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'total_loss': [],
        'reconstruction_loss': [],
        'kld_loss': []
    }
    
    # Training loop
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kld_loss = 0
        
        iterator = dataloader
        if verbose:
            iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in iterator:
            x = batch[0].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            results = model(x)
            
            # Compute loss
            loss_dict = model.loss_function(
                results['reconstruction'],
                x,
                results['mu'],
                results['logvar'],
                kld_weight
            )
            
            # Backward pass and optimize
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_recon_loss += loss_dict['reconstruction_loss'].item()
            epoch_kld_loss += loss_dict['kld_loss'].item()
        
        # Average losses
        avg_loss = epoch_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_kld_loss = epoch_kld_loss / len(dataloader)
        
        # Update history
        history['total_loss'].append(avg_loss)
        history['reconstruction_loss'].append(avg_recon_loss)
        history['kld_loss'].append(avg_kld_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {avg_loss:.4f} - "
                  f"Recon Loss: {avg_recon_loss:.4f} - "
                  f"KLD Loss: {avg_kld_loss:.4f} - "
                  f"Time: {elapsed:.2f}s")
    
    return history


def evaluate_vae(model, data, batch_size=64, device='cpu'):
    """
    Evaluate a trained VAE model on test data.
    
    Parameters
    ----------
    model : FinancialVAE
        The trained VAE model
    data : torch.Tensor or numpy.ndarray
        Test data of shape [n_samples, n_features]
    batch_size : int, optional
        Batch size for evaluation, by default 64
    device : str, optional
        Device to evaluate on ('cpu' or 'cuda'), by default 'cpu'
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'loss': Average total loss
        - 'reconstruction_loss': Average reconstruction loss
        - 'kld_loss': Average KL divergence loss
        - 'reconstructions': Reconstructed data
        - 'latent_representations': Encoded latent representations
    """
    # Convert data to torch tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.FloatTensor(data)
    
    # Create dataset and dataloader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Move model to device
    model = model.to(device)
    
    # Evaluation
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    
    reconstructions = []
    latent_representations = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            
            # Forward pass
            results = model(x)
            
            # Compute loss
            loss_dict = model.loss_function(
                results['reconstruction'],
                x,
                results['mu'],
                results['logvar']
            )
            
            # Update metrics
            total_loss += loss_dict['loss'].item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            total_kld_loss += loss_dict['kld_loss'].item()
            
            # Store reconstructions and latent representations
            reconstructions.append(results['reconstruction'].cpu())
            latent_representations.append(results['mu'].cpu())
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kld_loss = total_kld_loss / len(dataloader)
    
    # Concatenate results
    reconstructions = torch.cat(reconstructions, dim=0).numpy()
    latent_representations = torch.cat(latent_representations, dim=0).numpy()
    
    return {
        'loss': avg_loss,
        'reconstruction_loss': avg_recon_loss,
        'kld_loss': avg_kld_loss,
        'reconstructions': reconstructions,
        'latent_representations': latent_representations
    }
