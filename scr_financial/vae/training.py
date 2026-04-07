"""
Training utilities for the Variational Autoencoder.

This module provides functions for training and evaluating VAE models
on financial data.
"""

import logging
import time
from typing import Dict, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_vae(
    model: "FinancialVAE",  # noqa: F821
    data: Union[torch.Tensor, np.ndarray],
    batch_size: int = 64,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    kld_weight: float = 0.005,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """Train a VAE model on financial data.

    Args:
        model: The VAE model to train.
        data: Training data of shape [n_samples, n_features].
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        kld_weight: Weight for the KL divergence term in the loss.
        device: Device to train on ('cpu' or 'cuda').
        verbose: Whether to display progress bars.

    Returns:
        Dictionary containing training history:
        - ``'total_loss'``: List of total loss values.
        - ``'reconstruction_loss'``: List of reconstruction loss values.
        - ``'kld_loss'``: List of KL divergence loss values.
    """
    if isinstance(data, np.ndarray):
        data = torch.FloatTensor(data)

    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: Dict[str, list] = {
        "total_loss": [],
        "reconstruction_loss": [],
        "kld_loss": [],
    }

    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kld_loss = 0.0

        iterator = (
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            if verbose
            else dataloader
        )

        for batch in iterator:
            x = batch[0].to(device)

            optimizer.zero_grad()
            results = model(x)

            loss_dict = model.loss_function(
                results["reconstruction"],
                x,
                results["mu"],
                results["logvar"],
                kld_weight,
            )

            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += loss_dict["reconstruction_loss"].item()
            epoch_kld_loss += loss_dict["kld_loss"].item()

        avg_loss = epoch_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_kld_loss = epoch_kld_loss / len(dataloader)

        history["total_loss"].append(avg_loss)
        history["reconstruction_loss"].append(avg_recon_loss)
        history["kld_loss"].append(avg_kld_loss)

        if verbose and (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(
                "Epoch %d/%d - Loss: %.4f - Recon Loss: %.4f - "
                "KLD Loss: %.4f - Time: %.2fs",
                epoch + 1,
                epochs,
                avg_loss,
                avg_recon_loss,
                avg_kld_loss,
                elapsed,
            )

    return history


def evaluate_vae(
    model: "FinancialVAE",  # noqa: F821
    data: Union[torch.Tensor, np.ndarray],
    batch_size: int = 64,
    device: str = "cpu",
) -> Dict:
    """Evaluate a trained VAE model on test data.

    Args:
        model: The trained VAE model.
        data: Test data of shape [n_samples, n_features].
        batch_size: Batch size for evaluation.
        device: Device to evaluate on ('cpu' or 'cuda').

    Returns:
        Dictionary containing:
        - ``'loss'``: Average total loss.
        - ``'reconstruction_loss'``: Average reconstruction loss.
        - ``'kld_loss'``: Average KL divergence loss.
        - ``'reconstructions'``: Reconstructed data.
        - ``'latent_representations'``: Encoded latent representations.
    """
    if isinstance(data, np.ndarray):
        data = torch.FloatTensor(data)

    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_kld_loss = 0.0

    reconstructions = []
    latent_representations = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)

            results = model(x)
            loss_dict = model.loss_function(
                results["reconstruction"],
                x,
                results["mu"],
                results["logvar"],
            )

            total_loss += loss_dict["loss"].item()
            total_recon_loss += loss_dict["reconstruction_loss"].item()
            total_kld_loss += loss_dict["kld_loss"].item()

            reconstructions.append(results["reconstruction"].cpu())
            latent_representations.append(results["mu"].cpu())

    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kld_loss = total_kld_loss / len(dataloader)

    return {
        "loss": avg_loss,
        "reconstruction_loss": avg_recon_loss,
        "kld_loss": avg_kld_loss,
        "reconstructions": torch.cat(reconstructions, dim=0).numpy(),
        "latent_representations": torch.cat(
            latent_representations, dim=0
        ).numpy(),
    }
