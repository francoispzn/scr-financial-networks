"""
Sentiment analysis model for financial market data.

This module provides an LSTM-based model for analyzing market sentiment
from financial time series data.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class LSTMSentimentModel(nn.Module):
    """
    LSTM-based sentiment analysis model for financial time series.

    Attributes:
        hidden_dim: Dimension of hidden state.
        num_layers: Number of LSTM layers.
        lstm: LSTM layer.
        dropout: Dropout layer.
        fc: Fully connected output layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the LSTM sentiment model.

        Args:
            input_dim: Number of input features.
            hidden_dim: Dimension of hidden state.
            num_layers: Number of LSTM layers.
            output_dim: Dimension of output (typically 1 for sentiment score).
            dropout: Dropout probability.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Note:
            Hidden states are re-initialised to zeros on every forward call
            (stateless inference).

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            Output tensor of shape [batch_size, output_dim].
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(
            x.device
        )
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(
            x.device
        )

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out


def prepare_sentiment_data(
    data: np.ndarray,
    target: np.ndarray,
    seq_length: int,
    train_split: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare data for sentiment model training.

    Args:
        data: Feature data of shape [n_samples, n_features].
        target: Target values of shape [n_samples].
        seq_length: Length of sequence for LSTM.
        train_split: Proportion of data to use for training.

    Returns:
        Tuple containing (X_train, y_train, X_test, y_test).

    Raises:
        ValueError: If *seq_length* is not positive or *train_split* is
            outside (0, 1).
    """
    if seq_length <= 0:
        raise ValueError(f"seq_length must be > 0, got {seq_length}")
    if not 0.0 < train_split < 1.0:
        raise ValueError(
            f"train_split must be in (0, 1), got {train_split}"
        )

    X_list, y_list = [], []
    for i in range(len(data) - seq_length):
        X_list.append(data[i : i + seq_length])
        y_list.append(target[i + seq_length])

    X = np.array(X_list)
    y = np.array(y_list).reshape(-1, 1)

    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return (
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test),
    )


def train_sentiment_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 64,
    epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = "cpu",
) -> Dict:
    """Train the sentiment model.

    Args:
        model: The model to train.
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        batch_size: Batch size for training.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        device: Device to train on ('cpu' or 'cuda').

    Returns:
        Dictionary containing training history with keys
        ``'train_loss'`` and ``'test_loss'``.
    """
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history: Dict[str, list] = {"train_loss": [], "test_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            history["test_loss"].append(test_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(
                "Epoch [%d/%d], Train Loss: %.4f, Test Loss: %.4f",
                epoch + 1,
                epochs,
                train_loss,
                test_loss,
            )

    return history


def predict_sentiment(
    model: nn.Module,
    features: np.ndarray,
    seq_length: int,
    device: str = "cpu",
) -> float:
    """Predict sentiment score from features.

    Args:
        model: Trained sentiment model.
        features: Input features of shape [n_features] or
            [seq_length, n_features].
        seq_length: Length of sequence for LSTM.
        device: Device to run prediction on.

    Returns:
        Predicted sentiment score.

    Raises:
        ValueError: If *features* has fewer than *seq_length* time steps.
    """
    model.eval()

    if len(features.shape) == 1:
        features = features.reshape(1, -1)

    if features.shape[0] < seq_length:
        raise ValueError(
            f"Features must have at least {seq_length} time steps"
        )

    if features.shape[0] > seq_length:
        features = features[-seq_length:]

    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)

    with torch.no_grad():
        sentiment = model(features_tensor).item()

    return sentiment
