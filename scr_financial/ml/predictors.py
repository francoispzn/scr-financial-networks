"""
Predictors for financial network analysis.

This module provides predictors for forecasting network evolution
and market indicators.
"""

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet

logger = logging.getLogger(__name__)


class NetworkPredictor:
    """
    Predictor for financial network evolution.

    This class implements methods for predicting how the financial network
    will evolve over time based on historical data and external factors.

    Attributes:
        method: Prediction method identifier.
        model: Underlying sklearn estimator.
        scaler_X: Feature scaler.
        scaler_y: Target scaler.
    """

    def __init__(self, method: str = "rf") -> None:
        """Initialize the network predictor.

        Args:
            method: Prediction method ('rf' for Random Forest, 'gb' for
                Gradient Boosting, 'elastic' for Elastic Net).

        Raises:
            ValueError: If *method* is not one of the supported values.
        """
        self.method = method
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        if method == "rf":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif method == "gb":
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif method == "elastic":
            self.model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

    def prepare_features(
        self,
        network_history: List[Dict],
        system_indicators: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for network prediction.

        Args:
            network_history: List of network states over time.
            system_indicators: DataFrame of system-wide indicators.

        Returns:
            Tuple containing (X, y) where X is features and y is target values.
        """
        X: List[List[float]] = []
        y: List[float] = []

        for i in range(1, len(network_history)):
            prev_state = network_history[i - 1]
            curr_state = network_history[i]

            features: List[float] = []

            # Network density
            features.append(prev_state.get("network_density", 0))

            # Average degree
            features.append(prev_state.get("avg_degree", 0))

            # Add system indicators
            time_point = prev_state.get("time", 0)
            if time_point in system_indicators.index:
                for col in system_indicators.columns:
                    features.append(system_indicators.loc[time_point, col])

            # Target: current network density
            y.append(curr_state.get("network_density", 0))
            X.append(features)

        return np.array(X), np.array(y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the predictor to training data.

        Args:
            X: Feature matrix.
            y: Target values.
        """
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        self.model.fit(X_scaled, y_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the fitted model.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if not hasattr(self.model, "n_features_in_"):
            raise RuntimeError("Call fit() before predict()")

        X_scaled = self.scaler_X.transform(X)
        y_scaled_pred = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(
            y_scaled_pred.reshape(-1, 1)
        ).ravel()
        return y_pred

    def predict_network_evolution(
        self,
        current_state: Dict,
        system_forecast: pd.DataFrame,
        steps: int = 10,
    ) -> List[Dict]:
        """Predict network evolution over multiple time steps.

        Args:
            current_state: Current network state.
            system_forecast: Forecast of system indicators.
            steps: Number of steps to predict.

        Returns:
            List of predicted network states.
        """
        predictions: List[Dict] = []
        state = current_state.copy()

        for _ in range(steps):
            features: List[float] = []

            features.append(state.get("network_density", 0))
            features.append(state.get("avg_degree", 0))

            time_point = state.get("time", 0) + 1
            if time_point in system_forecast.index:
                for col in system_forecast.columns:
                    features.append(system_forecast.loc[time_point, col])
            else:
                for col in system_forecast.columns:
                    features.append(system_forecast.iloc[-1][col])

            pred = self.predict(np.array([features]))

            new_state = state.copy()
            new_state["time"] = time_point
            new_state["network_density"] = pred[0]

            predictions.append(new_state)
            state = new_state

        return predictions


class MarketPredictor:
    """
    Predictor for market indicators.

    This class implements a deep learning model for predicting market
    indicators based on historical data and network features.

    Attributes:
        model: Underlying LSTM model.
        scaler_X: Feature scaler.
        scaler_y: Target scaler.
        seq_length: Default sequence length used during prepare_sequences.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
    ) -> None:
        """Initialize the market predictor.

        Args:
            input_dim: Input dimension (number of features).
            hidden_dim: Hidden dimension for LSTM.
            output_dim: Output dimension (number of indicators to predict).
            num_layers: Number of LSTM layers.
        """
        self.model = LSTMPredictor(input_dim, hidden_dim, output_dim, num_layers)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.seq_length: int = 10

    def prepare_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training.

        Args:
            features: Feature matrix of shape [n_samples, n_features].
            targets: Target matrix of shape [n_samples, n_targets].
            seq_length: Length of sequences.

        Returns:
            Tuple (X, y) where X has shape [n_sequences, seq_length,
            n_features] and y has shape [n_sequences, n_targets].
        """
        self.seq_length = seq_length

        features_scaled = self.scaler_X.fit_transform(features)
        targets_scaled = self.scaler_y.fit_transform(targets)

        X: List[np.ndarray] = []
        y: List[np.ndarray] = []
        for i in range(len(features_scaled) - seq_length):
            X.append(features_scaled[i : i + seq_length])
            y.append(targets_scaled[i + seq_length])

        return np.array(X), np.array(y)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        epochs: int = 100,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ) -> Dict:
        """Train the market predictor.

        Args:
            X: Sequence data of shape [n_sequences, seq_length, n_features].
            y: Target data of shape [n_sequences, n_targets].
            batch_size: Batch size for training.
            epochs: Number of training epochs.
            learning_rate: Learning rate for optimizer.
            device: Device to train on ('cpu' or 'cuda').

        Returns:
            Dictionary containing training history.
        """
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        self.model = self.model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        history: Dict[str, List[float]] = {"loss": []}

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            history["loss"].append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Epoch [%d/%d], Loss: %.4f", epoch + 1, epochs, avg_loss
                )

        return history

    def predict(self, features: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Make predictions with the trained model.

        Args:
            features: Feature matrix of shape [seq_length, n_features] or
                [n_features].
            device: Device to run prediction on.

        Returns:
            Predicted values.

        Raises:
            RuntimeError: If :meth:`prepare_sequences` has not been called yet.
        """
        if not hasattr(self.scaler_X, "mean_"):
            raise RuntimeError("Call prepare_sequences() before predict()")

        self.model.eval()

        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        if features.shape[0] > self.seq_length:
            features = features[-self.seq_length :]

        if features.shape[0] < self.seq_length:
            padding = np.zeros(
                (self.seq_length - features.shape[0], features.shape[1])
            )
            features = np.vstack([padding, features])

        features_scaled = self.scaler_X.transform(features)
        features_tensor = (
            torch.FloatTensor(features_scaled).unsqueeze(0).to(device)
        )

        with torch.no_grad():
            pred_scaled = self.model(features_tensor).cpu().numpy()

        pred = self.scaler_y.inverse_transform(pred_scaled)
        return pred[0]


class LSTMPredictor(nn.Module):
    """
    LSTM-based predictor for time series forecasting.

    Attributes:
        hidden_dim: Hidden dimension for LSTM.
        num_layers: Number of LSTM layers.
        lstm: LSTM layer.
        fc: Fully connected output layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
    ) -> None:
        """Initialize the LSTM predictor.

        Args:
            input_dim: Input dimension (number of features).
            hidden_dim: Hidden dimension for LSTM.
            output_dim: Output dimension (number of indicators to predict).
            num_layers: Number of LSTM layers.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Note:
            Hidden states are re-initialised to zeros on every forward call
            (stateless inference).

        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim].

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
        out = self.fc(out)
        return out
