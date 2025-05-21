"""
Predictors for financial network analysis.

This module provides predictors for forecasting network evolution
and market indicators.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet


class NetworkPredictor:
    """
    Predictor for financial network evolution.
    
    This class implements methods for predicting how the financial network
    will evolve over time based on historical data and external factors.
    
    Parameters
    ----------
    method : str
        Prediction method ('rf' for Random Forest, 'gb' for Gradient Boosting,
        'elastic' for Elastic Net)
    """
    
    def __init__(self, method: str = 'rf'):
        """Initialize the network predictor."""
        self.method = method
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Initialize model based on method
        if method == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif method == 'gb':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif method == 'elastic':
            self.model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def prepare_features(self, network_history: List[Dict], system_indicators: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for network prediction.
        
        Parameters
        ----------
        network_history : list of dict
            List of network states over time
        system_indicators : pandas.DataFrame
            DataFrame of system-wide indicators
            
        Returns
        -------
        tuple
            Tuple containing (X, y) where X is features and y is target values
        """
        # Extract features and targets
        X = []
        y = []
        
        for i in range(1, len(network_history)):
            prev_state = network_history[i-1]
            curr_state = network_history[i]
            
            # Extract network features
            features = []
            
            # Network density
            prev_density = prev_state.get('network_density', 0)
            features.append(prev_density)
            
            # Average degree
            prev_avg_degree = prev_state.get('avg_degree', 0)
            features.append(prev_avg_degree)
            
            # Add system indicators
            time_point = prev_state.get('time', 0)
            if time_point in system_indicators.index:
                for col in system_indicators.columns:
                    features.append(system_indicators.loc[time_point, col])
            
            # Target: current network density
            target = curr_state.get('network_density', 0)
            
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the predictor to training data.
        
        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Target values
        """
        # Scale features and targets
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Fit model
        self.model.fit(X_scaled, y_scaled)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the fitted model.
        
        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix
            
        Returns
        -------
        numpy.ndarray
            Predicted values
        """
        # Scale features
        X_scaled = self.scaler_X.transform(X)
        
        # Make predictions
        y_scaled_pred = self.model.predict(X_scaled)
        
        # Inverse transform predictions
        y_pred = self.scaler_y.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()
        
        return y_pred
    
    def predict_network_evolution(self, current_state: Dict, system_forecast: pd.DataFrame, 
                                 steps: int = 10) -> List[Dict]:
        """
        Predict network evolution over multiple time steps.
        
        Parameters
        ----------
        current_state : dict
            Current network state
        system_forecast : pandas.DataFrame
            Forecast of system indicators
        steps : int
            Number of steps to predict
            
        Returns
        -------
        list
            List of predicted network states
        """
        predictions = []
        state = current_state.copy()
        
        for i in range(steps):
            # Extract features from current state and system forecast
            features = []
            
            # Network features
            features.append(state.get('network_density', 0))
            features.append(state.get('avg_degree', 0))
            
            # System indicators
            time_point = state.get('time', 0) + 1
            if time_point in system_forecast.index:
                for col in system_forecast.columns:
                    features.append(system_forecast.loc[time_point, col])
            else:
                # Use last available values
                for col in system_forecast.columns:
                    features.append(system_forecast.iloc[-1][col])
            
            # Make prediction
            pred = self.predict(np.array([features]))
            
            # Update state
            new_state = state.copy()
            new_state['time'] = time_point
            new_state['network_density'] = pred[0]
            
            # Add to predictions
            predictions.append(new_state)
            
            # Update current state for next iteration
            state = new_state
        
        return predictions


class MarketPredictor:
    """
    Predictor for market indicators.
    
    This class implements a deep learning model for predicting market indicators
    based on historical data and network features.
    
    Parameters
    ----------
    input_dim : int
        Input dimension (number of features)
    hidden_dim : int
        Hidden dimension for LSTM
    output_dim : int
        Output dimension (number of indicators to predict)
    num_layers : int
        Number of LSTM layers
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        """Initialize the market predictor."""
        self.model = LSTMPredictor(input_dim, hidden_dim, output_dim, num_layers)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.seq_length = 10  # Default sequence length
    
    def prepare_sequences(self, features: np.ndarray, targets: np.ndarray, 
                         seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Parameters
        ----------
        features : numpy.ndarray
            Feature matrix of shape [n_samples, n_features]
        targets : numpy.ndarray
            Target matrix of shape [n_samples, n_targets]
        seq_length : int
            Length of sequences
            
        Returns
        -------
        tuple
            Tuple containing (X, y) where X is of shape [n_sequences, seq_length, n_features]
            and y is of shape [n_sequences, n_targets]
        """
        self.seq_length = seq_length
        
        # Scale features and targets
        features_scaled = self.scaler_X.fit_transform(features)
        targets_scaled = self.scaler_y.fit_transform(targets)
        
        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - seq_length):
            X.append(features_scaled[i:i+seq_length])
            y.append(targets_scaled[i+seq_length])
        
        return np.array(X), np.array(y)
    
    def train(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, 
             epochs: int = 100, learning_rate: float = 0.001, device: str = 'cpu') -> Dict:
        """
        Train the market predictor.
        
        Parameters
        ----------
        X : numpy.ndarray
            Sequence data of shape [n_sequences, seq_length, n_features]
        y : numpy.ndarray
            Target data of shape [n_sequences, n_targets]
        batch_size : int
            Batch size for training
        epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate for optimizer
        device : str
            Device to train on ('cpu' or 'cuda')
            
        Returns
        -------
        dict
            Dictionary containing training history
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'loss': []
        }
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Average loss
            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        return history
    
    def predict(self, features: np.ndarray, device: str = 'cpu') -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Parameters
        ----------
        features : numpy.ndarray
            Feature matrix of shape [seq_length, n_features] or [n_features]
        device : str
            Device to run prediction on
            
        Returns
        -------
        numpy.ndarray
            Predicted values
        """
        self.model.eval()
        
        # Ensure features has the right shape
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # If we have more than seq_length samples, use only the last seq_length
        if features.shape[0] > self.seq_length:
            features = features[-self.seq_length:]
        
        # If we have less than seq_length samples, pad with zeros
        if features.shape[0] < self.seq_length:
            padding = np.zeros((self.seq_length - features.shape[0], features.shape[1]))
            features = np.vstack([padding, features])
        
        # Scale features
        features_scaled = self.scaler_X.transform(features)
        
        # Convert to tensor and add batch dimension
        features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            pred_scaled = self.model(features_tensor).cpu().numpy()
        
        # Inverse transform prediction
        pred = self.scaler_y.inverse_transform(pred_scaled)
        
        return pred[0]


class LSTMPredictor(nn.Module):
    """
    LSTM-based predictor for time series forecasting.
    
    Parameters
    ----------
    input_dim : int
        Input dimension (number of features)
    hidden_dim : int
        Hidden dimension for LSTM
    output_dim : int
        Output dimension (number of indicators to predict)
    num_layers : int
        Number of LSTM layers
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        """Initialize the LSTM predictor."""
        super(LSTMPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_length, input_dim]
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, output_dim]
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from last time step
        out = out[:, -1, :]
        
        # Linear layer
        out = self.fc(out)
        
        return out
