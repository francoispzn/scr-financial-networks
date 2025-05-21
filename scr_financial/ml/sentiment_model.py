"""
Sentiment analysis model for financial market data.

This module provides an LSTM-based model for analyzing market sentiment
from financial time series data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm


class LSTMSentimentModel(nn.Module):
    """
    LSTM-based sentiment analysis model for financial time series.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    hidden_dim : int
        Dimension of hidden state
    num_layers : int
        Number of LSTM layers
    output_dim : int
        Dimension of output (typically 1 for sentiment score)
    dropout : float
        Dropout probability
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 output_dim: int = 1, dropout: float = 0.2):
        """Initialize the LSTM sentiment model."""
        super(LSTMSentimentModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_len, input_dim]
            
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
        
        # Apply dropout
        out = self.dropout(out)
        
        # Linear layer
        out = self.fc(out)
        
        return out


def prepare_sentiment_data(data: np.ndarray, target: np.ndarray, 
                           seq_length: int, train_split: float = 0.8) -> Tuple:
    """
    Prepare data for sentiment model training.
    
    Parameters
    ----------
    data : numpy.ndarray
        Feature data of shape [n_samples, n_features]
    target : numpy.ndarray
        Target values of shape [n_samples]
    seq_length : int
        Length of sequence for LSTM
    train_split : float
        Proportion of data to use for training
        
    Returns
    -------
    tuple
        Tuple containing (X_train, y_train, X_test, y_test)
    """
    # Create sequences
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    # Split into train and test
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    return X_train, y_train, X_test, y_test


def train_sentiment_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                         X_test: torch.Tensor, y_test: torch.Tensor,
                         batch_size: int = 64, epochs: int = 100,
                         learning_rate: float = 0.001, device: str = 'cpu') -> Dict:
    """
    Train the sentiment model.
    
    Parameters
    ----------
    model : nn.Module
        The model to train
    X_train : torch.Tensor
        Training features
    y_train : torch.Tensor
        Training targets
    X_test : torch.Tensor
        Test features
    y_test : torch.Tensor
        Test targets
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
    # Move model to device
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'test_loss': []
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Average training loss
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            history['test_loss'].append(test_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    return history


def predict_sentiment(model: nn.Module, features: np.ndarray, seq_length: int, device: str = 'cpu') -> float:
    """
    Predict sentiment score from features.
    
    Parameters
    ----------
    model : nn.Module
        Trained sentiment model
    features : numpy.ndarray
        Input features of shape [n_features] or [seq_length, n_features]
    seq_length : int
        Length of sequence for LSTM
    device : str
        Device to run prediction on
        
    Returns
    -------
    float
        Predicted sentiment score
    """
    model.eval()
    
    # Reshape features if needed
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    # Ensure we have the right sequence length
    if features.shape[0] < seq_length:
        raise ValueError(f"Features must have at least {seq_length} time steps")
    
    # Use the last seq_length time steps
    if features.shape[0] > seq_length:
        features = features[-seq_length:]
    
    # Convert to tensor and add batch dimension
    features = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        sentiment = model(features).item()
    
    return sentiment