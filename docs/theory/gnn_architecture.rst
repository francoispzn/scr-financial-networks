GNN Architecture
================

Overview
--------

The temporal Graph Neural Network predicts spectral properties of the interbank network from sequences of daily graph snapshots.

Architecture
------------

.. code-block:: text

   Input: sequence of T daily graph snapshots (10 banks, 5 node features)
       |
   [GATConv (multi-head) -> BatchNorm -> ELU -> Dropout] x L layers
       |
   global_mean_pool -> graph-level embedding per snapshot
       |
   LSTM (M layers) over T time steps
       |
   FC -> [lambda_2, spectral_gap, spectral_radius]

Components
----------

**GAT Encoder** (per snapshot):

- Multi-layer Graph Attention Network (GATConv, multi-head)
- Each layer: GATConv -> BatchNorm -> ELU -> Dropout
- Aggregates neighbor features through actual network edges
- Output: node embeddings [N, hidden_dim]

**Graph Readout**:

- ``global_mean_pool``: averages node embeddings to produce a single graph-level vector
- Output: [1, hidden_dim] per snapshot

**Temporal Module**:

- LSTM processes the sequence of graph embeddings
- Captures how the network structure evolves over time
- Output: final hidden state

**Prediction Head**:

- Fully connected layer maps LSTM output to 3 spectral targets
- Targets: algebraic connectivity (lambda_2), spectral gap, spectral radius

Default Configuration
---------------------

- ``hidden_dim``: 64
- ``num_gcn_layers``: 3
- ``num_lstm_layers``: 2
- ``dropout``: 0.1
- ``seq_len``: 10 (number of graph snapshots per sequence)
- Learning rate: 3e-3 with cosine annealing
- Gradient clipping: max norm 1.0

Node Features
-------------

Each bank node has 5 features computed from daily market data:

1. **Volatility**: rolling standard deviation of returns
2. **Mean return**: rolling average daily return
3. **Log price**: log-transformed stock price
4. **Beta proxy**: rolling covariance with market / market variance
5. **Momentum**: cumulative return over lookback window

Edge Construction
-----------------

- Rolling Pearson correlation of daily returns between bank pairs
- Threshold: edges created where correlation > 0.3
- Edge weights: correlation values (stronger correlation = stronger link)

Training Data
-------------

- Source: 10 European bank stocks via yfinance (3 years of daily data)
- ~1000+ daily graph snapshots
- 80/20 train/test split
- Batched graph encoding via PyTorch Geometric ``Batch.from_data_list()``
