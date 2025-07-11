"""
Alternative GNN encoder architectures for architecture comparison study.

Provides GCN, Graph Transformer, and GIN encoders as drop-in replacements
for the GAT encoder in TemporalGNN.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, TransformerConv, global_mean_pool

logger = logging.getLogger(__name__)


class GCNEncoder(nn.Module):
    """Multi-layer GCN encoder (Kipf & Welling 2017).

    Fixed spectral filter — no learned attention weights.
    Simpler than GAT, serves as a baseline to test whether
    attention is genuinely beneficial.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(x, edge_index, edge_weight)
            h = bn(h)
            h = self.act(h)
            if i < len(self.convs) - 1:
                h = self.dropout(h)
            x = h
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(x, batch)


class GraphTransformerEncoder(nn.Module):
    """Graph Transformer encoder using TransformerConv from PyG.

    Multi-head attention with edge features and positional encoding.
    More expressive than GAT but also more expensive.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        head_dim = max(1, hidden_channels // heads)
        self.convs.append(TransformerConv(in_channels, head_dim, heads=heads,
                                           dropout=dropout, concat=True))
        self.bns.append(nn.BatchNorm1d(head_dim * heads))
        for _ in range(num_layers - 1):
            self.convs.append(TransformerConv(head_dim * heads, head_dim, heads=heads,
                                               dropout=dropout, concat=True))
            self.bns.append(nn.BatchNorm1d(head_dim * heads))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(x, edge_index)
            h = bn(h)
            h = self.act(h)
            if i < len(self.convs) - 1:
                h = self.dropout(h)
            x = h
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(x, batch)


class GINEncoder(nn.Module):
    """Graph Isomorphism Network encoder (Xu et al. 2019).

    Maximally powerful among 1-WL GNNs.
    Uses MLP for update function instead of simple linear + activation.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        def make_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )

        self.convs.append(GINConv(make_mlp(in_channels, hidden_channels)))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(make_mlp(hidden_channels, hidden_channels)))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(x, edge_index)
            h = bn(h)
            if i < len(self.convs) - 1:
                h = self.dropout(h)
            x = h
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return global_mean_pool(x, batch)


class MLPEncoder(nn.Module):
    """MLP baseline that ignores graph structure entirely.

    Flattens all node features and processes with a standard MLP.
    Serves as a non-graph baseline to quantify the value of
    graph structure in the prediction task.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 max_nodes: int = 100, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.max_nodes = max_nodes
        input_dim = in_channels * max_nodes
        layers = [nn.Linear(input_dim, hidden_channels), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # Pad/truncate node features to fixed size
        n_nodes = x.size(0)
        if batch is None:
            batch = torch.zeros(n_nodes, dtype=torch.long, device=x.device)

        # Process per graph in batch
        unique_batches = batch.unique()
        outputs = []
        for b in unique_batches:
            mask = batch == b
            node_feats = x[mask]  # (n_nodes_in_graph, in_channels)
            # Pad to max_nodes
            padded = torch.zeros(self.max_nodes, node_feats.size(1), device=x.device)
            n = min(node_feats.size(0), self.max_nodes)
            padded[:n] = node_feats[:n]
            flat = padded.reshape(1, -1)
            outputs.append(self.net(flat))

        return torch.cat(outputs, dim=0)


# ── Factory function ─────────────────────────────────────────────

ENCODER_REGISTRY = {
    "gcn": GCNEncoder,
    "transformer": GraphTransformerEncoder,
    "gin": GINEncoder,
    "mlp": MLPEncoder,
}


def create_encoder(encoder_type: str, in_channels: int, hidden_channels: int, **kwargs) -> nn.Module:
    """Create a GNN encoder by name.

    Args:
        encoder_type: One of 'gcn', 'gat', 'transformer', 'gin', 'mlp'.
        in_channels: Number of input node features.
        hidden_channels: Hidden dimension.
        **kwargs: Additional arguments (num_layers, heads, dropout, etc.)

    Returns:
        GNN encoder module outputting [num_graphs, hidden_channels].
    """
    if encoder_type == "gat":
        # Import from the main gnn_predictor to avoid circular deps
        from scr_financial.ml.gnn_predictor import GNNEncoder
        return GNNEncoder(in_channels, hidden_channels, **kwargs)

    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                         f"Available: {list(ENCODER_REGISTRY.keys()) + ['gat']}")

    cls = ENCODER_REGISTRY[encoder_type]

    # Filter kwargs to match the class __init__ signature
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(in_channels, hidden_channels, **valid_kwargs)
