"""
GNN Model for Narrator Reliability Scoring.

Architecture: Graph Convolutional Network (GCN) with:
  - 3 GCNConv layers (64 → 64 → 64 hidden)
  - BatchNorm + ReLU + Dropout at each layer
  - Final linear head: 64 → 1 with sigmoid output

Predicts a scalar reliability score (0-1) per narrator node.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm

from .config import HIDDEN_CHANNELS, NUM_LAYERS, DROPOUT


class ReliabilityGCN(nn.Module):
    """GCN for narrator structural representation learning (Graph Autoencoder encoder)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = HIDDEN_CHANNELS,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass.

        Args:
            x: Node features [N, F_in]
            edge_index: Adjacency [2, E]
            edge_attr: Edge features [E, F_edge]
        Returns:
            z: Node structural embeddings [N, hidden_channels]
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class ReliabilityGAT(nn.Module):
    """Graph Attention Network variant for structural representation learning."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = HIDDEN_CHANNELS,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        heads: int = 4,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer (multi-head)
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers (single head)
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1))
            self.bns.append(BatchNorm(hidden_channels))

        # Final GNN layer
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1))
            self.bns.append(BatchNorm(hidden_channels))

    def forward(self, x, edge_index, edge_attr=None):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


def build_model(
    in_channels: int,
    model_type: str = "gcn",
    **kwargs,
) -> nn.Module:
    """Factory: build a GNN encoder model by type."""
    if model_type == "gat":
        return ReliabilityGAT(in_channels, **kwargs)
    return ReliabilityGCN(in_channels, **kwargs)

