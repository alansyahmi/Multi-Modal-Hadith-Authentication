"""
GNN Models for Narrator Reliability Classification & Structural Fusion (PO2).

Architectures:
  1. ReliabilityGCN: Multi-layer Graph Convolutional Network with classification head.
  2. ReliabilityGAT: Multi-layer Graph Attention Network with edge-attention weighting.
  3. ReliabilityFusionGNN: Multimodal fusion network combining structural isnād topology
     with AraBERT semantic reliability embeddings (concatenation or attention gating).

Classes: 5 ordinal reliability tiers (0: matruk, 1: daif, 2: maqbul, 3: saduq, 4: thiqah).
Primary Evaluation Metric: Macro-F1 (falsifiable hypothesis H1 for PO2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm

from .config import HIDDEN_CHANNELS, NUM_LAYERS, DROPOUT

# Ordinal grade values for mapping discrete class predictions to continuous scores
CLASS_TO_SCORE = {0: 0.05, 1: 0.15, 2: 0.40, 3: 0.70, 4: 0.85}
CLASS_NAMES = ["matruk", "daif", "maqbul", "saduq", "thiqah"]


class ReliabilityGCN(nn.Module):
    """Supervised Graph Convolutional Network for narrator reliability classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = HIDDEN_CHANNELS,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        num_classes: int = 5,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        # Classification head
        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, num_classes),
            )
        else:
            self.classifier = None

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        """Extract node structural representations [N, hidden_channels]."""
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            If return_embeddings is True: (logits, embeddings)
            Else: logits [N, num_classes] (or embeddings if num_classes == 0)
        """
        z = self.encode(x, edge_index, edge_attr)
        if self.classifier is None:
            return z

        logits = self.classifier(z)
        if return_embeddings:
            return logits, z
        return logits


class ReliabilityGAT(nn.Module):
    """Graph Attention Network variant for structural representation and classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = HIDDEN_CHANNELS,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        heads: int = 4,
        num_classes: int = 5,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Multi-head input layer
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        self.bns.append(BatchNorm(hidden_channels))

        # Intermediate hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1))
            self.bns.append(BatchNorm(hidden_channels))

        # Final GNN representation layer
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1))
            self.bns.append(BatchNorm(hidden_channels))

        # Classification head
        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, num_classes),
            )
        else:
            self.classifier = None

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        """Extract node attention representations [N, hidden_channels]."""
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, edge_index, edge_attr)
        if self.classifier is None:
            return z

        logits = self.classifier(z)
        if return_embeddings:
            return logits, z
        return logits


class ReliabilityFusionGNN(nn.Module):
    """Multimodal Fusion GNN combining transmission topology with AraBERT semantic embeddings.

    Supports:
      - fusion_type="concat": Concatenates structural and semantic representations.
      - fusion_type="attention": Cross-modal gating mechanism between graph and text.
    """

    def __init__(
        self,
        graph_in_channels: int = 8,
        semantic_in_channels: int = 768,
        hidden_channels: int = HIDDEN_CHANNELS,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        num_classes: int = 5,
        fusion_type: str = "concat",
        backbone: str = "gcn",
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.dropout = dropout

        # Graph encoder backbone
        if backbone == "gat":
            self.graph_encoder = ReliabilityGAT(
                in_channels=graph_in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
                num_classes=0,
            )
        else:
            self.graph_encoder = ReliabilityGCN(
                in_channels=graph_in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                dropout=dropout,
                num_classes=0,
            )

        # Semantic projection layer
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if fusion_type == "attention":
            # Gating mechanism: computes dynamic modality weight alpha in [0, 1]
            self.gate = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.Tanh(),
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid(),
            )
            classifier_in = hidden_channels
        else:
            # Simple concatenation
            classifier_in = hidden_channels * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(
        self,
        x_graph: torch.Tensor,
        edge_index: torch.Tensor,
        x_semantic: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z_graph = self.graph_encoder.encode(x_graph, edge_index, edge_attr)
        z_sem = self.semantic_proj(x_semantic)

        if self.fusion_type == "attention":
            combined = torch.cat([z_graph, z_sem], dim=-1)
            alpha = self.gate(combined)
            fused = alpha * z_graph + (1.0 - alpha) * z_sem
        else:
            fused = torch.cat([z_graph, z_sem], dim=-1)

        logits = self.classifier(fused)
        if return_embeddings:
            return logits, fused
        return logits


def build_model(
    in_channels: int,
    model_type: str = "gcn",
    num_classes: int = 5,
    semantic_in_channels: int | None = None,
    fusion_type: str = "concat",
    **kwargs,
) -> nn.Module:
    """Factory: build GNN model by type.

    Supported types:
      - "gcn": Graph Convolutional Network classifier (PO2 graph-only arm)
      - "gat": Graph Attention Network classifier
      - "fusion": Multimodal Fusion GNN (PO2 fusion arm)
    """
    model_type = model_type.lower()
    if model_type == "gat":
        return ReliabilityGAT(in_channels=in_channels, num_classes=num_classes, **kwargs)
    elif model_type in ("fusion", "fusion_concat", "fusion_attention"):
        ft = "attention" if "attention" in model_type else fusion_type
        sem_dim = semantic_in_channels or 768
        return ReliabilityFusionGNN(
            graph_in_channels=in_channels,
            semantic_in_channels=sem_dim,
            num_classes=num_classes,
            fusion_type=ft,
            **kwargs,
        )
    return ReliabilityGCN(in_channels=in_channels, num_classes=num_classes, **kwargs)
