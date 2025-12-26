"""Neural network model components for link prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeEncoder(nn.Module):
    """Encodes node features into hidden representations."""

    def __init__(self, features_dim: int, hidden_dim: int):
        """Initialize node encoder.

        Args:
            features_dim: Input feature dimension
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(features_dim),
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through node encoder.

        Args:
            features: Node features [num_nodes, features_dim]

        Returns:
            Encoded node features [num_nodes, hidden_dim]
        """
        outputs = self.layers(features)
        return outputs


class PairEncoder(nn.Module):
    """Encodes pairs of node embeddings into edge representations."""

    def __init__(self, output_multiplier: int = 3):
        """Initialize pair encoder.

        Args:
            output_multiplier: Multiplier for output dimension
                (default: 3 for source, target, similarity)
        """
        super().__init__()
        self.output_multiplier = output_multiplier

    def forward(self, source_embeds: torch.Tensor, target_embeds: torch.Tensor) -> torch.Tensor:
        """Encode source and target node embeddings into edge representation.

        Args:
            source_embeds: Source node embeddings [num_edges, hidden_dim]
            target_embeds: Target node embeddings [num_edges, hidden_dim]

        Returns:
            Edge embeddings [num_edges, hidden_dim * output_multiplier]
        """
        features_to_concat = []

        if self.output_multiplier >= 1:
            features_to_concat.append(source_embeds)
        if self.output_multiplier >= 2:
            features_to_concat.append(target_embeds)
        if self.output_multiplier >= 3:
            features_to_concat.append(source_embeds * target_embeds)

        return torch.concat(features_to_concat, dim=-1)


class LinkPredictor(nn.Module):
    """Predicts link probability from edge embeddings."""

    def __init__(self, input_dim: int, reduction_factor: int = 2):
        """Initialize link predictor.

        Args:
            input_dim: Input dimension (edge embedding dimension)
            reduction_factor: Factor by which to reduce dimension in hidden layer
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // reduction_factor),
            nn.ReLU(),
            nn.LayerNorm(input_dim // reduction_factor),
            nn.Linear(input_dim // reduction_factor, 1),
        )

    def forward(self, edge_embeds: torch.Tensor) -> torch.Tensor:
        """Predict link probability from edge embeddings.

        Args:
            edge_embeds: Edge embeddings [num_edges, input_dim]

        Returns:
            Link predictions [num_edges, 1]
        """
        return self.layers(edge_embeds)


class SAGEConv(nn.Module):
    """GraphSAGE convolution layer with mean aggregation."""

    def __init__(self, in_dim: int, out_dim: int):
        """Initialize SAGEConv layer.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.sparse.FloatTensor,
        node_degrees: torch.Tensor = None,
        min_degree: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass through GraphSAGE convolution.

        Args:
            x: Node features [num_nodes, in_dim]
            adj: Sparse adjacency matrix [num_nodes, num_nodes]
            node_degrees: Pre-computed node degrees [num_nodes] (optional, for efficiency)
            min_degree: Minimum degree for normalization (to avoid division by zero)

        Returns:
            Output features [num_nodes, out_dim]
        """
        # Mean aggregation
        out = torch.sparse.mm(adj, x)

        if node_degrees is not None:
            row_sum = node_degrees
        else:
            row_sum = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=min_degree)

        out = out / row_sum.unsqueeze(1)

        out = self.linear(out)
        return out


class MessagePassingModel(nn.Module):
    """Graph neural network model for link prediction using message passing."""

    def __init__(
        self,
        node_encoder: NodeEncoder,
        pair_encoder: PairEncoder,
        link_predictor: LinkPredictor,
        num_layers: int,
        hidden_dim: int,
        dropout: float = 0.2,
        min_degree: float = 1.0,
    ):
        """Initialize message passing model.

        Args:
            node_encoder: Node encoder module
            pair_encoder: Pair encoder module
            link_predictor: Link predictor module
            num_layers: Number of message passing layers
            hidden_dim: Hidden dimension
            dropout: Dropout probability
            min_degree: Minimum node degree for normalization
        """
        super().__init__()
        self.node_encoder = node_encoder
        self.pair_encoder = pair_encoder
        self.link_predictor = link_predictor
        self.dropout = dropout
        self.min_degree = min_degree

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, data: dict, edges: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            data: Dictionary containing 'node_features', 'adj', and optionally 'node_degrees'
            edges: Edge indices [num_edges, 2]

        Returns:
            Link predictions [num_edges, 1]
        """
        feat = data["node_features"]
        adj = data["adj"]
        node_degrees = data.get("node_degrees", None)

        node_embeds = self.node_encoder(feat)

        for conv, norm in zip(self.convs, self.norms):
            node_embeds_in = node_embeds
            node_embeds = conv(
                node_embeds, adj, node_degrees=node_degrees, min_degree=self.min_degree
            )
            node_embeds = norm(node_embeds)
            node_embeds = F.relu(node_embeds)
            node_embeds = F.dropout(node_embeds, p=self.dropout, training=self.training)

            if node_embeds.shape == node_embeds_in.shape:
                node_embeds = node_embeds + node_embeds_in

        src_idx = edges[:, 0]
        dst_idx = edges[:, 1]
        src_embeds = node_embeds[src_idx]
        dst_embeds = node_embeds[dst_idx]
        edge_embeds = self.pair_encoder(src_embeds, dst_embeds)
        outputs = self.link_predictor(edge_embeds)
        return outputs
