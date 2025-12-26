"""PyTorch Lightning module for training and evaluation."""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn import metrics

from collaboration_prediction.data_modules.data import sample_negative_edges
from collaboration_prediction.model.model import (
    LinkPredictor,
    MessagePassingModel,
    NodeEncoder,
    PairEncoder,
)


def compute_hits(
    predictions: np.ndarray, labels: np.ndarray, k: int = 50, metric_precision: int = 4
) -> float:
    """Compute Hits@K metric.

    Args:
        predictions: Predicted scores
        labels: True labels
        k: Number of top predictions to consider

    Returns:
        Hits@K score
    """
    positive_predictions = predictions[labels == 1]
    negative_predictions = predictions[labels == 0]

    if len(positive_predictions) == 0:
        return 0.0
    if len(negative_predictions) == 0:
        return 1.0
    if len(negative_predictions) < k:
        k = len(negative_predictions)

    threshold = np.sort(negative_predictions)[-k]
    hits = np.sum(positive_predictions > threshold) / len(positive_predictions)
    return round(hits, metric_precision)


def compute_auroc(
    predictions: np.ndarray,
    labels: np.ndarray,
    auroc_default: float = 0.5,
    metric_precision: int = 4,
) -> float:
    """Compute AUROC metric.

    Args:
        predictions: Predicted scores
        labels: True labels

    Returns:
        AUROC score (0.5 if only one class is present)
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return auroc_default

    try:
        auroc = metrics.roc_auc_score(labels, predictions)
        return round(auroc, metric_precision)
    except ValueError:
        return auroc_default


class LinkPredictionLightningModule(pl.LightningModule):
    """PyTorch Lightning module for link prediction training and evaluation."""

    def __init__(
        self,
        features_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        factor_negative_edges: int = 1,
        hits_k: int = 50,
        pair_encoder_output_multiplier: int = 3,
        link_predictor_reduction_factor: int = 2,
        min_degree: float = 1.0,
        auroc_default: float = 0.5,
        metric_precision: int = 4,
        oversample_factor: float = 1.1,
    ):
        """Initialize the Lightning module.

        Args:
            features_dim: Dimension of input node features
            hidden_dim: Hidden dimension for the model
            num_layers: Number of message passing layers
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            factor_negative_edges: Factor for negative edge sampling during training
            hits_k: K value for Hits@K metric
            pair_encoder_output_multiplier: Multiplier for pair encoder output dimension
            link_predictor_reduction_factor: Reduction factor for link predictor
            min_degree: Minimum degree for normalization
            auroc_default: Default AUROC value when only one class is present
            metric_precision: Decimal places for metric rounding
            oversample_factor: Oversample factor for negative edge sampling
        """
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.factor_negative_edges = factor_negative_edges
        self.hits_k = hits_k
        self.min_degree = min_degree
        self.auroc_default = auroc_default
        self.metric_precision = metric_precision
        self.oversample_factor = oversample_factor

        node_encoder = NodeEncoder(features_dim, hidden_dim)
        pair_encoder = PairEncoder(output_multiplier=pair_encoder_output_multiplier)
        link_predictor = LinkPredictor(
            hidden_dim * pair_encoder_output_multiplier,
            reduction_factor=link_predictor_reduction_factor,
        )

        self.model = MessagePassingModel(
            node_encoder=node_encoder,
            pair_encoder=pair_encoder,
            link_predictor=link_predictor,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            min_degree=min_degree,
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.graph_data = None

    def setup(self, stage: str = None):
        """Set up method called by Lightning before training/validation/testing.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        # Graph data should be set via set_graph_data before training
        pass

    def set_graph_data(self, graph_data: dict):
        """Set graph data (adjacency matrix, node features, etc.).

        Args:
            graph_data: Dictionary containing graph data
        """
        self.graph_data = graph_data
        # Graph data will be moved to device in on_train_start/on_validation_start

    def _move_graph_data_to_device(self):
        """Move graph data to the current device."""
        if self.graph_data is not None:
            device = next(self.model.parameters()).device
            self.graph_data_device = {
                "edge_index": self.graph_data["edge_index"].to(device),
                "adj": self.graph_data["adj"].to(device),
                "node_features": self.graph_data["node_features"].to(device),
                "node_degrees": self.graph_data.get("node_degrees", None),
                "num_nodes": self.graph_data["num_nodes"],
                "existing_edges_set": self.graph_data.get("existing_edges_set", None),
            }
            if self.graph_data_device["node_degrees"] is not None:
                self.graph_data_device["node_degrees"] = self.graph_data_device["node_degrees"].to(
                    device
                )

    def on_train_start(self):
        """Move graph data to device when training starts."""
        self._move_graph_data_to_device()

    def on_validation_start(self):
        """Move graph data to device and initialize validation outputs list."""
        if not hasattr(self, "graph_data_device"):
            self._move_graph_data_to_device()
        self.validation_step_outputs = []

    def on_test_start(self):
        """Move graph data to device and initialize test outputs list."""
        if not hasattr(self, "graph_data_device"):
            self._move_graph_data_to_device()
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch: Batch of (positive_edges, positive_labels)
            batch_idx: Batch index

        Returns:
            Training loss
        """
        positive_edges, positive_labels = batch
        positive_edges = positive_edges.to(self.device)
        positive_labels = positive_labels.to(self.device)

        num_negative_edges = len(positive_edges) * self.factor_negative_edges
        negative_edges = sample_negative_edges(
            num_nodes=self.graph_data_device["num_nodes"],
            num_negative_edges=num_negative_edges,
            existing_edges_set=self.graph_data_device["existing_edges_set"],
            device=self.device,
            oversample_factor=self.oversample_factor,
        )
        negative_labels = torch.zeros(size=(len(negative_edges),), device=self.device)

        edges = torch.cat([positive_edges, negative_edges])
        labels = torch.cat([positive_labels, negative_labels]).view(-1, 1)

        outputs = self.model(self.graph_data_device, edges)
        loss = self.criterion(outputs, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validate a batch of data.

        Args:
            batch: Batch of (edges, labels)
            batch_idx: Batch index

        Returns:
            Dictionary with predictions and labels
        """
        if not hasattr(self, "graph_data_device"):
            self._move_graph_data_to_device()

        edges, labels = batch
        edges = edges.to(self.device)

        outputs = self.model(self.graph_data_device, edges)

        result = {"predictions": outputs.cpu().squeeze(), "labels": labels.cpu()}
        self.validation_step_outputs.append(result)

        return result

    def on_validation_epoch_end(self):
        """Compute validation metrics at end of epoch."""
        if not self.validation_step_outputs:
            return

        predictions = torch.cat(
            [out["predictions"] for out in self.validation_step_outputs]
        ).numpy()
        labels = torch.cat([out["labels"] for out in self.validation_step_outputs]).numpy()

        hits = compute_hits(
            predictions, labels, k=self.hits_k, metric_precision=self.metric_precision
        )
        auroc = compute_auroc(
            predictions,
            labels,
            auroc_default=self.auroc_default,
            metric_precision=self.metric_precision,
        )

        self.log("val/hits", hits, prog_bar=True)
        self.log("val/auroc", auroc, prog_bar=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step.

        Args:
            batch: Batch of (edges, labels)
            batch_idx: Batch index

        Returns:
            Dictionary with predictions and labels
        """
        if not hasattr(self, "graph_data_device"):
            self._move_graph_data_to_device()

        edges, labels = batch
        edges = edges.to(self.device)

        outputs = self.model(self.graph_data_device, edges)

        result = {"predictions": outputs.cpu().squeeze(), "labels": labels.cpu()}
        self.test_step_outputs.append(result)

        return result

    def on_test_epoch_end(self):
        """Compute test metrics at end of epoch."""
        if not self.test_step_outputs:
            return

        predictions = torch.cat([out["predictions"] for out in self.test_step_outputs]).numpy()
        labels = torch.cat([out["labels"] for out in self.test_step_outputs]).numpy()

        hits = compute_hits(
            predictions, labels, k=self.hits_k, metric_precision=self.metric_precision
        )
        auroc = compute_auroc(
            predictions,
            labels,
            auroc_default=self.auroc_default,
            metric_precision=self.metric_precision,
        )

        self.log("test/hits", hits, prog_bar=True)
        self.log("test/auroc", auroc, prog_bar=True)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer.

        Returns:
            Optimizer configuration
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
