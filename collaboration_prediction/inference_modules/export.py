"""Model export utilities for ONNX."""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from collaboration_prediction.data_modules.data import load_dataset

logger = logging.getLogger(__name__)

try:
    import onnx
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    onnx = None
    ort = None


class ONNXSAGEConv(nn.Module):
    """ONNX-compatible SAGEConv that uses edge_index instead of sparse matrix."""

    def __init__(self, in_dim: int, out_dim: int):
        """Initialize ONNX SAGEConv layer.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_degrees: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using edge_index for message passing.

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]
            node_degrees: Node degrees [num_nodes]

        Returns:
            Output features [num_nodes, out_dim]
        """
        num_nodes = x.shape[0]
        out_dim = x.shape[1]

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        neighbor_features = x[src_idx]

        # Aggregate messages from neighbors
        # Sort by destination to facilitate group-wise operations
        sorted_indices = torch.argsort(dst_idx)
        sorted_dst = dst_idx[sorted_indices]
        sorted_features = neighbor_features[sorted_indices]

        # Boundaries: True where a new group starts (first element is always True)
        # diff_mask has length num_edges - 1
        diff_mask = sorted_dst[1:] != sorted_dst[:-1]

        # boundaries has length num_edges
        # first element is always True (start of first group)
        first_boundary = torch.ones(1, device=x.device, dtype=torch.bool)
        boundaries = torch.cat([first_boundary, diff_mask])

        # Get unique destinations (at boundaries)
        unique_dst_indices = torch.nonzero(boundaries, as_tuple=False).squeeze(1)
        unique_dst = sorted_dst[unique_dst_indices]

        # Sum features for each unique destination group using cumsum and take differences
        # This avoids index_add with duplicates and is ONNX-compatible

        # Compute cumulative sum of features
        cumsum_features = torch.cumsum(sorted_features, dim=0)

        # Group end indices: one before each next start, plus the last index
        # unique_dst_indices[1:] are starts of 2nd, 3rd, ... groups
        # Subtracting 1 gives ends of 1st, 2nd, ... (k-1)th groups
        last_idx = (sorted_dst.shape[0] - 1) * torch.ones(
            1, device=sorted_dst.device, dtype=torch.long
        )
        group_end_indices = torch.cat([unique_dst_indices[1:] - 1, last_idx])

        # Get cumulative sums at group ends
        cumsum_at_ends = cumsum_features[group_end_indices]
        # Get cumulative sums at group starts (previous group's end, or 0 for first)
        cumsum_at_starts = torch.cat(
            [torch.zeros(1, out_dim, dtype=x.dtype, device=x.device), cumsum_at_ends[:-1]]
        )
        # Difference gives us group sums
        group_sums = cumsum_at_ends - cumsum_at_starts

        # Now scatter to output (unique_dst has no duplicates, so we can use direct indexing)
        out = torch.zeros(num_nodes, out_dim, dtype=x.dtype, device=x.device)
        out[unique_dst] = group_sums

        row_sum = node_degrees.unsqueeze(1)
        out = out / row_sum

        out = self.linear(out)
        return out


class ONNXMessagePassingModel(nn.Module):
    """ONNX-compatible version of MessagePassingModel using edge_index."""

    def __init__(self, model: nn.Module):
        """Initialize ONNX model wrapper.

        Args:
            model: The original MessagePassingModel
        """
        super().__init__()
        self.node_encoder = model.node_encoder
        self.pair_encoder = model.pair_encoder
        self.link_predictor = model.link_predictor
        self.dropout = model.dropout
        self.min_degree = model.min_degree

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for conv, norm in zip(model.convs, model.norms):
            in_dim = conv.linear.in_features
            out_dim = conv.linear.out_features
            onnx_conv = ONNXSAGEConv(in_dim, out_dim)
            onnx_conv.linear.weight.data = conv.linear.weight.data.clone()
            onnx_conv.linear.bias.data = conv.linear.bias.data.clone()
            self.convs.append(onnx_conv)
            self.norms.append(norm)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edges: torch.Tensor,
        node_degrees: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with edge_index.

        Args:
            node_features: Node features [num_nodes, features_dim]
            edge_index: Edge indices [2, num_edges]
            edges: Edge indices for prediction [num_edges, 2]
            node_degrees: Node degrees [num_nodes]

        Returns:
            Link predictions [num_edges, 1]
        """
        node_embeds = self.node_encoder(node_features)

        for conv, norm in zip(self.convs, self.norms):
            node_embeds_in = node_embeds
            node_embeds = conv(node_embeds, edge_index, node_degrees)
            node_embeds = norm(node_embeds)
            node_embeds = torch.relu(node_embeds)
            node_embeds = torch.dropout(node_embeds, p=self.dropout, train=self.training)

            node_embeds = node_embeds + node_embeds_in

        src_idx = edges[:, 0]
        dst_idx = edges[:, 1]
        src_embeds = node_embeds[src_idx]
        dst_embeds = node_embeds[dst_idx]
        edge_embeds = self.pair_encoder(src_embeds, dst_embeds)
        outputs = self.link_predictor(edge_embeds)
        return outputs


class ONNXModelWrapper(nn.Module):
    """Wrapper for MessagePassingModel to make it ONNX-compatible.

    Uses edge_index instead of sparse adjacency matrix.
    """

    def __init__(self, model: nn.Module):
        """Initialize ONNX wrapper.

        Args:
            model: The MessagePassingModel to wrap
        """
        super().__init__()
        self.onnx_model = ONNXMessagePassingModel(model)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edges: torch.Tensor,
        node_degrees: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with edge_index.

        Args:
            node_features: Node features [num_nodes, features_dim]
            edge_index: Edge indices [2, num_edges]
            edges: Edge indices for prediction [num_edges, 2]
            node_degrees: Node degrees [num_nodes]

        Returns:
            Link probabilities [num_edges, 1]
        """
        logits = self.onnx_model(node_features, edge_index, edges, node_degrees)
        return torch.sigmoid(logits)


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    example_data: Dict[str, torch.Tensor],
    opset_version: int = 14,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
) -> Path:
    """Export PyTorch model to ONNX format.

    Args:
        model: The PyTorch model to export
        output_path: Path where to save the ONNX model
        example_data: Dictionary with example inputs:
            - node_features: [num_nodes, features_dim]
            - adj: Sparse adjacency matrix
            - edges: [num_edges, 2]
            - node_degrees (optional): [num_nodes]
        opset_version: ONNX opset version
        dynamic_axes: Dictionary specifying dynamic axes (e.g., {'edges': {0: 'num_edges'}})

    Returns:
        Path to the exported ONNX model
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX is not installed. Install it with: uv sync --extra export")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapped_model = ONNXModelWrapper(model)
    wrapped_model.eval()

    node_features = example_data["node_features"]
    edge_index = example_data["edge_index"]
    edges = example_data["edges"]
    node_degrees = example_data.get("node_degrees", None)

    if node_degrees is None:
        num_nodes = node_features.shape[0]
        node_degrees = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
        node_degrees = node_degrees.index_add_(
            0,
            edge_index[1],
            torch.ones(edge_index.shape[1], dtype=torch.float32, device=edge_index.device),
        )
        node_degrees = node_degrees.clamp(min=1.0)

    inputs = (node_features, edge_index, edges, node_degrees)

    input_names = [
        "node_features",
        "edge_index",
        "edges",
        "node_degrees",
    ]

    output_names = ["predictions"]

    torch.onnx.export(
        wrapped_model,
        inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
        dynamic_axes=dynamic_axes,
        dynamo=False,  # Use legacy exporter for sparse tensor support
    )

    try:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        model_size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"ONNX model exported successfully to {output_path}")
        logger.info(f"Model size: {model_size_mb:.2f} MB")
    except Exception as e:
        logger.warning(f"ONNX model verification failed: {e}")

    return output_path


def load_example_data_for_export(
    checkpoint_path: str,
    data_root: str,
    dataset_name: str,
    make_undirected: bool = True,
    add_self_loops: bool = True,
    min_degree_clamp: int = 1,
    structural_features_cfg: Optional[Dict] = None,
) -> Dict[str, torch.Tensor]:
    """Load example data for model export.

    This function loads a small sample of the dataset to use as example
    inputs for ONNX export.

    Args:
        checkpoint_path: Path to the model checkpoint (unused, kept for compatibility)
        data_root: Root directory for data
        dataset_name: Name of the dataset
        make_undirected: Whether to make graph undirected
        add_self_loops: Whether to add self-loops
        min_degree_clamp: Minimum degree for clamping
        structural_features_cfg: Configuration for structural features

    Returns:
        Dictionary with example inputs for export
    """
    graph_data = load_dataset(
        name=dataset_name,
        root=data_root,
        make_undirected=make_undirected,
        add_self_loops=add_self_loops,
        min_degree_clamp=min_degree_clamp,
        structural_features_cfg=structural_features_cfg,
    )

    edge_index = graph_data["edge_index"]
    if edge_index.shape[1] == 0:
        raise ValueError(
            f"Graph has no edges. Cannot create example data for export. "
            f"Dataset: {dataset_name}, Root: {data_root}"
        )

    num_example_edges = 1
    example_edges = edge_index[:, :num_example_edges].T

    return {
        "node_features": graph_data["node_features"],
        "edge_index": graph_data["edge_index"],
        "adj": graph_data["adj"],
        "edges": example_edges,
        "node_degrees": graph_data.get("node_degrees", None),
    }


def verify_onnx_model(
    onnx_path: str,
    pytorch_model: nn.Module,
    example_data: Dict[str, torch.Tensor],
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """Verify that ONNX model produces similar outputs to PyTorch model.

    Args:
        onnx_path: Path to the ONNX model
        pytorch_model: The original PyTorch model
        example_data: Dictionary with example inputs (same as used for export)
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if outputs match within tolerance, False otherwise
    """
    if not ONNX_AVAILABLE:
        raise ImportError("onnxruntime is not installed. Install it with: uv sync --extra export")

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    node_features = example_data["node_features"].numpy()
    edge_index = example_data["edge_index"].numpy()
    edges = example_data["edges"].numpy()
    node_degrees = example_data.get("node_degrees", None)
    if node_degrees is None:
        num_nodes = node_features.shape[0]
        node_degrees_tensor = torch.zeros(num_nodes, dtype=torch.float32)
        node_degrees_tensor = node_degrees_tensor.index_add_(
            0,
            torch.from_numpy(edge_index[1]),
            torch.ones(edge_index.shape[1], dtype=torch.float32),
        )
        node_degrees = node_degrees_tensor.clamp(min=1.0).numpy()
    else:
        node_degrees = node_degrees.numpy()

    node_features = node_features.astype(np.float32)
    edge_index = edge_index.astype(np.int64)
    edges = edges.astype(np.int64)
    node_degrees = node_degrees.astype(np.float32)

    logger.debug("Input data info:")
    logger.debug(
        f"  node_features: shape={node_features.shape}, "
        f"range=[{node_features.min():.6f}, {node_features.max():.6f}]"
    )
    logger.debug(
        f"  edge_index: shape={edge_index.shape}, "
        f"range=[{edge_index.min()}, {edge_index.max()}]"
    )
    logger.debug(f"  edges: shape={edges.shape}, range=[{edges.min()}, {edges.max()}]")
    logger.debug(
        f"  node_degrees: shape={node_degrees.shape}, "
        f"range=[{node_degrees.min():.6f}, {node_degrees.max():.6f}]"
    )

    pytorch_model.eval()

    original_device = next(pytorch_model.parameters()).device
    pytorch_model_cpu = pytorch_model.cpu()

    try:
        with torch.no_grad():
            wrapped_model = ONNXModelWrapper(pytorch_model_cpu)
            wrapped_model.eval()

            pytorch_inputs = (
                torch.from_numpy(node_features).float(),
                torch.from_numpy(edge_index).long(),
                torch.from_numpy(edges).long(),
                torch.from_numpy(node_degrees).float(),
            )
            pytorch_inputs = tuple(tensor.cpu() for tensor in pytorch_inputs)
            pytorch_output = wrapped_model(*pytorch_inputs).numpy()
    finally:
        if original_device.type == "cuda":
            pytorch_model_cpu.to(original_device)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    onnx_inputs = {
        "node_features": node_features.astype(np.float32),
        "edge_index": edge_index.astype(np.int64),
        "edges": edges.astype(np.int64),
        "node_degrees": node_degrees.astype(np.float32),
    }

    onnx_outputs = session.run(None, onnx_inputs)
    onnx_output = onnx_outputs[0]

    if pytorch_output.shape != onnx_output.shape:
        logger.error(f"Shape mismatch: PyTorch {pytorch_output.shape} vs ONNX {onnx_output.shape}")
        return False

    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()

    logger.debug("Debug info:")
    logger.debug(
        f"  PyTorch output range: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]"
    )
    logger.debug(f"  ONNX output range: [{onnx_output.min():.6f}, {onnx_output.max():.6f}]")
    logger.debug(
        f"  PyTorch output mean: {pytorch_output.mean():.6f}, std: {pytorch_output.std():.6f}"
    )
    logger.debug(f"  ONNX output mean: {onnx_output.mean():.6f}, std: {onnx_output.std():.6f}")

    if np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol):
        logger.info("ONNX model verification passed!")
        logger.info(f"  Max difference: {max_diff:.6f}")
        logger.info(f"  Mean difference: {mean_diff:.6f}")
        logger.info(f"  Output shape: {onnx_output.shape}")
        return True
    else:
        logger.error("ONNX model verification failed!")
        logger.error(f"  Max difference: {max_diff:.6f} (tolerance: {atol})")
        logger.error(f"  Mean difference: {mean_diff:.6f}")
        logger.error(f"  Output shape: {onnx_output.shape}")
        logger.error(f"  Relative tolerance: {rtol}, Absolute tolerance: {atol}")

        diff = np.abs(pytorch_output - onnx_output)
        top_5_indices = np.argsort(diff.flatten())[-5:][::-1]
        logger.debug("  Top 5 differences:")
        for idx in top_5_indices:
            flat_idx = np.unravel_index(idx, pytorch_output.shape)
            logger.debug(
                f"    [{flat_idx}]: PyTorch={pytorch_output[flat_idx]:.6f}, "
                f"ONNX={onnx_output[flat_idx]:.6f}, diff={diff[flat_idx]:.6f}"
            )

        return False
