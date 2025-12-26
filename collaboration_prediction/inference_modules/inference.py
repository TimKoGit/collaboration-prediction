"""Inference server using FastAPI and Triton."""

import logging
import shutil
from pathlib import Path

import numpy as np
import onnx

logger = logging.getLogger(__name__)

try:
    import tritonclient.http as httpclient
    from tritonclient import utils as triton_utils

    TRITON_CLIENT_AVAILABLE = True
except ImportError:
    TRITON_CLIENT_AVAILABLE = False
    httpclient = None
    triton_utils = None


class TritonInferenceClient:
    """Client for Triton Inference Server."""

    def __init__(self, url: str = "localhost:8000", model_name: str = "link_prediction"):
        """Initialize Triton client.

        Args:
            url: Triton server URL
            model_name: Name of the model in Triton
        """
        if not TRITON_CLIENT_AVAILABLE:
            raise ImportError(
                "tritonclient is not installed. Install it with: uv sync --extra inference"
            )

        self.url = url
        self.model_name = model_name
        self.client = httpclient.InferenceServerClient(url=url, verbose=False)

    def predict(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edges: np.ndarray,
        node_degrees: np.ndarray,
    ) -> np.ndarray:
        """Run inference using Triton server.

        Args:
            node_features: Node features [num_nodes, features_dim]
            edge_index: Edge indices [2, num_edges]
            edges: Edge pairs for prediction [num_edges, 2]
            node_degrees: Node degrees [num_nodes]

        Returns:
            Predictions [num_edges, 1]
        """
        inputs = [
            httpclient.InferInput("node_features", node_features.shape, "FP32"),
            httpclient.InferInput("edge_index", edge_index.shape, "INT64"),
            httpclient.InferInput("edges", edges.shape, "INT64"),
            httpclient.InferInput("node_degrees", node_degrees.shape, "FP32"),
        ]

        inputs[0].set_data_from_numpy(node_features.astype(np.float32))
        inputs[1].set_data_from_numpy(edge_index.astype(np.int64))
        inputs[2].set_data_from_numpy(edges.astype(np.int64))
        inputs[3].set_data_from_numpy(node_degrees.astype(np.float32))

        outputs = [httpclient.InferRequestedOutput("predictions")]

        response = self.client.infer(self.model_name, inputs, outputs=outputs)
        predictions = response.as_numpy("predictions")
        return predictions


def prepare_triton_model_repository(
    onnx_model_path: str,
    model_repository: str,
    model_name: str = "link_prediction",
    version: int = 1,
) -> Path:
    """Prepare Triton model repository structure.

    Args:
        onnx_model_path: Path to ONNX model
        model_repository: Path to Triton model repository
        model_name: Name of the model
        version: Model version

    Returns:
        Path to the model directory
    """
    onnx_path = Path(onnx_model_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    model = onnx.load(str(onnx_path))
    input_info = {}
    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            shape.append(dim.dim_value)
        input_info[inp.name] = shape

    output_info = {}
    for out in model.graph.output:
        shape = []
        for dim in out.type.tensor_type.shape.dim:
            shape.append(dim.dim_value)
        output_info[out.name] = shape

    repo_path = Path(model_repository)
    model_dir = repo_path / model_name
    version_dir = model_dir / str(version)

    version_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(onnx_path, version_dir / "model.onnx")

    def format_dims(dims):
        return "[" + ", ".join(str(d) for d in dims) + "]"

    config_content = f"""name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 0
instance_group [
  {{
    count: 1
    kind: KIND_CPU
  }}
]
input [
  {{
    name: "node_features"
    data_type: TYPE_FP32
    dims: {format_dims(input_info["node_features"])}
  }},
  {{
    name: "edge_index"
    data_type: TYPE_INT64
    dims: {format_dims(input_info["edge_index"])}
  }},
  {{
    name: "edges"
    data_type: TYPE_INT64
    dims: {format_dims(input_info["edges"])}
  }},
  {{
    name: "node_degrees"
    data_type: TYPE_FP32
    dims: {format_dims(input_info["node_degrees"])}
  }}
]
output [
  {{
    name: "predictions"
    data_type: TYPE_FP32
    dims: {format_dims(output_info["predictions"])}
  }}
]
"""
    config_path = model_dir / "config.pbtxt"
    config_path.write_text(config_content)

    logger.info(f"Triton model repository prepared at {model_dir}")
    return model_dir
