"""FastAPI inference server for link prediction."""

import logging
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field

from collaboration_prediction.data_modules.data import load_dataset
from collaboration_prediction.inference_modules.inference import TritonInferenceClient
from collaboration_prediction.utils.dvc import ensure_data

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = None


class PredictionRequest(BaseModel):
    """Request model for link prediction.

    For static graph link prediction, only a single edge needs to be provided.
    The graph structure (node_features, edge_index, node_degrees) is loaded
    once at server startup.
    """

    edge: List[int] = Field(
        ...,
        description="Edge pair for prediction [src_node_id, dst_node_id]",
        min_length=2,
        max_length=2,
    )


class PredictionResponse(BaseModel):
    """Response model for link prediction."""

    prediction: float = Field(..., description="Link prediction score")


class InferenceServer:
    """FastAPI inference server for static graph link prediction."""

    def __init__(
        self,
        triton_url: str = "localhost:8000",
        triton_model_name: str = "link_prediction",
        data_root: str = "./data",
        dataset_name: str = "ogbl-collab",
        make_undirected: bool = True,
        add_self_loops: bool = True,
        min_degree_clamp: float = 1.0,
        structural_features_cfg: Optional[dict] = None,
    ):
        """Initialize inference server.

        Args:
            triton_url: Triton server URL
            triton_model_name: Name of the model in Triton
            data_root: Root directory for dataset
            dataset_name: Name of the dataset
            make_undirected: Whether to make graph undirected
            add_self_loops: Whether to add self-loops
            min_degree_clamp: Minimum degree for clamping
            structural_features_cfg: Configuration for structural features
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "fastapi is not installed. Install it with: uv sync --extra inference"
            )

        self.app = FastAPI(
            title="Link Prediction API",
            description="API for predicting links in collaboration graphs (static graph)",
            version="0.1.0",
        )

        self.client = TritonInferenceClient(url=triton_url, model_name=triton_model_name)

        logger.info("Ensuring graph data availability via DVC...")
        ensure_data(data_path=data_root, dataset_name=dataset_name)

        logger.info("Loading graph data...")

        graph_data = load_dataset(
            name=dataset_name,
            root=data_root,
            make_undirected=make_undirected,
            add_self_loops=add_self_loops,
            min_degree_clamp=min_degree_clamp,
            structural_features_cfg=structural_features_cfg,
        )

        self.node_features = graph_data["node_features"].numpy().astype(np.float32)
        self.edge_index = graph_data["edge_index"].numpy().astype(np.int64)
        self.node_degrees = graph_data.get("node_degrees", None)
        if self.node_degrees is None:
            num_nodes = self.node_features.shape[0]
            node_degrees_tensor = np.zeros(num_nodes, dtype=np.float32)
            dst_nodes = self.edge_index[1]
            unique, counts = np.unique(dst_nodes, return_counts=True)
            node_degrees_tensor[unique] = counts.astype(np.float32)
            self.node_degrees = np.clip(node_degrees_tensor, min_degree_clamp, None).astype(
                np.float32
            )
        else:
            self.node_degrees = self.node_degrees.numpy().astype(np.float32)

        logger.info(
            f"Graph loaded: {self.node_features.shape[0]} nodes, {self.edge_index.shape[1]} edges"
        )

        self._setup_routes()

    def _setup_routes(self):
        """Initialize API routes."""

        @self.app.get("/")
        async def root():
            return {
                "message": "Link Prediction API",
                "version": "0.1.0",
                "backend": "triton",
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            try:
                try:
                    self.client.client.is_server_live()
                    return {"status": "healthy", "backend": "triton"}
                except Exception:
                    return {
                        "status": "unhealthy",
                        "backend": "triton",
                        "error": "Triton server not available",
                    }
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Predict link for a given edge.

            Uses the pre-loaded static graph. Only one edge needs to be provided.

            Args:
                request: Prediction request with one edge [src, dst]

            Returns:
                Prediction response with link score
            """
            try:
                edge = np.array([request.edge], dtype=np.int64)  # Shape [1, 2]

                max_node_id = self.node_features.shape[0] - 1
                if edge.max() > max_node_id or edge.min() < 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Node indices must be in range [0, {max_node_id}], "
                        f"got range [{edge.min()}, {edge.max()}]",
                    )

                prediction_all = self.client.predict(
                    self.node_features,
                    self.edge_index,
                    edge,
                    self.node_degrees,
                )

                score = float(prediction_all[0, 0])

                return PredictionResponse(prediction=score)

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

        @self.app.get("/model/info")
        async def model_info():
            """Get model and graph information."""
            return {
                "backend": "triton",
                "model_name": getattr(self.client, "model_name", "link_prediction"),
                "graph": {
                    "num_nodes": int(self.node_features.shape[0]),
                    "num_edges": int(self.edge_index.shape[1]),
                    "features_dim": int(self.node_features.shape[1]),
                },
            }
