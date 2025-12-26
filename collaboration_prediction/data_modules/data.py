"""Data loading and preprocessing for link prediction."""

import contextlib
import functools
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import torch
import torch.utils.data as D
from ogb.linkproppred import PygLinkPropPredDataset
from omegaconf import OmegaConf

from collaboration_prediction.data_modules.features import compute_anchor_encodings, train_deepwalk
from collaboration_prediction.utils.dvc import dvc_add, dvc_pull

logger = logging.getLogger(__name__)

# Add numpy to safe globals for torch.load in PyTorch 2.6+
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

# Store original torch.load to avoid global patching
_original_torch_load = torch.load


@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that defaults to weights_only=False for OGB datasets.

    This function wraps the original torch.load and sets weights_only=False
    if not explicitly provided. This is necessary because PyTorch 2.6+ changed
    the default to weights_only=True, but OGB datasets contain PyG and NumPy
    objects that aren't in the safe globals list.

    Args:
        *args: Positional arguments passed to torch.load
        **kwargs: Keyword arguments passed to torch.load

    Returns:
        Loaded object from torch.load
    """
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


@contextlib.contextmanager
def _patch_torch_load() -> Generator[None, None, None]:
    """Context manager to temporarily patch torch.load for OGB dataset loading.

    This is needed because PyTorch 2.6+ changed the default to weights_only=True,
    but OGB datasets contain PyG and NumPy objects that aren't in the safe globals list.
    Since OGB datasets are from a trusted source, we can safely use weights_only=False.

    Yields:
        None: Context manager that patches torch.load during execution
    """
    torch.load = _patched_torch_load
    try:
        yield
    finally:
        torch.load = _original_torch_load


def load_dataset(
    name: str = "ogbl-collab",
    root: str = "./data",
    make_undirected: bool = True,
    add_self_loops: bool = True,
    min_degree_clamp: float = 1.0,
    structural_features_cfg: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Load OGB link prediction dataset and prepare graph data.

    Args:
        name: Dataset name
        root: Root directory for dataset storage
        make_undirected: Whether to make graph undirected
        add_self_loops: Whether to add self-loops
        min_degree_clamp: Minimum degree for clamping
        structural_features_cfg: Optional configuration for structural features

    Returns:
        Dictionary containing graph data and edge splits with keys:
        - 'edge_index': Edge indices tensor [2, num_edges]
        - 'adj': Sparse adjacency matrix
        - 'node_features': Node features tensor [num_nodes, total_features_dim]
        - 'node_degrees': Node degrees tensor [num_nodes]
        - 'num_nodes': Number of nodes
        - 'train': Tuple of (edges, labels) for training
        - 'valid': Tuple of (edges, labels) for validation
        - 'test': Tuple of (edges, labels) for testing

    Raises:
        FileNotFoundError: If dataset files are not found
        RuntimeError: If dataset loading fails
    """
    try:
        with _patch_torch_load():
            dataset = PygLinkPropPredDataset(name=name, root=root)
            graph_pyg = dataset[0]
            edge_split = dataset.get_edge_split()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Dataset files not found for '{name}' at '{root}'. "
            f"Please ensure the dataset is downloaded or available. Original error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset '{name}' from '{root}'. "
            f"This may be due to corrupted data or missing dependencies. Original error: {e}"
        ) from e

    edge_index = graph_pyg.edge_index
    node_features = graph_pyg.x
    num_nodes = graph_pyg.num_nodes

    if make_undirected:
        edge_index_undirected = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        edge_index_undirected = edge_index

    if structural_features_cfg:
        feature_list = [node_features]

        dataset_dir = Path(root) / name.replace("-", "_")
        processed_dir = dataset_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        serializable_cfg = OmegaConf.to_container(structural_features_cfg, resolve=True)

        cfg_str = json.dumps(serializable_cfg, sort_keys=True)
        cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()
        cache_path = processed_dir / f"structural_features_{cfg_hash}.pt"

        project_root = Path.cwd()
        try:
            current = Path(root).resolve()
            while current != current.parent:
                if (current / ".dvc").exists() or (current / "pyproject.toml").exists():
                    project_root = current
                    break
                current = current.parent
        except Exception:
            pass

        if not cache_path.exists():
            dvc_file = cache_path.with_suffix(".pt.dvc")
            if dvc_file.exists():
                logger.info(
                    f"Found DVC tracking for structural features, attempting to pull: {cache_path}"
                )
                dvc_pull(str(cache_path), root_dir=project_root)

        if cache_path.exists():
            logger.info(f"Loading structural features from cache: {cache_path}")
            cached_features = torch.load(cache_path, weights_only=False)
            node_features = cached_features
            dvc_add(str(cache_path), root_dir=project_root)
        else:
            logger.info("Computing structural features (not found in cache)...")
            dw_cfg = structural_features_cfg.get("deepwalk", {})
            if dw_cfg.get("enabled", False):
                logger.info("Computing DeepWalk embeddings...")
                dw_embeds = train_deepwalk(
                    edge_index=edge_index_undirected,
                    num_nodes=num_nodes,
                    embedding_dim=dw_cfg.get("embedding_dim", 128),
                    walk_length=dw_cfg.get("walk_length", 40),
                    window_size=dw_cfg.get("window_size", 5),
                    negative_size=dw_cfg.get("negative_size", 1),
                    epochs=dw_cfg.get("epochs", 1),
                    batch_size=dw_cfg.get("batch_size", 256),
                    num_workers=dw_cfg.get("num_workers", 4),
                    lr=dw_cfg.get("lr", 0.001),
                )
                feature_list.append(dw_embeds)

            anchor_cfg = structural_features_cfg.get("anchor_encodings", {})
            if anchor_cfg.get("enabled", False):
                logger.info("Computing Anchor Encodings...")
                anchor_encs, _ = compute_anchor_encodings(
                    edge_index=edge_index_undirected,
                    num_nodes=num_nodes,
                    num_anchor_nodes=anchor_cfg.get("num_anchor_nodes", 32),
                    num_unique_values=anchor_cfg.get("num_unique_values", 16),
                )
                feature_list.append(anchor_encs)

            if len(feature_list) > 1:
                node_features = torch.cat(feature_list, dim=-1)
                logger.info(f"Final node features dimension: {node_features.shape[1]}")

            logger.info(f"Saving computed structural features to cache: {cache_path}")
            torch.save(node_features, cache_path)

            dvc_add(str(cache_path), root_dir=project_root)

    if add_self_loops:
        self_loops = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
        edge_index_with_self_loops = torch.cat([edge_index_undirected, self_loops], dim=1)
    else:
        edge_index_with_self_loops = edge_index_undirected

    values = torch.ones(edge_index_with_self_loops.shape[1], dtype=torch.float32)
    adj = torch.sparse_coo_tensor(
        indices=edge_index_with_self_loops, values=values, size=(num_nodes, num_nodes)
    ).coalesce()

    node_degrees = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=min_degree_clamp)

    positive_edges_train = edge_split["train"]["edge"]
    num_positive_edges_train = len(positive_edges_train)

    positive_edges_valid = edge_split["valid"]["edge"]
    num_positive_edges_valid = len(positive_edges_valid)

    negative_edges_valid = edge_split["valid"]["edge_neg"]
    num_negative_edges_valid = len(negative_edges_valid)

    positive_edges_test = edge_split["test"]["edge"]
    num_positive_edges_test = len(positive_edges_test)

    negative_edges_test = edge_split["test"]["edge_neg"]
    num_negative_edges_test = len(negative_edges_test)

    edges_train = positive_edges_train
    edges_valid = torch.cat([positive_edges_valid, negative_edges_valid])
    edges_test = torch.cat([positive_edges_test, negative_edges_test])

    labels_train = torch.ones(size=(num_positive_edges_train,))

    labels_valid = torch.cat(
        [
            torch.ones(size=(num_positive_edges_valid,)),
            torch.zeros(size=(num_negative_edges_valid,)),
        ]
    )

    labels_test = torch.cat(
        [torch.ones(size=(num_positive_edges_test,)), torch.zeros(size=(num_negative_edges_test,))]
    )

    edge_index_np = edge_index_undirected.cpu().numpy()
    existing_edges_set = set(zip(edge_index_np[0], edge_index_np[1]))

    data = {
        "edge_index": edge_index_undirected,
        "adj": adj,
        "node_features": node_features,
        "node_degrees": node_degrees,
        "num_nodes": num_nodes,
        "train": (edges_train, labels_train),
        "valid": (edges_valid, labels_valid),
        "test": (edges_test, labels_test),
        "existing_edges_set": existing_edges_set,
    }

    return data


def sample_negative_edges(
    num_nodes: int,
    num_negative_edges: int,
    existing_edges_set: Optional[set[tuple[int, int]]] = None,
    device: Optional[torch.device] = None,
    oversample_factor: float = 1.1,
) -> torch.Tensor:
    """Sample negative edges uniformly at random.

    Samples edges that do not exist in the graph, ensuring no positive edges are included.

    Args:
        num_nodes: Number of nodes in the graph
        num_negative_edges: Number of negative edges to sample
        existing_edges_set: Set of existing edge tuples (src, dst) for fast filtering
        device: Device to create tensors on
        oversample_factor: Factor to oversample when filtering out positive edges

    Returns:
        Negative edge indices [num_negative_edges, 2]
    """
    negative_edges_list = []
    max_samples = int(num_negative_edges * oversample_factor * 10)
    samples_generated = 0

    while len(negative_edges_list) < num_negative_edges and samples_generated < max_samples:
        batch_size = min(num_negative_edges * 2, max_samples - samples_generated)
        src = torch.randint(0, num_nodes, (batch_size,), device="cpu", dtype=torch.long)
        dst = torch.randint(0, num_nodes, (batch_size,), device="cpu", dtype=torch.long)

        mask = src != dst
        src = src[mask]
        dst = dst[mask]

        for i in range(len(src)):
            if len(negative_edges_list) >= num_negative_edges:
                break

            s, d = int(src[i]), int(dst[i])
            edge = (s, d)

            if existing_edges_set is not None:
                if edge not in existing_edges_set:
                    negative_edges_list.append([s, d])
            else:
                negative_edges_list.append([s, d])

        samples_generated += batch_size

    if negative_edges_list:
        neg_edges = torch.tensor(negative_edges_list, device=device, dtype=torch.long)
        return neg_edges
    else:
        return torch.empty((0, 2), device=device, dtype=torch.long)


def get_dataloaders(data: dict, batch_size: int, num_workers: int = 0) -> dict[str, D.DataLoader]:
    """Create PyTorch DataLoaders for train, validation, and test sets.

    Args:
        data: Dictionary containing train, valid, and test data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading

    Returns:
        Dictionary containing train, valid, and test DataLoaders
    """
    dataloader_train = D.DataLoader(
        dataset=D.TensorDataset(*data["train"]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    dataloader_valid = D.DataLoader(
        dataset=D.TensorDataset(*data["valid"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    dataloader_test = D.DataLoader(
        dataset=D.TensorDataset(*data["test"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return {
        "train": dataloader_train,
        "valid": dataloader_valid,
        "test": dataloader_test,
    }
