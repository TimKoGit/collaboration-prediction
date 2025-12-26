"""Structural feature generation for graph nodes."""

import logging
import math
import random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SkipGramModel(nn.Module):
    """Simple Skip-gram model for node embeddings."""

    def __init__(self, num_nodes: int, embedding_dim: int):
        """Initialize Skip-gram model.

        Args:
            num_nodes: Total number of nodes
            embedding_dim: Embedding dimension
        """
        super().__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.output_weights = nn.Embedding(num_nodes, embedding_dim)

    def forward(self, target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Forward pass for Skip-gram.

        Args:
            target: Target node indices [batch_size]
            context: Context node indices [batch_size]

        Returns:
            Logits for the target-context pair
        """
        target_embeds = self.embeddings(target)
        context_weights = self.output_weights(context)

        return torch.sum(target_embeds * context_weights, dim=-1)


class SkipGramDataset(IterableDataset):
    """Highly optimized dataset that yields batches of tensors to minimize Python overhead."""

    def __init__(
        self,
        walks: List[List[int]],
        window_size: int,
        num_nodes: int,
        negative_size: int,
        batch_size: int,
    ):
        """Initialize SkipGramDataset.

        Args:
            walks: List of random walks
            window_size: Window size for context
            num_nodes: Total number of nodes
            negative_size: Number of negative samples
            batch_size: Batch size for iteration
        """
        super().__init__()
        self.walks = np.array(walks, dtype=np.int32)
        self.window_size = window_size
        self.num_nodes = num_nodes
        self.negative_size = negative_size
        self.batch_size = batch_size

        num_walks, walk_len = self.walks.shape
        total_samples = num_walks * walk_len * (window_size * 2) * (1 + negative_size)
        self.est_len = total_samples // batch_size

    def __iter__(self):
        """Iterate over the dataset and yield batches."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_walks = self.walks
        else:
            per_worker = int(math.ceil(len(self.walks) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_walks = self.walks[worker_id * per_worker : (worker_id + 1) * per_worker]

        targets, contexts, labels = [], [], []

        for walk in iter_walks:
            walk_len = len(walk)
            for i in range(walk_len):
                target = walk[i]
                start = max(0, i - self.window_size)
                end = min(walk_len, i + self.window_size + 1)

                for j in range(start, end):
                    if i == j:
                        continue

                    targets.append(int(target))
                    contexts.append(int(walk[j]))
                    labels.append(1.0)

                    if self.negative_size > 0:
                        neg_nodes = np.random.randint(0, self.num_nodes, size=self.negative_size)
                        for neg_node in neg_nodes:
                            targets.append(int(target))
                            contexts.append(int(neg_node))
                            labels.append(0.0)

                    if len(targets) >= self.batch_size:
                        yield (
                            torch.tensor(targets[: self.batch_size], dtype=torch.long),
                            torch.tensor(contexts[: self.batch_size], dtype=torch.long),
                            torch.tensor(labels[: self.batch_size], dtype=torch.float32),
                        )
                        targets = targets[self.batch_size :]
                        contexts = contexts[self.batch_size :]
                        labels = labels[self.batch_size :]

    def __len__(self) -> int:
        """Return the estimated number of batches."""
        return self.est_len


def generate_random_walks(
    edge_index: torch.Tensor, num_nodes: int, walk_length: int, num_walks_per_node: int = 1
) -> List[List[int]]:
    """Fast random walks using NumPy and CSR-like structure.

    Args:
        edge_index: Graph structure [2, num_edges]
        num_nodes: Total number of nodes
        walk_length: Length of each walk
        num_walks_per_node: Number of walks to start from each node

    Returns:
        List of walks (each walk is a list of node indices)
    """
    logger.info("Building CSR-like graph for fast random walks...")
    edge_index_np = edge_index.cpu().numpy()

    row, col = edge_index_np
    idx = np.argsort(row)
    row, col = row[idx], col[idx]

    ptr = np.zeros(num_nodes + 1, dtype=np.int32)
    counts = np.bincount(row, minlength=num_nodes)
    ptr[1:] = np.cumsum(counts)

    walks = np.zeros((num_nodes * num_walks_per_node, walk_length), dtype=np.int32)

    for i in range(num_walks_per_node):
        nodes = np.arange(num_nodes)
        np.random.shuffle(nodes)

        for j, start_node in enumerate(tqdm(nodes, desc=f"Generating walks pass {i+1}")):
            walk = walks[i * num_nodes + j]
            walk[0] = start_node
            curr = start_node
            for step in range(1, walk_length):
                start_idx, end_idx = ptr[curr], ptr[curr + 1]
                if start_idx == end_idx:
                    walk[step:] = curr
                    break
                curr = col[np.random.randint(start_idx, end_idx)]
                walk[step] = curr

    return walks.tolist()


def train_deepwalk(
    edge_index: torch.Tensor,
    num_nodes: int,
    embedding_dim: int = 128,
    walk_length: int = 40,
    window_size: int = 5,
    negative_size: int = 1,
    epochs: int = 1,
    batch_size: int = 256,
    num_workers: int = 4,
    lr: float = 0.001,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Train DeepWalk embeddings.

    Args:
        edge_index: Graph structure
        num_nodes: Number of nodes
        embedding_dim: Dimension of DeepWalk embeddings
        walk_length: Length of random walks
        window_size: Skip-gram window size
        negative_size: Number of negative samples per positive
        epochs: Training epochs
        batch_size: Training batch size
        num_workers: Number of workers for data loading
        lr: Learning rate for optimizer
        device: Device to use for training

    Returns:
        Node embeddings [num_nodes, embedding_dim]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Generating random walks for DeepWalk...")
    walks = generate_random_walks(edge_index, num_nodes, walk_length, num_walks_per_node=1)

    logger.info("Training DeepWalk (Skip-gram)...")
    dataset = SkipGramDataset(walks, window_size, num_nodes, negative_size, batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = SkipGramModel(num_nodes, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"DeepWalk Epoch {epoch+1}", total=len(dataset))
        for target, context, label in pbar:
            target, context, label = target.to(device), context.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(target, context)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (pbar.n + 1)})

    return model.embeddings.weight.detach().cpu()


def compute_anchor_encodings(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_anchor_nodes: int = 32,
    num_unique_values: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute anchor-based distance encodings using BFS.

    Args:
        edge_index: Graph structure
        num_nodes: Number of nodes
        num_anchor_nodes: Number of anchor nodes to select
        num_unique_values: Max distance to clip and encode

    Returns:
        Tuple of (encodings, raw_distances)
        encodings: [num_nodes, num_anchor_nodes * num_unique_values]
        raw_distances: [num_nodes, num_anchor_nodes]
    """
    logger.info(f"Computing anchor encodings with {num_anchor_nodes} anchors...")

    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    edge_index_np = edge_index.cpu().numpy()
    for i in range(edge_index_np.shape[1]):
        u, v = edge_index_np[0, i], edge_index_np[1, i]
        adj[u].append(v)

    anchors = random.sample(range(num_nodes), num_anchor_nodes)
    distances = torch.full((num_nodes, num_anchor_nodes), float(num_unique_values + 1))

    for i, anchor in enumerate(tqdm(anchors, desc="BFS from anchors")):
        q = deque([(anchor, 0)])
        distances[anchor, i] = 0

        while q:
            curr, d = q.popleft()
            if d >= num_unique_values:
                continue

            for neighbor in adj[curr]:
                if distances[neighbor, i] > d + 1:
                    distances[neighbor, i] = d + 1
                    q.append((neighbor, d + 1))

    distances = torch.clamp(distances, max=num_unique_values)

    to_compare = torch.arange(1, num_unique_values + 1)
    encodings = (distances.unsqueeze(-1) >= to_compare).float()
    encodings = encodings.reshape(num_nodes, num_anchor_nodes * num_unique_values)

    return encodings, distances
