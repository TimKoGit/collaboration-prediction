"""Evaluation script for link prediction model."""

import logging
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from collaboration_prediction.data_modules.data import get_dataloaders, load_dataset
from collaboration_prediction.model.lightning_module import LinkPredictionLightningModule

logger = logging.getLogger(__name__)


def evaluate_model(cfg: DictConfig, checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Evaluate a trained model on the test set.

    Args:
        cfg: Hydra configuration object
        checkpoint_path: Path to the model checkpoint

    Returns:
        Dictionary containing test metrics, or None if evaluation failed
    """
    logger.info(f"Evaluating model from checkpoint: {checkpoint_path}")

    if torch.cuda.is_available():
        precision = cfg.trainer.precision.get("float32_matmul_precision", "medium")
        torch.set_float32_matmul_precision(precision)

    logger.info("Loading dataset...")
    graph_data = load_dataset(
        name=cfg.data.dataset_name,
        root=cfg.data.data_root,
        make_undirected=cfg.data.preprocessing.make_undirected,
        add_self_loops=cfg.data.preprocessing.add_self_loops,
        min_degree_clamp=cfg.data.preprocessing.negative_sampling.min_degree_clamp,
        structural_features_cfg=cfg.data.preprocessing.get("structural_features"),
        dvc_repo_path=cfg.data.get("dvc_repo_path"),
    )

    features_dim = graph_data["node_features"].shape[1]
    logger.info(f"Dataset loaded: {graph_data['num_nodes']} nodes, features_dim={features_dim}")

    logger.info("Creating data loaders...")
    dataloaders = get_dataloaders(
        data=graph_data, batch_size=cfg.training.batch_size, num_workers=cfg.data.num_workers
    )

    try:
        model = LinkPredictionLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )
    except Exception as e:
        logger.error(f"Failed to load model from checkpoint: {e}")
        return None

    model.set_graph_data(graph_data)

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
    )

    logger.info("Running evaluation on test set...")
    test_results = trainer.test(model=model, dataloaders=dataloaders["test"])

    if test_results:
        results = test_results[0]
        logger.info("\n" + "=" * 50)
        logger.info("Evaluation Results:")
        logger.info("=" * 50)
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info("=" * 50)
        return results
    else:
        logger.warning("No test results returned")
        return None
