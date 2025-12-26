"""Training script for link prediction model using PyTorch Lightning and Hydra."""

import logging
import subprocess
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from collaboration_prediction.data_modules.data import get_dataloaders, load_dataset
from collaboration_prediction.model.lightning_module import LinkPredictionLightningModule

logger = logging.getLogger(__name__)


def get_git_commit_id() -> str:
    """Get current git commit ID.

    Returns:
        Git commit ID or 'unknown' if not available
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


class PlotCallback(Callback):
    """Callback to save training plots."""

    def __init__(self, plots_dir: Path, plotting_cfg):
        """Initialize callback.

        Args:
            plots_dir: Directory to save plots
            plotting_cfg: Plotting configuration from config
        """
        super().__init__()
        self.plots_dir = plots_dir
        self.plotting_cfg = plotting_cfg
        self.train_losses = []
        self.val_hits = []
        self.val_auroc = []
        self.epochs = []

    def on_train_epoch_end(self, trainer, pl_module):
        """Collect all training and validation metrics at the end of an epoch."""
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        loss = metrics.get("train/loss_epoch") or metrics.get("train/loss")
        if loss is not None:
            val = loss.item() if hasattr(loss, "item") else loss
            self.train_losses.append(val)
            logger.info(f"Epoch {epoch} - Train Loss: {val:.4f}")

        hits = metrics.get("val/hits_epoch") or metrics.get("val/hits")
        auroc = metrics.get("val/auroc_epoch") or metrics.get("val/auroc")

        if hits is not None or auroc is not None:
            if epoch not in self.epochs:
                self.epochs.append(epoch)

                log_msg = f"Epoch {epoch} - Validation Metrics:"
                if hits is not None:
                    val_hits = hits.item() if hasattr(hits, "item") else hits
                    self.val_hits.append(val_hits)
                    log_msg += f" Hits@K: {val_hits:.4f}"
                if auroc is not None:
                    val_auroc = auroc.item() if hasattr(auroc, "item") else auroc
                    self.val_auroc.append(val_auroc)
                    log_msg += f" AUROC: {val_auroc:.4f}"
                logger.info(log_msg)

    def on_train_end(self, trainer, pl_module):
        """Save plots at the end of training."""
        if not self.train_losses and not self.val_hits and not self.val_auroc:
            logger.warning("No metrics collected during training, skipping plot generation")
            return

        logger.info(
            f"Generating plots. Collected: {len(self.train_losses)} train losses, "
            f"{len(self.val_hits)} val hits, {len(self.val_auroc)} val aurocs"
        )

        self.plots_dir.mkdir(exist_ok=True, parents=True)

        # Get the active MLflow run ID from the logger
        # This ensures we log to the same run created by MLFlowLogger
        run_id = None
        if trainer.logger and hasattr(trainer.logger, "run_id") and trainer.logger.run_id:
            run_id = trainer.logger.run_id

        log_file = self.plots_dir / "train.log"
        if log_file.exists():
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifact(str(log_file), "logs")
            else:
                mlflow.log_artifact(str(log_file), "logs")

        # Plot 1: Training Loss
        if self.train_losses:
            figsize = self.plotting_cfg.single_plot_size
            plt.figure(figsize=figsize)
            epochs_loss = list(range(len(self.train_losses)))
            plt.plot(
                epochs_loss,
                self.train_losses,
                color=self.plotting_cfg.colors.loss,
                linestyle="-",
                linewidth=self.plotting_cfg.linewidth,
                label="Training Loss",
            )
            plt.xlabel("Epoch", fontsize=self.plotting_cfg.label_fontsize)
            plt.ylabel("Loss", fontsize=self.plotting_cfg.label_fontsize)
            plt.title(
                "Training Loss Over Epochs",
                fontsize=self.plotting_cfg.title_fontsize,
                fontweight="bold",
            )
            plt.grid(True, alpha=self.plotting_cfg.grid_alpha)
            plt.legend(fontsize=self.plotting_cfg.legend_fontsize)
            plt.tight_layout()
            plt.savefig(
                self.plots_dir / "training_loss.png", dpi=self.plotting_cfg.dpi, bbox_inches="tight"
            )
            plt.close()

            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifact(str(self.plots_dir / "training_loss.png"), "plots")
            else:
                mlflow.log_artifact(str(self.plots_dir / "training_loss.png"), "plots")

        # Plot 2: Validation Hits@K
        if self.val_hits and self.epochs:
            figsize = self.plotting_cfg.single_plot_size
            plt.figure(figsize=figsize)
            plt.plot(
                self.epochs[: len(self.val_hits)],
                self.val_hits,
                color=self.plotting_cfg.colors.hits,
                linestyle="-",
                linewidth=self.plotting_cfg.linewidth,
                marker=self.plotting_cfg.markers.hits,
                label="Hits@K",
            )
            plt.xlabel("Epoch", fontsize=self.plotting_cfg.label_fontsize)
            plt.ylabel("Hits@K", fontsize=self.plotting_cfg.label_fontsize)
            plt.title(
                "Validation Hits@K Over Epochs",
                fontsize=self.plotting_cfg.title_fontsize,
                fontweight="bold",
            )
            plt.grid(True, alpha=self.plotting_cfg.grid_alpha)
            plt.legend(fontsize=self.plotting_cfg.legend_fontsize)
            plt.tight_layout()
            plt.savefig(
                self.plots_dir / "validation_hits.png",
                dpi=self.plotting_cfg.dpi,
                bbox_inches="tight",
            )
            plt.close()

            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifact(str(self.plots_dir / "validation_hits.png"), "plots")
            else:
                mlflow.log_artifact(str(self.plots_dir / "validation_hits.png"), "plots")

        # Plot 3: Validation AUROC
        if self.val_auroc and self.epochs:
            figsize = self.plotting_cfg.single_plot_size
            plt.figure(figsize=figsize)
            plt.plot(
                self.epochs[: len(self.val_auroc)],
                self.val_auroc,
                color=self.plotting_cfg.colors.auroc,
                linestyle="-",
                linewidth=self.plotting_cfg.linewidth,
                marker=self.plotting_cfg.markers.auroc,
                label="AUROC",
            )
            plt.xlabel("Epoch", fontsize=self.plotting_cfg.label_fontsize)
            plt.ylabel("AUROC", fontsize=self.plotting_cfg.label_fontsize)
            plt.title(
                "Validation AUROC Over Epochs",
                fontsize=self.plotting_cfg.title_fontsize,
                fontweight="bold",
            )
            plt.grid(True, alpha=self.plotting_cfg.grid_alpha)
            plt.legend(fontsize=self.plotting_cfg.legend_fontsize)
            plt.tight_layout()
            plt.savefig(
                self.plots_dir / "validation_auroc.png",
                dpi=self.plotting_cfg.dpi,
                bbox_inches="tight",
            )
            plt.close()

            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifact(str(self.plots_dir / "validation_auroc.png"), "plots")
            else:
                mlflow.log_artifact(str(self.plots_dir / "validation_auroc.png"), "plots")

        # Combined plot: All metrics
        if self.train_losses and self.val_hits and self.val_auroc:
            figsize = self.plotting_cfg.combined_plot_size
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            # Left plot: Loss
            epochs_loss = list(range(len(self.train_losses)))
            ax1.plot(
                epochs_loss,
                self.train_losses,
                color=self.plotting_cfg.colors.loss,
                linestyle="-",
                linewidth=self.plotting_cfg.linewidth,
                label="Training Loss",
            )
            ax1.set_xlabel("Epoch", fontsize=self.plotting_cfg.label_fontsize)
            ax1.set_ylabel("Loss", fontsize=self.plotting_cfg.label_fontsize)
            ax1.set_title(
                "Training Loss", fontsize=self.plotting_cfg.title_fontsize - 1, fontweight="bold"
            )
            ax1.grid(True, alpha=self.plotting_cfg.grid_alpha)
            ax1.legend(fontsize=self.plotting_cfg.legend_fontsize)

            # Right plot: Validation metrics
            ax2.plot(
                self.epochs[: len(self.val_hits)],
                self.val_hits,
                color=self.plotting_cfg.colors.hits,
                linestyle="-",
                linewidth=self.plotting_cfg.linewidth,
                marker=self.plotting_cfg.markers.hits,
                label="Hits@K",
            )
            ax2.plot(
                self.epochs[: len(self.val_auroc)],
                self.val_auroc,
                color=self.plotting_cfg.colors.auroc,
                linestyle="-",
                linewidth=self.plotting_cfg.linewidth,
                marker=self.plotting_cfg.markers.auroc,
                label="AUROC",
            )
            ax2.set_xlabel("Epoch", fontsize=self.plotting_cfg.label_fontsize)
            ax2.set_ylabel("Score", fontsize=self.plotting_cfg.label_fontsize)
            ax2.set_title(
                "Validation Metrics",
                fontsize=self.plotting_cfg.title_fontsize - 1,
                fontweight="bold",
            )
            ax2.grid(True, alpha=self.plotting_cfg.grid_alpha)
            ax2.legend(fontsize=self.plotting_cfg.legend_fontsize)

            plt.tight_layout()
            plt.savefig(
                self.plots_dir / "all_metrics.png", dpi=self.plotting_cfg.dpi, bbox_inches="tight"
            )
            plt.close()

            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifact(str(self.plots_dir / "all_metrics.png"), "plots")
            else:
                mlflow.log_artifact(str(self.plots_dir / "all_metrics.png"), "plots")


def train_model(cfg: DictConfig) -> None:
    """Train the link prediction model.

    Args:
        cfg: Hydra configuration object
    """
    plots_dir = Path(cfg.logging.plots_dir)
    plots_dir.mkdir(exist_ok=True, parents=True)
    log_file = plots_dir / "train.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    project_logger = logging.getLogger("collaboration_prediction")
    for handler in project_logger.handlers[:]:
        if isinstance(handler, (logging.FileHandler, logging.StreamHandler)):
            handler.close()
            project_logger.removeHandler(handler)

    project_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    project_logger.addHandler(console_handler)
    project_logger.setLevel(logging.INFO)
    project_logger.propagate = False

    try:
        for plot_file in [
            "training_loss.png",
            "validation_hits.png",
            "validation_auroc.png",
            "all_metrics.png",
        ]:
            path = plots_dir / plot_file
            if path.exists():
                path.unlink()

        if log_file.exists():
            log_file.write_text("")

        if torch.cuda.is_available():
            precision = cfg.trainer.precision.get("float32_matmul_precision", "medium")
            torch.set_float32_matmul_precision(precision)

        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

        logger.info("Loading dataset...")
        graph_data = load_dataset(
            name=cfg.data.dataset_name,
            root=cfg.data.data_root,
            make_undirected=cfg.data.preprocessing.make_undirected,
            add_self_loops=cfg.data.preprocessing.add_self_loops,
            min_degree_clamp=cfg.data.preprocessing.negative_sampling.min_degree_clamp,
            structural_features_cfg=cfg.data.preprocessing.get("structural_features"),
        )

        features_dim = graph_data["node_features"].shape[1]
        logger.info(f"Dataset loaded: {graph_data['num_nodes']} nodes, features_dim={features_dim}")

        logger.info("Creating data loaders...")
        dataloaders = get_dataloaders(
            data=graph_data, batch_size=cfg.training.batch_size, num_workers=cfg.data.num_workers
        )

        logger.info("Initializing model...")
        model = LinkPredictionLightningModule(
            features_dim=features_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            learning_rate=cfg.training.learning_rate,
            factor_negative_edges=cfg.training.factor_negative_edges,
            hits_k=cfg.training.evaluation.hits_k,
            pair_encoder_output_multiplier=cfg.model.pair_encoder.output_multiplier,
            link_predictor_reduction_factor=cfg.model.link_predictor.reduction_factor,
            min_degree=cfg.model.sage_conv.min_degree,
            auroc_default=cfg.training.evaluation.auroc_default,
            metric_precision=cfg.training.evaluation.metric_precision,
            oversample_factor=cfg.data.preprocessing.negative_sampling.oversample_factor,
        )

        model.set_graph_data(graph_data)

        git_commit_id = get_git_commit_id()

        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.logging.mlflow.experiment_name,
            run_name=cfg.logging.mlflow.run_name,
            tracking_uri=cfg.logging.mlflow.tracking_uri,
            log_model=False,
        )

        original_log_metrics = mlflow_logger.log_metrics

        def filtered_log_metrics(metrics, step=None):
            filtered_metrics = {k: v for k, v in metrics.items() if k != "epoch"}
            if filtered_metrics:
                original_log_metrics(filtered_metrics, step)

        mlflow_logger.log_metrics = filtered_log_metrics

        hyperparameters = {
            "model/features_dim": features_dim,
            "model/hidden_dim": cfg.model.hidden_dim,
            "model/num_layers": cfg.model.num_layers,
            "model/dropout": cfg.model.dropout,
            "training/learning_rate": cfg.training.learning_rate,
            "training/batch_size": cfg.training.batch_size,
            "training/num_epochs": cfg.training.num_epochs,
            "training/factor_negative_edges": cfg.training.factor_negative_edges,
            "training/hits_k": cfg.training.evaluation.hits_k,
            "data/dataset_name": cfg.data.dataset_name,
            "data/num_workers": cfg.data.num_workers,
            "git_commit_id": git_commit_id,
        }

        mlflow_logger.log_hyperparams(hyperparameters)

        plot_callback = PlotCallback(plots_dir=plots_dir, plotting_cfg=cfg.logging.plotting)

        callbacks = [plot_callback]
        if cfg.trainer.checkpoint.enable:
            checkpoint_dir = Path(cfg.trainer.default_root_dir) / cfg.trainer.checkpoint.dirpath
            checkpoint_callback = ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                monitor=cfg.trainer.checkpoint.monitor,
                mode=cfg.trainer.checkpoint.mode,
                save_top_k=cfg.trainer.checkpoint.save_top_k,
                save_last=cfg.trainer.checkpoint.save_last,
                filename=cfg.trainer.checkpoint.filename,
                every_n_epochs=cfg.trainer.checkpoint.get("every_n_epochs"),
                every_n_train_steps=cfg.trainer.checkpoint.get("every_n_train_steps"),
                verbose=True,
            )
            callbacks.append(checkpoint_callback)
            logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

        trainer = pl.Trainer(
            accelerator=cfg.trainer.accelerator,
            devices=cfg.trainer.devices,
            max_epochs=cfg.trainer.max_epochs,
            check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
            log_every_n_steps=cfg.trainer.log_every_n_steps,
            enable_progress_bar=cfg.trainer.enable_progress_bar,
            enable_model_summary=cfg.trainer.enable_model_summary,
            default_root_dir=cfg.trainer.default_root_dir,
            logger=mlflow_logger,
            callbacks=callbacks,
        )

        logger.info("Starting training...")
        trainer.fit(
            model=model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["valid"],
        )

        logger.info("Evaluating on test set...")
        test_results = trainer.test(
            model=model,
            dataloaders=dataloaders["test"],
        )
        if test_results:
            for i, results in enumerate(test_results):
                logger.info(f"Test Set {i} Results:")
                for metric, value in results.items():
                    logger.info(f"  {metric}: {value:.4f}")

        if cfg.export.get("auto_export_onnx", False):
            logger.info("Exporting model to ONNX format...")
            try:
                from collaboration_prediction.inference_modules.export import (
                    export_to_onnx,
                    load_example_data_for_export,
                )

                checkpoint_callback = None
                for callback in callbacks:
                    if isinstance(callback, pl.callbacks.ModelCheckpoint):
                        checkpoint_callback = callback
                        break

                if checkpoint_callback and checkpoint_callback.best_model_path:
                    checkpoint_path = checkpoint_callback.best_model_path
                else:
                    checkpoint_path = trainer.checkpoint_callback.last_model_path

                if checkpoint_path and Path(checkpoint_path).exists():
                    example_data = load_example_data_for_export(
                        checkpoint_path=checkpoint_path,
                        data_root=cfg.data.data_root,
                        dataset_name=cfg.data.dataset_name,
                        structural_features_cfg=cfg.data.preprocessing.get("structural_features"),
                    )

                    onnx_path = cfg.export.onnx.export_path
                    export_to_onnx(
                        model=model.model,
                        output_path=onnx_path,
                        example_data=example_data,
                        opset_version=cfg.export.onnx.opset_version,
                    )
                    logger.info(f"Model exported to ONNX: {onnx_path}")
                else:
                    logger.warning("No checkpoint found, skipping ONNX export")
            except ImportError:
                logger.warning("ONNX not available. Install with: uv sync --extra export")
            except Exception as e:
                logger.error(f"ONNX export failed: {e}")

        logger.info("Training completed!")

    finally:
        file_handler.close()
        project_logger.removeHandler(file_handler)
        if "console_handler" in locals():
            console_handler.close()
            project_logger.removeHandler(console_handler)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Entry point for training when run directly with Hydra."""
    train_model(cfg)
