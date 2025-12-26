"""Command-line interface entry point for all commands."""

import os
from pathlib import Path

import click
import hydra
import uvicorn

from collaboration_prediction.evaluate import evaluate_model
from collaboration_prediction.inference_modules.export import (
    export_to_onnx,
    load_example_data_for_export,
    verify_onnx_model,
)
from collaboration_prediction.inference_modules.inference import prepare_triton_model_repository
from collaboration_prediction.inference_modules.inference_server import InferenceServer
from collaboration_prediction.model.lightning_module import LinkPredictionLightningModule
from collaboration_prediction.train import train_model
from collaboration_prediction.utils.dvc import ensure_data, init_external_data


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Collaboration Prediction - Link prediction pipeline for scientist collaborations."""
    pass


@cli.command()
@click.option(
    "--config-path",
    default="configs",
    help="Path to Hydra configuration directory",
    type=click.Path(exists=True),
)
@click.option(
    "--config-name",
    default="config",
    help="Name of the configuration file (without .yaml)",
)
@click.option(
    "--force-download",
    is_flag=True,
    help="Force re-download of data even if it exists",
)
@click.argument("overrides", nargs=-1)
def init_data(config_path: str, config_name: str, force_download: bool, overrides: tuple):
    """Initialize external DVC repo and upload data.

    Example:
        collab-pred init-data
        collab-pred init-data --force-download
    """
    overrides_list = list(overrides)
    config_path_abs = os.path.abspath(config_path)

    with hydra.initialize_config_dir(config_dir=config_path_abs, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides_list)

        dvc_repo_path = cfg.data.dvc_repo_path

        dvc_root = Path(dvc_repo_path).resolve()
        data_root_abs = Path(cfg.data.data_root).resolve()

        try:
            data_path_rel = data_root_abs.relative_to(dvc_root)
        except ValueError:
            click.echo(
                f"Error: data_root ({data_root_abs}) must be inside dvc_repo_path ({dvc_root})",
                err=True,
            )
            raise click.Abort()

        dataset_name = cfg.data.dataset_name

        click.echo(f"Initializing DVC repo at {dvc_root}...")
        if init_external_data(
            dvc_repo_path=str(dvc_root),
            data_path=str(data_path_rel),
            dataset_name=dataset_name,
            force_download=force_download,
        ):
            click.echo("✓ Data initialized and added to DVC successfully")
        else:
            click.echo("❌ Failed to initialize data", err=True)
            raise click.Abort()


@cli.command()
@click.option(
    "--config-path",
    default="configs",
    help="Path to Hydra configuration directory",
    type=click.Path(exists=True),
)
@click.option(
    "--config-name",
    default="config",
    help="Name of the configuration file (without .yaml)",
)
@click.argument("overrides", nargs=-1)
def train(config_path: str, config_name: str, overrides: tuple):
    """Train the link prediction model.

    Examples:
        python commands.py train
        python commands.py train --config-name config training.batch_size=128
        python commands.py train training.learning_rate=0.0005 model.hidden_dim=512
    """
    overrides_list = list(overrides)

    config_path_abs = os.path.abspath(config_path)
    if not os.path.exists(config_path_abs):
        click.echo(f"Error: Config directory not found: {config_path_abs}", err=True)
        raise click.Abort()

    with hydra.initialize_config_dir(config_dir=config_path_abs, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides_list)

        data_path = cfg.data.data_root
        dataset_name = cfg.data.dataset_name
        if not ensure_data(data_path, dataset_name):
            raise click.Abort()

        train_model(cfg)


@cli.command()
@click.option(
    "--checkpoint",
    required=True,
    help="Path to model checkpoint",
    type=click.Path(exists=True),
)
@click.option(
    "--config-path",
    default="configs",
    help="Path to Hydra configuration directory",
    type=click.Path(exists=True),
)
@click.option(
    "--config-name",
    default="config",
    help="Name of the configuration file (without .yaml)",
)
@click.argument("overrides", nargs=-1)
def evaluate(checkpoint: str, config_path: str, config_name: str, overrides: tuple):
    """Evaluate a trained model on test set.

    Examples:
        python commands.py evaluate --checkpoint lightning_logs/version_0/checkpoints/best.ckpt
    """
    overrides_list = list(overrides)

    config_path_abs = os.path.abspath(config_path)
    if not os.path.exists(config_path_abs):
        click.echo(f"Error: Config directory not found: {config_path_abs}", err=True)
        raise click.Abort()

    with hydra.initialize_config_dir(config_dir=config_path_abs, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides_list)

        data_path = cfg.data.data_root
        dataset_name = cfg.data.dataset_name
        if not ensure_data(data_path, dataset_name):
            raise click.Abort()

        results = evaluate_model(cfg, checkpoint)
        if results is None:
            raise click.Abort()


@cli.command()
@click.option(
    "--checkpoint",
    required=True,
    help="Path to model checkpoint",
    type=click.Path(exists=True),
)
@click.option(
    "--output",
    required=True,
    help="Output path for ONNX model",
    type=click.Path(),
)
@click.option(
    "--config-path",
    default="configs",
    help="Path to Hydra configuration directory",
    type=click.Path(exists=True),
)
@click.option(
    "--config-name",
    default="config",
    help="Name of the configuration file (without .yaml)",
)
@click.argument("overrides", nargs=-1)
def export_onnx(checkpoint: str, output: str, config_path: str, config_name: str, overrides: tuple):
    """Export model to ONNX format.

    Examples:
        collab-pred export-onnx --checkpoint best.ckpt --output model.onnx
    """
    overrides_list = list(overrides)

    config_path_abs = os.path.abspath(config_path)
    if not os.path.exists(config_path_abs):
        click.echo(f"Error: Config directory not found: {config_path_abs}", err=True)
        raise click.Abort()

    with hydra.initialize_config_dir(config_dir=config_path_abs, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides_list)

        data_path = cfg.data.data_root
        dataset_name = cfg.data.dataset_name
        if not ensure_data(data_path, dataset_name):
            raise click.Abort()

        click.echo(f"Loading checkpoint from {checkpoint}...")
        try:
            model = LinkPredictionLightningModule.load_from_checkpoint(
                checkpoint,
                map_location="cpu",
            )
        except Exception as e:
            click.echo(f"Error: Failed to load model from checkpoint: {e}", err=True)
            raise click.Abort()
        model.eval()

        click.echo("Loading example data for export...")
        example_data = load_example_data_for_export(
            checkpoint_path=checkpoint,
            data_root=cfg.data.data_root,
            dataset_name=cfg.data.dataset_name,
            structural_features_cfg=cfg.data.preprocessing.get("structural_features"),
            dvc_repo_path=cfg.data.get("dvc_repo_path"),
        )

        click.echo(f"Exporting model to ONNX format: {output}...")
        export_to_onnx(
            model=model.model,
            output_path=output,
            example_data=example_data,
            opset_version=cfg.export.onnx.opset_version,
            dynamic_axes=cfg.export.onnx.get("dynamic_axes", None),
        )

        click.echo(f"✓ ONNX model exported successfully to {output}")

        if cfg.export.get("verify_after_export", True):
            click.echo("\nVerifying ONNX model...")
            verification_cfg = cfg.export.onnx.get("verification", {})
            verification_kwargs = {
                k: verification_cfg[k] for k in ["rtol", "atol"] if k in verification_cfg
            }
            is_valid = verify_onnx_model(
                onnx_path=output,
                pytorch_model=model.model,
                example_data=example_data,
                **verification_kwargs,
            )
            if not is_valid:
                click.echo("⚠ Warning: ONNX model verification failed, but export completed.")


@cli.command()
@click.option(
    "--onnx-model",
    required=True,
    help="Path to ONNX model to verify",
    type=click.Path(exists=True),
)
@click.option(
    "--checkpoint",
    required=True,
    help="Path to PyTorch checkpoint for comparison",
    type=click.Path(exists=True),
)
@click.option(
    "--config-path",
    default="configs",
    help="Path to Hydra configuration directory",
    type=click.Path(exists=True),
)
@click.option(
    "--config-name",
    default="config",
    help="Name of the configuration file (without .yaml)",
)
@click.argument("overrides", nargs=-1)
def verify_onnx(
    onnx_model: str,
    checkpoint: str,
    config_path: str,
    config_name: str,
    overrides: tuple,
):
    """Verify that ONNX model produces similar outputs to PyTorch model."""
    config_path = os.path.abspath(config_path)

    with hydra.initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=list(overrides))

    click.echo(f"Loading checkpoint from {checkpoint}...")
    model = LinkPredictionLightningModule.load_from_checkpoint(checkpoint)
    model.eval()

    click.echo("Loading example data for verification...")
    make_undirected = cfg.data.preprocessing.get("make_undirected", True)
    add_self_loops = cfg.data.preprocessing.get("add_self_loops", True)
    min_degree_clamp = cfg.data.preprocessing.get("min_degree_clamp", 1.0)

    example_data = load_example_data_for_export(
        checkpoint_path=checkpoint,
        data_root=cfg.data.data_root,
        dataset_name=cfg.data.dataset_name,
        make_undirected=make_undirected,
        add_self_loops=add_self_loops,
        min_degree_clamp=min_degree_clamp,
        structural_features_cfg=cfg.data.preprocessing.get("structural_features"),
        dvc_repo_path=cfg.data.get("dvc_repo_path"),
    )

    verification_cfg = cfg.export.onnx.get("verification", {})
    verification_kwargs = {
        k: verification_cfg[k] for k in ["rtol", "atol"] if k in verification_cfg
    }
    is_valid = verify_onnx_model(
        onnx_path=onnx_model,
        pytorch_model=model.model,
        example_data=example_data,
        **verification_kwargs,
    )

    if is_valid:
        click.echo("✓ ONNX model verification passed!")
        return 0
    else:
        click.echo("❌ ONNX model verification failed!")
        return 1


@cli.command()
@click.option(
    "--onnx-model",
    required=True,
    help="Path to ONNX model",
    type=click.Path(exists=True),
)
@click.option(
    "--model-repository",
    default="./models/triton_models",
    help="Path to Triton model repository",
    type=click.Path(),
)
@click.option(
    "--model-name",
    default="link_prediction",
    help="Name of the model in Triton",
)
@click.option(
    "--version",
    default=1,
    help="Model version",
    type=int,
)
def prepare_triton(
    onnx_model: str,
    model_repository: str,
    model_name: str,
    version: int,
):
    """Prepare ONNX model for Triton Inference Server.

    Creates the model repository structure and config.pbtxt file.
    """
    click.echo("Preparing Triton model repository...")
    click.echo(f"  ONNX model: {onnx_model}")
    click.echo(f"  Repository: {model_repository}")
    click.echo(f"  Model name: {model_name}")
    click.echo(f"  Version: {version}")

    model_dir = prepare_triton_model_repository(
        onnx_model_path=onnx_model,
        model_repository=model_repository,
        model_name=model_name,
        version=version,
    )

    click.echo(f"✓ Triton model repository prepared at {model_dir}")
    click.echo("  You can now start Triton server with:")
    click.echo("    docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \\")
    click.echo(f"      -v {Path(model_repository).absolute()}:/models \\")
    click.echo(
        "      nvcr.io/nvidia/tritonserver:24.01-py3 tritonserver --model-repository=/models"
    )


@cli.command()
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind the server to",
)
@click.option(
    "--port",
    default=8080,
    help="Port to bind the server to",
    type=int,
)
@click.option(
    "--triton-url",
    default="localhost:8000",
    help="Triton server URL",
)
@click.option(
    "--triton-model-name",
    default="link_prediction",
    help="Name of the model in Triton",
)
@click.option(
    "--data-root",
    default=None,
    help="Root directory for dataset",
    type=click.Path(),
)
@click.option(
    "--dataset-name",
    default=None,
    help="Name of the dataset",
)
@click.option(
    "--config-path",
    default="configs",
    help="Path to Hydra configuration directory",
    type=click.Path(exists=True),
)
@click.option(
    "--config-name",
    default="config",
    help="Name of the configuration file (without .yaml)",
)
@click.argument("overrides", nargs=-1)
def serve(
    host: str,
    port: int,
    triton_url: str,
    triton_model_name: str,
    data_root: str,
    dataset_name: str,
    config_path: str,
    config_name: str,
    overrides: tuple,
):
    """Start FastAPI inference server with Triton backend.

    The server provides a REST API for link prediction on a static graph.
    The graph is loaded once at startup. Only edges need to be provided in requests.
    Make sure Triton server is running before starting this server.
    """
    config_dir_abs = os.path.abspath(config_path)
    with hydra.initialize_config_dir(config_dir=config_dir_abs, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=list(overrides))

    data_root = data_root or cfg.data.data_root
    dataset_name = dataset_name or cfg.data.dataset_name
    make_undirected = cfg.data.preprocessing.get("make_undirected", True)
    add_self_loops = cfg.data.preprocessing.get("add_self_loops", True)
    min_degree_clamp = cfg.data.preprocessing.negative_sampling.get("min_degree_clamp", 1.0)

    click.echo("Starting inference server...")
    click.echo(f"  Host: {host}")
    click.echo(f"  Port: {port}")
    click.echo("  Backend: Triton")
    click.echo(f"  Triton URL: {triton_url}")
    click.echo(f"  Model name: {triton_model_name}")
    click.echo(f"  Dataset: {dataset_name}")
    click.echo(f"  Data root: {data_root}")

    try:
        server = InferenceServer(
            triton_url=triton_url,
            triton_model_name=triton_model_name,
            data_root=data_root,
            dataset_name=dataset_name,
            make_undirected=make_undirected,
            add_self_loops=add_self_loops,
            min_degree_clamp=min_degree_clamp,
            structural_features_cfg=cfg.data.preprocessing.get("structural_features"),
            dvc_repo_path=cfg.data.get("dvc_repo_path"),
        )

        click.echo(f"\n✓ Server starting at http://{host}:{port}")
        click.echo(f"  API docs: http://{host}:{port}/docs")
        click.echo(f"  Health check: http://{host}:{port}/health")

        uvicorn.run(server.app, host=host, port=port, log_level="info")

    except Exception as e:
        click.echo(f"Error starting server: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()
