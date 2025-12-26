"""Data downloading utilities for DVC local storage."""

import logging
from pathlib import Path

from ogb.linkproppred import PygLinkPropPredDataset

logger = logging.getLogger(__name__)


def download_data(
    dataset_name: str = "ogbl-collab", data_root: str = "./data", force_download: bool = False
) -> Path:
    """Download dataset from OGB to local storage.

    This function downloads the dataset from Open Graph Benchmark (OGB)
    and saves it to the specified data root directory. The downloaded data
    will then be tracked by DVC for version control.

    Args:
        dataset_name: Name of the OGB dataset (default: 'ogbl-collab')
        data_root: Root directory where data will be stored
        force_download: If True, force re-download even if data exists

    Returns:
        Path to the downloaded dataset directory
    """
    data_root_path = Path(data_root)
    # OGB uses underscores in directory names (ogbl_collab) but hyphens in dataset names
    dataset_name_underscore = dataset_name.replace("-", "_")
    dataset_path = data_root_path / dataset_name
    dataset_path_alt = data_root_path / dataset_name_underscore

    for path_to_check in [dataset_path, dataset_path_alt]:
        if path_to_check.exists() and not force_download:
            logger.info(f"Dataset already exists at {path_to_check}")
            logger.info("Use force_download=True to re-download")
            return path_to_check

    data_root_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading dataset '{dataset_name}' to {data_root_path}...")

    try:
        PygLinkPropPredDataset(name=dataset_name, root=str(data_root_path))
        logger.info("Dataset object created successfully")

        for path_to_check in [dataset_path_alt, dataset_path]:
            if path_to_check.exists():
                if any(path_to_check.iterdir()):
                    logger.info(f"Dataset verified at {path_to_check}")
                    return path_to_check
                else:
                    logger.warning(f"Dataset directory exists but is empty: {path_to_check}")
                    return path_to_check

        raise FileNotFoundError(
            f"Dataset download completed but path not found. "
            f"Checked: {dataset_path} and {dataset_path_alt}"
        )

    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}") from e
