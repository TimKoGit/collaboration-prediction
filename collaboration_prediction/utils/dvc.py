"""DVC utilities for data versioning and management."""

import logging
import os
from pathlib import Path
from typing import Optional

from dvc.exceptions import DvcException
from dvc.repo import Repo

from collaboration_prediction.data_modules.data_download import download_data

logger = logging.getLogger(__name__)


def ensure_dvc_repo(root_dir: Optional[Path] = None) -> bool:
    """Ensure DVC repository is initialized.

    Args:
        root_dir: Root directory of the project (default: current working directory)

    Returns:
        True if DVC repo exists or was successfully initialized, False otherwise
    """
    if root_dir is None:
        root_dir = Path.cwd()
    else:
        root_dir = Path(root_dir)

    dvc_dir = root_dir / ".dvc"

    if dvc_dir.exists() and (dvc_dir / "config").exists():
        return True

    try:
        logger.info("Initializing DVC repository...")
        original_cwd = os.getcwd()
        try:
            os.chdir(str(root_dir))
            Repo.init()
            logger.info(f"DVC repository initialized at {root_dir}")
            return True
        finally:
            os.chdir(original_cwd)
    except Exception as e:
        logger.warning(f"Failed to initialize DVC repository: {e}")
        return False


def dvc_pull(data_path: str, root_dir: Optional[Path] = None) -> bool:
    """Pull data from DVC storage.

    Args:
        data_path: Path to data file or directory tracked by DVC
        root_dir: Root directory of the project (default: current working directory)

    Returns:
        True if data was successfully pulled, False otherwise
    """
    if root_dir is None:
        root_dir = Path.cwd()
    else:
        root_dir = Path(root_dir)

    if not ensure_dvc_repo(root_dir):
        return False

    try:
        repo = Repo(str(root_dir))

        data_path_obj = Path(data_path)
        if not data_path_obj.is_absolute():
            data_path_obj = root_dir / data_path_obj

        logger.info(f"Pulling data from DVC: {data_path}")
        repo.pull([str(data_path_obj.relative_to(root_dir))])
        logger.info(f"Data successfully pulled to {data_path_obj}")
        return True

    except DvcException as e:
        logger.info(f"DVC pull failed (data may not be in cache): {e}")
        logger.info("Will attempt to download data from source...")
        return False
    except Exception as e:
        logger.warning(f"Failed to pull data from DVC: {e}")
        return False


def dvc_add(data_path: str, root_dir: Optional[Path] = None) -> bool:
    """Add data to DVC tracking.

    Args:
        data_path: Path to data file or directory to track with DVC
        root_dir: Root directory of the project (default: current working directory)

    Returns:
        True if data was successfully added to DVC, False otherwise
    """
    if root_dir is None:
        root_dir = Path.cwd()
    else:
        root_dir = Path(root_dir)

    if not ensure_dvc_repo(root_dir):
        return False

    try:
        repo = Repo(str(root_dir))

        data_path_obj = Path(data_path)
        if data_path_obj.is_absolute():
            try:
                data_path_rel = data_path_obj.relative_to(root_dir)
            except ValueError:
                logger.warning(f"{data_path} is not under {root_dir}")
                return False
        else:
            data_path_rel = data_path_obj

        dvc_file = root_dir / f"{data_path_rel}.dvc"
        if dvc_file.exists():
            logger.info(f"Data already tracked by DVC: {data_path_rel}")
            return True

        logger.info(f"Adding data to DVC tracking: {data_path_rel}")
        repo.add(str(data_path_rel))
        logger.info(f"Data successfully added to DVC: {data_path_rel}")
        return True

    except Exception as e:
        logger.warning(f"Failed to add {data_path} to DVC: {e}")
        return False


def ensure_data(
    data_path: str,
    dataset_name: str,
    root_dir: Optional[Path] = None,
    download_if_missing: bool = True,
) -> bool:
    """Ensure data is available, pulling from DVC or downloading if needed.

    Args:
        data_path: Path to data directory
        dataset_name: Name of the dataset (for downloading)
        root_dir: Root directory of the project
        download_if_missing: If True, download data if DVC pull fails

    Returns:
        True if data is available, False otherwise
    """
    if root_dir is None:
        root_dir = Path.cwd()
    else:
        root_dir = Path(root_dir)

    data_path_obj = Path(data_path)
    if not data_path_obj.is_absolute():
        data_path_obj = root_dir / data_path_obj

    dataset_name_underscore = dataset_name.replace("-", "_")
    dataset_path = data_path_obj / dataset_name
    dataset_path_alt = data_path_obj / dataset_name_underscore

    data_path_rel = (
        data_path_obj.relative_to(root_dir)
        if data_path_obj.is_relative_to(root_dir)
        else Path(data_path)
    )

    for path_to_check in [dataset_path, dataset_path_alt]:
        if path_to_check.exists() and any(path_to_check.iterdir()):
            logger.info(f"Data already exists at {path_to_check}")
            dvc_add(str(data_path_rel), root_dir)
            return True

    logger.info(
        f"Data not found at {dataset_path} or {dataset_path_alt}, attempting to pull from DVC..."
    )
    if dvc_pull(str(data_path_rel), root_dir):
        for path_to_check in [dataset_path, dataset_path_alt]:
            if path_to_check.exists() and any(path_to_check.iterdir()):
                return True

    if download_if_missing:
        logger.info("DVC pull failed, downloading data from source...")

        try:
            download_data(
                dataset_name=dataset_name, data_root=str(data_path_obj), force_download=False
            )

            for path_to_check in [dataset_path, dataset_path_alt]:
                if path_to_check.exists() and any(path_to_check.iterdir()):
                    logger.info(f"Download verified at {path_to_check}")
                    dvc_add(str(data_path_rel), root_dir)
                    return True

            logger.error(
                f"Download completed but data not found at {dataset_path} or {dataset_path_alt}"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            return False

    return False
