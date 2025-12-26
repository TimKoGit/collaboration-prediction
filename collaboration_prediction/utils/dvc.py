"""DVC utilities for data versioning and management."""

import logging
import os
from pathlib import Path
from typing import Optional

from dvc.exceptions import DvcException
from dvc.repo import Repo

from collaboration_prediction.data_modules.data_download import download_data

logger = logging.getLogger(__name__)


def ensure_dvc_repo(dvc_root: Path) -> bool:
    """Ensure DVC repository is initialized in the specified directory.

    Args:
        dvc_root: Path to the DVC repository root

    Returns:
        True if DVC repo exists or was successfully initialized, False otherwise
    """
    dvc_root = Path(dvc_root).resolve()
    dvc_dir = dvc_root / ".dvc"

    if dvc_dir.exists() and (dvc_dir / "config").exists():
        return True

    try:
        dvc_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing DVC repository at {dvc_root}...")
        original_cwd = os.getcwd()
        try:
            os.chdir(str(dvc_root))
            Repo.init(no_scm=True)
            logger.info(f"DVC repository initialized at {dvc_root} (no-scm)")
            return True
        finally:
            os.chdir(original_cwd)
    except Exception as e:
        logger.warning(f"Failed to initialize DVC repository at {dvc_root}: {e}")
        return False


def dvc_pull(data_path: str, dvc_root: Path) -> bool:
    """Pull data from DVC storage.

    Args:
        data_path: Path to data file or directory tracked by DVC (relative to dvc_root)
        dvc_root: Path to the DVC repository root

    Returns:
        True if data was successfully pulled, False otherwise
    """
    dvc_root = Path(dvc_root).resolve()
    if not ensure_dvc_repo(dvc_root):
        return False

    original_cwd = os.getcwd()
    try:
        os.chdir(str(dvc_root))
        repo = Repo(".")
        logger.info(f"Pulling data from DVC: {data_path}")
        repo.pull([str(data_path)])
        logger.info(f"Data successfully pulled to {dvc_root / data_path}")
        return True

    except DvcException as e:
        logger.info(f"DVC pull failed: {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to pull data from DVC: {e}")
        return False
    finally:
        os.chdir(original_cwd)


def dvc_add(data_path: str, dvc_root: Path) -> bool:
    """Add data to DVC tracking.

    Args:
        data_path: Path to data file or directory to track (relative to dvc_root or absolute)
        dvc_root: Path to the DVC repository root

    Returns:
        True if data was successfully added to DVC, False otherwise
    """
    dvc_root = Path(dvc_root).resolve()
    if not ensure_dvc_repo(dvc_root):
        return False

    original_cwd = os.getcwd()
    try:
        os.chdir(str(dvc_root))
        repo = Repo(".")

        data_path_obj = Path(data_path)
        if data_path_obj.is_absolute():
            try:
                data_path_rel = data_path_obj.relative_to(dvc_root)
            except ValueError:
                logger.warning(f"{data_path} is not under {dvc_root}")
                return False
        else:
            data_path_rel = data_path_obj

        dvc_file = dvc_root / f"{data_path_rel}.dvc"
        if dvc_file.exists():
            logger.info(f"Data already tracked by DVC: {data_path_rel}")
            return True

        logger.info(f"Adding data to DVC tracking: {data_path_rel}")
        repo.add(str(data_path_rel))
        logger.info(f"Data successfully added to DVC: {data_path_rel}")
        return True

    except Exception as e:
        logger.warning(f"Failed to add {data_path} to DVC in {dvc_root}: {e}")
        return False
    finally:
        os.chdir(original_cwd)


def init_external_data(
    dvc_repo_path: str, data_path: str, dataset_name: str, force_download: bool = False
) -> bool:
    """Manually initialize external DVC repo and upload data.

    Args:
        dvc_repo_path: Path to the external DVC repo
        data_path: Path where data should be stored (relative to dvc_repo_path)
        dataset_name: Name of the dataset to download
        force_download: Whether to force download even if data exists

    Returns:
        True if successfully initialized and data added, False otherwise
    """
    dvc_root = Path(dvc_repo_path).resolve()
    if not ensure_dvc_repo(dvc_root):
        return False

    abs_data_path = dvc_root / data_path
    abs_data_path.mkdir(parents=True, exist_ok=True)

    try:
        download_data(
            dataset_name=dataset_name, data_root=str(abs_data_path), force_download=force_download
        )

        dataset_name_underscore = dataset_name.replace("-", "_")
        found = False
        for path_to_check in [
            abs_data_path / dataset_name,
            abs_data_path / dataset_name_underscore,
        ]:
            if path_to_check.exists() and any(path_to_check.iterdir()):
                found = True
                break

        if found:
            return dvc_add(data_path, dvc_root)
        else:
            logger.error(f"Download completed but data not found in {abs_data_path}")
            return False

    except Exception as e:
        logger.error(f"Failed to initialize data: {e}")
        return False


def ensure_data(
    data_path: str,
    dataset_name: str,
    dvc_repo_path: Optional[str] = None,
) -> bool:
    """Check if data is available. No automatic DVC pull/add here as per new requirements.

    Args:
        data_path: Path to data directory
        dataset_name: Name of the dataset
        dvc_repo_path: Path to DVC repository (unused for now, but kept for signature)

    Returns:
        True if data is available, False otherwise
    """
    data_path_obj = Path(data_path).resolve()

    dataset_name_underscore = dataset_name.replace("-", "_")
    dataset_path = data_path_obj / dataset_name
    dataset_path_alt = data_path_obj / dataset_name_underscore

    for path_to_check in [dataset_path, dataset_path_alt]:
        if path_to_check.exists() and any(path_to_check.iterdir()):
            logger.info(f"Data already exists at {path_to_check}")
            return True

    logger.error(
        f"Data not found at {dataset_path} or {dataset_path_alt}. "
        f"Please run 'collab-pred init-data' first."
    )
    return False
