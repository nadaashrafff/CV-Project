"""Mount Google Drive and extract the dataset ZIP to Colab local storage."""
import zipfile
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.paths import COLAB_DATASET, DATASET_ZIP

logger = get_logger(__name__)


def mount_drive() -> None:
    from google.colab import drive
    drive.mount("/content/drive")
    logger.info("Google Drive mounted")


def extract_dataset(
    zip_path: Path = DATASET_ZIP,
    target: Path = COLAB_DATASET,
    force: bool = False,
) -> Path:
    if target.exists() and not force:
        logger.info(f"Dataset already at {target}, skipping extraction")
        return target
    logger.info(f"Extracting {zip_path} → {target}")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target)
    logger.info("Extraction complete")
    return target


def discover_dataset_root(base: Path = COLAB_DATASET) -> Path:
    """Walk extracted directory to find the folder containing data.yaml."""
    for p in sorted(base.rglob("data.yaml")):
        return p.parent
    raise FileNotFoundError(f"data.yaml not found under {base}")
