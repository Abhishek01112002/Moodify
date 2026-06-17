from pathlib import Path
import zipfile
import logging
import kaggle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_dataset():
    """Almost Million Songs Dataset 2025 (~1 Million tracks)"""
    dataset_slug = "anantsinghal786/almost-million-songs-dataset-2025-16-features"
    
    logger.info(f"📥 Downloading {dataset_slug} ... (approx 63 MB)")
    kaggle.api.dataset_download_files(dataset_slug, path=DATA_DIR, unzip=False)
    
    # Unzip
    zip_path = next(DATA_DIR.glob("*.zip"))
    logger.info(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    # Show downloaded files
    files = list(DATA_DIR.glob("*.csv"))
    logger.info(f"✅ Download complete! Files found: {len(files)}")
    for f in files:
        logger.info(f"   → {f.name} ({f.stat().st_size / 1_000_000:.1f} MB)")
    
    logger.info("Raw data ready for preprocessing!")

if __name__ == "__main__":
    download_dataset()
