from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

PREPROCESSING = 0
TRAINING = 1
TRACKING_MLFLOW = "https://dagshub.com/PabloGete/TAED2_scooby_doo_team.mlflow"

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
METRICS_DIR = PROJ_ROOT / "metrics"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_TRAIN_IMAGES = PROCESSED_DATA_DIR / "train_images"
PROCESSED_VALID_IMAGES = PROCESSED_DATA_DIR / "valid_images"
PROCESSED_TEST_IMAGES = PROCESSED_DATA_DIR / "test_images"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
