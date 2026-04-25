from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
DATA_FILE = PACKAGE_DIR / "snake_dataset.csv"
SMALL_MODEL_PATH = PACKAGE_DIR / "snake_smallnet.pth"
BIG_MODEL_PATH = PACKAGE_DIR / "snake_biggernet.pth"
MODEL_PATH = PACKAGE_DIR / "snake_model_Y.pth"
