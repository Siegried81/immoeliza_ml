"""Where the data and the model artefacts live, relative to the repository.

Every script used to carry its own absolute path to one developer's desktop,
drive letter included, so a clone on any other machine failed at the first
`read_json`. The repository root is derived from this file's
location, which holds wherever the repository is checked out and whatever the
current working directory is. Override with the `IMMOELIZA_DATA_DIR` environment
variable when the data sits elsewhere (an external drive, a CI cache).
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("IMMOELIZA_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA = DATA_DIR / "raw" / "dataframe.json"
CLEAN_DIR = DATA_DIR / "clean"
CLEAN_DATA = CLEAN_DIR / "cleaned_data.json"
MISSING_VALUES_PLOT = DATA_DIR / "missing_values_plot.png"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_xgboost.joblib"
IMAGE_DIR = PROJECT_ROOT / "images"
