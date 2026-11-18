from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CV_DIR = RESULTS / "cv"
TEST_DIR = RESULTS / "test"
FIG_DIR = RESULTS / "figures"
MODEL_DIR = RESULTS / "models"
LOG_DIR = RESULTS / "logs"

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 1          # tang len 3 cho cac so sanh chinh
PRIMARY_METRIC = "macro_f1"