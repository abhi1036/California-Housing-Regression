from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model Paths
MODEL_PATH = BASE_DIR / "models" / "linear_regression.pkl"
FEATURE_PATH = BASE_DIR / "models" / "features.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.pkl"

# Train-Test Split
TEST_SIZE = 0.2
RANDOM_STATE = 42