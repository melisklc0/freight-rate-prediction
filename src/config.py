from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TRAIN_FILE = DATA_DIR / "train_test.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
DECEMBER_FILE = DATA_DIR / "december_chart_inputs.csv"
TEMPLATE_FILE = DATA_DIR / "validation_predictions_template.csv"

SEED = 42

# noise thresholds, applied to training data only
MIN_RATE_PER_MILE = 1.0
MAX_RATE_PER_MILE = 3.5
MAX_DISTANCE_RATIO = 1.6

# validation mimics the real task: train on the past, predict the next two months
HOLDOUT_START = "2025-09-01"

# extra two-month windows to check that a decision holds in more than one period
TIME_FOLDS = [
    ("2025-05-01", "2025-07-01"),
    ("2025-07-01", "2025-09-01"),
    ("2025-09-01", "2025-11-01"),
]

LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "random_state": SEED,
    "verbose": -1,
}

UNSEEN_CITIES = 8
UNSEEN_CITY_ROUNDS = 5
