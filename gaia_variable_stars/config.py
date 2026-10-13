from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
INTERIM_ROOT = DATA_ROOT / "interim"
PROCESSED_ROOT = DATA_ROOT / "processed"

MODELS_ROOT = PROJECT_ROOT / "models"
REPORTS_ROOT = PROJECT_ROOT / "reports"
FIGURES_ROOT = REPORTS_ROOT / "figures"

SOURCE_FILE = RAW_ROOT / "gaia_sources.csv"
BATCH_ROOT = RAW_ROOT / "photometry_batches"
PHOTOMETRY_FILE = RAW_ROOT / "gaia_photometry.csv"

LABELED_FILE = INTERIM_ROOT / "gaia_labeled.csv"
CLEAN_FILE = INTERIM_ROOT / "gaia_clean.csv"

FEATURE_FILE = PROCESSED_ROOT / "gaia_features.csv"
MIN_POINTS = 20

SEED = 42
TEST_SIZE = 0.2
FOLDS = 5

CLASSES = [
    "RR Lyrae",
    "Cepheid",
    "Eclipsing Binary",
    "Long-Period Variable",
]

FEATURES = [
    "magnitude_mean",
    "magnitude_std",
    "magnitude_skew",
    "magnitude_kurt",
    "amplitude",
    "log_period",
    "power",
    "fourier_amp",
    "fourier_ratio",
    "bp_rp_color",
]

MODEL_FILE = MODELS_ROOT / "model.joblib"
CV_FILE = MODELS_ROOT / "cv.csv"
RANK_FILE = MODELS_ROOT / "rank.csv"
TEST_FILE = MODELS_ROOT / "test.csv"
PRED_FILE = MODELS_ROOT / "pred.csv"