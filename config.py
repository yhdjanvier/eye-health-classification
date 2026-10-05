"""
config.py - single source of truth for paths, constants and class information.

WHAT : plain Python constants shared by every script (training, evaluation, app).
WHY  : one place to change image size / paths / hyper-parameters guarantees that
       training, evaluation and the Streamlit app always use the SAME settings
       (this is what prevents "training/serving skew" bugs).
WHERE: imported by src/*.py and app.py.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (all relative to the project root, so the project can be moved/cloned)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"      # put the Kaggle dataset here
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SAMPLES_DIR = PROJECT_ROOT / "sample_images"

MODEL_PATH = MODELS_DIR / "eye_health_cnn.keras"           # trained model
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"         # label order
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"       # preprocessing + hyper-params
MANIFEST_PATH = RESULTS_DIR / "dataset_manifest.csv"       # every valid image + metadata
SPLITS_PATH = RESULTS_DIR / "splits.csv"                   # train/val/test assignment
HISTORY_PATH = RESULTS_DIR / "training_history.csv"
EXPERIMENTS_PATH = RESULTS_DIR / "experiments.csv"         # log of every training run
METRICS_PATH = RESULTS_DIR / "metrics.json"
PER_CLASS_PATH = RESULTS_DIR / "per_class_metrics.csv"

# ----------------------------------------------------------------------------
# Data / preprocessing
# ----------------------------------------------------------------------------
IMG_SIZE = (224, 224)      # MobileNetV2's native size; good speed/accuracy trade-off
BATCH_SIZE = 32            # fits in 8 GB RAM / CPU-only laptops; stable gradients
SEED = 42                  # reproducible splits, shuffling and weight init
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# The dataset folder that means "no disease". Used for the Healthy-vs-Diseased view.
HEALTHY_CLASS = "normal"

# ----------------------------------------------------------------------------
# Training hyper-parameters (defaults; can be overridden from the command line)
# ----------------------------------------------------------------------------
DROPOUT = 0.30
DENSE_UNITS = 256
L2_WEIGHT = 1e-4
LR_HEAD = 1e-3             # phase 1: new classifier head on frozen backbone
LR_FINETUNE = 1e-5         # phase 2: gentle fine-tuning of the top backbone layers
EPOCHS_HEAD = 12
EPOCHS_FINETUNE = 20
EPOCHS_CUSTOM = 40
UNFREEZE_LAYERS = 40       # top layers of MobileNetV2 unfrozen in phase 2
EARLY_STOP_PATIENCE = 6
DEFAULT_CONFIDENCE_THRESHOLD = 0.70   # replaced by a value chosen on validation data

# ----------------------------------------------------------------------------
# Class information shown in the app (keys = dataset folder names, lower-case)
# ----------------------------------------------------------------------------
CLASS_INFO = {
    "normal": {
        "display": "Healthy (Normal)",
        "healthy": True,
        "description": (
            "The retinal image does not show the visual patterns the model associates "
            "with cataract, diabetic retinopathy or glaucoma."
        ),
        "advice": "Keep up routine eye check-ups, especially if you have diabetes or a family history of eye disease.",
    },
    "cataract": {
        "display": "Cataract",
        "healthy": False,
        "description": (
            "Cataract is a clouding of the eye's lens. In the training data, cataract images "
            "typically look hazy or blurred with low contrast in the retinal details."
        ),
        "advice": "Please have an ophthalmologist examine your eyes to confirm and discuss treatment options.",
    },
    "diabetic_retinopathy": {
        "display": "Diabetic Retinopathy",
        "healthy": False,
        "description": (
            "Diabetic retinopathy is damage to the retina's blood vessels caused by diabetes. "
            "Images often show tiny haemorrhages, micro-aneurysms or yellowish exudates."
        ),
        "advice": "Please see an eye specialist soon; early treatment greatly reduces the risk of vision loss.",
    },
    "glaucoma": {
        "display": "Glaucoma",
        "healthy": False,
        "description": (
            "Glaucoma is a group of conditions that damage the optic nerve, often linked to raised "
            "eye pressure. Images may show an enlarged optic cup or thinning of the nerve rim."
        ),
        "advice": "Please arrange an eye-pressure test and optic-nerve examination with an ophthalmologist.",
    },
}


def get_class_info(class_name: str) -> dict:
    """Return display info for a class; works even for unknown dataset folders."""
    key = class_name.strip().lower()
    if key in CLASS_INFO:
        return CLASS_INFO[key]
    return {
        "display": class_name.replace("_", " ").title(),
        "healthy": key == HEALTHY_CLASS,
        "description": "A class defined by the training dataset.",
        "advice": "Please consult a qualified eye-care professional.",
    }


def ensure_dirs() -> None:
    for d in (RAW_DATA_DIR, MODELS_DIR, RESULTS_DIR, SAMPLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
