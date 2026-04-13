import logging
import os
import json
from pathlib import Path
import numpy as np

# ── Logging ────────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler("logs/isl.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


log = get_logger("utils")

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR= DATA_DIR / "processed"
TRAIN_DIR    = DATA_DIR / "train"
TEST_DIR     = DATA_DIR / "test"
MODEL_DIR    = PROJECT_ROOT / "models"
LOGS_DIR     = PROJECT_ROOT / "logs"

for _d in [RAW_DIR, PROCESSED_DIR, TRAIN_DIR, TEST_DIR, MODEL_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

MODEL_H5_PATH   = MODEL_DIR / "cnn_model.h5"
LANDMARK_H5_PATH= MODEL_DIR / "landmark_model.h5"
LABEL_MAP_PATH  = MODEL_DIR / "label_map.json"

IMG_SIZE   = 128
NUM_CHANNELS = 1          # grayscale
LANDMARK_DIM = 63         # 21 landmarks × 3 (x,y,z)

# Simulated special gesture timings (seconds)
SPACE_TIMEOUT  = 2.0      # no gesture for 2s → insert space
DELETE_HOLD    = 1.5      # same gesture held for 1.5s → delete last letter


# ── Label helpers ──────────────────────────────────────────────────────────────
def build_label_map(class_names: list) -> dict:
    """int → class_name"""
    return {i: c for i, c in enumerate(sorted(class_names))}


def save_label_map(label_map: dict, path=LABEL_MAP_PATH):
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
    log.info(f"Label map saved → {path}")


def load_label_map(path=LABEL_MAP_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label map not found: {path}")
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def labels_to_int(label_map: dict) -> dict:
    """class_name → int"""
    return {v: k for k, v in label_map.items()}


# ── Normalise landmarks ────────────────────────────────────────────────────────
def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    landmarks: (21, 3) array of (x, y, z) in [0,1] range from MediaPipe.
    Centre around wrist (lm[0]), scale by hand span.
    """
    if landmarks is None or len(landmarks) == 0:
        return np.zeros(LANDMARK_DIM, dtype=np.float32)
    lm = landmarks.reshape(21, 3).copy()
    wrist = lm[0].copy()
    lm -= wrist
    scale = np.linalg.norm(lm[9])   # middle-finger MCP as reference
    if scale > 0:
        lm /= scale
    return lm.flatten().astype(np.float32)


# ── Confusion-pair rule fix ────────────────────────────────────────────────────
CONFUSION_PAIRS = {
    frozenset(["D", "R", "U"]): None,
    frozenset(["T", "K", "I"]): None,
    frozenset(["S", "M", "N"]): None,
}

def confusion_correct(label: str, confidence: float, top_preds: list) -> str:
    """
    top_preds: [(label, prob), ...] sorted descending.
    If top-2 labels are a known confusion pair AND gap < 0.15, return higher-voted
    label from smoothing buffer (caller should pass buffer result here).
    This function acts as a gate — it returns the original label if no confusion.
    """
    if len(top_preds) < 2:
        return label
    second_label, second_conf = top_preds[1]
    gap = confidence - second_conf
    if gap < 0.15:
        pair = frozenset([label, second_label])
        for known_pair in CONFUSION_PAIRS:
            if pair <= known_pair:
                log.debug(f"Confusion detected {label}/{second_label} gap={gap:.3f}")
                return label   # smoothing buffer downstream will stabilise
    return label