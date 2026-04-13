"""
preprocessing.py
────────────────
Two pipelines:
  1. image_pipeline  – grayscale → blur → threshold → resize (for CNN)
  2. landmark_pipeline – MediaPipe extraction → normalise (for Landmark model)

Also provides extract_landmarks_from_dataset() which pre-computes landmarks
from dataset images and saves them as .npy for fast landmark model training.
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, normalize_landmarks, TRAIN_DIR, TEST_DIR, PROCESSED_DIR, IMG_SIZE, LANDMARK_DIM

log = get_logger("preprocessing")

mp_hands = mp.solutions.hands
_hands_static = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)


# ── Image pipeline ─────────────────────────────────────────────────────────────
def image_pipeline(frame: np.ndarray) -> np.ndarray:
    """
    Input : BGR or grayscale frame (any size).
    Output: (IMG_SIZE, IMG_SIZE, 1) float32 in [0,1].
    """
    if frame is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2,
    )
    resized = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE))
    return (resized.astype(np.float32) / 255.0)[..., np.newaxis]


# ── Landmark pipeline ──────────────────────────────────────────────────────────
def extract_landmarks_from_frame(frame: np.ndarray, hands_instance=None):
    """
    Extract normalised landmark vector from a live BGR frame.
    Returns (np.ndarray shape (LANDMARK_DIM,), annotated_frame) or (None, frame).
    """
    if frame is None:
        return None, frame
    h_inst = hands_instance or _hands_static
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = h_inst.process(rgb)
    if not result.multi_hand_landmarks:
        return None, frame

    hand = result.multi_hand_landmarks[0]
    raw = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
    normalised = normalize_landmarks(raw)

    # draw
    mp.solutions.drawing_utils.draw_landmarks(
        frame, hand, mp_hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style(),
    )
    return normalised, frame


# ── Pre-compute landmarks from dataset images ──────────────────────────────────
def extract_landmarks_from_dataset(split: str = "train"):
    """
    Iterates TRAIN_DIR or TEST_DIR images, extracts landmarks, saves:
      data/processed/{split}_landmarks.npy  – shape (N, LANDMARK_DIM)
      data/processed/{split}_labels.npy     – shape (N,)  int labels
      data/processed/class_names.txt
    """
    from utils import load_label_map
    try:
        label_map = load_label_map()
    except FileNotFoundError:
        log.error("label_map.json not found — run dataset_loader.py first.")
        sys.exit(1)

    inv = {v: k for k, v in label_map.items()}
    root = TRAIN_DIR if split == "train" else TEST_DIR

    out_lm  = PROCESSED_DIR / f"{split}_landmarks.npy"
    out_lbl = PROCESSED_DIR / f"{split}_labels.npy"

    if out_lm.exists() and out_lbl.exists():
        log.info(f"Processed {split} landmarks already exist — skipping.")
        return np.load(out_lm), np.load(out_lbl)

    log.info(f"Extracting landmarks for {split} split …")
    from tqdm import tqdm

    X, Y = [], []
    for cls_dir in tqdm(sorted(root.iterdir()), desc="Classes"):
        cls = cls_dir.name.upper()
        if cls not in inv:
            log.warning(f"Class {cls} not in label_map — skipping.")
            continue
        label_idx = inv[cls]
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            lm, _ = extract_landmarks_from_frame(img)
            if lm is None:
                # MediaPipe failed → zero vector (will be low-confidence at inference)
                lm = np.zeros(LANDMARK_DIM, dtype=np.float32)
            X.append(lm)
            Y.append(label_idx)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)
    np.save(out_lm, X)
    np.save(out_lbl, Y)
    log.info(f"Saved {split}: X={X.shape}  Y={Y.shape}")

    # save class names
    with open(PROCESSED_DIR / "class_names.txt", "w") as f:
        for k in sorted(label_map.keys()):
            f.write(f"{k}:{label_map[k]}\n")

    return X, Y


# ── Augment image batch ────────────────────────────────────────────────────────
def augment_image(img: np.ndarray) -> np.ndarray:
    """Random flip, rotation, brightness — applied to (H,W,1) float32."""
    sq = img[..., 0]
    if np.random.rand() > 0.5:
        sq = cv2.flip(sq, 1)
    angle = np.random.uniform(-15, 15)
    h, w = sq.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    sq = cv2.warpAffine(sq, M, (w, h))
    sq = np.clip(sq * np.random.uniform(0.8, 1.2), 0, 1)
    return sq[..., np.newaxis].astype(np.float32)