"""
dataset_loader.py
─────────────────
Downloads the ASL Alphabet dataset from Kaggle (used as ISL proxy when
the ISL-digits dataset is not large enough for A-Z letters).

WHY ASL AS PROXY:
  The Kaggle ISL dataset (ardamavi/indian-sign-language-digits) covers only
  digits 0-9.  For a full A-Z letter system we use the ASL Alphabet dataset
  (grassknoted/asl-alphabet), which contains 87,000 images across 29 classes
  (A-Z + SPACE + DELETE + NOTHING).  Hand shapes for many letters overlap
  between ASL and ISL, making it a valid proxy for a demo system.

DATASET LINK:
  https://www.kaggle.com/datasets/grassknoted/asl-alphabet

SETUP (one-time):
  1. pip install kaggle
  2. Place ~/.kaggle/kaggle.json  (API credentials from kaggle.com/account)
  3. python src/dataset_loader.py
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

# allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    get_logger, RAW_DIR, TRAIN_DIR, TEST_DIR, PROCESSED_DIR,
    build_label_map, save_label_map,
)
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from tqdm import tqdm

log = get_logger("dataset_loader")

KAGGLE_DATASET = "grassknoted/asl-alphabet"
DATASET_ZIP    = RAW_DIR / "asl-alphabet.zip"
EXTRACTED_DIR  = RAW_DIR / "asl_alphabet_train" / "asl_alphabet_train"

# Expected class names from the ASL dataset
ASL_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

IMG_SIZE = 128


# ── Download ───────────────────────────────────────────────────────────────────
def download_dataset():
    if EXTRACTED_DIR.exists() and any(EXTRACTED_DIR.iterdir()):
        log.info("Dataset already extracted — skipping download.")
        return
    try:
        import kaggle  # noqa: F401
    except ImportError:
        log.error("kaggle package not installed.  Run: pip install kaggle")
        sys.exit(1)

    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists():
        log.error(
            "Kaggle credentials not found at ~/.kaggle/kaggle.json\n"
            "  1. Go to https://www.kaggle.com/account\n"
            "  2. Create API token\n"
            "  3. Place kaggle.json at ~/.kaggle/kaggle.json"
        )
        sys.exit(1)

    log.info(f"Downloading {KAGGLE_DATASET} …")
    os.makedirs(RAW_DIR, exist_ok=True)
    import kaggle as kg
    kg.api.authenticate()
    kg.api.dataset_download_files(
        KAGGLE_DATASET, path=str(RAW_DIR), unzip=False, quiet=False
    )

    # rename whatever zip was downloaded
    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        log.error("No zip found after download.")
        sys.exit(1)
    zips[0].rename(DATASET_ZIP)
    log.info(f"Downloaded → {DATASET_ZIP}")


def extract_dataset():
    if EXTRACTED_DIR.exists() and any(EXTRACTED_DIR.iterdir()):
        log.info("Already extracted.")
        return
    log.info("Extracting …")
    with zipfile.ZipFile(DATASET_ZIP, "r") as z:
        z.extractall(RAW_DIR)
    log.info("Extraction complete.")


# ── Build train/test splits ────────────────────────────────────────────────────
def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def build_splits(test_size: float = 0.15, max_per_class: int = 1000):
    """
    Walk EXTRACTED_DIR, read images, split, save to TRAIN_DIR / TEST_DIR.
    max_per_class caps images per class to keep memory manageable.
    """
    if (TRAIN_DIR / "A").exists():
        log.info("Splits already exist — skipping.")
        return _collect_class_names()

    log.info("Building train/test splits …")
    source = EXTRACTED_DIR
    if not source.exists():
        log.error(f"Extracted dir not found: {source}")
        sys.exit(1)

    class_dirs = sorted([d for d in source.iterdir() if d.is_dir()])
    class_names = [d.name.upper() for d in class_dirs]

    for cls_dir in tqdm(class_dirs, desc="Classes"):
        cls_name = cls_dir.name.upper()
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        imgs = imgs[:max_per_class]

        train_imgs, test_imgs = train_test_split(imgs, test_size=test_size, random_state=42)

        for split_name, split_imgs in [("train", train_imgs), ("test", test_imgs)]:
            out_dir = (TRAIN_DIR if split_name == "train" else TEST_DIR) / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for src in split_imgs:
                shutil.copy2(src, out_dir / src.name)

    log.info(f"Splits ready  train={TRAIN_DIR}  test={TEST_DIR}")
    return class_names


def _collect_class_names():
    return sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])


# ── Label map ──────────────────────────────────────────────────────────────────
def create_label_map():
    class_names = _collect_class_names()
    if not class_names:
        log.error("No classes found in TRAIN_DIR.")
        sys.exit(1)
    lm = build_label_map(class_names)
    save_label_map(lm)
    log.info(f"Label map: {len(lm)} classes → {list(lm.values())[:5]} …")
    return lm


# ── Generators for training ────────────────────────────────────────────────────
def image_generator(split: str, label_map: dict, batch_size: int = 32):
    """Yields (batch_images, batch_labels) indefinitely."""
    inv = {v: k for k, v in label_map.items()}
    root = TRAIN_DIR if split == "train" else TEST_DIR
    paths, labels = [], []
    for cls_dir in sorted(root.iterdir()):
        cls = cls_dir.name.upper()
        if cls not in inv:
            continue
        for img_path in cls_dir.glob("*.jpg"):
            paths.append(img_path)
            labels.append(inv[cls])
        for img_path in cls_dir.glob("*.png"):
            paths.append(img_path)
            labels.append(inv[cls])

    paths = np.array(paths)
    labels = np.array(labels)
    idx = np.arange(len(paths))
    np.random.shuffle(idx)
    paths, labels = paths[idx], labels[idx]

    i = 0
    while True:
        batch_p = paths[i: i + batch_size]
        batch_l = labels[i: i + batch_size]
        i += batch_size
        if i >= len(paths):
            i = 0

        imgs = []
        for p in batch_p:
            img = _load_image(p)
            if img is None:
                img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            imgs.append(img)

        X = np.array(imgs, dtype=np.float32) / 255.0
        X = X[..., np.newaxis]           # (B, 128, 128, 1)
        from tensorflow.keras.utils import to_categorical  # noqa
        Y = to_categorical(batch_l, num_classes=len(label_map))
        yield X, Y


def steps_per_epoch(split: str, batch_size: int = 32) -> int:
    root = TRAIN_DIR if split == "train" else TEST_DIR
    total = sum(1 for _ in root.rglob("*.jpg")) + sum(1 for _ in root.rglob("*.png"))
    return max(1, total // batch_size)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    download_dataset()
    extract_dataset()
    build_splits()
    create_label_map()
    log.info("Dataset ready.")