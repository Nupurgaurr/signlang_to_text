"""
predict.py
──────────
Real-time inference using the Landmark model (PRIMARY).
Falls back to CNN model if landmark model not found.

Provides:
  Predictor class – load once, call predict_frame() per frame
"""

import os
import sys
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    get_logger, load_label_map, normalize_landmarks,
    LANDMARK_H5_PATH, MODEL_H5_PATH, LANDMARK_DIM, IMG_SIZE,
    confusion_correct,
)
from preprocessing import image_pipeline

log = get_logger("predict")


class Predictor:
    """
    Loads model and label map once.  Thread-safe for read operations.

    Usage:
        pred = Predictor()
        label, confidence, top_preds, annotated_frame = pred.predict_frame(frame)
    """

    CONFIDENCE_THRESHOLD = 0.55

    def __init__(self):
        self.label_map  = load_label_map()
        self.num_classes = len(self.label_map)
        self.model, self.mode = self._load_model()
        self._init_mediapipe()
        log.info(f"Predictor ready  mode={self.mode}  classes={self.num_classes}")

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_model(self):
        import tensorflow as tf
        if LANDMARK_H5_PATH.exists():
            log.info(f"Loading landmark model → {LANDMARK_H5_PATH}")
            return tf.keras.models.load_model(str(LANDMARK_H5_PATH)), "landmark"
        elif MODEL_H5_PATH.exists():
            log.info(f"Loading CNN model → {MODEL_H5_PATH}")
            return tf.keras.models.load_model(str(MODEL_H5_PATH)), "cnn"
        else:
            raise FileNotFoundError(
                "No trained model found. Run:\n"
                "  python src/train_model.py --model landmark"
            )

    # ── MediaPipe ──────────────────────────────────────────────────────────────
    def _init_mediapipe(self):
        mp_hands = mp.solutions.hands
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.55,
            model_complexity=0,      # fastest
        )
        self._draw_utils  = mp.solutions.drawing_utils
        self._draw_styles = mp.solutions.drawing_styles
        self._mp_hands    = mp_hands

    # ── Core inference ─────────────────────────────────────────────────────────
    def predict_frame(self, frame: np.ndarray):
        """
        Args:
            frame: BGR numpy array from OpenCV.

        Returns:
            label       : str  – predicted class name or None
            confidence  : float
            top_preds   : list of (label, prob) sorted desc
            annotated   : BGR frame with hand landmarks drawn
        """
        annotated = frame.copy()
        if frame is None:
            return None, 0.0, [], annotated

        if self.mode == "landmark":
            return self._predict_landmark(frame, annotated)
        else:
            return self._predict_cnn(frame, annotated)

    # ── Landmark path ──────────────────────────────────────────────────────────
    def _predict_landmark(self, frame, annotated):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None, 0.0, [], annotated

        hand = result.multi_hand_landmarks[0]
        self._draw_utils.draw_landmarks(
            annotated, hand,
            self._mp_hands.HAND_CONNECTIONS,
            self._draw_styles.get_default_hand_landmarks_style(),
            self._draw_styles.get_default_hand_connections_style(),
        )

        raw = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
        feat = normalize_landmarks(raw).reshape(1, -1)

        probs = self.model.predict(feat, verbose=0)[0]
        return self._decode(probs)

    # ── CNN path ───────────────────────────────────────────────────────────────
    def _predict_cnn(self, frame, annotated):
        # Try to isolate hand region via MediaPipe bounding box
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            self._draw_utils.draw_landmarks(
                annotated, hand, self._mp_hands.HAND_CONNECTIONS,
            )
            h, w = frame.shape[:2]
            xs = [lm.x * w for lm in hand.landmark]
            ys = [lm.y * h for lm in hand.landmark]
            pad = 30
            x1, y1 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
            x2, y2 = min(w, int(max(xs)) + pad), min(h, int(max(ys)) + pad)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                roi = frame
        else:
            roi = frame

        inp = image_pipeline(roi).reshape(1, IMG_SIZE, IMG_SIZE, 1)
        probs = self.model.predict(inp, verbose=0)[0]
        return self._decode(probs)

    # ── Decode ─────────────────────────────────────────────────────────────────
    def _decode(self, probs: np.ndarray):
        top_idx = np.argsort(probs)[::-1][:5]
        top_preds = [(self.label_map.get(int(i), str(i)), float(probs[i])) for i in top_idx]

        label, confidence = top_preds[0]
        if confidence < self.CONFIDENCE_THRESHOLD:
            return None, confidence, top_preds, None   # caller will use annotated

        label = confusion_correct(label, confidence, top_preds)
        return label, confidence, top_preds, None

    # ── Get probability vector ─────────────────────────────────────────────────
    from typing import Optional

    def get_probs(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return raw probability vector for MovingAverageBuffer."""
        if frame is None:
            return None
        if self.mode == "landmark":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._hands.process(rgb)
            if not result.multi_hand_landmarks:
                return None
            raw = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.multi_hand_landmarks[0].landmark],
                dtype=np.float32,
            )
            feat = normalize_landmarks(raw).reshape(1, -1)
        else:
            inp = image_pipeline(frame).reshape(1, IMG_SIZE, IMG_SIZE, 1)
            feat = inp
        probs = self.model.predict(feat, verbose=0)[0]
        return probs