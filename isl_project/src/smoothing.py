

from typing import Optional, Tuple
from collections import deque, Counter
import time
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, SPACE_TIMEOUT, DELETE_HOLD

log = get_logger("smoothing")


class MajorityVoteBuffer:
    """
    Collect `size` predictions, return the majority label if it exceeds
    `min_confidence` fraction of votes; else return None.
    """

    def __init__(self, size: int = 15, min_confidence: float = 0.6):
        self.size = size
        self.min_confidence = min_confidence
        self._buf: deque = deque(maxlen=size)

    from typing import Optional

    def push(self, label: str) -> Optional[str]:
        if label is None:
            return None
        self._buf.append(label)
        if len(self._buf) < self.size:
            return None
        counts = Counter(self._buf)
        top_label, top_count = counts.most_common(1)[0]
        frac = top_count / self.size
        if frac >= self.min_confidence:
            return top_label
        return None

    def clear(self):
        self._buf.clear()

    @property
    def fill(self) -> float:
        return len(self._buf) / self.size


class MovingAverageBuffer:
    """
    Average probability vectors over `size` frames and return the argmax label.
    Requires label_map to convert index → name.
    """

    def __init__(self, size: int = 10, num_classes: int = 29):
        self.size = size
        self.num_classes = num_classes
        self._buf: deque = deque(maxlen=size)

    def push(self, probs: np.ndarray, label_map: dict) -> Tuple[Optional[str], float]:
        """Returns (smoothed_label, smoothed_confidence)."""
        if probs is None or len(probs) != self.num_classes:
            return None, 0.0
        self._buf.append(probs)
        avg = np.mean(self._buf, axis=0)
        idx = int(np.argmax(avg))
        conf = float(avg[idx])
        label = label_map.get(idx)
        return label, conf

    def clear(self):
        self._buf.clear()


class SpaceDeleteDetector:
    """
    Detects:
      SPACE  – no confident gesture for `space_timeout` seconds
      DELETE – same gesture held for `delete_hold` seconds
    """

    def __init__(
        self,
        space_timeout: float = SPACE_TIMEOUT,
        delete_hold:   float = DELETE_HOLD,
    ):
        self.space_timeout = space_timeout
        self.delete_hold   = delete_hold
        self._last_gesture_time = time.time()
        self._hold_label  = None
        self._hold_start  = None
        self._space_fired = False
        self._delete_fired= False

    def update(self, label: Optional[str]) -> Optional[str]:
        """
        Call every frame with the current stable label (or None).
        Returns: 'SPACE', 'DELETE', or None.
        """
        now = time.time()

        if label is None:
            # no gesture — check space timeout
            elapsed = now - self._last_gesture_time
            if elapsed >= self.space_timeout and not self._space_fired:
                self._space_fired = True
                self._hold_label  = None
                self._hold_start  = None
                log.debug("SPACE triggered (timeout)")
                return "SPACE"
            return None

        # gesture present
        self._last_gesture_time = now
        self._space_fired = False

        if label != self._hold_label:
            self._hold_label   = label
            self._hold_start   = now
            self._delete_fired = False
            return None

        held = now - (self._hold_start or now)
        if held >= self.delete_hold and not self._delete_fired:
            self._delete_fired = True
            log.debug(f"DELETE triggered (held {label} for {held:.1f}s)")
            return "DELETE"

        return None

    def reset(self):
        self._last_gesture_time = time.time()
        self._hold_label  = None
        self._hold_start  = None
        self._space_fired = False
        self._delete_fired= False