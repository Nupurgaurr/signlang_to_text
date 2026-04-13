
import threading
from typing import Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger

log = get_logger("tts")

# ── pyttsx3 ────────────────────────────────────────────────────────────────────
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 150)
    _tts_engine.setProperty("volume", 1.0)
    _HAS_PYTTSX3 = True
    log.info("pyttsx3 TTS ready.")
except Exception as e:
    _HAS_PYTTSX3 = False
    log.warning(f"pyttsx3 unavailable ({e}). Will try gTTS.")

# ── gTTS fallback ──────────────────────────────────────────────────────────────
try:
    from gtts import gTTS
    import tempfile, os
    try:
        import pygame
        pygame.mixer.init()
        _HAS_GTTS = True
        log.info("gTTS + pygame TTS ready.")
    except Exception:
        _HAS_GTTS = False
except Exception:
    _HAS_GTTS = False


_lock = threading.Lock()
_current_thread: Optional[threading.Thread] = None

def _speak_pyttsx3(text: str):
    with _lock:
        try:
            _tts_engine.say(text)
            _tts_engine.runAndWait()
        except Exception as e:
            log.error(f"pyttsx3 speak error: {e}")


def _speak_gtts(text: str):
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tmp = f.name
        tts.save(tmp)
        pygame.mixer.music.load(tmp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        os.unlink(tmp)
    except Exception as e:
        log.error(f"gTTS speak error: {e}")


def speak(text: str):
    """
    Speak text in a background thread (non-blocking).
    Skips if a previous utterance is still running.
    """
    global _current_thread
    if not text or not text.strip():
        return

    if _current_thread and _current_thread.is_alive():
        log.debug("TTS busy — skipping.")
        return

    if _HAS_PYTTSX3:
        _current_thread = threading.Thread(target=_speak_pyttsx3, args=(text,), daemon=True)
    elif _HAS_GTTS:
        _current_thread = threading.Thread(target=_speak_gtts, args=(text,), daemon=True)
    else:
        log.warning("No TTS engine available.")
        return

    _current_thread.start()
    log.info(f"Speaking: '{text}'")


def is_speaking() -> bool:
    return _current_thread is not None and _current_thread.is_alive()