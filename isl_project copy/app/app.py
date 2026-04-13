"""
app/app.py
──────────
Streamlit real-time ISL recognition — OpenCV webcam (no WebRTC/av).

Run:
  streamlit run app/app.py
"""

import sys
import os
import time
import numpy as np
import cv2
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(
    page_title="ISL Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from predict import Predictor
from smoothing import MajorityVoteBuffer, MovingAverageBuffer, SpaceDeleteDetector
from spell_corrector import correct_sentence
from text_to_speech import speak, is_speaking
from utils import get_logger

log = get_logger("app")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, html, body {
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background-color: #0a0c10;
}
[data-testid="stSidebar"] {
    background-color: #0e1117;
    border-right: 1px solid #1a1d26;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 0rem !important;
    max-width: 1200px;
}

/* ── top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 4px 16px 4px;
    margin-bottom: 4px;
    border-bottom: 1px solid #13161f;
}
.topbar-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e2e8f0;
    letter-spacing: 0.3px;
}
.topbar-badge {
    font-size: 0.67rem;
    font-weight: 600;
    color: #00d4aa;
    background: #00d4aa12;
    border: 1px solid #00d4aa35;
    border-radius: 20px;
    padding: 3px 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── camera panel ── */
.cam-label {
    font-size: 0.67rem;
    font-weight: 500;
    color: #374151;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    margin-bottom: 8px;
    padding-left: 2px;
}
[data-testid="stImage"] img {
    border-radius: 14px;
    width: 100% !important;
    border: 1px solid #1a1d26;
}
.cam-placeholder {
    background: #0e1117;
    border: 1px dashed #1a1d26;
    border-radius: 14px;
    height: 340px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
}
.cam-placeholder-icon { font-size: 2.2rem; opacity: 0.12; }
.cam-placeholder-text {
    font-size: 0.75rem;
    color: #252836;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* ── detected letter card ── */
.letter-card {
    background: #0d1117;
    border: 1px solid #161b25;
    border-radius: 18px;
    padding: 28px 20px 20px 20px;
    text-align: center;
    margin-bottom: 14px;
}
.card-label {
    font-size: 0.63rem;
    font-weight: 600;
    color: #374151;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.letter-glyph {
    font-size: 7rem;
    font-weight: 700;
    line-height: 1;
    min-height: 88px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #00d4aa 30%, #4f8ef7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -3px;
}
.letter-glyph.empty {
    background: none;
    -webkit-text-fill-color: #1e2535;
    color: #1e2535;
}
.conf-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-top: 14px;
}
.conf-chip {
    font-size: 0.7rem;
    font-weight: 500;
    padding: 3px 11px;
    border-radius: 20px;
    background: #111827;
    color: #374151;
    border: 1px solid #1a2030;
}
.conf-chip.on {
    background: #00d4aa0e;
    color: #00d4aa;
    border-color: #00d4aa30;
}
.dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 4px;
    vertical-align: middle;
}
.dot.live { background:#00d4aa; box-shadow:0 0 5px #00d4aa70; }
.dot.idle { background:#252836; }
.conf-bar-track {
    background: #111827;
    border-radius: 4px;
    height: 3px;
    width: 100%;
    margin-top: 12px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 3px;
    border-radius: 4px;
    background: linear-gradient(90deg, #00d4aa, #4f8ef7);
}

/* ── status strip ── */
.strip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2px;
    margin-bottom: 12px;
}
.strip-txt { font-size: 0.67rem; color: #252836; font-weight: 500; }
.strip-txt.on { color: #00d4aa; }

/* ── sentence card ── */
.sentence-card {
    background: #0d1117;
    border: 1px solid #161b25;
    border-radius: 16px;
    padding: 18px 18px 14px 18px;
    margin-bottom: 14px;
}
.sentence-body {
    font-size: 1.5rem;
    font-weight: 500;
    color: #e2e8f0;
    min-height: 48px;
    line-height: 1.45;
    word-break: break-all;
    letter-spacing: 1.5px;
}
.sentence-body.empty {
    font-size: 1.1rem;
    color: #1e2535;
    font-weight: 400;
    font-style: italic;
}

/* ── letter tiles strip ── */
.tiles-card {
    background: #0d1117;
    border: 1px solid #161b25;
    border-radius: 16px;
    padding: 16px 18px 14px 18px;
    margin-bottom: 14px;
}
.tiles-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    min-height: 44px;
    align-items: center;
}
.tile {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 600;
    background: #111827;
    color: #00d4aa;
    border: 1px solid #1a2a3a;
    letter-spacing: 0;
    flex-shrink: 0;
}
.tile.space-tile {
    width: 24px;
    background: #0d1117;
    border: 1px dashed #1a2030;
    color: transparent;
}
.tile.latest {
    background: #00d4aa18;
    border-color: #00d4aa55;
    color: #00d4aa;
    box-shadow: 0 0 8px #00d4aa22;
}
.tiles-empty {
    font-size: 0.78rem;
    color: #1e2535;
    font-style: italic;
}

/* ── full sentence line ── */
.sentence-line {
    font-size: 1.35rem;
    font-weight: 500;
    color: #e2e8f0;
    min-height: 40px;
    line-height: 1.5;
    word-break: break-word;
    letter-spacing: 1px;
    margin-top: 4px;
}
.sentence-line.empty {
    font-size: 0.9rem;
    color: #1e2535;
    font-style: italic;
    font-weight: 400;
}

/* ── buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 9px 0 !important;
    border: 1px solid #161b25 !important;
    background: #0d1117 !important;
    color: #4a5568 !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.3px !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #111827 !important;
    border-color: #00d4aa45 !important;
    color: #00d4aa !important;
    box-shadow: 0 0 14px #00d4aa15 !important;
}
.stButton > button[kind="primary"] {
    background: #00d4aa12 !important;
    border-color: #00d4aa45 !important;
    color: #00d4aa !important;
}
.stButton > button[kind="primary"]:hover {
    background: #00d4aa20 !important;
    box-shadow: 0 0 18px #00d4aa22 !important;
}

/* ── speaking badge ── */
.speaking {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.7rem;
    color: #4f8ef7;
    padding: 6px 12px;
    background: #4f8ef710;
    border: 1px solid #4f8ef728;
    border-radius: 8px;
    margin-top: 8px;
    font-weight: 500;
}

/* slider / input cleanup */
[data-testid="stSlider"] > div { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "running":        False,
        "sentence":       [],
        "current_letter": "",
        "confidence":     0.0,
        "top_preds":      [],
        "fps":            0.0,
        "hand_detected":  False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Load Predictor ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def get_predictor():
    try:
        return Predictor()
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None

predictor = get_predictor()
if predictor is None:
    st.stop()

num_classes = predictor.num_classes
label_map   = predictor.label_map


# ── Smoothing (persist across reruns via session_state) ───────────────────────
if "maj_buf" not in st.session_state:
    st.session_state.maj_buf = MajorityVoteBuffer(size=30, min_confidence=0.75)
if "avg_buf" not in st.session_state:
    st.session_state.avg_buf = MovingAverageBuffer(size=12, num_classes=num_classes)
if "sd" not in st.session_state:
    st.session_state.sd = SpaceDeleteDetector()
if "last_appended" not in st.session_state:
    st.session_state.last_appended = None
if "last_append_time" not in st.session_state:
    st.session_state.last_append_time = 0.0

maj_buf = st.session_state.maj_buf
avg_buf = st.session_state.avg_buf
sd      = st.session_state.sd
APPEND_COOLDOWN = 3.0


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider("Confidence threshold", 0.40, 0.95, 0.55, 0.05)
    predictor.CONFIDENCE_THRESHOLD = confidence_threshold
    cam_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
    st.markdown("---")
    maj_size = st.slider("Vote buffer size", 5, 40, 30, 1)
    maj_buf.size = maj_size
    maj_buf.min_confidence = st.slider("Vote fraction", 0.5, 0.95, 0.75, 0.05)
    st.markdown("---")
    st.caption("⏱ No hand 3 s → **SPACE**")
    st.caption("🤚 Hold sign 2.5 s → **DELETE**")
    st.markdown("---")
    with st.expander(f"Classes ({num_classes})"):
        st.write(", ".join(sorted(label_map.values())))


# ── Top bar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <span class="topbar-title">🤟 &nbsp; ISL Recognition</span>
  <span class="topbar-badge">● Live</span>
</div>
""", unsafe_allow_html=True)


# ── 2-panel layout ─────────────────────────────────────────────────────────────
col_cam, col_out = st.columns([3, 2], gap="large")

# ── LEFT ──────────────────────────────────────────────────────────────────────
with col_cam:
    st.markdown("<div class='cam-label'>Camera Feed</div>", unsafe_allow_html=True)
    cl, cr = st.columns(2)
    with cl:
        start_btn = st.button("▶  Start", use_container_width=True, type="primary")
    with cr:
        stop_btn  = st.button("⏹  Stop",  use_container_width=True)
    frame_ph = st.empty()

# ── RIGHT ─────────────────────────────────────────────────────────────────────
with col_out:
    letter_ph    = st.empty()
    status_ph    = st.empty()
    tiles_ph     = st.empty()   # letter-by-letter tiles
    sentence_ph  = st.empty()   # full sentence line

    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
    ba, bb = st.columns(2)
    with ba:
        speak_btn = st.button("🔊  Speak",  use_container_width=True)
    with bb:
        clear_btn = st.button("🗑  Clear",  use_container_width=True)
    correct_btn = st.button("✏️  Spell Correct", use_container_width=True)
    speaking_ph = st.empty()


# ── Button actions ─────────────────────────────────────────────────────────────
if start_btn:
    st.session_state.running = True

if stop_btn:
    st.session_state.running = False

if clear_btn:
    st.session_state.sentence       = []
    st.session_state.current_letter = ""
    st.session_state.confidence     = 0.0
    maj_buf.clear()
    avg_buf.clear()
    sd.reset()
    st.session_state.last_appended = None
    st.rerun()

if speak_btn:
    text = "".join(st.session_state.sentence).strip()
    if text:
        speak(text)

if correct_btn:
    text = "".join(st.session_state.sentence).strip()
    if text:
        corrected = correct_sentence(text)
        st.session_state.sentence = list(corrected)
    st.rerun()


# ── Info panel renderer ────────────────────────────────────────────────────────
def render_info(letter, conf, top_preds, hand_detected, fps, sentence_str):
    # ── detected letter card ──────────────────────────────────────────────────
    display    = letter if letter and letter not in ("NOTHING",) else "—"
    conf_pct   = int(conf * 100)
    glyph_cls  = "letter-glyph" if display != "—" else "letter-glyph empty"
    chip_cls   = "conf-chip on" if hand_detected and display != "—" else "conf-chip"
    dot_cls    = "dot live"     if hand_detected else "dot idle"

    letter_ph.markdown(f"""
<div class="letter-card">
  <div class="card-label">Detecting</div>
  <div class="{glyph_cls}">{display}</div>
  <div class="conf-row">
    <span class="{chip_cls}">
      <span class="{dot_cls}"></span>{conf_pct}% confidence
    </span>
  </div>
  <div class="conf-bar-track">
    <div class="conf-bar-fill" style="width:{conf_pct}%"></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── status strip ─────────────────────────────────────────────────────────
    strip_cls = "strip-txt on" if hand_detected else "strip-txt"
    hand_txt  = "Hand detected" if hand_detected else "No hand"
    status_ph.markdown(f"""
<div class="strip">
  <span class="{strip_cls}">● {hand_txt}</span>
  <span class="strip-txt">{fps:.1f} fps</span>
</div>
""", unsafe_allow_html=True)

    # ── letter tiles ──────────────────────────────────────────────────────────
    # Build individual tile HTML for every character in sentence_str
    if sentence_str:
        chars = list(sentence_str)
        tile_html = ""
        for i, ch in enumerate(chars):
            is_latest = (i == len(chars) - 1)
            if ch == " ":
                tile_html += '<span class="tile space-tile"> </span>'
            else:
                cls = "tile latest" if is_latest else "tile"
                tile_html += f'<span class="{cls}">{ch}</span>'
        tiles_inner = f'<div class="tiles-row">{tile_html}</div>'
    else:
        tiles_inner = '<div class="tiles-row"><span class="tiles-empty">Letters will appear here…</span></div>'

    tiles_ph.markdown(f"""
<div class="tiles-card">
  <div class="card-label">Letters</div>
  {tiles_inner}
</div>
""", unsafe_allow_html=True)

    # ── full sentence line ────────────────────────────────────────────────────
    # Split into words and render sentence as readable text
    words = sentence_str.strip()
    if words:
        sentence_ph.markdown(f"""
<div class="sentence-card">
  <div class="card-label">Sentence</div>
  <div class="sentence-line">{words}</div>
</div>
""", unsafe_allow_html=True)
    else:
        sentence_ph.markdown("""
<div class="sentence-card">
  <div class="card-label">Sentence</div>
  <div class="sentence-line empty">Sign letters to form a sentence…</div>
</div>
""", unsafe_allow_html=True)

    # ── speaking indicator ────────────────────────────────────────────────────
    if is_speaking():
        speaking_ph.markdown(
            '<div class="speaking">🔊 &nbsp; Speaking…</div>',
            unsafe_allow_html=True,
        )
    else:
        speaking_ph.empty()


# ── Frame overlay (backend unchanged) ─────────────────────────────────────────
def _draw_overlay(img, label, conf, hand_detected):
    h, w = img.shape[:2]
    dot_color = (0, 212, 170) if hand_detected else (60, 60, 60)
    cv2.circle(img, (w - 20, 20), 8, dot_color, -1)
    if label and label not in ("NOTHING",):
        cv2.putText(img, label, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 212, 170), 4, cv2.LINE_AA)
        bar_w = int((w - 40) * conf)
        cv2.rectangle(img, (20, h - 30), (w - 20, h - 16), (30, 30, 50), -1)
        cv2.rectangle(img, (20, h - 30), (20 + bar_w, h - 16), (0, 212, 170), -1)
        cv2.putText(img, f"{conf:.0%}", (w - 70, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return img


# ── Webcam loop (backend unchanged) ───────────────────────────────────────────
if st.session_state.running:
    cap = cv2.VideoCapture(int(cam_index))
    if not cap.isOpened():
        st.error(
            f"❌ Cannot open camera index {int(cam_index)}. "
            "Try changing the Camera index in the sidebar."
        )
        st.session_state.running = False
        st.stop()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_times = []

    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                status_ph.warning("⚠️ Frame read failed, retrying…")
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)

            # ── Inference ──────────────────────────────────────────────────────
            raw_probs                 = predictor.get_probs(frame)
            label, conf, top_preds, _ = predictor.predict_frame(frame)
            hand_detected             = raw_probs is not None

            # Moving-average smoothing
            if raw_probs is not None:
                smooth_label, smooth_conf = avg_buf.push(raw_probs, label_map)
                if smooth_conf > predictor.CONFIDENCE_THRESHOLD:
                    label, conf = smooth_label, smooth_conf

            # Majority-vote stabilisation
            stable = maj_buf.push(label)

            # Space / Delete detection
            action = sd.update(stable)

            # ── Word formation ─────────────────────────────────────────────────
            now = time.time()
            if action == "SPACE":
                if st.session_state.sentence and st.session_state.sentence[-1] != " ":
                    st.session_state.sentence.append(" ")
            elif action == "DELETE":
                if st.session_state.sentence:
                    st.session_state.sentence.pop()
            elif (
                stable
                and stable not in ("NOTHING", "DEL", "DELETE", "SPACE", None)
                and (
                    stable != st.session_state.last_appended
                    or now - st.session_state.last_append_time > APPEND_COOLDOWN
                )
            ):
                st.session_state.sentence.append(stable)
                st.session_state.last_appended    = stable
                st.session_state.last_append_time = now

            # ── FPS ────────────────────────────────────────────────────────────
            frame_times.append(now)
            frame_times = [t for t in frame_times if now - t < 2]
            fps = len(frame_times) / 2.0

            # ── Draw + push to UI ──────────────────────────────────────────────
            annotated = _draw_overlay(frame.copy(), label, conf, hand_detected)
            rgb       = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_ph.image(rgb, channels="RGB", use_column_width=True)

            sentence_str = "".join(st.session_state.sentence)
            render_info(label, conf, top_preds, hand_detected, fps, sentence_str)

    except Exception as e:
        st.error(f"❌ Camera error: {e}")
        log.exception(e)
    finally:
        cap.release()
        st.session_state.running = False

else:
    frame_ph.markdown("""
<div class="cam-placeholder">
  <div class="cam-placeholder-icon">📷</div>
  <div class="cam-placeholder-text">Press Start to activate camera</div>
</div>
""", unsafe_allow_html=True)

    render_info(
        st.session_state.current_letter,
        st.session_state.confidence,
        st.session_state.top_preds,
        False, 0.0,
        "".join(st.session_state.sentence),
    )