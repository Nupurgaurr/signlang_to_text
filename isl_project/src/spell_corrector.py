

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger

log = get_logger("spell_corrector")

# ── Try pyenchant ──────────────────────────────────────────────────────────────
try:
    import enchant
    _dict = enchant.Dict("en_US")
    _HAS_ENCHANT = True
    log.info("pyenchant loaded.")
except Exception as e:
    _HAS_ENCHANT = False
    log.warning(f"pyenchant unavailable ({e}). Using fallback corrector.")


# ── Fallback: edit-distance against 500 common words ──────────────────────────
_COMMON_WORDS = [
    "hello", "world", "good", "morning", "night", "please", "thank", "you",
    "sorry", "help", "yes", "no", "okay", "fine", "home", "food", "water",
    "name", "love", "hate", "come", "go", "here", "there", "now", "later",
    "stop", "start", "more", "less", "big", "small", "hot", "cold", "fast",
    "slow", "new", "old", "first", "last", "back", "front", "left", "right",
    "up", "down", "open", "close", "eat", "drink", "sleep", "walk", "run",
    "talk", "read", "write", "play", "work", "school", "house", "family",
    "friend", "doctor", "hospital", "money", "time", "day", "night", "week",
    "month", "year", "today", "tomorrow", "yesterday", "always", "never",
    "sign", "language", "hand", "finger", "letter", "word", "sentence",
]


def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def _fallback_correct(word: str) -> str:
    word_l = word.lower()
    if not word_l:
        return word
    best, best_dist = word_l, 3   # max edit distance to accept suggestion
    for w in _COMMON_WORDS:
        d = _edit_distance(word_l, w)
        if d < best_dist:
            best, best_dist = w, d
    return best if best != word_l else word


# ── Public API ─────────────────────────────────────────────────────────────────
def correct_word(word: str) -> str:
    if not word.strip():
        return word
    if _HAS_ENCHANT:
        try:
            if _dict.check(word):
                return word
            suggestions = _dict.suggest(word)
            return suggestions[0] if suggestions else word
        except Exception as e:
            log.warning(f"enchant error: {e}")
    return _fallback_correct(word)


def correct_sentence(sentence: str) -> str:
    """Correct each word in the sentence and return the corrected sentence."""
    if not sentence.strip():
        return sentence
    words = sentence.split()
    corrected = [correct_word(w) for w in words]
    result = " ".join(corrected)
    log.debug(f"Spell correction: '{sentence}' → '{result}'")
    return result