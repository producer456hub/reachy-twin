"""Shared voice-loop pieces: audio helpers, wake-word routing, brain construction.

Used by twin.hub (the panel's RobotHub) and twin.app (the standalone CLI loop).
"""
import re

import numpy as np

from twin.config import CLAUDE_VOICE, LOCAL_VOICE
from twin.brains import make_claude, make_local

SR = 16000
SILENCE_HANG = 80         # ~0.8 s trailing silence ends an utterance
SPEECH_START = 3          # consecutive loud chunks to trigger
MIN_CHUNKS = 30           # ignore blips shorter than ~0.3 s
EXIT_WORDS = ("goodbye", "good bye", "shut down", "go to sleep")
WAKE = {"maintop": re.compile(r"\bmaintop\b", re.I), "claude": re.compile(r"\bclaude\b", re.I)}


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2))) if len(x) else 0.0


def _mono(s):
    return s.mean(axis=1) if getattr(s, "ndim", 1) == 2 else s


def calibrate_floor(mini, n=40):
    vals = []
    for _ in range(n):
        s = mini.media.get_audio_sample()
        if s is not None and len(s):
            vals.append(_rms(_mono(s)))
    # clamp the ceiling too: calibrating while audio happens to be playing
    # (e.g. a reconnect mid-speech) would otherwise leave the mic half-deaf
    return min(max((float(np.median(vals)) if vals else 0.001) * 4.0, 0.012), 0.045)


def detect_switch(text, current):
    """Return the brain key named earliest in the utterance, else None."""
    hits = [(m.start(), k) for k, rx in WAKE.items() if (m := rx.search(text))]
    if not hits:
        return None
    return min(hits)[1]


def strip_wake(text):
    """Drop a leading 'hey <name>,' / '<name>,' so the brain doesn't hear its own name."""
    t = re.sub(r"^\s*(hey|hi|ok|okay)?\s*(maintop|claude)[\s,.:!-]*", "", text, count=1, flags=re.I)
    return t.strip()


def build_brains():
    """Maintop (local Ollama, the default) + Claude (subscription CLI, optional).

    This host is a self-contained entity: NO Marcus, nothing reaches vr-2/the tailnet.
    """
    brains, voices = {}, {}
    brains["maintop"] = make_local()
    voices["maintop"] = LOCAL_VOICE
    brains["claude"] = make_claude()
    voices["claude"] = CLAUDE_VOICE
    return brains, voices
