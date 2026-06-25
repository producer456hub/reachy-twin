"""faster-whisper speech-to-text. Input: float32 mono @ 16 kHz. Output: text."""
import numpy as np
from faster_whisper import WhisperModel

from .config import WHISPER_MODEL, WHISPER_COMPUTE, WHISPER_DEVICE


class STT:
    def __init__(self, model: str = WHISPER_MODEL, compute: str = WHISPER_COMPUTE,
                 device: str = WHISPER_DEVICE):
        # Prefer the GPU (float16) on this host; CTranslate2 needs its own cuDNN/cuBLAS,
        # so if CUDA init fails for any reason, fall back to CPU/int8 rather than crash.
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        comp = compute
        if compute == "auto":
            comp = "float16" if device == "cuda" else "int8"
        try:
            self.model = WhisperModel(model, device=device, compute_type=comp)
            self.device = device
        except Exception:
            self.model = WhisperModel(model, device="cpu", compute_type="int8")
            self.device = "cpu"

    def transcribe(self, audio_16k_mono: np.ndarray) -> str:
        if audio_16k_mono.dtype != np.float32:
            audio_16k_mono = audio_16k_mono.astype(np.float32)
        segments, _ = self.model.transcribe(
            audio_16k_mono, language="en", vad_filter=True, beam_size=1,
        )
        return " ".join(s.text.strip() for s in segments).strip()
