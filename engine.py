"""faster-whisper wrapper with per-call timing."""

import sys
import time

from faster_whisper import WhisperModel

# Aliases so --model tiny works, and so the turbo repo name is discoverable.
ALIASES = {
    "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "distil-small": "distil-whisper/distil-small.en",
    "distil-medium": "distil-whisper/distil-medium.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}

# Whisper's default temperature schedule. When a decode trips the compression
# ratio or logprob threshold it silently re-runs at the next temperature - up to
# six passes for one clip. Great for accuracy, ruinous for a stopwatch: it is
# why an unstabilised benchmark shows 12s on a 1s chunk. Off by default here,
# --fallback puts it back.
FALLBACK_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def use_utf8_stdout():
    """Windows consoles default to cp1252 and crash on Devanagari, CJK, emoji."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


class Engine:
    def __init__(self, model="base", device="auto", compute_type=None,
                 language=None, beam_size=1, cpu_threads=0, vad_filter=False,
                 fallback=False, timestamps=False):
        name = ALIASES.get(model, model)
        if device == "auto":
            device = _pick_device()
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"
        t0 = time.perf_counter()
        self.model = WhisperModel(
            name, device=device, compute_type=compute_type, cpu_threads=cpu_threads
        )
        self.load_seconds = time.perf_counter() - t0
        self.name = model
        self.repo = name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.temperature = FALLBACK_TEMPERATURES if fallback else 0.0
        self.without_timestamps = not timestamps

    def transcribe(self, audio, sample_rate=16000):
        """Returns (text, seconds_spent, info). Timing covers the full generator drain."""
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            temperature=self.temperature,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,
            without_timestamps=self.without_timestamps,
            word_timestamps=False,
        )
        text = "".join(s.text for s in segments).strip()
        return text, time.perf_counter() - t0, info

    def warmup(self, seconds=1.0, sample_rate=16000):
        import numpy as np
        silence = np.zeros(int(seconds * sample_rate), dtype="float32")
        _, dt, _ = self.transcribe(silence, sample_rate)
        return dt

    def describe(self):
        bits = "%s [%s] on %s/%s, beam=%d, load=%.2fs" % (
            self.name, self.repo, self.device, self.compute_type,
            self.beam_size, self.load_seconds)
        if self.temperature != 0.0:
            bits += ", temp-fallback on"
        if self.language:
            bits += ", lang=%s" % self.language
        return bits


def _pick_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
