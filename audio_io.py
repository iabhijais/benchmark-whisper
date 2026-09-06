"""Audio sources for the live transcription harness.

Two sources, one interface: they both yield float32 mono frames at 16 kHz,
tagged with the wall-clock time at which the frame became available.

  MicSource   - real capture via sounddevice
  FileSource  - a wav/mp3/anything ffmpeg can decode, paced in real time so
                latency numbers mean the same thing as they do on a mic
"""

import queue
import subprocess
import sys
import time

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480


class MicSource:
    def __init__(self, device=None, samplerate=SAMPLE_RATE):
        import sounddevice as sd

        self.sd = sd
        self.device = device
        self.samplerate = samplerate
        self.q = queue.Queue()
        self.stream = None
        self.overflows = 0

    def _callback(self, indata, frames, time_info, status):
        if status and status.input_overflow:
            self.overflows += 1
        # copy: sounddevice reuses the buffer
        self.q.put((time.perf_counter(), indata[:, 0].copy()))

    def __enter__(self):
        self.stream = self.sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            device=self.device,
            callback=self._callback,
        )
        self.stream.start()
        return self

    def __exit__(self, *exc):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

    def frames(self):
        """Yield (wall_time, frame) forever. Blocks on the mic."""
        while True:
            yield self.q.get()

    def describe(self):
        info = self.sd.query_devices(self.device, "input") if self.device is not None \
            else self.sd.query_devices(kind="input")
        return f"mic: {info['name']}"


class FileSource:
    """Decode a file to 16k mono f32 with ffmpeg, then hand it out in real time."""

    def __init__(self, path, samplerate=SAMPLE_RATE, speed=1.0):
        self.path = path
        self.samplerate = samplerate
        self.speed = speed
        self.audio = decode_to_mono16k(path, samplerate)
        self.overflows = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def frames(self):
        start = time.perf_counter()
        n = len(self.audio)
        for i in range(0, n - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            frame = self.audio[i:i + FRAME_SAMPLES]
            due = start + ((i + FRAME_SAMPLES) / self.samplerate) / self.speed
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            yield (time.perf_counter(), frame)
        # trailing silence so the VAD closes the last utterance
        silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        for _ in range(40):
            time.sleep(FRAME_MS / 1000.0 / self.speed)
            yield (time.perf_counter(), silence)

    def describe(self):
        return f"file: {self.path} ({len(self.audio)/self.samplerate:.1f}s, {self.speed:g}x)"

    def ambient_estimate(self):
        """Noise floor over the whole file.

        A file's first second is usually speech, so calibrating on it the way we
        do for a mic would set the gate above the voice and detect nothing.
        The 10th percentile frame is the quiet part, wherever it happens to be.
        """
        n = len(self.audio) // FRAME_SAMPLES
        if n < 4:
            return 0.0
        frames = self.audio[:n * FRAME_SAMPLES].reshape(n, FRAME_SAMPLES)
        rms = np.sqrt((frames * frames).mean(axis=1) + 1e-12)
        return float(np.percentile(rms, 10))


def decode_to_mono16k(path, samplerate=SAMPLE_RATE):
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(samplerate), "-",
    ]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError:
        sys.exit("ffmpeg not found on PATH - needed to decode audio files.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffmpeg failed on {path}:\n{e.stderr.decode(errors='replace')}")
    return np.frombuffer(raw, dtype=np.float32).copy()
