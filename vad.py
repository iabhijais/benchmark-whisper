"""Energy VAD with an ambient calibration pass.

webrtcvad isn't installed and Silero costs a torch forward pass per frame,
which would pollute the very latency numbers this harness exists to measure.
An RMS gate over 30 ms frames is enough to decide "is this an utterance",
and it costs nothing.
"""

import numpy as np


class EnergyVAD:
    def __init__(self, start_ms=90, end_ms=600, sensitivity=3.0, floor=0.004,
                 frame_ms=30):
        self.frame_ms = frame_ms
        self.start_frames = max(1, start_ms // frame_ms)
        self.end_frames = max(1, end_ms // frame_ms)
        self.sensitivity = sensitivity
        self.floor = floor
        self.threshold = floor
        self.speaking = False
        self._run_speech = 0
        self._run_silence = 0
        self._noise = None

    def calibrate(self, frames):
        """frames: iterable of numpy arrays of ambient room noise."""
        rms = np.array([float(np.sqrt(np.mean(f * f) + 1e-12)) for f in frames])
        if len(rms) == 0:
            return self.threshold
        return self.set_noise(float(np.percentile(rms, 75)))

    def set_noise(self, noise):
        """Set the gate from a known noise floor, skipping the live calibration."""
        self._noise = noise
        self.threshold = max(self.floor, noise * self.sensitivity)
        return self.threshold

    def push(self, frame):
        """Feed one frame. Returns 'start', 'end', or None."""
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        loud = rms > self.threshold
        event = None
        if loud:
            self._run_speech += 1
            self._run_silence = 0
            if not self.speaking and self._run_speech >= self.start_frames:
                self.speaking = True
                event = "start"
        else:
            self._run_silence += 1
            self._run_speech = 0
            if self.speaking and self._run_silence >= self.end_frames:
                self.speaking = False
                event = "end"
        return event

    @property
    def hangover_ms(self):
        """Silence we deliberately wait through before calling an utterance done."""
        return self.end_frames * self.frame_ms
