"""Live speech-to-text with Whisper, instrumented for latency.

Pipeline:  capture thread -> energy VAD -> utterance buffer -> worker thread
running faster-whisper.  Every transcription call is timed, and the report at
the end is the point of the whole thing.

Metrics reported per utterance:
  audio      length of the speech that was transcribed
  infer      wall time inside faster_whisper.transcribe()
  RTF        infer / audio  (below 1.0 = faster than real time)
  latency    last loud frame captured -> final text on screen.  This is what a
             user actually feels, and it includes the VAD hangover we chose to
             wait through, any queueing behind a partial, and inference.

Usage:
  python live_transcribe.py --model base
  python live_transcribe.py --model small --partial-interval 0.8
  python live_transcribe.py --file ../input/voiceover.wav --model tiny
  python live_transcribe.py --list-devices
"""

import argparse
import json
import queue
import sys
import threading
import time
from collections import deque

import numpy as np

from audio_io import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, FileSource, MicSource
from engine import Engine, use_utf8_stdout
from vad import EnergyVAD


# --------------------------------------------------------------------------- #
# worker


class Worker(threading.Thread):
    """One transcription at a time, off the capture path."""

    def __init__(self, engine):
        super().__init__(daemon=True)
        self.engine = engine
        self.jobs = queue.Queue()
        self.results = queue.Queue()
        self.busy = False
        self._stop = threading.Event()

    def submit(self, kind, audio, meta):
        self.busy = True
        self.jobs.put((kind, audio, meta))

    def run(self):
        while not self._stop.is_set():
            try:
                kind, audio, meta = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if kind == "quit":
                break
            queued_at = meta["submitted"]
            started = time.perf_counter()
            text, infer, _ = self.engine.transcribe(audio)
            meta.update(
                text=text,
                infer=infer,
                queue_wait=started - queued_at,
                done=time.perf_counter(),
                audio_seconds=len(audio) / SAMPLE_RATE,
            )
            self.busy = self.jobs.qsize() > 0
            self.results.put((kind, meta))

    def stop(self):
        self._stop.set()
        self.jobs.put(("quit", None, {}))


# --------------------------------------------------------------------------- #
# capture thread


def start_capture(source, out_q, stop_event):
    def pump():
        try:
            for item in source.frames():
                if stop_event.is_set():
                    return
                out_q.put(item)
        except Exception as e:  # keep the main loop alive to print the report
            out_q.put(("error", e))
        out_q.put(("eof", None))

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    return t


# --------------------------------------------------------------------------- #
# stats


def pct(values, p):
    return float(np.percentile(values, p)) if values else float("nan")


def report(finals, partials, engine, source, args, vad, overflows):
    print("\n" + "=" * 72)
    print("LIVE TRANSCRIPTION - SPEED REPORT")
    print("=" * 72)
    print("model    : " + engine.describe())
    print("source   : " + source.describe())
    print("vad      : threshold=%.5f hangover=%dms" % (vad.threshold, vad.hangover_ms))
    print("partials : every %ss" % args.partial_interval if args.partial_interval
          else "partials : off")
    if overflows:
        print("WARNING  : %d input overflow(s) - audio was dropped at capture" % overflows)

    if not finals:
        print("\nNo utterances captured. Nothing to measure.")
        return {}

    print("\n  #   audio    infer     RTF   latency   qwait   text")
    print("  " + "-" * 68)
    for i, u in enumerate(finals, 1):
        preview = (u["text"][:34] + "..") if len(u["text"]) > 36 else u["text"]
        print("  %-3d %6.2fs %7.2fs %6.2f %8.2fs %6.2fs   %s"
              % (i, u["audio_seconds"], u["infer"], u["rtf"], u["latency"],
                 u["queue_wait"], preview))

    rtfs = [u["rtf"] for u in finals]
    lats = [u["latency"] for u in finals]
    infers = [u["infer"] for u in finals]
    audio_total = sum(u["audio_seconds"] for u in finals)

    print("\n  finals            n=%d   speech=%.1fs   inference=%.1fs"
          % (len(finals), audio_total, sum(infers)))
    print("  RTF               median %.2f   p95 %.2f   min %.2f   max %.2f"
          % (pct(rtfs, 50), pct(rtfs, 95), min(rtfs), max(rtfs)))
    print("  latency (s)       median %.2f   p95 %.2f   min %.2f   max %.2f"
          % (pct(lats, 50), pct(lats, 95), min(lats), max(lats)))
    print("  latency less the %.2fs VAD hangover: median %.2f"
          % (vad.hangover_ms / 1000, pct(lats, 50) - vad.hangover_ms / 1000))
    agg = sum(infers) / audio_total
    print("  aggregate RTF     %.2f  (%s)"
          % (agg, "keeps up with live speech" if agg < 1
             else "SLOWER THAN REAL TIME - backlog will grow"))

    summary = {
        "model": engine.name, "repo": engine.repo, "device": engine.device,
        "compute_type": engine.compute_type, "beam_size": engine.beam_size,
        "load_seconds": engine.load_seconds,
        "utterances": len(finals), "speech_seconds": audio_total,
        "inference_seconds": sum(infers),
        "rtf_median": pct(rtfs, 50), "rtf_p95": pct(rtfs, 95),
        "rtf_aggregate": agg,
        "latency_median": pct(lats, 50), "latency_p95": pct(lats, 95),
        "vad_hangover_s": vad.hangover_ms / 1000,
        "overflows": overflows,
        "finals": finals,
    }

    if partials:
        p_infer = [p["infer"] for p in partials]
        p_lag = [p["lag"] for p in partials]
        print("\n  partials          n=%d   infer median %.2fs   p95 %.2fs"
              % (len(partials), pct(p_infer, 50), pct(p_infer, 95)))
        print("  partial lag (s)   median %.2f   p95 %.2f"
              % (pct(p_lag, 50), pct(p_lag, 95)))
        print("  (lag = newest audio in the buffer -> partial text on screen)")
        summary["partials"] = {
            "n": len(partials),
            "infer_median": pct(p_infer, 50), "infer_p95": pct(p_infer, 95),
            "lag_median": pct(p_lag, 50), "lag_p95": pct(p_lag, 95),
        }

    print("\n  transcript")
    print("  " + "-" * 68)
    for u in finals:
        if u["text"]:
            print("  " + u["text"])
    print()
    return summary


# --------------------------------------------------------------------------- #
# level meter


def run_meter(args):
    """Tune --sensitivity by eye before spending a model load on a bad gate."""
    source = FileSource(args.file, speed=args.speed) if args.file \
        else MicSource(device=args.device_index)
    vad = EnergyVAD(end_ms=args.silence_ms, sensitivity=args.sensitivity,
                    frame_ms=FRAME_MS)
    calib = []
    calib_target = max(1, int(args.calibrate_s * 1000 / FRAME_MS))
    print("meter: %s" % source.describe())

    with source:
        if hasattr(source, "ambient_estimate"):
            vad.set_noise(source.ambient_estimate())
            calib = None
        else:
            print("calibrating %gs - stay quiet ..." % args.calibrate_s, flush=True)
        t0 = time.perf_counter()
        try:
            for stamp, frame in source.frames():
                if args.seconds and time.perf_counter() - t0 > args.seconds:
                    break
                if calib is not None:
                    calib.append(frame)
                    if len(calib) >= calib_target:
                        vad.calibrate(calib)
                        calib = None
                        print("threshold = %.5f\n" % vad.threshold, flush=True)
                    continue
                rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
                vad.push(frame)
                bars = int(min(rms / max(vad.threshold, 1e-6), 3.0) * 20)
                gate = int(20)
                line = "".join("#" if i < bars else ("|" if i == gate else " ")
                               for i in range(60))
                print("  %.4f [%s] %s" % (rms, line,
                                          "SPEECH" if vad.speaking else "      "),
                      end="\r", flush=True)
        except KeyboardInterrupt:
            pass
    print("\nthreshold %.5f - the | mark is the gate; speech should push well past it."
          % vad.threshold)


# --------------------------------------------------------------------------- #
# main


def main():
    ap = argparse.ArgumentParser(
        description="Live Whisper transcription with latency metrics")
    ap.add_argument("--model", default="base",
                    help="tiny|base|small|medium|large-v3|turbo|distil-small|<hf repo>")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--compute-type", default=None,
                    help="int8 (cpu default) | int8_float32 | float32 | float16 (cuda)")
    ap.add_argument("--language", default=None, help="e.g. en, hi. omit to auto-detect")
    ap.add_argument("--beam-size", type=int, default=1)
    ap.add_argument("--cpu-threads", type=int, default=0, help="0 = ctranslate2 default")
    ap.add_argument("--vad-filter", action="store_true",
                    help="also run whisper's own Silero VAD (costs time)")
    ap.add_argument("--fallback", action="store_true",
                    help="enable whisper's temperature fallback: better on hard audio, "
                         "but one clip can silently decode up to 6 times")
    ap.add_argument("--timestamps", action="store_true",
                    help="decode segment timestamps too (slower)")

    ap.add_argument("--file", default=None,
                    help="feed a file in real time instead of a mic")
    ap.add_argument("--speed", type=float, default=1.0, help="file playback rate")
    ap.add_argument("--device-index", type=int, default=None, help="input device index")
    ap.add_argument("--list-devices", action="store_true")

    ap.add_argument("--partial-interval", type=float, default=0.0,
                    help="seconds between in-flight partial decodes (0 = off)")
    ap.add_argument("--silence-ms", type=int, default=600, help="VAD hangover")
    ap.add_argument("--min-utterance-ms", type=int, default=350)
    ap.add_argument("--max-utterance-s", type=float, default=25.0,
                    help="force a cut in long speech")
    ap.add_argument("--sensitivity", type=float, default=3.0,
                    help="VAD gate = noise floor x this")
    ap.add_argument("--calibrate-s", type=float, default=1.0)

    ap.add_argument("--seconds", type=float, default=None,
                    help="auto-stop after N seconds")
    ap.add_argument("--max-utterances", type=int, default=None)
    ap.add_argument("--json", default=None, help="write the summary to this path")
    ap.add_argument("--meter", action="store_true",
                    help="show a live input level meter and the VAD gate, no model")
    args = ap.parse_args()
    use_utf8_stdout()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    if args.meter:
        run_meter(args)
        return

    print("loading model ...", flush=True)
    engine = Engine(args.model, args.device, args.compute_type, args.language,
                    args.beam_size, args.cpu_threads, args.vad_filter,
                    fallback=args.fallback, timestamps=args.timestamps)
    warm = engine.warmup()
    print("ready: %s  warmup=%.2fs" % (engine.describe(), warm))

    source = FileSource(args.file, speed=args.speed) if args.file \
        else MicSource(device=args.device_index)

    vad = EnergyVAD(end_ms=args.silence_ms, sensitivity=args.sensitivity,
                    frame_ms=FRAME_MS)

    worker = Worker(engine)
    worker.start()

    audio_q = queue.Queue()
    stop_event = threading.Event()

    finals, partials = [], []
    pending_finals = {}
    seq = 0

    preroll = deque(maxlen=int(300 / FRAME_MS))  # 300 ms so we don't clip onsets
    buf = []
    utt = None
    last_partial_at = 0.0
    calibrating = True
    calib_frames = []
    calib_target = max(1, int(args.calibrate_s * 1000 / FRAME_MS))

    with source:
        start_capture(source, audio_q, stop_event)
        t_start = time.perf_counter()
        if hasattr(source, "ambient_estimate"):
            calibrating = False
            print("vad threshold = %.5f (from the file's own noise floor)\n"
                  % vad.set_noise(source.ambient_estimate()), flush=True)
        else:
            print("calibrating ambient noise for %gs - stay quiet ..."
                  % args.calibrate_s, flush=True)

        try:
            while True:
                if args.seconds and time.perf_counter() - t_start > args.seconds:
                    break
                if args.max_utterances and len(finals) >= args.max_utterances:
                    break

                # ---- audio ----
                try:
                    stamp, frame = audio_q.get(timeout=0.05)
                except queue.Empty:
                    stamp = frame = None
                if stamp == "eof":
                    break
                if stamp == "error":
                    raise frame

                if frame is not None:
                    if calibrating:
                        calib_frames.append(frame)
                        if len(calib_frames) >= calib_target:
                            th = vad.calibrate(calib_frames)
                            calibrating = False
                            print("vad threshold = %.5f. speak now (ctrl-c to stop)\n"
                                  % th, flush=True)
                    else:
                        event = vad.push(frame)
                        preroll.append(frame)

                        if event == "start":
                            seq += 1
                            utt = {"id": seq, "t_start": stamp, "t_last_speech": stamp}
                            buf = list(preroll)
                            last_partial_at = stamp
                        elif utt is not None:
                            buf.append(frame)
                            if vad._run_silence == 0:
                                utt["t_last_speech"] = stamp

                        force_cut = (utt is not None and
                                     len(buf) * FRAME_SAMPLES / SAMPLE_RATE
                                     > args.max_utterance_s)

                        if utt is not None and (event == "end" or force_cut):
                            audio = np.concatenate(buf)
                            # drop the hangover silence we waited through
                            if event == "end":
                                trim = int((vad.hangover_ms - 120) / 1000 * SAMPLE_RATE)
                                if len(audio) > trim + SAMPLE_RATE // 4:
                                    audio = audio[:-trim]
                            dur_ms = len(audio) / SAMPLE_RATE * 1000
                            if dur_ms >= args.min_utterance_ms:
                                meta = {"id": utt["id"],
                                        "submitted": time.perf_counter(),
                                        "t_last_speech": utt["t_last_speech"]}
                                pending_finals[utt["id"]] = meta
                                worker.submit("final", audio, meta)
                            utt, buf = None, []
                            if force_cut:
                                vad.speaking = False

                        # ---- partials ----
                        if (utt is not None and args.partial_interval > 0
                                and not worker.busy
                                and stamp - last_partial_at >= args.partial_interval):
                            last_partial_at = stamp
                            worker.submit("partial", np.concatenate(buf),
                                          {"id": utt["id"],
                                           "submitted": time.perf_counter(),
                                           "newest_audio": stamp})

                # ---- results ----
                while True:
                    try:
                        kind, meta = worker.results.get_nowait()
                    except queue.Empty:
                        break
                    if kind == "partial":
                        meta["lag"] = meta["done"] - meta["newest_audio"]
                        partials.append({k: meta[k] for k in
                                         ("id", "infer", "lag", "audio_seconds", "text")})
                        print("  ~ " + meta["text"][-90:], end="\r", flush=True)
                    else:
                        meta["latency"] = meta["done"] - meta["t_last_speech"]
                        meta["rtf"] = meta["infer"] / max(meta["audio_seconds"], 1e-6)
                        finals.append({k: meta[k] for k in
                                       ("id", "text", "infer", "latency", "rtf",
                                        "audio_seconds", "queue_wait")})
                        print(" " * 100, end="\r")
                        print("[%.1fs audio | %.2fs infer | RTF %.2f | latency %.2fs] %s"
                              % (meta["audio_seconds"], meta["infer"], meta["rtf"],
                                 meta["latency"], meta["text"]), flush=True)
                        pending_finals.pop(meta["id"], None)

        except KeyboardInterrupt:
            print("\nstopping ...")
        finally:
            stop_event.set()

    # let anything in flight land
    deadline = time.perf_counter() + 30
    while pending_finals and time.perf_counter() < deadline:
        try:
            kind, meta = worker.results.get(timeout=0.2)
        except queue.Empty:
            continue
        if kind == "final":
            meta["latency"] = meta["done"] - meta["t_last_speech"]
            meta["rtf"] = meta["infer"] / max(meta["audio_seconds"], 1e-6)
            finals.append({k: meta[k] for k in
                           ("id", "text", "infer", "latency", "rtf",
                            "audio_seconds", "queue_wait")})
            pending_finals.pop(meta["id"], None)
    worker.stop()

    finals.sort(key=lambda u: u["id"])
    summary = report(finals, partials, engine, source, args, vad,
                     getattr(source, "overflows", 0))

    if args.json and summary:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print("wrote " + args.json)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
