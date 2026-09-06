"""Offline speed sweep: which Whisper model can actually keep up live, on this box.

The live harness answers "how did that session feel". This answers "what should
I run" - it sweeps models, compute types and chunk sizes over one fixed piece of
audio, so the numbers are comparable.

The chunk sweep is the important half. A live system never hands Whisper a
10-minute file; it hands it 1-8 second utterances, and short clips have a fixed
overhead that RTF on a long file hides completely.

Usage:
  python bench.py --audio sample.wav --models tiny,base,small
  python bench.py --record 12 --models tiny,base,small,medium
  python bench.py --audio sample.wav --models base --compute-types int8,float32
"""

import argparse
import json
import statistics
import sys
import time

import numpy as np

from audio_io import SAMPLE_RATE, decode_to_mono16k
from engine import Engine, use_utf8_stdout


def record(seconds, device=None):
    import sounddevice as sd
    print("recording %gs - speak now ..." % seconds, flush=True)
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32", device=device)
    sd.wait()
    print("done.")
    return audio[:, 0].copy()


def save_wav(path, audio, samplerate=SAMPLE_RATE):
    import wave
    pcm = np.clip(audio, -1, 1)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(samplerate)
        w.writeframes(pcm.tobytes())


def run(engine, audio, repeat):
    times = []
    text = ""
    for _ in range(repeat):
        text, dt, _ = engine.transcribe(audio)
        times.append(dt)
    return min(times), statistics.median(times), text


def main():
    ap = argparse.ArgumentParser(description="Whisper speed sweep")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", help="audio file (anything ffmpeg reads)")
    src.add_argument("--record", type=float, help="record N seconds from the mic first")
    ap.add_argument("--device-index", type=int, default=None)
    ap.add_argument("--save-recording", default="bench_sample.wav")

    ap.add_argument("--models", default="tiny,base,small")
    ap.add_argument("--compute-types", default=None,
                    help="default: int8 on cpu, float16 on cuda")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--language", default=None)
    ap.add_argument("--beam-size", type=int, default=1)
    ap.add_argument("--cpu-threads", type=int, default=0)
    ap.add_argument("--chunks", default="1,2,4,8",
                    help="chunk lengths in seconds for the latency sweep; '' to skip")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--fallback", action="store_true",
                    help="enable whisper's temperature fallback (accurate, unstable timing)")
    ap.add_argument("--timestamps", action="store_true",
                    help="decode segment timestamps too (slower)")
    ap.add_argument("--full", action="store_true",
                    help="also transcribe the whole clip in one call")
    ap.add_argument("--json", default="bench_results.json")
    args = ap.parse_args()
    use_utf8_stdout()

    if args.record:
        audio = record(args.record, args.device_index)
        save_wav(args.save_recording, audio)
        print("saved " + args.save_recording)
        label = args.save_recording
    else:
        audio = decode_to_mono16k(args.audio)
        label = args.audio

    total = len(audio) / SAMPLE_RATE
    if total < 1.0:
        sys.exit("audio is only %.2fs - need at least 1s" % total)
    print("audio: %s  (%.1fs, 16k mono)\n" % (label, total))

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ctypes = [c.strip() for c in args.compute_types.split(",")] if args.compute_types \
        else [None]
    chunks = [float(c) for c in args.chunks.split(",") if c.strip()] if args.chunks else []
    chunks = [c for c in chunks if c <= total]

    rows = []
    for model in models:
        for ctype in ctypes:
            try:
                engine = Engine(model, args.device, ctype, args.language,
                                args.beam_size, args.cpu_threads,
                                fallback=args.fallback, timestamps=args.timestamps)
            except Exception as e:
                print("SKIP %s/%s: %s" % (model, ctype, e))
                continue
            engine.warmup()
            tag = "%s/%s" % (model, engine.compute_type)
            row = {"model": model, "repo": engine.repo, "device": engine.device,
                   "compute_type": engine.compute_type,
                   "beam_size": args.beam_size,
                   "load_seconds": round(engine.load_seconds, 3),
                   "chunks": {}, "full": None}

            for c in chunks:
                clip = audio[:int(c * SAMPLE_RATE)]
                best, med, text = run(engine, clip, args.repeat)
                row["chunks"][str(c)] = {"best": round(best, 3),
                                         "median": round(med, 3),
                                         "rtf": round(med / c, 3),
                                         "text": text}
                print("  %-22s %5.1fs chunk -> %6.2fs  RTF %5.2f"
                      % (tag, c, med, med / c))

            if args.full:
                best, med, text = run(engine, audio, 1)
                row["full"] = {"seconds": round(med, 3), "rtf": round(med / total, 3),
                               "text": text}
                print("  %-22s full %.1fs   -> %6.2fs  RTF %5.2f"
                      % (tag, total, med, med / total))

            rows.append(row)
            del engine
            print()

    if not rows:
        sys.exit("nothing ran")

    # ---- table ----
    print("=" * 78)
    print("WHISPER SPEED SWEEP  -  %s, beam=%d" % (rows[0]["device"], args.beam_size))
    print("audio: %s (%.1fs)" % (label, total))
    print("=" * 78)
    header = "%-14s %-12s %7s" % ("model", "compute", "load")
    for c in chunks:
        header += "  %8s" % ("%gs" % c)
    if args.full:
        header += "  %8s" % "full"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "%-14s %-12s %6.2fs" % (r["model"], r["compute_type"], r["load_seconds"])
        for c in chunks:
            d = r["chunks"].get(str(c))
            line += "  %7.2fs" % d["median"] if d else "  %8s" % "-"
        if args.full and r["full"]:
            line += "  %7.2fs" % r["full"]["seconds"]
        print(line)

    print("\nsame numbers as RTF (inference / audio; < 1.0 keeps up live)")
    header2 = "%-14s %-12s" % ("model", "compute")
    for c in chunks:
        header2 += "  %8s" % ("%gs" % c)
    if args.full:
        header2 += "  %8s" % "full"
    print(header2)
    print("-" * len(header2))
    for r in rows:
        line = "%-14s %-12s" % (r["model"], r["compute_type"])
        for c in chunks:
            d = r["chunks"].get(str(c))
            line += "  %8.2f" % d["rtf"] if d else "  %8s" % "-"
        if args.full and r["full"]:
            line += "  %8.2f" % r["full"]["rtf"]
        print(line)

    print("\ntranscript check (shortest chunk, so quality and speed are read together)")
    if chunks:
        c0 = str(chunks[0])
        for r in rows:
            d = r["chunks"].get(c0)
            if d:
                print("  %-14s %s" % (r["model"], d["text"][:100]))

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump({"audio": label, "audio_seconds": total,
                   "beam_size": args.beam_size, "rows": rows}, fh, indent=2)
    print("\nwrote " + args.json)


if __name__ == "__main__":
    main()
