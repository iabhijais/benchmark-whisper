# live-transcribe

Live microphone transcription with Whisper, built to be measured rather than
admired. Every decode is timed, and the run ends in a report.

Two entry points:

| | |
|---|---|
| `live_transcribe.py` | the live system — mic in, text out, latency measured |
| `bench.py` | offline sweep — which model can keep up on this machine |

Both sit on `faster-whisper` (CTranslate2), which is already installed here
along with the tiny → large-v3 weights.

---

## Quick start

Live from the microphone:

```bash
python live_transcribe.py --model base --language en
```

Speak. Each utterance prints as it is finalised, with its own timing, and a
summary table prints on Ctrl-C.

Find out what your machine can run before you commit to a model:

```bash
python bench.py --record 15 --models tiny,base,small,medium
```

---

## How the live path works

```
mic (30 ms frames) ─▶ energy VAD ─▶ utterance buffer ─▶ worker thread ─▶ text
                          │                                  │
                   start / end events              faster-whisper.transcribe()
```

The capture thread never blocks on inference, so a slow model shows up as
growing latency rather than dropped audio. Transcription runs on a single
worker: one decode at a time, which is what a real deployment does and what
keeps the timings interpretable.

Segmentation is an RMS gate over 30 ms frames, calibrated against the room at
startup (against the file's own noise floor in `--file` mode, since a file's
first second is usually speech). A neural VAD was deliberately not used for
segmentation — a torch forward pass per frame would land inside the very
measurement this tool exists to take. Whisper's own Silero VAD is still
available as `--vad-filter` if you want it inside the decode.

### The metrics

| metric | meaning |
|---|---|
| `audio` | length of speech handed to Whisper |
| `infer` | wall time inside `transcribe()` |
| `RTF` | `infer / audio`. Below 1.0 means it keeps up with live speech |
| `latency` | **last loud frame captured → final text on screen.** What a user feels |
| `qwait` | how long the final sat behind an in-flight partial |

`latency` is the honest number and it is always larger than `infer`, because it
includes the VAD hangover — the silence you deliberately wait through before
declaring the utterance over. The report prints latency both raw and with the
hangover subtracted, so you can see how much is inference and how much is a
tuning choice you made.

`RTF < 1.0` is necessary but not sufficient for live use. A model with RTF 0.5
still leaves you a full second behind on a 2-second utterance.

### Partial results

```bash
python live_transcribe.py --model base --partial-interval 0.8
```

Re-decodes the growing buffer while the person is still talking, so text
appears mid-sentence. It costs something real: a final can land behind an
in-flight partial, which the report shows as `qwait`. In a measured run here,
partials pushed median final latency from ~1.1 s to ~1.75 s. Worth it for
perceived responsiveness, not worth it if you need the fastest final.

---

## Measured on this machine

CPU-only (no CUDA), 16 cores, int8, `beam_size=1`, language pinned.
Numbers are seconds of inference for one chunk of that length — see
`bench_results.json` for the full run.

Seconds of inference for one chunk of that length:

| model | load | 2 s | 4 s | 8 s | 15 s | full 59 s |
|---|---|---|---|---|---|---|
| tiny | 0.77 s | 0.36 | 0.43 | 1.21 | 1.13 | 1.91 |
| base | 0.76 s | 0.73 | 0.89 | 1.09 | 1.52 | 4.17 |
| small | 1.40 s | 6.06 | 3.35 | 6.01 | 5.94 | 11.72 |
| medium | 4.33 s | 8.14 | 10.87 | 17.63 | 17.76 | 20.13 |
| large-v3 | 8.41 s | 15.63 | 19.45 | 27.59 | 31.70 | 63.58 |

The same numbers as RTF — **below 1.0 keeps up with live speech**:

| model | 2 s | 4 s | 8 s | 15 s | full |
|---|---|---|---|---|---|
| tiny | **0.18** | **0.11** | **0.15** | **0.07** | **0.03** |
| base | **0.36** | **0.22** | **0.14** | **0.10** | **0.07** |
| small | 3.03 | **0.84** | **0.75** | **0.40** | **0.20** |
| medium | 4.07 | 2.72 | 2.20 | 1.18 | **0.34** |
| large-v3 | 7.82 | 4.86 | 3.45 | 2.11 | **1.08** |

**On this CPU, `tiny` and `base` are the only genuinely live-capable models.**
Both stay near RTF 0.1–0.2 at every realistic utterance length, and `base` is
the better transcript of the two for a cost of roughly half a second.

`small` is the interesting one: comfortable at 4–8 s (RTF 0.75–0.84) but RTF
3.03 at 2 s. That is not noise — the transcript check for that cell shows
`पूलिसर पूलिसर पूलिसर …`, a repetition loop running to the 448-token limit.
A degenerate decode costs full price, and short clips provoke it.

`medium` and `large-v3` are batch tools here, not live ones. large-v3 needs
27.6 s to transcribe 8 s of speech, and 8.4 s just to load. They are in the
table because they are what the accuracy ladder actually costs — read the
transcript check at the bottom of a sweep alongside the timings.

The shape to notice: **short chunks are disproportionately expensive.** Whisper
pads every input to 30 s internally, so a 1-second clip does not cost a
thirtieth of a 30-second clip. That is the single most important fact for a
live system, and it is invisible if you only ever benchmark whole files.

---

## The trap that makes benchmarks lie

Whisper's default decoding retries at temperatures `0.0, 0.2, 0.4, 0.6, 0.8,
1.0` whenever a decode trips the compression-ratio or log-probability
threshold. One clip can silently decode **six times**.

The first version of this sweep, before the flag existed, produced:

```
base   1s chunk -> 12.60s   (RTF 12.60)
base   8s chunk ->  1.41s   (RTF  0.18)
```

A 1-second clip appearing to take nine times longer than an 8-second one is not
a measurement, it is a fallback cascade. Both tools therefore **pin
`temperature=0.0` by default** — one decode, one number — and put the standard
behaviour behind `--fallback`.

Use `--fallback` when you care about transcript quality; leave it off when you
care about the stopwatch. Measured cost on an 8 s chunk here: 0.84 s → 1.25 s,
about 50%, on audio where the text came out identical either way. On harder
audio it is the difference between a clean transcript and a repetition loop.

---

## Tuning the gate

If nothing is detected, or silence is transcribed as speech, look at the input
before touching the model:

```bash
python live_transcribe.py --meter
```

A live bar with the VAD gate marked. Speech should push well past the mark.
Adjust with `--sensitivity` (gate = noise floor × this; default 3.0).

The built-in mic array on this machine calibrates to a fairly high noise floor
(~0.035), so `--sensitivity 2.0` may segment better than the default here.

---

## Options worth knowing

**`live_transcribe.py`**

| flag | default | notes |
|---|---|---|
| `--model` | `base` | `tiny…large-v3`, `turbo`, `distil-small`, or any HF repo |
| `--language` | auto | pin it. Auto-detection costs time on every call |
| `--partial-interval` | `0` (off) | seconds between mid-utterance decodes |
| `--silence-ms` | `600` | VAD hangover. Lower = snappier, more mid-sentence cuts |
| `--max-utterance-s` | `25` | forces a cut in continuous speech |
| `--sensitivity` | `3.0` | VAD gate multiplier |
| `--fallback` | off | Whisper's temperature retries |
| `--file` | — | feed a file in real time instead of a mic |
| `--seconds`, `--max-utterances` | — | auto-stop, for scripted runs |
| `--json` | — | write the summary out |

**`bench.py`**

| flag | default | notes |
|---|---|---|
| `--audio` / `--record N` | — | one is required |
| `--models` | `tiny,base,small` | comma-separated |
| `--chunks` | `1,2,4,8` | chunk lengths for the latency sweep |
| `--compute-types` | `int8` on CPU | try `int8,float32` to see the trade |
| `--full` | off | also time the whole clip in one call |

---

## Notes

- `--file` mode paces the audio in real time through the identical pipeline, so
  latency numbers mean the same thing as they do on a mic. Use `--speed 2` to
  run a long file faster, but then the latency figures are no longer wall-clock
  honest — RTF still is.
- `turbo` and the `distil-*` aliases point at HF repos that are **not** cached
  locally; first use downloads them.
- On a CUDA box, `--device cuda --compute-type float16` typically moves RTF by
  an order of magnitude. Nothing in the harness needs to change.
