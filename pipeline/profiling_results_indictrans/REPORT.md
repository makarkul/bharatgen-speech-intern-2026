# Speech-Translation Pipeline — Latency Profile (IndicTrans2 translator)

Generated: **2026-06-23T05:20:24+00:00**  ·  commit `b2eb43e`

> This is the **IndicTrans2** run. The Param-2 counterpart (same clips, same matrix, same pod) is in [`../profiling_results/`](../profiling_results/). Only the translator differs between the two reports.

## Environment

| Field | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3, 81559 MiB |
| Date (UTC) | 2026-06-23T05:20:24+00:00 |
| Git commit | `b2eb43e` |
| ASR model | `bharatgenai/Shrutam-2` |
| Translate model | `ai4bharat/indictrans2-indic-indic-1B` |
| TTS model | `bharatgenai/sooktam2` |
| transformers (main server) | unknown (the IndicTrans2 translate service runs transformers 4.56.2 in its own venv) |

## How to read this

The pipeline runs four stages **sequentially**, so total latency is their sum. Each clip from the benchmark corpus (`datasets/shrutilipi/{hindi,tamil}`) was POSTed through the live `/speak` endpoint, so these are real end-to-end request times (including the HTTP hop to the IndicTrans2 translate service), not isolated micro-benchmarks. We report **median** (typical) and **p90 / max** (worst-case, which matters for a live demo) rather than a mean, because a single slow run shouldn't define the story.

## Cold start

The server was freshly booted for this run, so the first request pays one-time GPU/CUDA warmup. It is **excluded from the summary below** and shown here on its own:

- First request `hindi->tamil` (clip_001.wav): **2.967s total** (asr 0.357s · translate 0.461s · tts 2.081s)

## Summary — all warm runs

_n = 24 runs_

| Stage | Median (s) | p90 (s) | Max (s) | % of total (median) |
|---|---|---|---|---|
| Transcode (ffmpeg) | 0.07 | 0.07 | 0.07 | 2% |
| ASR (Shrutam-2) | 0.57 | 0.85 | 0.92 | 15% |
| Translate (IndicTrans2) | 0.39 | 0.68 | 0.73 | 10% |
| TTS (Sooktam-2) | 2.98 | 3.85 | 5.85 | 79% |
| **TOTAL** | 3.78 | 4.58 | 7.57 | 100% |

## Cross-language vs same-language (the cost translation adds)

Same-language runs skip translation (it's a no-op), so comparing them to cross-language runs isolates how much the IndicTrans2 step actually costs.

### Cross-language runs (translate active)

_n = 14 runs_

| Stage | Median (s) | p90 (s) | Max (s) | % of total (median) |
|---|---|---|---|---|
| Transcode (ffmpeg) | 0.07 | 0.07 | 0.07 | 2% |
| ASR (Shrutam-2) | 0.57 | 0.85 | 0.92 | 14% |
| Translate (IndicTrans2) | 0.48 | 0.69 | 0.73 | 12% |
| TTS (Sooktam-2) | 3.05 | 3.2 | 5.85 | 74% |
| **TOTAL** | 4.14 | 4.58 | 7.57 | 100% |

### Same-language runs (translate skipped — control)

_n = 10 runs_

| Stage | Median (s) | p90 (s) | Max (s) | % of total (median) |
|---|---|---|---|---|
| Transcode (ffmpeg) | 0.07 | 0.07 | 0.07 | 2% |
| ASR (Shrutam-2) | 0.57 | 0.84 | 0.87 | 17% |
| Translate (IndicTrans2) | 0.0 | 0.0 | 0.0 | 0% |
| TTS (Sooktam-2) | 2.77 | 3.85 | 4.07 | 80% |
| **TOTAL** | 3.45 | 4.46 | 4.76 | 100% |

## Inside the translate step (cross-language runs)

Splitting translate into model compute (reported by the IndicTrans2 service) vs HTTP/network overhead shows whether the bottleneck is the model or the service hop.

| Component | Median (s) | p90 (s) |
|---|---|---|
| Model generation | 0.47 | 0.68 |
| HTTP overhead | 0.007 | 0.007 |

## Per target language

| Target | n | Median total (s) | p90 total (s) |
|---|---|---|---|
| hindi | 10 | 4.16 | 4.76 |
| marathi | 5 | 4.2 | 4.5 |
| tamil | 9 | 3.28 | 3.67 |

## Raw data

Per-run measurements are in [`results.csv`](results.csv) (24 warm runs). Each row is one `/speak` request with every stage time, input duration, and output length, so any number above can be traced back to its source runs.
