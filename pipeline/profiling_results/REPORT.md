# Speech-Translation Pipeline — Latency Profile

Generated: **2026-06-17T10:03:56+00:00**  ·  commit `1a64f65`

## Key findings (TL;DR)

1. **Translation (Param-2) is the bottleneck — ~67% of total latency.** On a cross-language request it is **77%** of the time (median 10.3s of a 13.3s total). ASR and transcode are negligible; TTS is a distant second.
2. **Measured proof, not a guess.** A same-language request skips translation entirely. Same-language median total is **3.3s**; cross-language is **13.3s** → translation adds **~10s**.
3. **The slowness is the model, not the architecture.** Splitting the translate step shows model compute = **10.26s** vs the HTTP hop to the separate Param-2 service = **0.009s** (9 ms). The two-service design costs effectively nothing; the cost is the 17B model.
4. **Latency is highly variable because Param-2 is a "thinking" model.** Translate ranges from ~6.8s to ~35s on similar inputs — it spends a variable amount of reasoning before answering, so input length does *not* predict latency (e.g. a 139-char input took 6.8s while a 175-char input took 35.3s). Median 10.3s but **p90 28s**. This variance, not just the average, is the headline risk.
5. **Cold start is ~60s.** The first request after a server boot pays a one-time warmup (translate alone took 59.6s on the first call). A live demo should send a throwaway warmup request at startup.

**Biggest lever:** reduce Param-2's reasoning cost — e.g. cap the thinking-token budget, try a non-thinking decode, or evaluate a smaller/faster translator. Everything else in the pipeline is already fast.

**Scope of this measurement:** single-request latency (one request at a time), Hindi and Tamil source clips, n = 24 warm runs on one H100. This is *not* a throughput / concurrent-load test.

## Environment

| Field | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3, 81559 MiB |
| Date (UTC) | 2026-06-17T10:03:56+00:00 |
| Git commit | `1a64f65` |
| ASR model | `bharatgenai/Shrutam-2` |
| Translate model | `bharatgenai/Param2-17B-A2.4B-Thinking` |
| TTS model | `bharatgenai/sooktam2` |
| transformers (main server) | 4.56.2 (Param-2 service runs 4.52.3 in its own venv) |

## How to read this

The pipeline runs four stages **sequentially**, so total latency is their sum. Each clip from the benchmark corpus (`shrutilipi` Hindi + Tamil) was POSTed through the live `/speak` endpoint, so these are real end-to-end request times (including the HTTP hop to the Param-2 translate service), not isolated micro-benchmarks. We report **median** (typical) and **p90 / max** (worst-case, which matters for a live demo) rather than a mean, because a single slow run shouldn't define the story.

## Cold start

The server was freshly booted for this run, so the first request pays one-time GPU/CUDA warmup. It is **excluded from the summary below** and shown here on its own:

- First request `hindi->tamil` (clip_001.wav): **62.682s total** (asr 0.878s · translate 59.62s · tts 2.084s)

## Summary — all warm runs

_n = 24 runs_

| Stage | Median (s) | p90 (s) | Max (s) | % of total (median) |
|---|---|---|---|---|
| Transcode (ffmpeg) | 0.1 | 0.1 | 0.11 | 1% |
| ASR (Shrutam-2) | 0.56 | 0.9 | 1.14 | 5% |
| Translate (Param-2) | 6.88 | 20.91 | 35.26 | 67% |
| TTS (Sooktam-2) | 2.73 | 3.54 | 5.1 | 27% |
| **TOTAL** | 10.25 | 26.99 | 37.95 | 100% |

## Cross-language vs same-language (the cost translation adds)

Same-language runs skip translation (it's a no-op), so comparing them to cross-language runs isolates how much the Param-2 step actually costs.

### Cross-language runs (translate active)

_n = 14 runs_

| Stage | Median (s) | p90 (s) | Max (s) | % of total (median) |
|---|---|---|---|---|
| Transcode (ffmpeg) | 0.1 | 0.1 | 0.11 | 1% |
| ASR (Shrutam-2) | 0.56 | 0.9 | 0.91 | 4% |
| Translate (Param-2) | 10.27 | 28.28 | 35.26 | 77% |
| TTS (Sooktam-2) | 2.75 | 3.07 | 5.1 | 21% |
| **TOTAL** | 13.27 | 31.05 | 37.95 | 100% |

### Same-language runs (translate skipped — control)

_n = 10 runs_

| Stage | Median (s) | p90 (s) | Max (s) | % of total (median) |
|---|---|---|---|---|
| Transcode (ffmpeg) | 0.09 | 0.1 | 0.1 | 3% |
| ASR (Shrutam-2) | 0.56 | 0.88 | 1.14 | 17% |
| Translate (Param-2) | 0.0 | 0.0 | 0.0 | 0% |
| TTS (Sooktam-2) | 2.52 | 3.54 | 3.63 | 77% |
| **TOTAL** | 3.26 | 4.15 | 4.3 | 100% |

## Inside the translate step (cross-language runs)

Splitting translate into model compute (reported by the Param-2 service) vs HTTP/network overhead shows whether the bottleneck is the 17B model or the service hop.

| Component | Median (s) | p90 (s) |
|---|---|---|
| Model generation | 10.26 | 28.27 |
| HTTP overhead | 0.009 | 0.01 |

**Takeaway:** the HTTP hop between the two services is ~9 ms — negligible. The translate cost is entirely the 17B model's generation.

## Per target language

> **Caveat:** these totals mix cross-language and same-language runs for each target (e.g. the Tamil row includes both slow Hindi→Tamil and fast Tamil→Tamil runs), so the median can look low while p90 catches a slow cross-language run. Read this table only for rough per-language feel; the cross-vs-same tables above are the controlled comparison.

| Target | n | Median total (s) | p90 total (s) |
|---|---|---|---|
| hindi | 10 | 8.55 | 18.48 |
| marathi | 5 | 10.54 | 13.75 |
| tamil | 9 | 3.62 | 31.05 |

## Raw data

Per-run measurements are in [`results.csv`](results.csv) (24 warm runs + 1 cold, flagged). Each row is one `/speak` request with every stage time, input duration, and output length, so any number above can be traced back to its source runs. Environment captured in [`environment.json`](environment.json).
