# Speech-Translation Pipeline — Detailed Latency Analysis

**H100 80GB HBM3** · commit `1a64f65` · 2026-06-17T10:03:56Z
Pipeline: ffmpeg transcode → **Shrutam-2** (ASR) → **Param2-17B-A2.4B-Thinking** (translate) → **sooktam2** (TTS), run sequentially over the live `/speak` endpoint.

| Field | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3, 81559 MiB |
| ASR model | `bharatgenai/Shrutam-2` |
| Translate model | `bharatgenai/Param2-17B-A2.4B-Thinking` |
| TTS model | `bharatgenai/sooktam2` |
| transformers | 4.56.2 main server (Param-2 service runs 4.52.3 in its own venv) |
| Source corpus | shrutilipi — **10 unique clips** (5 Hindi + 5 Tamil), 109.6s total audio |
| Runs | **24 warm + 1 cold** = each clip × its matrix of target languages |

> **On the corpus size:** the profile uses **10 unique source clips**, each sent through several target languages (Hindi→{Tamil, Marathi, Hindi}; Tamil→{Hindi, Tamil}), producing 24 warm request-runs. So "24 runs" ≠ "24 clips" — it's 10 clips exercised across language pairs, including same-language controls.

---

## Key findings (TL;DR)

1. **Translation (Param-2) is the bottleneck — ~67% of total latency**, rising to **77%** on cross-language requests (median 10.3s of a 13.3s total).
2. **It's the model, not the architecture.** Inside translate: model generation = **10.26s**, HTTP hop to the separate Param-2 service = **9 ms**. The two-service split costs effectively nothing.
3. **Proven by a control.** Same-language requests skip translation → median total **3.3s** vs cross-language **13.3s**. Translation adds **~10s**.
4. **Latency is driven by the model's "thinking," not input size.** Translate time barely correlates with input audio length (r = +0.05) or source character count (r = +0.08) — see [Correlation analysis](#correlation-what-actually-drives-each-stage). A 139-char input took 6.8s while a 175-char input took 35.3s. Median 10.3s but **p90 28s, max 35s**. The variance, not the average, is the real demo risk.
5. **Cold start ~60s** (first post-boot request pays one-time GPU/CUDA warmup; translate alone took 59.6s). Mitigation: send a throwaway warmup request at startup.

**Biggest lever:** cut Param-2's reasoning cost — cap the thinking-token budget, try a non-thinking decode, or evaluate a smaller/faster translator. Everything else is already fast.

> **Scope:** single-request latency (one request at a time), Hindi + Tamil sources. **Not** a throughput / concurrent-load test. Output (TTS) audio duration was not captured in this run.

---

## Summary — all warm runs (n = 24)

| Stage | Median (s) | p90 (s) | Max (s) | % of total (median) |
|---|---|---|---|---|
| Transcode (ffmpeg) | 0.1 | 0.1 | 0.11 | 1% |
| ASR (Shrutam-2) | 0.56 | 0.9 | 1.14 | 5% |
| Translate (Param-2) | 6.88 | 20.91 | 35.26 | 67% |
| TTS (Sooktam-2) | 2.73 | 3.54 | 5.1 | 27% |
| **TOTAL** | **10.25** | **26.99** | **37.95** | 100% |

## Cross-language vs same-language (the cost translation adds)

Same-language runs skip translation (no-op), so the gap isolates Param-2's real cost.

**Cross-language (translate active), n = 14**

| Stage | Median (s) | p90 (s) | Max (s) | % of total |
|---|---|---|---|---|
| Transcode | 0.1 | 0.1 | 0.11 | 1% |
| ASR | 0.56 | 0.9 | 0.91 | 4% |
| Translate | 10.27 | 28.28 | 35.26 | 77% |
| TTS | 2.75 | 3.07 | 5.1 | 21% |
| **TOTAL** | **13.27** | **31.05** | **37.95** | 100% |

**Same-language (translate skipped — control), n = 10**

| Stage | Median (s) | p90 (s) | Max (s) | % of total |
|---|---|---|---|---|
| Transcode | 0.09 | 0.1 | 0.1 | 3% |
| ASR | 0.56 | 0.88 | 1.14 | 17% |
| Translate | 0.0 | 0.0 | 0.0 | 0% |
| TTS | 2.52 | 3.54 | 3.63 | 77% |
| **TOTAL** | **3.26** | **4.15** | **4.3** | 100% |

## Inside the translate step (cross-language runs)

| Component | Median (s) | p90 (s) |
|---|---|---|
| Model generation | 10.26 | 28.27 |
| HTTP overhead | 0.009 | 0.01 |

**Takeaway:** the service hop is ~9 ms — negligible. The cost is entirely the 17B model's generation.

---

## Input clip duration analysis

The 10 source clips and their measured durations (from `transcripts.tsv`):

| Source | Clip | Input duration (s) |
|---|---|---|
| Hindi | clip_001 | 8.60 |
| Hindi | clip_002 | 10.12 |
| Hindi | clip_003 | 12.52 |
| Hindi | clip_004 | 12.24 |
| Hindi | clip_005 | 9.36 |
| Tamil | clip_001 | 12.48 |
| Tamil | clip_002 | 10.00 |
| Tamil | clip_003 | 11.00 |
| Tamil | clip_004 | 8.76 |
| Tamil | clip_005 | 14.56 |

| | Value |
|---|---|
| Range | 8.60s – 14.56s |
| Median | 10.56s |
| Mean | 10.98s |
| Unique audio | 109.6s across 10 clips |

These are short, conversational-length news clips — a representative single-utterance demo workload, not long-form. The narrow 8.6–14.6s band matters for the next section: it means input length is roughly held constant, so the wide swing in translate time **cannot** be explained by input size.

## Real-time factor (RTF)

RTF = total latency ÷ input audio duration. Below 1.0× means the pipeline is faster than real-time (it returns the answer in less time than the clip plays); above 1.0× means slower than real-time.

| Run type | Median RTF | Min | Max |
|---|---|---|---|
| All warm | 0.96× | 0.24× | 3.75× |
| **Cross-language** | **1.15×** | 0.84× | **3.75×** |
| Same-language (control) | 0.32× | 0.24× | 0.38× |

Per-stage RTF (median, share of input audio length consumed):

| Stage | RTF (median) | Range |
|---|---|---|
| ASR (Shrutam-2) | 0.05× | 0.035×–0.086× |
| TTS (Sooktam-2) | 0.24× | 0.165×–0.409× |

**Reading it:**
- **Same-language is comfortably real-time (0.32×)** — ASR + TTS alone process a 10s clip in ~3s.
- **Cross-language tips past real-time (1.15× median, up to 3.75×)** purely because translation is bolted on. For a live demo, a cross-language request can take ~3.75× the clip's own length in the worst case.
- ASR (0.05×) and TTS (0.24×) are both well under real-time and scale predictably with audio length — neither is a concern.

## Correlation — what actually drives each stage

Pearson r between **input audio duration** and each stage's latency (n = 24 warm):

| vs input duration | r | Interpretation |
|---|---|---|
| ASR | **+0.65** | Strong — ASR scales with audio length, as expected. |
| TTS | +0.26 | Weak-positive — driven more by output text length than input audio. |
| Translate | **+0.05** | ~None — input length does not predict translate time. |
| Total | +0.08 | ~None — total is dominated by the uncorrelated translate stage. |

And translate vs **source character count** (cross-language, n = 14): **r = +0.08** — also no relationship.

**Conclusion:** ASR behaves like a normal audio model (longer clip → more time). Translation does **not** — neither audio length nor text length predicts its latency. The driver is Param-2's variable internal reasoning ("thinking") budget per request. This is the quantitative backing for finding #4: optimizing the translator (thinking budget / decode strategy / model choice) is the only lever that moves total latency.

---

## Raw data

Per-run measurements (all 24 warm + 1 cold, flagged) are in [`results.csv`](results.csv) — every stage time, input duration, and source/target character count. Environment in [`environment.json`](environment.json). Condensed version for quick sharing: [`SUMMARY.md`](SUMMARY.md).

*Note on output duration:* this run did not record synthesized-audio length, so output RTF / speaking-rate metrics aren't available. The profiler ([`../profile_pipeline.py`](../profile_pipeline.py)) would need to measure the returned WAV's duration to add that on a future run.
