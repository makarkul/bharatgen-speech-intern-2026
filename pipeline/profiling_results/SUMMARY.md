# Speech-Translation Pipeline — Latency Profile (Summary)

H100 80GB · commit `1a64f65` · 2026-06-17 · **n = 24 warm runs + 1 cold** (5 Hindi + 5 Tamil clips × multiple target languages, run one-at-a-time through the live `/speak` endpoint).

Models: ASR `Shrutam-2` · Translate `Param2-17B-A2.4B-Thinking` · TTS `sooktam2`.

## Findings

1. **Translation (Param-2) is the bottleneck — ~67% of total latency**, rising to **77%** on a cross-language request.
2. **It's the model, not the architecture.** Inside the translate step: model generation = **10.26s**, the HTTP hop to the separate Param-2 service = **9ms**. The two-service split costs effectively nothing — the cost is the 17B model.
3. **Proven by control.** Same-language requests skip translation entirely → median total **3.3s** vs cross-language **13.3s**. Translation adds **~10s**.
4. **High variance because Param-2 is a "thinking" model.** Translate ranges ~6.8s → 35s on similar inputs; input length does *not* predict latency (a 139-char input took 6.8s, a 175-char input took 35.3s). Median 10.3s but **p90 28s** — the variance, not just the average, is the real demo risk.
5. **Cold start ~60s** (first request after boot pays a one-time GPU/CUDA warmup — translate alone took 59.6s on the first call). Mitigation: fire a throwaway warmup request at startup.

**Biggest lever:** cut Param-2's reasoning cost — cap the thinking-token budget, try a non-thinking decode, or evaluate a smaller/faster translator. Everything else in the pipeline is already fast.

> Scope: single-request latency (one request at a time), Hindi + Tamil sources. **Not** a throughput / concurrent-load test.

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

---

*Full methodology, per-language breakdown, and raw per-run data: see `REPORT.md` and `results.csv` in this folder.*
