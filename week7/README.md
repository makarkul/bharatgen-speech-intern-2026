# Week 7 — Profile, document, hand off

**Goal:** Everything captured well enough that the next intern starts at Week 6, not Week 0.

**Primary resources:** [PyTorch Profiler recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) (30m) · [Diátaxis framework](https://diataxis.fr/) (30m — structures the README / SETUP / EXPERIMENTS / NEXT_STEPS split).

## Hands-on

- **End-to-end profiling:** total pipeline latency for a 10-second Hindi → English audio task. Break it down by stage: ASR ms, translate ms, TTS ms, overhead ms. You should be able to point to where the time goes and why.
- **Memory profiling:** peak GPU memory for each model loaded individually and concurrently. Implication: what GPU tier do we actually need?
- **`LIMITATIONS.md`** — single document covering:
  - Language coverage gaps
  - Code-mixing failures
  - Accent sensitivity (Sooktam-2 is reference-bounded)
  - Translation degradation patterns
  - Latency bottlenecks
  - Models that don't fit on a single 16 GB GPU
- **Handoff package — promote to repo root:**
  - `README.md` — "How to run this in 30 minutes from scratch."
  - `SETUP.md` — env, GPU, HF tokens, model downloads.
  - `EXPERIMENTS.md` — what was tried, what worked, what didn't, what to try next.
  - `NEXT_STEPS.md` — three concrete projects the next intern could pick up (e.g., streaming ASR, on-device quantization, fine-tuning Sooktam-2 on a new speaker).
- **30-minute walkthrough video** (Loom) — demo the pipeline and walk through the repo structure.

## Deliverable

- Complete repo, tagged `v1.0-intern-handoff`.

## Anti-patterns

- Leaving docs for the last day. (Docs accumulate weekly from Week 1.)

## Week-7 success questions

If you can answer these without hedging, the plan worked:

1. For a 10-second Hindi audio file end-to-end translated to English speech, where does the time go? Give it in milliseconds per stage.
2. Why can't all three models run on a 16 GB GPU simultaneously without something giving?
3. What's the difference between how Shrutam-2 and Whisper handle code-mixed Hindi-English, based on your measurements?
4. Sooktam-2 is reference-guided. What does that mean for output quality, and for inference latency vs. a description-conditioned model like Indic Parler-TTS?
5. If I gave you a 30-minute audio file, what breaks and why?
