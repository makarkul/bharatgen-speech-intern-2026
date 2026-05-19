# BharatGen Speech Translation — Intern Project (2026)

8-week project to build a working `STT → Translate → TTS` demo using:

- **Shrutam-2** (`bharatgenai/Shrutam-2`) — ASR
- **IndicTrans2** (`ai4bharat/indictrans2-indic-en-1B` / `en-indic-1B`) — Translation
- **Sooktam-2** (`bharatgenai/sooktam2`) — TTS

End state: a Gradio demo a non-engineer can click through, plus a handoff package that lets the next intern start at Week 6, not Week 0.

## Repo layout

```
week0/    Python + ML bootstrap (5 days)
week1/    Audio and tensors
week2/    Hugging Face mental model
week3/    Shrutam-2 (ASR) evaluation
week4/    Sooktam-2 (TTS) evaluation
week5/    IndicTrans2 (translation) evaluation
week6/    Pipeline integration + Gradio UI
week7/    Profile, document, hand off
docs/     The 8-week plan and tracker (xlsx + generated views)
scripts/  Tooling — render_tracker.py regenerates docs/TRACKER.md + docs/tracker.csv
```

## Tracker

- [`docs/TRACKER.md`](docs/TRACKER.md) — status summary, per-week progress, deliverables, cadence, glossary, resources (GitHub-rendered Markdown)
- [`docs/tracker.csv`](docs/tracker.csv) — full 98-item tracker (GitHub renders CSV as a sortable, searchable table)
- [`docs/bharatgen_intern_tracker.xlsx`](docs/bharatgen_intern_tracker.xlsx) — source of truth; edit and push, and the [`Render tracker`](../../actions/workflows/render-tracker.yml) GitHub Action regenerates the Markdown + CSV views automatically. To regenerate locally: `python scripts/render_tracker.py`.

Each `weekN/` has a `README.md` with that week's goals, hands-on tasks, deliverables, and anti-patterns.

## Getting started

1. Read [`docs/intern_8week_plan.md`](docs/intern_8week_plan.md) end-to-end (~30 min).
2. Complete the pre-Week-0 setup in [`SETUP.md`](SETUP.md) — HF token, RunPod pod, `HF_HOME` on persistent volume, tooling.
3. Start with [`week0/README.md`](week0/README.md).

## Working agreement

| When | What |
|---|---|
| Monday morning | Intern posts the week's plan in Slack thread; mentor reacts |
| Wednesday | Optional 10-min blocker check |
| Friday EOD | Commit week's deliverable + 5-bullet progress note |
| Friday next-day | Mentor reviews commits as PR comments |

No daily standups. Async by default. The Excel tracker in `docs/` is the single source of truth for status.

## Guiding principles (from the plan)

1. **Build first, theorize second.** Theory is introduced just-in-time when it explains something you've already seen break.
2. **One model per week, one deliverable per week.** No skipping.
3. **The inference stack is taught by profiling, not lecturing.** Latency + memory measurements are part of every deliverable from Week 3 onward.
4. **Limitations are not afterthoughts.** Every model exercise ends with "where does this fail?" written down.

## Things to deliberately *not* do

- No fine-tuning (not even LoRA). Inference-only for 7 weeks.
- No training-from-scratch toy models.
- No vLLM / TensorRT / quantization. Week 8+ territory.
- No real-time streaming. The demo is request-response.
- No Docker until Week 6, and only if asked.
