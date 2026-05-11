# Week 6 — Pipeline integration

**Goal:** The three models stitched into one working `STT → Translate → TTS` demo with a Gradio UI.

**Primary resources:** [HF Audio Course Unit 7: Putting it all together](https://huggingface.co/learn/audio-course/chapter7/introduction) (2h) · [Gradio Quickstart](https://www.gradio.app/guides/quickstart) (45m) · [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html) (30m).

## Concepts (lean)

- Pipeline architecture: where state lives (audio buffers, intermediate text, language metadata).
- Failure modes that only appear when models compose.
- GPU memory budgeting: three models, finite GPU. All loaded? Lazy load? Offload to CPU? Trade-offs.

## Hands-on

- Wire Shrutam-2 + IndicTrans2 + Sooktam-2 together. Input: Hindi audio. Output: English audio (or Hindi → Tamil, etc.).
- Build a simple Gradio UI: file upload or mic input, source/target language selectors, audio player output.
- **GPU placement strategy:** pick one (all loaded / lazy load / evict between calls). **Defend it in writing.** The defending is more valuable than the optimum.
- Handle exactly **3 failure modes** — don't try to handle every edge:
  1. Empty transcription
  2. Very long input audio (>5 min)
  3. Translation timeout
- Profiling: end-to-end wall clock for a 10s audio input, per-stage breakdown.

## Deliverable

- Working Gradio demo (`week6/app.py` or similar)
- `week6/architecture.md` with a **hand-drawn** block diagram (photo committed) showing data flow, model placements, memory budget, known failure modes.

## Anti-patterns

- Streaming / realtime / WebSockets. (Its own multi-week project.)
- Containerizing prematurely. Plain Python + Gradio. (Containerization without a deployment target is theater.)
