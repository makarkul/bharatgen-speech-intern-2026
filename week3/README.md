# Week 3 — Shrutam-2 (ASR)

**Goal:** BharatGen's ASR runs reliably on your machine. You can quantify how good it is and where it breaks.

**Primary resources:** [HF Audio Course Unit 5: ASR](https://huggingface.co/learn/audio-course/chapter5/introduction) (2-3h) · [jiwer library](https://github.com/jitsi/jiwer) (15m) · [Whisper paper](https://arxiv.org/abs/2212.04356) (skim, supplementary).

## Concepts (lean)

- What ASR is
- WER (Word Error Rate) and CER (Character Error Rate)
- Why code-mixing is hard: tokenizer coverage, training-data distribution
- Beam search vs greedy decoding (1 paragraph each — no implementation)

## Hands-on

- Run `bharatgenai/Shrutam-2` on:
  - 10 Hindi clips
  - 10 Tamil clips
  - 5 code-mixed Hindi-English clips
  - Use **real** audio (YouTube interviews, podcast snippets, your own recordings) — **not** model-card examples.
- Compute WER against ground-truth transcripts using `jiwer`.
- Run `openai/whisper-large-v3` on the same clips. Compare WER per language.
- **Profiling (introduce this week, reuse every week after):**
  - End-to-end latency
  - GPU memory peak (`torch.cuda.max_memory_allocated()`)
  - Throughput (audio-seconds per inference-second)
  - **Always note hardware, batch size, audio length, language.** A number without context is noise.

## Deliverable

`week3/week3_shrutam2_eval.md` containing:

- WER table — Shrutam-2 vs Whisper, per language
- Latency / memory table
- 5 transcription failures with your hypothesis on *why* each failed

## Anti-patterns

- Trying to "improve" the model. (Improvement is meaningless before measurement is solid.)
- Confusing audio length with model latency.
- Reporting numbers without describing the hardware.
