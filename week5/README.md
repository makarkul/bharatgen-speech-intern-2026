# Week 5 — Translation (IndicTrans2)

**Goal:** Text-to-text translation between Indic languages and English works cleanly, and you have a clear mental model of the encoder-decoder pattern.

**Primary resources:** [HF LLM Course Ch 7.4: Translation](https://huggingface.co/learn/llm-course/chapter7/4) (1.5h) · [Jay Alammar: Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (full read this time, 1h) · [IndicTrans2 paper](https://arxiv.org/abs/2305.16307) (skim Sections 1-3).

## Concepts (lean)

- Sequence-to-sequence (seq2seq): ASR, TTS, translation are all seq2seq across different modalities.
- Encoder (reads whole input, builds representation) vs decoder (generates one token at a time, looking at encoder output). Why encoder-decoder beats encoder-only for translation.
- Autoregressive generation — latency scales with output length.
- Beam search — now actually run `num_beams=1, 4, 8`. Observe output and latency changes.

## Hands-on

- Run both `ai4bharat/indictrans2-indic-en-1B` and `ai4bharat/indictrans2-en-indic-1B` on 30 sample sentences across 4 languages.
- Edge cases:
  - Technical/legal Hindi
  - Casual WhatsApp-style text
  - Numbers and dates
  - Code-mixed input
- Round-trip: Hindi → English → Hindi for 10 sentences. Where does meaning drift?
- Profiling: latency, GPU memory, tokens/sec.

## Deliverable

- `week5/week5_indictrans2_eval.md` — eval doc with tables and observations
- `week5/translate.py` — small CLI: `python translate.py --src hi --tgt en --text "..."`

The CLI gets imported by the Week 6 pipeline. Keep it small.

## Anti-patterns

- Treating BLEU as the only metric. (Brittle for morphologically rich Indic languages — use BLEU + round-trip + human spot-checks.)
- Forgetting that translation quality degrades with sentence length.
