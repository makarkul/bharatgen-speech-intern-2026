# Week 4 — Sooktam-2 (TTS)

**Goal:** TTS pipeline runs. You understand reference-guided voice cloning. You have informed opinions on output quality, backed by listenable artifacts.

**Primary resource:** [HF Audio Course Unit 6: From text to speech](https://huggingface.co/learn/audio-course/chapter6/introduction) (2h).

## Concepts (lean)

- Embeddings — a vector of numbers representing something (token, speaker, image). Similar things → similar vectors.
- Reference-guided voice conditioning: Sooktam-2 takes a reference WAV + reference transcript and produces a speaker embedding. The TTS is conditioned on it — output mimics the reference voice.
- Vocoder vs acoustic model:
  - Acoustic model: text → mel-spectrogram
  - Vocoder: mel-spectrogram → waveform
  - Buzzy output = vocoder issue. Wrong pronunciation = acoustic model issue. Different fixes.
- RTF (Real-Time Factor) = generation_time / audio_duration. RTF < 1 = faster than real-time.

## Hands-on

- Run `bharatgenai/sooktam2` with `trust_remote_code=True`. Supply a clean 3–10s reference WAV + reference transcript + target text.
- Generate the same sentence in **Hindi, Marathi, Tamil** with **3 different reference speakers** each. Listen carefully.
- Edge cases — document what breaks:
  - Very long target text (>500 chars)
  - English loanwords inside Hindi
  - Numbers and currency ("2026", "₹450")
  - Code-mixed input
- Compare with `ai4bharat/indic-parler-tts` on the same Hindi sentences. Reference-conditioned vs description-conditioned paradigms — articulate the trade-off.
- Profiling table (same as Week 3): latency, peak GPU memory, **RTF**.

## Deliverable

- `week4/week4_sooktam2_eval.md` — failure cases, latency/RTF table, Sooktam-2 vs Indic-Parler-TTS comparison
- `week4/samples/` — committed audio (or linked if too large) for every quality claim made in the markdown

## Anti-patterns

- Claiming quality without samples. ("Sounds good" → folk wisdom.) Every quality assertion links to a WAV.
- Forgetting that the reference clip's quality is a ceiling on the output. Garbage in → garbage out.
