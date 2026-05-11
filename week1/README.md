# Week 1 — Audio and tensors

**Goal:** Understand what audio looks like to a model, and move tensors around without panicking.

**Primary resource:** [HF Audio Course Unit 1: Working with audio data](https://huggingface.co/learn/audio-course/chapter1/introduction) (2h) — this unit is the curriculum for the week.

## Concepts (lean)

- Sampling rate (16 kHz vs 44.1 kHz; why ASR wants 16 kHz)
- Mono vs stereo (speech models want mono)
- Waveform vs spectrogram, mel scale
- Tensor basics: `.to(device)`, `.shape`, `.dtype`, `.cpu()`, `.numpy()`

No backprop, no loss functions, no architectures.

## Hands-on

- Record 5 audio clips on your phone in different languages (Hindi, English + 3 of Marathi/Tamil/Kannada/Bengali). Save as WAV.
- Load each with both `librosa` and `torchaudio`.
- Plot waveform and mel-spectrogram side by side. Zoom in on silence vs speech regions; annotate.
- Resample 44.1 kHz → 16 kHz with `torchaudio.transforms.Resample`. Listen to both. Convince yourself why ASR models want 16 kHz.
- Time a 10000×10000 matmul on CPU vs GPU. Compare to Week 0's 1000×1000 numbers.

## Deliverable

- `week1/week1_audio_basics.ipynb` — plots + your annotations
- `week1/audio_cheatsheet.md` — 1-page note for the next intern: sampling rate, mono/stereo, resampling, mel-spectrogram

## Anti-patterns

- Reading three Stanford CS224S lectures.
- Trying to "understand" Fourier transforms before plotting one. Pattern-match what librosa produces.
