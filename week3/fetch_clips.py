#!/usr/bin/env python
"""
Fetch Week 3 benchmark clips from AI4Bharat Shrutilipi (Hindi + Tamil).

What it does:
  - Streams the gated Shrutilipi dataset (you must run `hf auth login` +
    accept the dataset terms on the website first).
  - For each language, saves the FIRST N_CLIPS clips longer than MIN_DURATION.
  - Writes a transcripts.tsv (filename, transcript, duration) = the ground
    truth used to score WER later. You do NOT need to read Tamil — these
    transcripts are human-verified by AI4Bharat and are what the model is
    graded against.

Why deterministic: streaming reads clips in the dataset's fixed stored order,
so "first N over Xs" gives the same clips on every run. No random seed needed.

Schema (confirmed by peeking at the live dataset):
    audio_filepath : Audio  -> .["array"] (samples) + .["sampling_rate"]
    text           : str    -> transcript
    duration       : float  -> length in seconds (filter on this, no decode)
    lang           : str

Output:
  ~/datasets/shrutilipi/<lang>/clips/clip_001.wav ... clip_010.wav
  ~/datasets/shrutilipi/<lang>/transcripts.tsv
"""

import os
import sys
from pathlib import Path

import soundfile as sf
from datasets import load_dataset

OUT_ROOT = Path.home() / "datasets" / "shrutilipi"
MIN_DURATION = 8.0      # seconds — long enough for meaningful WER
N_CLIPS = 10            # per language
LANGS = ["hindi", "tamil"]


def fetch_language(lang: str) -> None:
    out_dir = OUT_ROOT / lang / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = OUT_ROOT / lang / "transcripts.tsv"

    print(f"\n=== {lang}: streaming Shrutilipi (first {N_CLIPS} clips > {MIN_DURATION}s) ===")
    ds = load_dataset("ai4bharat/Shrutilipi", lang, split="train", streaming=True)

    rows = []
    scanned = 0
    for example in ds:
        scanned += 1

        # Filter on the provided duration first — cheap, no audio decode needed.
        duration = example.get("duration")
        if duration is None or duration <= MIN_DURATION:
            continue

        # Only now touch the audio column (this triggers the decode).
        audio = example["audio_filepath"]
        array, sr = audio["array"], audio["sampling_rate"]

        n = len(rows) + 1
        fname = f"clip_{n:03d}.wav"
        sf.write(out_dir / fname, array, sr)
        text = (example.get("text") or "").strip()
        rows.append((fname, text, f"{float(duration):.2f}"))
        print(f"  [{n:>2}] {fname}  {float(duration):5.1f}s  {text[:60]}")

        if len(rows) >= N_CLIPS:
            break

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("filename\ttranscript\tduration\n")
        for fname, text, dur in rows:
            f.write(f"{fname}\t{text}\t{dur}\n")

    print(f"  -> {len(rows)} clips saved to {out_dir}")
    print(f"  -> transcripts: {tsv_path}  (scanned {scanned} clips to find {len(rows)})")


if __name__ == "__main__":
    for lang in LANGS:
        fetch_language(lang)
    print("\nDone. Verify with:  ls ~/datasets/shrutilipi/*/clips/ | grep -c wav")

    # torchcodec spawns decoder threads that crash on normal interpreter
    # shutdown (a benign SIGABRT during finalization). Everything is already
    # written and flushed by here, so skip the buggy finalizer with a hard exit.
    sys.stdout.flush()
    os._exit(0)
