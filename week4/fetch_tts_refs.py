#!/usr/bin/env python
"""
Fetch TTS reference clips for Week 4 (Sooktam-2 voice cloning).

Builds ~/datasets/tts_refs/<lang>/ for Hindi, Marathi, Tamil — separate from
the Week 3 eval sets, which must stay untouched (WER results depend on them).

A good voice-cloning reference clip must be three things at once, and
Shrutilipi guarantees none of them, so this script checks all three:

  1. Right length: 8-10s. Long enough to clone a voice from, short enough
     for the Week 4 README's reference-clip limit.
  2. Different speakers: Shrutilipi has no speaker IDs, and consecutive
     clips come from the same news broadcast (= same anchor). Each clip is
     fingerprinted with ECAPA-TDNN (speechbrain), a speaker-verification
     model: same speaker scores cosine similarity ~0.6+, different ~0.3.
     This is checked WITHIN each clip too — chunks of one clip are compared
     to each other to reject clips where a second person starts talking
     (a whole-clip fingerprint alone blends both voices and misses this).
  3. Clean audio: scored with torchaudio SQUIM, which estimates speech
     quality without a clean reference. PESQ (1.0-4.5) is the ranking key;
     STOI (0-1, intelligibility) and SI-SDR (dB, noise) are reported too.

Per language: collect a pool of N_POOL candidates that pass the length
filter (spaced CAND_SPACING rows apart to hop between broadcasts), then
sort by PESQ and greedily keep the best N_CLIPS whose fingerprints are all
mutually below SAME_SPEAKER_THRESH. Noisy clips lose the ranking and are
never saved — the output needs no manual quality pass.

Output per language:
  ~/datasets/tts_refs/<lang>/clips/clip_001.wav ...
  ~/datasets/tts_refs/<lang>/transcripts.tsv   (filename, transcript, duration)
  plus a printed quality + pairwise-similarity report.

Needs: `hf auth login` + accepted Shrutilipi terms on the HF website.
"""

import os
import sys
from pathlib import Path

import soundfile as sf
import torch
from datasets import load_dataset

OUT_ROOT = Path.home() / "datasets" / "tts_refs"
LANGS = ["hindi", "marathi", "tamil"]
MIN_DURATION = 8.0         # seconds — enough voice to clone from
MAX_DURATION = 10.0        # Week 4 README: reference clips are 3-10s
N_CLIPS = 5                # distinct speakers to keep per language
N_POOL = 15                # candidates to score before picking the best 5
CAND_SPACING = 200         # rows to skip between candidates (same broadcast = same anchor)
SAME_SPEAKER_THRESH = 0.4  # cosine sim above this = probably the same person
INTRA_CLIP_THRESH = 0.45   # 3s chunks of one clip below this = speaker change mid-clip
                           # (calibrated: single-speaker clips score 0.54+, a real
                           #  two-speaker clip scored 0.18)
MAX_SCAN = 40000           # safety cap on rows scanned per language


def load_models():
    from speechbrain.inference.speaker import EncoderClassifier
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    embedder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=Path.home() / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb",
    )
    squim = SQUIM_OBJECTIVE.get_model()
    return embedder, squim


def embed(embedder, array) -> torch.Tensor:
    wav = torch.tensor(array, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return embedder.encode_batch(wav).squeeze()


def quality(squim, array) -> dict:
    wav = torch.tensor(array, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        stoi, pesq, si_sdr = squim(wav)
    return {"pesq": pesq.item(), "stoi": stoi.item(), "si_sdr": si_sdr.item()}


def min_intra_clip_sim(embedder, array, sr) -> float:
    """Lowest voice similarity between 3s chunks of one clip.

    One person talking throughout -> all chunks sound alike (0.54+ in
    calibration). A second speaker taking over mid-clip drags this way
    down (0.18 observed)."""
    import itertools
    chunk = int(3.0 * sr)
    chunks = [array[i:i + chunk] for i in range(0, len(array), chunk)]
    if len(chunks[-1]) < 2 * sr:  # trailing remnant too short to fingerprint
        chunks = chunks[:-1]
    embs = [embed(embedder, c) for c in chunks]
    return min(
        torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        for a, b in itertools.combinations(embs, 2)
    )


def collect_pool(lang: str, embedder, squim) -> list[dict]:
    """Stream Shrutilipi and return N_POOL scored, fingerprinted candidates."""
    ds = load_dataset("ai4bharat/Shrutilipi", lang, split="train", streaming=True)
    pool = []
    scanned = 0
    next_candidate_at = 0
    for example in ds:
        scanned += 1
        if scanned > MAX_SCAN:
            print(f"  ! hit scan cap ({MAX_SCAN}) with {len(pool)}/{N_POOL} candidates")
            break
        if scanned < next_candidate_at:
            continue

        # Duration filter first — cheap, no audio decode needed.
        duration = example.get("duration")
        if duration is None or not (MIN_DURATION <= duration <= MAX_DURATION):
            continue

        # Only now decode the audio, fingerprint the voice, score the quality.
        audio = example["audio_filepath"]
        array, sr = audio["array"], audio["sampling_rate"]
        assert sr == 16000, f"expected 16 kHz, got {sr}"

        intra = min_intra_clip_sim(embedder, array, sr)
        if intra < INTRA_CLIP_THRESH:
            print(f"  [row {scanned:>6}] rejected: speaker change mid-clip (intra sim {intra:.2f})")
            next_candidate_at = scanned + CAND_SPACING
            continue

        q = quality(squim, array)
        pool.append({
            "array": array, "sr": sr,
            "text": (example.get("text") or "").strip(),
            "duration": float(duration),
            "emb": embed(embedder, array),
            **q,
        })
        print(f"  [row {scanned:>6}] candidate {len(pool):>2}: {duration:4.1f}s  "
              f"PESQ {q['pesq']:.2f}  STOI {q['stoi']:.2f}  SI-SDR {q['si_sdr']:5.1f} dB")
        if len(pool) >= N_POOL:
            break
        next_candidate_at = scanned + CAND_SPACING
    return pool


def pick_best_distinct(pool: list[dict]) -> list[dict]:
    """Best quality first; keep a clip only if it's a new voice vs all kept."""
    picked = []
    for cand in sorted(pool, key=lambda c: c["pesq"], reverse=True):
        sims = [
            torch.nn.functional.cosine_similarity(cand["emb"], p["emb"], dim=0).item()
            for p in picked
        ]
        if sims and max(sims) >= SAME_SPEAKER_THRESH:
            continue  # same voice as a better-quality clip we already kept
        picked.append(cand)
        if len(picked) >= N_CLIPS:
            break
    return picked


def save(lang: str, picked: list[dict]) -> None:
    out_dir = OUT_ROOT / lang / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, c in enumerate(picked, start=1):
        fname = f"clip_{n:03d}.wav"
        sf.write(out_dir / fname, c["array"], c["sr"])
        rows.append(f"{fname}\t{c['text']}\t{c['duration']:.2f}")
        print(f"  KEPT {fname}  {c['duration']:4.1f}s  PESQ {c['pesq']:.2f}  "
              f"STOI {c['stoi']:.2f}  SI-SDR {c['si_sdr']:5.1f} dB")
        if c["pesq"] < 2.5:
            print(f"       ^ warning: lowest-quality pick (PESQ < 2.5) — worth a listen")
    tsv_path = OUT_ROOT / lang / "transcripts.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("filename\ttranscript\tduration\n")
        f.write("\n".join(rows) + "\n")
    print(f"  -> {len(picked)} clips in {out_dir}, transcripts in {tsv_path}")

    # Pairwise voice similarity — every off-diagonal value should be < THRESH.
    print("  pairwise speaker similarity (same person would be ~0.6+):")
    print("        " + "  ".join(f"c{j+1:02d} " for j in range(len(picked))))
    for i, ci in enumerate(picked):
        vals = "  ".join(
            f"{torch.nn.functional.cosine_similarity(ci['emb'], cj['emb'], dim=0).item():.2f}"
            for cj in picked
        )
        print(f"   c{i+1:02d}  {vals}")


def main() -> None:
    embedder, squim = load_models()
    for lang in LANGS:
        print(f"\n=== {lang}: scoring {N_POOL} candidates, keeping best {N_CLIPS} distinct voices ===")
        pool = collect_pool(lang, embedder, squim)
        picked = pick_best_distinct(pool)
        if len(picked) < N_CLIPS:
            print(f"  ! only {len(picked)}/{N_CLIPS} distinct voices in the pool — raise N_POOL and rerun")
        save(lang, picked)
    print("\nDone. Clips are quality-ranked and speaker-verified — ready to use as references.")


if __name__ == "__main__":
    main()
    # torchcodec spawns decoder threads that crash on normal interpreter
    # shutdown (a benign SIGABRT during finalization). Everything is already
    # written and flushed by here, so skip the buggy finalizer with a hard exit.
    sys.stdout.flush()
    os._exit(0)
