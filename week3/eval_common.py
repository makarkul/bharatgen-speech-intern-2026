"""Shared helpers for the Week 3 ASR eval (Shrutam-2 vs Whisper large-v3).

Discovers the 25 benchmark clips + their ground-truth transcripts, and defines
the single per-clip result schema. Both run_whisper.py and run_shrutam2.py
import from here so the two runs stay strictly comparable (same clips, same
order, same fields). No GPU needed — pure bookkeeping.
"""
import csv
import json
from pathlib import Path

# Each clip set: its folder under DATA_ROOT, and the language we decode it as.
# Code-mixed (Hindi-English) is decoded as HINDI: Shrutam-2 has no "Hinglish"
# mode, so we pick its base language. That mismatch is itself a likely failure
# source worth calling out in the 5-failure analysis.
CLIP_SETS = [
    {"name": "hindi",     "dir": "shrutilipi/hindi", "lang": "hindi"},
    {"name": "tamil",     "dir": "shrutilipi/tamil", "lang": "tamil"},
    {"name": "codemixed", "dir": "codemixed",        "lang": "hindi"},
]

# Whisper wants an ISO 639-1 code (or None to auto-detect).
WHISPER_LANG = {"hindi": "hi", "tamil": "ta"}

# Shrutam-2's inference() takes a natural-language prompt, not a lang code.
SHRUTAM2_PROMPT = {
    "hindi": "Transcribe speech to Hindi text.",
    "tamil": "Transcribe speech to Tamil text.",
}


def iter_clips(data_root):
    """Yield one dict per benchmark clip, in a fixed order.

    Keys: clip_id, set, lang, path, reference, duration.
    """
    data_root = Path(data_root)
    for cs in CLIP_SETS:
        tsv = data_root / cs["dir"] / "transcripts.tsv"
        with open(tsv, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                fname = row["filename"]
                yield {
                    "clip_id": f"{cs['name']}/{fname}",
                    "set": cs["name"],
                    "lang": cs["lang"],
                    "path": str(data_root / cs["dir"] / "clips" / fname),
                    "reference": row["transcript"],
                    "duration": float(row["duration"]),
                }


def save_results(path, model_name, rows):
    """Merge one model's rows into the shared results JSON (keyed by model)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[model_name] = rows
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} rows for '{model_name}' -> {path}")
