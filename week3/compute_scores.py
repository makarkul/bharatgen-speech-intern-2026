"""Week 3 scoring: WER + CER tables (Shrutam-2 vs Whisper) + a profiling table.

Reads week3_results.json (produced on the pod) and prints paste-ready markdown
tables for week3_shrutam2_eval.md. Pure CPU — no GPU needed.

Light text normalization is applied before scoring (lowercase, strip Unicode
punctuation, collapse whitespace) so we don't over-count errors on punctuation
or English casing — this matters most for the code-mixed clips.

WER/CER are aggregated at the corpus level per group (total edits / total
reference length), which is the correct way to combine clips — not an average
of per-clip rates.

Usage:
    python compute_scores.py --results week3_results.json
"""
import argparse
import json
import re
import unicodedata

import jiwer

# --- run configuration (NOT stored in the results file, so we record it here) ---
HARDWARE = "RunPod H100 80GB (SXM)"
BATCH_SIZE = 1

SETS = [("hindi", "Hindi"), ("tamil", "Tamil"), ("codemixed", "Code-mixed")]
MODEL_LABELS = {"shrutam2": "Shrutam-2", "whisper-large-v3": "Whisper large-v3"}


def normalize(text):
    """Lowercase, drop Unicode punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFC", str(text)).lower().strip()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return re.sub(r"\s+", " ", text).strip()


def score_group(rows):
    refs = [normalize(r["reference"]) for r in rows]
    hyps = [normalize(r["prediction"]) for r in rows]
    return jiwer.wer(refs, hyps), jiwer.cer(refs, hyps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="week3_results.json")
    args = ap.parse_args()
    data = json.loads(open(args.results, encoding="utf-8").read())
    models = [m for m in ("shrutam2", "whisper-large-v3") if m in data]

    # ---- WER / CER table ----
    print("\n### WER / CER by language  (lower is better)\n")
    print("| Language | " + " | ".join(
        f"{MODEL_LABELS[m]} WER | {MODEL_LABELS[m]} CER" for m in models) + " |")
    print("|" + "---|" * (1 + 2 * len(models)))
    for key, label in SETS:
        cells = [("%.2f | %.2f" % score_group([r for r in data[m] if r["set"] == key]))
                 for m in models]
        print(f"| {label} | " + " | ".join(cells) + " |")
    cells = [("%.2f | %.2f" % score_group(data[m])) for m in models]
    print(f"| **Overall** | " + " | ".join(cells) + " |")

    # ---- profiling table ----
    print(f"\n### Profiling  (hardware: {HARDWARE} · batch size: {BATCH_SIZE})\n")
    print("| Model | Avg latency/clip (s) | Peak GPU mem (GB) | Throughput (audio-s / inference-s) |")
    print("|---|---|---|---|")
    for m in models:
        rows = data[m]
        tot_lat = sum(r["latency_s"] for r in rows)
        tot_aud = sum(r["audio_duration_s"] for r in rows)
        peak_gb = max(r["peak_gpu_mem_bytes"] for r in rows) / 1e9
        print(f"| {MODEL_LABELS[m]} | {tot_lat/len(rows):.2f} | {peak_gb:.1f} | {tot_aud/tot_lat:.1f} |")

    # ---- worst Shrutam-2 clips, to seed the 5-failure analysis ----
    if "shrutam2" in data:
        print("\n### Shrutam-2 — worst clips by WER  (candidates for your 5 failures)\n")
        scored = sorted(
            ((jiwer.wer(normalize(r["reference"]), normalize(r["prediction"])), r)
             for r in data["shrutam2"]),
            key=lambda x: -x[0])
        for w, r in scored[:6]:
            print(f"- {r['clip_id']}  (WER={w:.2f})")
            print(f"    ref:  {r['reference'][:95]}")
            print(f"    pred: {r['prediction'][:95]}")


if __name__ == "__main__":
    main()
