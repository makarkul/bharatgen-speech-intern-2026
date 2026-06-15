"""Phase 3: run openai/whisper-large-v3 on the 25 benchmark clips.

Loads the model ONCE, loops clip-by-clip (batch size 1), records prediction +
profiling (latency, peak GPU memory) per clip, and writes into the shared
week3_results.json under the "whisper-large-v3" key.

On the pod:
    python run_whisper.py --data-root /workspace/datasets

Our clips are all 8-16 s (< 30 s), so no long-form chunking is needed.
"""
import argparse
import time

import torch
from transformers import pipeline

from eval_common import iter_clips, save_results, WHISPER_LANG

MODEL = "openai/whisper-large-v3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="week3_results.json")
    args = ap.parse_args()

    on_gpu = torch.cuda.is_available()
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL,
        torch_dtype=torch.float16 if on_gpu else torch.float32,
        device=0 if on_gpu else -1,
    )

    rows = []
    for clip in iter_clips(args.data_root):
        # Force the decoding language when we know it; omit the key entirely
        # (rather than passing None) to let Whisper auto-detect otherwise.
        gen_kwargs = {"task": "transcribe"}
        lang = WHISPER_LANG.get(clip["lang"])
        if lang is not None:
            gen_kwargs["language"] = lang
        if on_gpu:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = pipe(clip["path"], generate_kwargs=gen_kwargs)
        latency = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() if on_gpu else 0

        rows.append({
            "clip_id": clip["clip_id"],
            "set": clip["set"],
            "lang": clip["lang"],
            "prediction": out["text"].strip(),
            "reference": clip["reference"],
            "latency_s": round(latency, 4),
            "peak_gpu_mem_bytes": int(peak),
            "audio_duration_s": clip["duration"],
        })
        print(f"[whisper] {clip['clip_id']:<22} {latency:6.2f}s  {out['text'][:60]}")

    save_results(args.out, "whisper-large-v3", rows)


if __name__ == "__main__":
    main()
