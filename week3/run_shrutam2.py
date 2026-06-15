"""Phase 3: run bharatgenai/Shrutam-2 on the 25 benchmark clips.

Shrutam-2 is NOT a standard transformers model. Its repo ships an
`inference_script.py` that we drive directly. Verified against the real repo
source (snapshot c0bdaba, pulled 2026-06-08):

  * The model loads at MODULE TOP LEVEL (`model, tokenizer = model_factory()`),
    so `import inference_script` loads everything ONCE. We then call its
    `inference(wav_path, prompt)` in a loop and reuse the loaded model.

  * Its config paths are RELATIVE: ckpt_path="model.pt", LLM_PATH="llm". And
    line 80 `print(inference("844...wav", ...))` is NOT guarded by __main__, so
    it runs at import using a relative wav path. => the module only works when
    the current working directory IS the repo dir. We os.chdir() into it before
    importing. (encoder.pt and the yaml are __file__-relative, so those are safe.)

  * `inference()` returns a ONE-ELEMENT LIST (tokenizer.batch_decode), e.g.
    ['transcript'] — we unwrap it so WER isn't scored against "['...']".

  * Importing runs the built-in Hindi sample once and prints it. That's
    expected — it also warms up CUDA so the first timed clip isn't cold-biased.

On an H100 the LLM (fp32) + checkpoints fit easily, so the Colab `mmap=True`
edit is unnecessary here.

Run on the pod (from anywhere — paths are resolved to absolute):
    python run_shrutam2.py \
        --data-root /workspace/datasets \
        --repo-dir  /workspace/Shrutam-2 \
        --out       /workspace/week3_results.json
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch

from eval_common import iter_clips, save_results, SHRUTAM2_PROMPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--repo-dir", required=True,
                    help="Path to the snapshot_download'd Shrutam-2 repo")
    ap.add_argument("--out", default="week3_results.json")
    args = ap.parse_args()

    # Resolve everything to ABSOLUTE before we chdir, so the eval data, the
    # output file, and the clip paths don't depend on the working directory.
    data_root = Path(args.data_root).resolve()
    repo_dir = Path(args.repo_dir).resolve()
    out_path = Path(args.out).resolve()

    if not torch.cuda.is_available():
        sys.exit("ERROR: Shrutam-2 requires a CUDA GPU (model.to('cuda') is hardcoded).")

    # Fail early and clearly if the snapshot is incomplete, rather than dying
    # deep inside the model's import with a cryptic FileNotFoundError.
    for need in ("inference_script.py", "model.pt", "llm", "844424930324970-261-f.wav"):
        if not (repo_dir / need).exists():
            sys.exit(f"ERROR: '{need}' not found under --repo-dir {repo_dir}. "
                     "Did snapshot_download finish?")

    # The two lines that make Shrutam-2's relative imports + relative file
    # paths resolve: put the repo on the import path, and make it the cwd.
    sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)

    print(">>> Importing Shrutam-2 (loads the model once; the ['...'] line below "
          "is its built-in warmup sample — expected) ...", flush=True)
    import inference_script  # noqa: E402 — must follow sys.path/chdir setup
    print(">>> Model loaded. Starting eval loop.\n", flush=True)

    rows = []
    for clip in iter_clips(data_root):
        prompt = SHRUTAM2_PROMPT[clip["lang"]]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        prediction = inference_script.inference(clip["path"], prompt)
        latency = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated()

        # inference() returns ['transcript']; unwrap to the bare string.
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0] if prediction else ""

        rows.append({
            "clip_id": clip["clip_id"],
            "set": clip["set"],
            "lang": clip["lang"],
            "prompt": prompt,
            "prediction": str(prediction).strip(),
            "reference": clip["reference"],
            "latency_s": round(latency, 4),
            "peak_gpu_mem_bytes": int(peak),
            "audio_duration_s": clip["duration"],
        })
        print(f"[shrutam2] {clip['clip_id']:<22} {latency:6.2f}s  {str(prediction)[:60]}", flush=True)

    save_results(str(out_path), "shrutam2", rows)


if __name__ == "__main__":
    main()
