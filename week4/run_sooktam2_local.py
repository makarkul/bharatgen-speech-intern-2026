#!/usr/bin/env python
"""
Run Sooktam-2 TTS locally (WSL, GTX 1650) — the simple way.

The official setup (`setup-cls.sh` / trust_remote_code) is broken on modern
Python: it pins tensorflow==2.15.0 (py3.12 wheels don't exist) and downgrades
the whole torch stack — all for a dataset-prep script inference never uses.
This runs the model directly instead:

  - code: git clone of https://huggingface.co/bharatgenai/sooktam2 at
    ~/models/sooktam2 (only `src/` is used, put on sys.path below)
  - weights + vocab: auto-downloaded once into the HF cache (~2.3 GB)
  - deps: pip install cached_path hydra-core omegaconf pydub vocos \
          torchdiffeq x_transformers jieba pypinyin indic_unified_parser

GOTCHA: ckpt_file/vocab_file must be passed explicitly — left empty, F5TTS
silently downloads the ORIGINAL English F5-TTS weights instead of Sooktam-2.

Local perf for reference: ~18s model load, ~40s per sentence (RTF ~9).
Fine for one-off samples; use Colab T4 for the grid + profiling.
"""

import sys
import time
from pathlib import Path

SOOKTAM_SRC = Path.home() / "models" / "sooktam2" / "src"
sys.path.insert(0, str(SOOKTAM_SRC))

from huggingface_hub import hf_hub_download

# ---- edit these three for each generation ----
REF_FILE = Path.home() / "datasets" / "tts_refs" / "hindi" / "clips" / "clip_001.wav"
REF_TEXT = "इस बीच कल शिमला से वीडियो कांफ्रेसिंग के माध्यम से संगठनात्मक जिला नूरपुर के अन्य पिछड़ा वर्ग मोर्चा की वर्घुअल रैली को भी मुख्यमंत्री ने संबोधित किया"
GEN_TEXT = "आज मौसम बहुत अच्छा है और बाज़ार में काफ़ी चहल-पहल देखने को मिल रही है।"
LANGUAGE = "hindi"
OUT_WAV = "sooktam_output.wav"
# ----------------------------------------------

ckpt = hf_hub_download("bharatgenai/sooktam2", "model_1250000.pt")
vocab = hf_hub_download("bharatgenai/sooktam2", "vocab.txt")

from f5_tts.api import F5TTS

t0 = time.time()
model = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab)
print(f"model loaded in {time.time() - t0:.0f}s")

t0 = time.time()
wav, sr, _ = model.infer(
    ref_file=str(REF_FILE),
    ref_text=REF_TEXT,
    gen_text=GEN_TEXT,
    tokenizer="cls",
    cls_language=LANGUAGE,
    file_wave=OUT_WAV,
)
gen_time = time.time() - t0
audio_len = len(wav) / sr
print(f"generated {audio_len:.1f}s of audio in {gen_time:.0f}s (RTF {gen_time / audio_len:.1f})")
print(f"saved: {OUT_WAV}")
