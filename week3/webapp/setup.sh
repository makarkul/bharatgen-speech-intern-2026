#!/usr/bin/env bash
# One-time setup for the speech-to-speech backend on a RunPod GPU pod.
# Run this AFTER `git clone`-ing the bharatgen repo onto the pod, from inside
# the week3/webapp/ folder:
#
#     cd bharatgen-speech-intern-2026/week3/webapp
#     bash setup.sh
#
# It is idempotent — safe to re-run. It does NOT start the server; that's a
# separate command (see README / the end of this script).
#
# What it does:
#   1. installs Python deps (requirements.txt)
#   2. clones the two model repos (Shrutam-2 ASR, Sooktam-2 TTS)
#   3. applies the two required source patches (see comments below)
#   4. pre-downloads Sooktam-2's checkpoint + vocab
set -euo pipefail

# Where the model repos get cloned. Override by exporting MODELS_DIR first.
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
SHRUTAM_DIR="$MODELS_DIR/Shrutam-2"
SOOKTAM_DIR="$MODELS_DIR/sooktam2"

echo ">>> [1/4] Installing Python deps ..."
pip install -r requirements.txt

mkdir -p "$MODELS_DIR"

echo ">>> [2/4] Cloning model repos (skips if already present) ..."
# Shrutam-2: the whole repo is needed (custom inference_script.py + weights).
if [ ! -d "$SHRUTAM_DIR" ]; then
    git clone https://huggingface.co/bharatgenai/Shrutam-2 "$SHRUTAM_DIR"
else
    echo "    Shrutam-2 already at $SHRUTAM_DIR"
fi
# Sooktam-2: only src/ is imported, but the clone also LFS-pulls the 2.3GB ckpt.
if [ ! -d "$SOOKTAM_DIR" ]; then
    git clone https://huggingface.co/bharatgenai/sooktam2 "$SOOKTAM_DIR"
else
    echo "    Sooktam-2 already at $SOOKTAM_DIR"
fi

echo ">>> [3/4] Applying required source patches ..."
# Both patches are done in Python so they're exact and idempotent (re-running
# just sees the patch is already there and does nothing).
python3 - "$SHRUTAM_DIR" "$SOOKTAM_DIR" <<'PYEOF'
import sys
from pathlib import Path

shrutam_dir, sooktam_dir = Path(sys.argv[1]), Path(sys.argv[2])

# --- Patch A: Shrutam-2 OOM fix (mmap the 5GB checkpoint instead of loading
# it fully into system RAM). Harmless on big-RAM pods, essential on small ones.
infer = shrutam_dir / "inference_script.py"
txt = infer.read_text()
needle = 'torch.load(CKPT_PATH, map_location="cpu")'
if needle in txt and "mmap=True" not in txt:
    txt = txt.replace(needle, 'torch.load(CKPT_PATH, map_location="cpu", mmap=True)')
    infer.write_text(txt)
    print("    [A] Shrutam-2 mmap patch applied.")
elif "mmap=True" in txt:
    print("    [A] Shrutam-2 mmap patch already present.")
else:
    print("    [A] WARNING: expected torch.load line not found in inference_script.py "
          "(repo may have changed) — check manually.")

# --- Patch B: Sooktam-2 'robotic head' cut bug. The boundary search slides the
# ref mel over the WHOLE generated mel; a later passage can match better and
# collapse a 13s output to ~1s. Window the search to the first ~1s of offsets.
utils = sooktam_dir / "src" / "f5_tts" / "infer" / "utils_infer.py"
txt = utils.read_text()
old = "                best = int(torch.argmin(mse).item())"
new = (
    "                # PATCH: ref is always at the START; a similar later passage\n"
    "                # can match better and catastrophically cut (13s -> 1s).\n"
    "                max_offset = max(1, min(mse.numel(), int(1.0 * target_sample_rate / hop_length)))\n"
    "                best = int(torch.argmin(mse[:max_offset]).item())"
)
if old in txt:
    txt = txt.replace(old, new)
    utils.write_text(txt)
    print("    [B] Sooktam-2 robotic-head patch applied.")
elif "max_offset" in txt:
    print("    [B] Sooktam-2 robotic-head patch already present.")
else:
    print("    [B] WARNING: expected argmin line not found in utils_infer.py "
          "(repo may have changed) — check manually.")
PYEOF

echo ">>> [4/4] Pre-downloading Sooktam-2 checkpoint + vocab ..."
python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download
hf_hub_download("bharatgenai/sooktam2", "model_1250000.pt")
hf_hub_download("bharatgenai/sooktam2", "vocab.txt")
print("    Sooktam-2 weights cached.")
PYEOF

echo ""
echo ">>> Setup complete. Model repos in $MODELS_DIR"
echo ">>> Start the server with:"
echo "        export SHRUTAM_DIR=$SHRUTAM_DIR"
echo "        export SOOKTAM_DIR=$SOOKTAM_DIR"
echo "        uvicorn server:app --host 0.0.0.0 --port 8000"
