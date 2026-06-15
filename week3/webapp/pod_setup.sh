#!/usr/bin/env bash
# Full bring-up of the speech-to-speech backend on a fresh Blackwell GPU pod
# (RTX PRO 4500 / sm_120), OR replay after a pod restart. Run from anywhere:
#
#     bash /workspace/bharatgen-speech-intern-2026/week3/webapp/pod_setup.sh
#
# Then watch:  tail -f /workspace/server.log   (models take ~30-40s to load)
#
# It is idempotent — safe to re-run. It downloads the model weights if missing
# (e.g. on a brand-new pod where /workspace was never populated), and skips the
# download if they're already present and the correct size.
#
# This encodes every hard-won fix from the first bring-up:
#   0. Download weights cleanly: git clone, DROP the .git dirs (git-lfs keeps a
#      duplicate copy that doubles disk use), then re-fetch the two big files via
#      hf_hub_download and verify exact byte sizes (git-lfs silently truncated
#      them last time).
#   1. Blackwell (sm_120) needs cu128 torch — the template's cu124 torch has no
#      kernels for this GPU. --force-reinstall is required (plain install skips).
#   2. torchaudio 2.11 dropped built-in .load() -> we patch both models to use
#      soundfile instead (avoids the torchcodec/libnvrtc rabbit hole).
#   3. Shrutam mmap + Sooktam robotic-head patches (as before).
#   4. Extra deps Sooktam imports at module top: matplotlib, librosa, ffmpeg.
set -e

export HF_HOME=/root/hf_cache
export SHRUTAM_DIR=/workspace/models/Shrutam-2
export SOOKTAM_DIR=/workspace/models/sooktam2

# The two big weight files and their EXACT expected sizes (bytes). git clone
# truncated these last time, so we always verify the size, not just existence.
SHRUTAM_PT="$SHRUTAM_DIR/model.pt";              SHRUTAM_PT_BYTES=5159634998
SOOKTAM_PT="$SOOKTAM_DIR/model_1250000.pt";      SOOKTAM_PT_BYTES=5377795177

# Returns 0 (true) if $1 exists AND is exactly $2 bytes.
file_ok() { [ -f "$1" ] && [ "$(stat -c%s "$1" 2>/dev/null)" = "$2" ]; }

if file_ok "$SHRUTAM_PT" "$SHRUTAM_PT_BYTES" && file_ok "$SOOKTAM_PT" "$SOOKTAM_PT_BYTES"; then
    echo ">>> [0/4] Model weights already present and correct size — skipping download."
else
    echo ">>> [0/4] Downloading model weights (fresh pod or missing/truncated) ..."
    # git-lfs isn't in the pod template; needed so the repos' code (and pointers) clone.
    apt-get update -qq && apt-get install -y -qq git-lfs
    git lfs install

    mkdir -p /workspace/models
    cd /workspace/models

    # Clone the model REPOS for their code + small files. The big LFS files may
    # come down truncated, so we re-fetch them explicitly below. We DROP each
    # .git dir immediately to avoid git-lfs's duplicate copy filling the disk.
    [ -d "$SHRUTAM_DIR" ] || git clone https://huggingface.co/bharatgenai/Shrutam-2 "$SHRUTAM_DIR"
    rm -rf "$SHRUTAM_DIR/.git"
    [ -d "$SOOKTAM_DIR" ] || git clone https://huggingface.co/bharatgenai/sooktam2 "$SOOKTAM_DIR"
    rm -rf "$SOOKTAM_DIR/.git"

    # Re-fetch the two big checkpoints via hf_hub_download (integrity-checked,
    # unlike the silent git-lfs truncation), then copy into place. Idempotent:
    # only does it if the size is wrong/missing.
    python3 - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os

targets = [
    ("bharatgenai/Shrutam-2", "model.pt",          "/workspace/models/Shrutam-2/model.pt",        5159634998),
    ("bharatgenai/sooktam2",  "model_1250000.pt",  "/workspace/models/sooktam2/model_1250000.pt", 5377795177),
]
for repo, fname, dst, want in targets:
    if os.path.exists(dst) and os.path.getsize(dst) == want:
        print(f"  OK (already correct): {dst}")
        continue
    print(f"  downloading {repo}/{fname} ...")
    src = hf_hub_download(repo, fname)
    shutil.copy(src, dst)
    got = os.path.getsize(dst)
    assert got == want, f"SIZE MISMATCH {dst}: got {got}, want {want}"
    print(f"  OK: {dst} ({got} bytes)")
PY
    echo ">>> [0/4] Weights ready."
fi

echo ">>> [1/4] Installing ffmpeg (decodes the browser's webm uploads) ..."
# soundfile can't read the Opus/webm a browser records; ffmpeg transcodes it to
# WAV in /speak. Also satisfies pydub's ffmpeg lookup.
apt-get update -qq && apt-get install -y -qq ffmpeg

echo ">>> [2/4] Installing Blackwell-capable torch (cu128) + all deps ..."
# cu128 wheels carry sm_120 kernels for the RTX PRO 4500. --force-reinstall is
# mandatory: a plain install sees torch 'already satisfied' and does nothing.
pip install --no-cache-dir --force-reinstall \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir \
    fastapi uvicorn python-multipart numpy \
    transformers==4.56.2 huggingface_hub==0.36.0 cffi sympy soundfile \
    matplotlib librosa cached_path hydra-core omegaconf pydub vocos \
    torchdiffeq x_transformers jieba pypinyin indic_unified_parser

echo ">>> [3/4] Re-applying the 4 source patches ..."
python3 - <<'PY'
from pathlib import Path

# Patch A: Shrutam mmap (avoid system-RAM OOM staging the 5GB checkpoint)
f = Path("/workspace/models/Shrutam-2/inference_script.py"); t = f.read_text()
n = 'torch.load(CKPT_PATH, map_location="cpu")'
if n in t and "mmap=True" not in t:
    f.write_text(t.replace(n, 'torch.load(CKPT_PATH, map_location="cpu", mmap=True)'))
    print("  [A] Shrutam mmap applied")
else:
    print("  [A] Shrutam mmap:", "already present" if "mmap=True" in t else "WARN line not found")

# Patch: Shrutam torchaudio.load -> soundfile
t = f.read_text()
old = "    wav, sr = torchaudio.load(wav_path)"
new = ("    import soundfile as _sf, torch as _torch, numpy as _np\n"
       "    _data, sr = _sf.read(wav_path, dtype='float32')\n"
       "    if _data.ndim == 1: _data = _data[:, None]\n"
       "    wav = _torch.from_numpy(_np.ascontiguousarray(_data.T))")
if old in t:
    f.write_text(t.replace(old, new)); print("  [C] Shrutam soundfile applied")
else:
    print("  [C] Shrutam soundfile:", "already present" if "_sf.read(wav_path" in t else "WARN line not found")

# Patch B: Sooktam robotic-head (window the boundary search to first ~1s)
u = Path("/workspace/models/sooktam2/src/f5_tts/infer/utils_infer.py"); t = u.read_text()
old2 = "                best = int(torch.argmin(mse).item())"
new2 = ("                max_offset = max(1, min(mse.numel(), int(1.0 * target_sample_rate / hop_length)))\n"
        "                best = int(torch.argmin(mse[:max_offset]).item())")
if old2 in t:
    u.write_text(t.replace(old2, new2)); print("  [B] Sooktam robotic-head applied")
else:
    print("  [B] Sooktam robotic-head:", "already present" if "max_offset" in t else "WARN line not found")

# Patch: Sooktam torchaudio.load -> soundfile
t = u.read_text()
old3 = "    audio, sr = torchaudio.load(ref_audio)"
new3 = ("    import soundfile as _sf, torch as _torch, numpy as _np\n"
        "    _data, sr = _sf.read(ref_audio, dtype='float32')\n"
        "    if _data.ndim == 1: _data = _data[:, None]\n"
        "    audio = _torch.from_numpy(_np.ascontiguousarray(_data.T))")
if old3 in t:
    u.write_text(t.replace(old3, new3)); print("  [D] Sooktam soundfile applied")
else:
    print("  [D] Sooktam soundfile:", "already present" if "_sf.read(ref_audio" in t else "WARN line not found")
PY

echo ">>> [4/4] Starting the server on :8000 ..."
cd /workspace/bharatgen-speech-intern-2026/week3/webapp
# kill any previous instance on 8000
fuser -k 8000/tcp 2>/dev/null || true
sleep 1
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > /workspace/server.log 2>&1 &
echo ""
echo ">>> Server starting (PID $!). Models load in ~30-40s."
echo ">>> Watch:   tail -f /workspace/server.log"
echo ">>> Ready when you see: 'Uvicorn running on http://0.0.0.0:8000'"
