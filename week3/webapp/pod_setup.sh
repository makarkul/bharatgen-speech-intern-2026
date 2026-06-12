#!/usr/bin/env bash
# Replay the pod environment after a restart (or first-time setup on a fresh
# Blackwell pod). Model weights are expected to already live on /workspace
# (they survive a pod restart). Run from anywhere:
#
#     bash /workspace/bharatgen-speech-intern-2026/week3/webapp/pod_setup.sh
#
# Then watch:  tail -f /workspace/server.log   (models take ~30-40s to load)
#
# This encodes every hard-won fix from the first bring-up:
#   1. Blackwell (sm_120) needs cu128 torch — the template's cu124 torch has no
#      kernels for this GPU. --force-reinstall is required (plain install skips).
#   2. torchaudio 2.11 dropped built-in .load() -> we patch both models to use
#      soundfile instead (avoids the torchcodec/libnvrtc rabbit hole).
#   3. Shrutam mmap + Sooktam robotic-head patches (as before).
#   4. Extra deps Sooktam imports at module top: matplotlib, librosa.
set -e

export HF_HOME=/root/hf_cache
export SHRUTAM_DIR=/workspace/models/Shrutam-2
export SOOKTAM_DIR=/workspace/models/sooktam2

MODELS_OK=1
[ -f "$SHRUTAM_DIR/model.pt" ] || { echo "MISSING $SHRUTAM_DIR/model.pt"; MODELS_OK=0; }
[ -f "$SOOKTAM_DIR/model_1250000.pt" ] || { echo "MISSING $SOOKTAM_DIR/model_1250000.pt"; MODELS_OK=0; }
if [ "$MODELS_OK" = 0 ]; then
    echo "ERROR: model weights not found on /workspace. They should have survived the restart."
    echo "If the volume was wiped, re-download with hf_hub_download (see project notes)."
    exit 1
fi

echo ">>> [0/3] Installing ffmpeg (decodes the browser's webm uploads) ..."
# soundfile can't read the Opus/webm a browser records; ffmpeg transcodes it to
# WAV in /speak. Also satisfies pydub's ffmpeg lookup.
apt-get update -qq && apt-get install -y -qq ffmpeg

echo ">>> [1/3] Installing Blackwell-capable torch (cu128) + all deps ..."
# cu128 wheels carry sm_120 kernels for the RTX PRO 4500. --force-reinstall is
# mandatory: a plain install sees torch 'already satisfied' and does nothing.
pip install --no-cache-dir --force-reinstall \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir \
    fastapi uvicorn python-multipart numpy \
    transformers==4.56.2 huggingface_hub==0.36.0 cffi sympy soundfile \
    matplotlib librosa cached_path hydra-core omegaconf pydub vocos \
    torchdiffeq x_transformers jieba pypinyin indic_unified_parser

echo ">>> [2/3] Re-applying the 4 source patches ..."
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

echo ">>> [3/3] Starting the server on :8000 ..."
cd /workspace/bharatgen-speech-intern-2026/week3/webapp
# kill any previous instance on 8000
fuser -k 8000/tcp 2>/dev/null || true
sleep 1
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > /workspace/server.log 2>&1 &
echo ""
echo ">>> Server starting (PID $!). Models load in ~30-40s."
echo ">>> Watch:   tail -f /workspace/server.log"
echo ">>> Ready when you see: 'Uvicorn running on http://0.0.0.0:8000'"
