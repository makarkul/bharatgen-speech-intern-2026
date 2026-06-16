#!/usr/bin/env bash
# Full bring-up of the speech-TRANSLATION pipeline on a fresh GPU pod (built/
# tested for H100 SXM 80GB; the GPU-probe below also handles Blackwell sm_120),
# OR replay after a pod restart. Run from anywhere:
#
#     bash /workspace/bharatgen-speech-intern-2026/pipeline/pod_setup.sh
#
# Two services come up:
#   * main server :8000  — Shrutam-2 (ASR) + Sooktam-2 (TTS), transformers 4.56.2
#   * param service :8500 — Param-2 translation, transformers 4.52.3, OWN venv
# They run separately because their transformers versions clash. The main server
# calls the param service over HTTP (PARAM_URL).
#
# Then watch:  tail -f /workspace/server.log        (ASR/TTS load ~30-40s)
#              tail -f /workspace/param_service.log  (Param-2 17B load is slower)
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
#   1. The GPU is PROBED: keep stock torch if it runs (H100/Hopper does), else
#      install cu128 torch (a too-new Blackwell sm_120 card needs it).
#   2. torchaudio 2.11 dropped built-in .load() -> we patch both models to use
#      soundfile instead (avoids the torchcodec/libnvrtc rabbit hole).
#   3. Shrutam mmap + Sooktam robotic-head patches (as before).
#   4. Extra deps Sooktam imports at module top: matplotlib, librosa, ffmpeg.
set -e

# HF_HOME MUST live on the persistent volume (/workspace), NOT /root. Param-2's
# ~34GB weights download into this cache on first load — if it were on /root
# (ephemeral disk), it'd be WIPED on every pod stop and re-downloaded each time,
# defeating the whole point of the network volume. On /workspace it downloads
# ONCE and persists. (The old pod used /root to split two disks; with a single
# persistent volume, everything goes on /workspace.)
export HF_HOME=/workspace/hf_cache
export SHRUTAM_DIR=/workspace/models/Shrutam-2
export SOOKTAM_DIR=/workspace/models/sooktam2

# The two big weight files and their EXACT expected sizes (bytes). git clone
# truncated these last time, so we always verify the size, not just existence.
SHRUTAM_PT="$SHRUTAM_DIR/model.pt";              SHRUTAM_PT_BYTES=5159634998
SOOKTAM_PT="$SOOKTAM_DIR/model_1250000.pt";      SOOKTAM_PT_BYTES=5377795177

# Returns 0 (true) if $1 exists AND is exactly $2 bytes.
file_ok() { [ -f "$1" ] && [ "$(stat -c%s "$1" 2>/dev/null)" = "$2" ]; }

if file_ok "$SHRUTAM_PT" "$SHRUTAM_PT_BYTES" && file_ok "$SOOKTAM_PT" "$SOOKTAM_PT_BYTES"; then
    echo ">>> [0/7] Model weights already present and correct size — skipping download."
else
    echo ">>> [0/7] Downloading model weights (fresh pod or missing/truncated) ..."
    # git-lfs isn't in the pod template; needed so the repos' code (and pointers) clone.
    apt-get update -qq && apt-get install -y -qq git-lfs
    git lfs install
    # huggingface_hub is needed RIGHT HERE for the integrity-checked re-fetch below
    # (git-lfs silently truncates the big checkpoints). The full deps install runs
    # later in [3/7], but this step can't wait for it — install it up front.
    python3 -m pip install --no-cache-dir huggingface_hub

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
    echo ">>> [0/7] Weights ready."
fi

echo ">>> [1/7] Installing ffmpeg (decodes the browser's webm uploads) ..."
# soundfile can't read the Opus/webm a browser records; ffmpeg transcodes it to
# WAV in /speak. Also satisfies pydub's ffmpeg lookup.
apt-get update -qq && apt-get install -y -qq ffmpeg

echo ">>> [2/7] Probing the GPU to choose the right CUDA build of torch ..."
# Stock torch runs on H100/Ampere/etc but NOT on a too-new Blackwell card
# (sm_120: "no kernel image is available"). PROBE with a tiny GPU op rather than
# hardcoding a GPU list, and remember which CUDA wheel index to use. We do the
# actual (re)install AFTER the deps below — because a dep (vocos) silently
# upgrades torch from default PyPI, which would mismatch torchvision and break
# the whole ASR/TTS import ("operator torchvision::nms does not exist"). Pinning
# the matched trio LAST is the only thing that sticks.
if python3 -c "import torch; assert torch.cuda.is_available(); (torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().item()" 2>/dev/null; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    echo "    Stock torch runs on this GPU (e.g. H100) -> cu124 trio."
else
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    echo "    Stock torch can't run (new Blackwell card) -> cu128 trio."
fi

echo ">>> [3/7] Installing the rest of the Python deps, then pinning a matched torch trio ..."
pip install --no-cache-dir \
    fastapi uvicorn python-multipart numpy requests \
    transformers==4.56.2 huggingface_hub==0.36.0 cffi sympy soundfile \
    matplotlib librosa cached_path hydra-core omegaconf pydub vocos \
    torchdiffeq x_transformers jieba pypinyin indic_unified_parser

# Pin a MATCHED torch trio from the probed CUDA index, LAST, so the vocos-induced
# torch upgrade (which mismatches torchvision -> torchvision::nms error) is undone
# and nothing else can override it. --force-reinstall is required (plain install
# sees torch 'already satisfied' and skips).
echo "    pinning matched torch/torchvision/torchaudio from $TORCH_INDEX_URL ..."
pip install --no-cache-dir --force-reinstall \
    torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
# Fail fast if they still don't agree (e.g. index lacked a matched set).
python3 -c "from transformers import pipeline" || {
    echo "    !!! torch/torchvision still mismatched — check versions before proceeding."; exit 1; }

echo ">>> [4/7] Re-applying the source patches ..."
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

# Patch E: Shrutam runs an UNGUARDED demo at import bottom —
# print(inference("blindtest_250138.wav", ...)) — but that sample file isn't in
# the repo, so the import crashes. We import inference_script (not run it), and
# call inference() per request, so drop the demo line.
import re as _re
lines = f.read_text().splitlines(keepends=True)
kept = [ln for ln in lines if not ln.startswith('print(inference("blindtest_250138.wav"')]
if len(kept) != len(lines):
    f.write_text("".join(kept)); print("  [E] Shrutam import-time demo removed")
else:
    print("  [E] Shrutam import-time demo: not found (already removed or absent)")
PY

echo ">>> [5/7] Starting the MAIN server (ASR+TTS) on :8000 ..."
cd /workspace/bharatgen-speech-intern-2026/pipeline
# Param service uses :8500, NOT :8001 — RunPod's own nginx squats 8001 inside the
# container, so binding it fails. 8500 is free. It's internal (localhost) only.
export PARAM_URL=http://localhost:8500   # where the main server finds the translator
# kill any previous instance on 8000
fuser -k 8000/tcp 2>/dev/null || true
sleep 1
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > /workspace/server.log 2>&1 &
echo ">>> Main server starting (PID $!). ASR/TTS load in ~30-40s."

echo ">>> [6/7] Setting up + starting the PARAM-2 service on :8500 (own venv) ..."
# Param-2 needs transformers==4.52.3, which clashes with the main stack's 4.56.2.
# So it lives in its own venv. We create it once (idempotent) on /workspace so it
# survives restarts. The 17B weights download into HF_HOME on first model load.
PARAM_VENV=/workspace/param_venv
if [ ! -d "$PARAM_VENV" ]; then
    python3 -m venv "$PARAM_VENV"
    "$PARAM_VENV/bin/pip" install --no-cache-dir --upgrade pip
    # Match the main stack's GPU torch (cu128 for Blackwell) — reuse the probe result.
    if python3 -c "import torch; assert torch.cuda.is_available(); (torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().item()" 2>/dev/null; then
        "$PARAM_VENV/bin/pip" install --no-cache-dir torch
    else
        "$PARAM_VENV/bin/pip" install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
    fi
    "$PARAM_VENV/bin/pip" install --no-cache-dir \
        "transformers==4.52.3" accelerate fastapi uvicorn pydantic
else
    echo "    Param venv already exists — skipping create."
fi
fuser -k 8500/tcp 2>/dev/null || true
sleep 1
nohup "$PARAM_VENV/bin/uvicorn" param_service:app --host 0.0.0.0 --port 8500 \
    > /workspace/param_service.log 2>&1 &
echo ">>> Param-2 service starting (PID $!). It PRE-LOADS the 17B model at startup,"
echo "    so it takes a while before /health reports loaded=true (or fails loudly)."
echo ""
echo ">>> [7/7] Waiting for the Param-2 service to finish loading the model ..."
# Poll /health until the model loads, fails, or we give up. This turns "did it
# even come up?" from a log-reading chore into a clear PASS/FAIL line.
PARAM_OK=0
for i in $(seq 1 60); do          # up to ~10 min (17B load + first-time weight download)
    H=$(curl -s http://localhost:8500/health 2>/dev/null || true)
    case "$H" in
        *'"loaded":true'*)  PARAM_OK=1; break ;;
    esac
    sleep 10
done
if [ "$PARAM_OK" = 1 ]; then
    echo "    PASS — Param-2 loaded; translation is live."
else
    echo "    WARN — Param-2 not loaded after waiting. Check /workspace/param_service.log"
    echo "           (likely causes: OOM with all 3 models, transformers 4.52.3 clash, or"
    echo "            still downloading the 17B weights). The ASR/TTS server still runs;"
    echo "            same-language requests work, cross-language will error until this loads."
fi
echo ""
echo ">>> Watch:   tail -f /workspace/server.log /workspace/param_service.log"
echo ">>> Main ready when you see: 'Uvicorn running on http://0.0.0.0:8000'"
echo ">>> NOTE: expose port 8000 (and 8500 only if testing the translator directly)."
