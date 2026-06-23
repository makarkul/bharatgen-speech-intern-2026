#!/usr/bin/env bash
# Full bring-up of the speech-TRANSLATION pipeline on a fresh GPU pod (built/
# tested for H100 SXM 80GB), OR replay after a pod restart. Run from anywhere:
#
#     bash /workspace/bharatgen-speech-intern-2026/pipeline/pod_setup.sh
#
# Choose the translator with the TRANSLATOR env var (default: param):
#     TRANSLATOR=param      bash .../pod_setup.sh   # main server -> Param-2 :8500   (17B) — DEFAULT
#     TRANSLATOR=indictrans bash .../pod_setup.sh   # main server -> IndicTrans2 :8501 (fast, OPT-IN)
#
# Param-2 IS the pipeline's translator. IndicTrans2 is a SEPARATE, opt-in A/B
# experiment — it lives in its own venv/port/script and only comes up when you
# explicitly ask for it, so it never affects the default Param-2 pipeline.
#
# Services that come up:
#   * main server :8000  — Shrutam-2 (ASR) + Sooktam-2 (TTS), transformers 4.56.2
#   * the chosen translator service (Param-2 :8500 OR IndicTrans2 :8501)
# They run in SEPARATE venvs (their transformers versions clash). The main server
# calls the translator over HTTP (PARAM_URL).
#
# ─────────────────────────────────────────────────────────────────────────────
# WHY THIS VERSION EXISTS (the expensive lesson): the old script installed the
# MAIN stack with a BARE `pip install` into the pod's EPHEMERAL base python. That
# env is WIPED on every pod restart -> a ~10-min reinstall EVERY restart, and a
# dead :8000 with "uvicorn: command not found". This version puts the main stack
# in a VENV on /workspace (/workspace/main_venv), exactly like the translator
# venvs that already survive restarts. After the one-time build, a restart costs
# ~30s (relaunch only). A `.ready` sentinel + an import canary make the skip safe
# even if a build was interrupted.
# ─────────────────────────────────────────────────────────────────────────────
#
# Then watch:  tail -f /workspace/server.log         (ASR/TTS load ~30-40s)
#              tail -f /workspace/translator.log      (translator load)
#
# Idempotent — safe to re-run. Downloads weights if missing (verifies exact byte
# sizes), skips them otherwise. Self-heals a broken venv (rebuilds it).
#
# Encodes every hard-won fix; see the inline notes flagged [GOTCHA #N].
set -e

# ── Persistent locations (all on /workspace, which survives pod restarts) ──────
export HF_HOME=/workspace/hf_cache                  # [GOTCHA #9] big weights cache persists
export SHRUTAM_DIR=/workspace/models/Shrutam-2
export SOOKTAM_DIR=/workspace/models/sooktam2
MAIN_VENV=/workspace/main_venv                      # the main ASR/TTS stack (NEW: persistent)
PARAM_VENV=/workspace/param_venv                    # Param-2 translator
INDIC_VENV=/workspace/indictrans_venv               # IndicTrans2 translator (built by indictrans_setup.sh)
REPO=/workspace/bharatgen-speech-intern-2026

# Which translator the main server talks to. param (8500, the 17B model) IS the
# pipeline's translator and the default. indictrans (8501) is an opt-in A/B
# alternative. Override with TRANSLATOR=indictrans.
TRANSLATOR="${TRANSLATOR:-param}"

# [GOTCHA #11] Both translator models are GATED on HuggingFace -> need a token.
# Source it SOFTLY from /workspace/.hf_token (persists, never committed) so it's
# not hand-typed every restart. We do NOT hard-abort here — the ASR/TTS path
# needs no token; we require it only when building a translator venv below.
#   To set it once:  echo 'HF_TOKEN=hf_xxx' > /workspace/.hf_token
if [ -f /workspace/.hf_token ]; then
    set -a; . /workspace/.hf_token; set +a
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_TOKEN="$HF_TOKEN"     # some hf_hub versions read this name

# The two big weight files and their EXACT expected sizes (bytes).
SHRUTAM_PT="$SHRUTAM_DIR/model.pt";              SHRUTAM_PT_BYTES=5159634998
SOOKTAM_PT="$SOOKTAM_DIR/model_1250000.pt";      SOOKTAM_PT_BYTES=5377795177
file_ok() { [ -f "$1" ] && [ "$(stat -c%s "$1" 2>/dev/null)" = "$2" ]; }

# Kill a uvicorn instance by its FULL module:app arg. [GOTCHA: fuser not installed]
# fuser is in psmisc (ephemeral, absent after restart) — we hit "fuser: command
# not found". pkill (procps) is always present. Match the FULL "module:app" so we
# never kill the wrong service (server:app vs param_service:app vs
# indictrans_service:app). `|| true` is MANDATORY: pkill exits 1 when nothing
# matches (the normal first-run case), which would abort under `set -e`.
kill_uvicorn() { pkill -f "uvicorn $1" 2>/dev/null || true; pkill -f "/uvicorn $1" 2>/dev/null || true; }

echo ">>> ============================================================"
echo ">>> Pipeline bring-up. Translator = $TRANSLATOR"
echo ">>> ============================================================"

# ── [0/8] Model weights ───────────────────────────────────────────────────────
if file_ok "$SHRUTAM_PT" "$SHRUTAM_PT_BYTES" && file_ok "$SOOKTAM_PT" "$SOOKTAM_PT_BYTES"; then
    echo ">>> [0/8] Model weights already present and correct size — skipping download."
else
    echo ">>> [0/8] Downloading model weights (fresh pod or missing/truncated) ..."
    # [GOTCHA #7] git-lfs isn't in the pod template; clone truncates big .pt files.
    apt-get update -qq || true
    apt-get install -y -qq git-lfs || { sleep 5; apt-get install -y -qq git-lfs; }
    command -v git-lfs >/dev/null || { echo "    !!! git-lfs missing — cannot fetch weights"; exit 1; }
    git lfs install
    python3 -m pip install --no-cache-dir huggingface_hub

    mkdir -p /workspace/models
    cd /workspace/models
    [ -d "$SHRUTAM_DIR" ] || git clone https://huggingface.co/bharatgenai/Shrutam-2 "$SHRUTAM_DIR"
    rm -rf "$SHRUTAM_DIR/.git"
    [ -d "$SOOKTAM_DIR" ] || git clone https://huggingface.co/bharatgenai/sooktam2 "$SOOKTAM_DIR"
    rm -rf "$SOOKTAM_DIR/.git"

    # [GOTCHA #7] Re-fetch the two big checkpoints via hf_hub_download (integrity-
    # checked, unlike silent git-lfs truncation) and verify exact byte sizes.
    # Wrapped so a failure (e.g. a gated 401) gives a clear message, not a raw
    # traceback that aborts the whole bring-up under set -e.
    python3 - <<'PY' || { echo "    !!! weight download failed (gated repo? set HF_TOKEN in /workspace/.hf_token)"; exit 1; }
from huggingface_hub import hf_hub_download
import shutil, os
targets = [
    ("bharatgenai/Shrutam-2", "model.pt",          "/workspace/models/Shrutam-2/model.pt",        5159634998),
    ("bharatgenai/sooktam2",  "model_1250000.pt",  "/workspace/models/sooktam2/model_1250000.pt", 5377795177),
]
for repo, fname, dst, want in targets:
    if os.path.exists(dst) and os.path.getsize(dst) == want:
        print(f"  OK (already correct): {dst}"); continue
    print(f"  downloading {repo}/{fname} ...")
    src = hf_hub_download(repo, fname)
    shutil.copy(src, dst)
    got = os.path.getsize(dst)
    assert got == want, f"SIZE MISMATCH {dst}: got {got}, want {want}"
    print(f"  OK: {dst} ({got} bytes)")
PY
    echo ">>> [0/8] Weights ready."
fi

# ── [1/8] System packages (ephemeral — reinstall EVERY run) ───────────────────
# [GOTCHA #10] ffmpeg decodes the browser's webm/Opus uploads; soundfile can't.
# These live on ephemeral disk so they vanish on restart and MUST reinstall each
# run. A transient dpkg lock at boot is common -> retry once, then VERIFY (never
# blanket `|| true`, which would silently ship without ffmpeg and break /speak).
# psmisc gives fuser (not used directly now, but harmless); curl is for health polls.
echo ">>> [1/8] Installing system packages (ffmpeg, curl, psmisc) ..."
apt-get update -qq || true
apt-get install -y -qq ffmpeg curl psmisc \
    || { echo "    apt lock? retrying in 5s ..."; sleep 5; apt-get install -y -qq ffmpeg curl psmisc; }
command -v ffmpeg >/dev/null || { echo "    !!! ffmpeg missing — /speak webm decode will fail"; exit 1; }
command -v curl   >/dev/null || { echo "    !!! curl missing — health polls will give false WARNs"; exit 1; }

# ── [2/8] Pick the CUDA wheel index ───────────────────────────────────────────
# [GOTCHA #1, #14] H100/Hopper runs the cu124 trio; a too-new Blackwell (sm_120)
# needs cu128. The OLD probe ran `import torch` in BASE python — but torch now
# lives only in the venvs, so base python has none -> the probe threw -> it
# silently picked cu128 on an H100 (WRONG -> CPU fallback / "no kernel image").
# We instead read the GPU's compute capability from nvidia-smi, DEFAULT to cu124
# (the documented H100 target), and pick cu128 only on a positively-detected
# Blackwell. The `case` guard is load-bearing: a non-numeric value ([N/A] on
# MIG/virtual GPUs, or empty) would otherwise crash `[ -ge ]` under set -e.
echo ">>> [2/8] Choosing the CUDA wheel index from the GPU's compute capability ..."
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
case "$CC" in (*[!0-9]*|"") CC=0 ;; esac     # non-numeric/empty -> 0 -> cu124 default
if [ "$CC" -ge 120 ]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    echo "    compute_cap=$CC (Blackwell sm_120+) -> cu128 trio."
else
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    echo "    compute_cap=$CC -> cu124 trio (H100/Hopper default)."
fi
# Belt-and-suspenders: if a venv build below still can't see the GPU, it
# force-reinstalls cu124 (the value proven to work on this pod). See build_venv_torch.

# ── Helper: build a venv's torch correctly (probe in the VENV, not base) ──────
# Installs torch into the venv from $TORCH_INDEX_URL, then verifies the venv's
# OWN python can see the GPU; if not, force-reinstalls cu124 (the indictrans_setup
# pattern that's proven on this pod). $1 = venv path.
build_venv_torch() {
    local venv="$1"
    # [GOTCHA #15] vocos (installed earlier without --index-url) drags torchaudio in from
    # PyPI.  New PyTorch nightly torchaudio on PyPI links against libcudart.so.13 which is
    # absent on cu124 pods.  --force-reinstall guarantees we overwrite any PyPI-sourced
    # torchaudio with the correct CUDA-matched build from the index.
    "$venv/bin/pip" install --no-cache-dir --force-reinstall torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
    if ! "$venv/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "    torch can't see the GPU from $venv — forcing cu124 ..."
        "$venv/bin/pip" install --no-cache-dir --force-reinstall \
            torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu124"
    fi
}

# ── [3/8] MAIN stack venv on /workspace (the headline fix) ────────────────────
# [GOTCHA #2] vocos (a dep) silently upgrades torch from default PyPI, mismatching
# torchvision -> "operator torchvision::nms does not exist" -> the ASR/TTS import
# dies. The ONLY reliable fix: install all deps FIRST, then force-reinstall a
# MATCHED torch trio LAST. A venv doesn't change pip's resolver, so this ordering
# still holds inside the venv.
#
# A `.ready` sentinel guards the skip: it's written ONLY after a clean build, so
# an INTERRUPTED build (OOM/network — common on the multi-GB trio) doesn't leave a
# half-built venv that the skip would relaunch (which was the original "uvicorn not
# found" bug in disguise). On EVERY path we then run a strong import canary.
MAIN_READY="$MAIN_VENV/.ready"
echo ">>> [3/8] Main ASR/TTS stack venv ($MAIN_VENV) ..."
if [ -f "$MAIN_READY" ]; then
    echo "    main_venv already built (.ready present) — skipping the ~10-min install."
else
    echo "    Building main_venv (one-time ~10 min; survives future restarts) ..."
    rm -rf "$MAIN_VENV"                      # clear any half-built remnant
    python3 -m venv "$MAIN_VENV"
    "$MAIN_VENV/bin/pip" install --no-cache-dir --upgrade pip
    # All deps EXCEPT the torch trio (vocos will drag in a mismatched torch here).
    "$MAIN_VENV/bin/pip" install --no-cache-dir \
        fastapi uvicorn python-multipart numpy requests \
        transformers==4.56.2 huggingface_hub==0.36.0 cffi sympy soundfile \
        matplotlib librosa cached_path hydra-core omegaconf pydub vocos \
        torchdiffeq x_transformers jieba pypinyin indic_unified_parser \
        || { echo "    !!! main_venv dep install failed — rerun to retry (no .ready written)"; exit 1; }
    # Matched torch trio LAST, into the venv, probed in the venv python.
    build_venv_torch "$MAIN_VENV"
    touch "$MAIN_READY"                      # mark clean ONLY after the trio is in
    echo "    main_venv built."
fi
# [GOTCHA #15 self-heal] torchaudio may be a PyPI nightly (e.g. 2.11.0) that links
# against libcudart.so.13, which is absent on cu124 pods.  Detect and fix in-place
# before running the canary — much faster than a full venv rebuild.
if ! "$MAIN_VENV/bin/python" -c "import torchaudio" 2>/dev/null; then
    echo "    torchaudio import failed (wrong CUDA build from PyPI) — force-reinstalling from $TORCH_INDEX_URL ..."
    "$MAIN_VENV/bin/pip" install --no-cache-dir --force-reinstall \
        torchaudio --index-url "$TORCH_INDEX_URL" \
        || { echo "    !!! torchaudio reinstall failed"; rm -f "$MAIN_READY"; exit 1; }
fi

# [GOTCHA #2] Import canary on EVERY path (build AND skip). `from transformers
# import pipeline` ALONE has historically passed even when torchvision::nms is
# unregistered (it resolves lazily), so we also `import torchvision.ops` to force
# the op to register here, where we can fail fast and self-heal, instead of at the
# server's import (a confusing downstream crash). torchaudio is also tested so a
# wrong-CUDA build is caught here rather than at uvicorn startup. If it fails, drop
# .ready so the next run rebuilds.
"$MAIN_VENV/bin/python" -c "from transformers import pipeline; import torchvision.ops; import torchaudio; import torch; assert torch.cuda.is_available()" \
    || { echo "    !!! main_venv broken (trio mismatch / no GPU). Removing .ready — rerun to rebuild."; rm -f "$MAIN_READY"; exit 1; }
echo "    main_venv import canary OK."

# ── [4/8] Source patches (idempotent; edit persistent /workspace/models/*) ────
# [GOTCHA #3,#4,#5,#6] These run in BASE python (pure pathlib string-replace, no
# torch) and persist on /workspace. The torchaudio.load->soundfile patches (C,D)
# are LOAD-BEARING: without them the server crashes at import. So if a critical
# patch's target line is missing AND its result isn't already present, we ABORT
# loudly (the repo layout changed) instead of letting it degrade to a mystery
# crash at server start.
echo ">>> [4/8] Re-applying source patches ..."
python3 - <<'PY' || { echo "    !!! a load-bearing patch failed — see message above"; exit 1; }
from pathlib import Path
import sys

fail = []

# Patch A: Shrutam mmap (avoid system-RAM OOM staging the 5GB checkpoint)
# Variable is ckpt_path (lowercase) in the actual file — not CKPT_PATH.
f = Path("/workspace/models/Shrutam-2/inference_script.py"); t = f.read_text()
n = 'torch.load(ckpt_path, map_location="cpu")'
if n in t and "mmap=True" not in t:
    f.write_text(t.replace(n, 'torch.load(ckpt_path, map_location="cpu", mmap=True)'))
    print("  [A] Shrutam mmap applied")
else:
    print("  [A] Shrutam mmap:", "already present" if "mmap=True" in t else "WARN line not found")

# Patch C: Shrutam torchaudio.load -> soundfile  (LOAD-BEARING)
t = f.read_text()
old = "    wav, sr = torchaudio.load(wav_path)"
new = ("    import soundfile as _sf, torch as _torch, numpy as _np\n"
       "    _data, sr = _sf.read(wav_path, dtype='float32')\n"
       "    if _data.ndim == 1: _data = _data[:, None]\n"
       "    wav = _torch.from_numpy(_np.ascontiguousarray(_data.T))")
if old in t:
    f.write_text(t.replace(old, new)); print("  [C] Shrutam soundfile applied")
elif "_sf.read(wav_path" in t:
    print("  [C] Shrutam soundfile: already present")
else:
    fail.append("C (Shrutam soundfile): neither target line nor result found — repo layout changed")

# Patch B: Sooktam robotic-head (window the boundary search to first ~1s)
u = Path("/workspace/models/sooktam2/src/f5_tts/infer/utils_infer.py"); t = u.read_text()
old2 = "                best = int(torch.argmin(mse).item())"
new2 = ("                max_offset = max(1, min(mse.numel(), int(1.0 * target_sample_rate / hop_length)))\n"
        "                best = int(torch.argmin(mse[:max_offset]).item())")
if old2 in t:
    u.write_text(t.replace(old2, new2)); print("  [B] Sooktam robotic-head applied")
else:
    print("  [B] Sooktam robotic-head:", "already present" if "max_offset" in t else "WARN line not found")

# Patch D: Sooktam torchaudio.load -> soundfile  (LOAD-BEARING)
t = u.read_text()
old3 = "    audio, sr = torchaudio.load(ref_audio)"
new3 = ("    import soundfile as _sf, torch as _torch, numpy as _np\n"
        "    _data, sr = _sf.read(ref_audio, dtype='float32')\n"
        "    if _data.ndim == 1: _data = _data[:, None]\n"
        "    audio = _torch.from_numpy(_np.ascontiguousarray(_data.T))")
if old3 in t:
    u.write_text(t.replace(old3, new3)); print("  [D] Sooktam soundfile applied")
elif "_sf.read(ref_audio" in t:
    print("  [D] Sooktam soundfile: already present")
else:
    fail.append("D (Sooktam soundfile): neither target line nor result found — repo layout changed")

# Patch E: strip Shrutam's unguarded import-time demo line
lines = f.read_text().splitlines(keepends=True)
kept = [ln for ln in lines if not ln.startswith('print(inference("blindtest_250138.wav"')]
if len(kept) != len(lines):
    f.write_text("".join(kept)); print("  [E] Shrutam import-time demo removed")
else:
    print("  [E] Shrutam import-time demo: not found (already removed or absent)")

if fail:
    print("LOAD-BEARING PATCH FAILURE:"); [print("   -", x) for x in fail]
    sys.exit(1)
PY

# ── [5/8] Start the MAIN server (:8000) from main_venv ────────────────────────
echo ">>> [5/8] Starting the MAIN server (ASR+TTS) on :8000 from main_venv ..."
cd "$REPO/pipeline"
# Point the main server at the chosen translator. [GOTCHA #8] 8500=Param-2,
# 8501=IndicTrans2 (8001 is squatted by RunPod's nginx — never use it).
if [ "$TRANSLATOR" = "param" ]; then
    export PARAM_URL=http://localhost:8500
else
    export PARAM_URL=http://localhost:8501
fi
echo "    Main server -> translator at $PARAM_URL"
kill_uvicorn "server:app"
sleep 1
nohup "$MAIN_VENV/bin/uvicorn" server:app --host 0.0.0.0 --port 8000 > /workspace/server.log 2>&1 &
MAIN_PID=$!
echo "    Main server starting (PID $MAIN_PID). ASR/TTS load in ~30-40s."

# ── [6/8] Build (if needed) + start the chosen TRANSLATOR ─────────────────────
if [ "$TRANSLATOR" = "param" ]; then
    echo ">>> [6/8] Param-2 translator on :8500 (venv $PARAM_VENV) ..."
    : "${HF_TOKEN:?Param-2 is gated — run: echo 'HF_TOKEN=hf_xxx' > /workspace/.hf_token}"
    PARAM_READY="$PARAM_VENV/.ready"
    if [ ! -f "$PARAM_READY" ]; then
        echo "    Building param_venv (one-time) ..."
        rm -rf "$PARAM_VENV"
        python3 -m venv "$PARAM_VENV"
        "$PARAM_VENV/bin/pip" install --no-cache-dir --upgrade pip
        build_venv_torch "$PARAM_VENV"      # [GOTCHA #14] correct CUDA wheel, probed in-venv
        "$PARAM_VENV/bin/pip" install --no-cache-dir \
            "transformers==4.52.3" accelerate fastapi uvicorn pydantic \
            || { echo "    !!! param_venv install failed — rerun to retry"; exit 1; }
        touch "$PARAM_READY"
    fi
    "$PARAM_VENV/bin/python" -c "import torch; from transformers import AutoModelForCausalLM; assert torch.cuda.is_available()" \
        || { echo "    !!! param_venv broken — removing .ready, rerun to rebuild"; rm -f "$PARAM_READY"; exit 1; }
    kill_uvicorn "param_service:app"
    sleep 1
    nohup "$PARAM_VENV/bin/uvicorn" param_service:app --host 0.0.0.0 --port 8500 \
        > /workspace/translator.log 2>&1 &
    echo "    Param-2 service starting (PID $!) — pre-loads the 17B model (slow)."
    TRANSLATOR_PORT=8500
else
    echo ">>> [6/8] IndicTrans2 translator on :8501 (venv $INDIC_VENV) ..."
    # NOTE: ai4bharat/indictrans2-indic-indic-1B is a PUBLIC model — no HF token
    # needed (indictrans_service.py loads it tokenless). So, unlike the Param-2
    # branch, we do NOT guard on HF_TOKEN here.
    if [ ! -f "$INDIC_VENV/.ready" ] && [ ! -x "$INDIC_VENV/bin/uvicorn" ]; then
        # Delegate the full build to the dedicated, already-correct script.
        echo "    indictrans_venv not built — running indictrans_setup.sh ..."
        bash "$REPO/pipeline/indictrans_setup.sh"
        # indictrans_setup.sh starts the service itself; mark and move on.
        TRANSLATOR_PORT=8501
    else
        "$INDIC_VENV/bin/python" -c "import torch; from IndicTransToolkit.processor import IndicProcessor; assert torch.cuda.is_available()" 2>/dev/null \
            || { echo "    !!! indictrans_venv broken — run: rm -rf $INDIC_VENV && bash pipeline/indictrans_setup.sh"; exit 1; }
        kill_uvicorn "indictrans_service:app"
        sleep 1
        nohup "$INDIC_VENV/bin/uvicorn" indictrans_service:app --host 0.0.0.0 --port 8501 \
            > /workspace/translator.log 2>&1 &
        echo "    IndicTrans2 service starting (PID $!)."
        TRANSLATOR_PORT=8501
    fi
fi

# ── [7/8] Wait for the MAIN server to be ready ────────────────────────────────
# [GOTCHA: no /health endpoint] server.py exposes only POST /speak and GET / .
# Models load at import, so uvicorn doesn't answer GET / until ASR+TTS finish —
# a 200 on / IS the readiness signal. We also `kill -0 $MAIN_PID` so a crash-on-
# import is caught in seconds, not a 5-min poll. The OLD script never checked the
# main server at all — a dead :8000 looked "successful".
echo ">>> [7/8] Waiting for the MAIN server (:8000) to be ready ..."
MAIN_OK=0
for i in $(seq 1 30); do          # ~5 min
    if ! kill -0 "$MAIN_PID" 2>/dev/null; then
        echo "    !!! Main server process died during load — see /workspace/server.log"; break
    fi
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://localhost:8000/ 2>/dev/null || true)
    if [ "$code" = "200" ]; then MAIN_OK=1; break; fi
    sleep 10
done
if [ "$MAIN_OK" = 1 ]; then echo "    PASS — main server live on :8000."
else echo "    WARN — main server not ready. Check /workspace/server.log"; fi

# ── [8/8] Wait for the TRANSLATOR to report loaded ────────────────────────────
echo ">>> [8/8] Waiting for the translator (:$TRANSLATOR_PORT) to load ..."
T_OK=0
for i in $(seq 1 60); do          # ~10 min (Param-2 17B load / first-time weight download)
    H=$(curl -s "http://localhost:$TRANSLATOR_PORT/health" 2>/dev/null || true)
    case "$H" in (*'"loaded":true'*) T_OK=1; break ;; esac
    sleep 10
done
if [ "$T_OK" = 1 ]; then echo "    PASS — translator loaded on :$TRANSLATOR_PORT."
else echo "    WARN — translator not loaded. Check /workspace/translator.log"; fi

echo ""
echo ">>> ============================================================"
echo ">>> Done. Translator=$TRANSLATOR (port $TRANSLATOR_PORT)."
echo ">>> Watch:  tail -f /workspace/server.log /workspace/translator.log"
echo ">>> Default translator is Param-2. Opt into the A/B alternative with TRANSLATOR=indictrans."
echo ">>> Expose port 8000 in the RunPod UI to reach the web app."
echo ">>> ============================================================"
