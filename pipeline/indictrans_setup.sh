#!/usr/bin/env bash
# Set up + start the IndicTrans2 translation service on :8501, in its OWN venv,
# WITHOUT touching the working Param-2 / ASR / TTS setup. Run on the pod:
#
#     bash /workspace/bharatgen-speech-intern-2026/pipeline/indictrans_setup.sh
#
# It is idempotent: re-running reuses the venv and just (re)starts the service.
# This is a PARALLEL experiment — pod_setup.sh (the main pipeline) is untouched.
#
# After it's up, point the main server at IndicTrans2 instead of Param-2 by
# restarting server.py with PARAM_URL=http://localhost:8501 (see the note at the
# end). To go back to Param-2, restart it with PARAM_URL=http://localhost:8500.
set -e

# Same persistent-volume cache as everything else, so the ~4GB download survives
# a pod stop. Must be exported BEFORE the service imports transformers.
export HF_HOME=/workspace/hf_cache

INDIC_VENV=/workspace/indictrans_venv
LOG=/workspace/indictrans_service.log
PORT=8501     # 8500 is Param-2; 8001 is squatted by RunPod's nginx — 8501 is free.

echo ">>> [1/4] Creating the IndicTrans2 venv (if missing) ..."
if [ ! -d "$INDIC_VENV" ]; then
    python3 -m venv "$INDIC_VENV"
    "$INDIC_VENV/bin/pip" install --no-cache-dir --upgrade pip

    # GPU probe — same logic as pod_setup.sh. On H100 stock torch runs, so we
    # keep it; only a too-new Blackwell (sm_120) needs the cu128 wheel. NOTE:
    # IndicTrans2 does NOT pull vocos, so the old "vocos silently upgrades torch
    # and breaks torchvision" trap can't fire here — but we still install torch
    # explicitly and FIRST so nothing downstream surprises us.
    echo ">>> [2/4] Probing GPU + installing torch ..."
    if python3 -c "import torch; assert torch.cuda.is_available(); (torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().item()" 2>/dev/null; then
        echo "    Stock torch runs on this GPU (e.g. H100) -> default wheel."
        "$INDIC_VENV/bin/pip" install --no-cache-dir torch
    else
        echo "    Stock torch can't run (new Blackwell card) -> cu128 wheel."
        "$INDIC_VENV/bin/pip" install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
    fi

    echo ">>> [3/4] Installing IndicTrans2 deps + IndicTransToolkit ..."
    # PIN transformers==4.56.2 — NOT latest. transformers 5.x removed the old
    # `transformers.tokenization_utils.PreTrainedTokenizerBase` import path that
    # IndicTransToolkit still uses -> ImportError at import time. 4.56.2 is the
    # same version the main ASR/TTS stack already runs (proven good) and is within
    # the toolkit's recommended transformers>=4.51.
    "$INDIC_VENV/bin/pip" install --no-cache-dir \
        "transformers==4.56.2" accelerate sentencepiece fastapi uvicorn pydantic
    # The toolkit ships the IndicProcessor (preprocess/postprocess) — MANDATORY,
    # the model mistranslates without it. Installed from GitHub (not on PyPI under
    # this name). Needs a C compiler for its Cython bits; pod images have gcc.
    "$INDIC_VENV/bin/pip" install --no-cache-dir \
        "git+https://github.com/VarunGumma/IndicTransToolkit.git"

    # Re-pin transformers AFTER the toolkit install: the toolkit doesn't pin
    # transformers, so its dependency resolution can quietly pull 5.x back in and
    # re-break the PreTrainedTokenizerBase import. Snap it back to 4.56.2 last.
    "$INDIC_VENV/bin/pip" install --no-cache-dir "transformers==4.56.2"

    # Fail fast if the install is somehow inconsistent — surfaces problems NOW,
    # not on the first translate request.
    "$INDIC_VENV/bin/python" -c "import torch; from transformers import AutoModelForSeq2SeqLM; from IndicTransToolkit.processor import IndicProcessor; print('    import check OK — torch', torch.__version__)" || {
        echo "    !!! IndicTrans2 venv import check FAILED — fix before starting."; exit 1; }
else
    echo "    venv already exists — skipping create (delete $INDIC_VENV to rebuild)."
fi

echo ">>> [4/4] Starting the IndicTrans2 service on :$PORT ..."
cd /workspace/bharatgen-speech-intern-2026/pipeline
fuser -k "$PORT/tcp" 2>/dev/null || true   # kill any stale instance holding the port
sleep 1
nohup "$INDIC_VENV/bin/uvicorn" indictrans_service:app --host 0.0.0.0 --port "$PORT" \
    > "$LOG" 2>&1 &
echo ">>> IndicTrans2 service starting (PID $!). It eager-loads the model at startup."
echo ""
echo ">>> Waiting for it to report loaded ..."
OK=0
for i in $(seq 1 30); do          # up to ~5 min (small model + first-time ~4GB download)
    H=$(curl -s "http://localhost:$PORT/health" 2>/dev/null || true)
    case "$H" in
        *'"loaded":true'*) OK=1; break ;;
    esac
    sleep 10
done
if [ "$OK" = 1 ]; then
    echo "    PASS — IndicTrans2 loaded on :$PORT."
else
    echo "    WARN — not loaded after waiting. Check $LOG"
fi
echo ""
echo ">>> Smoke-test it directly:"
echo "    curl -s -X POST http://localhost:$PORT/translate \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"text\":\"नमस्ते, आप कैसे हैं?\",\"src\":\"hindi\",\"tgt\":\"tamil\"}'"
echo ""
echo ">>> To make the WEB APP use IndicTrans2 instead of Param-2, restart the main"
echo ">>> server pointing at this port (this does NOT touch the Param-2 service):"
echo "    fuser -k 8000/tcp; sleep 1"
echo "    cd /workspace/bharatgen-speech-intern-2026/pipeline"
echo "    PARAM_URL=http://localhost:$PORT nohup uvicorn server:app --host 0.0.0.0 --port 8000 > /workspace/server.log 2>&1 &"
echo ">>> To switch BACK to Param-2: same command with PARAM_URL=http://localhost:8500"
