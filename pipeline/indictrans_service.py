"""IndicTrans2 translation as a STANDALONE service — a DROP-IN alternative to
the Param-2 service (param_service.py), for comparing latency without touching
the working pipeline.

Why a separate service (same reasoning as Param-2)?
  IndicTrans2 has its own dependency stack (IndicTransToolkit + sentencepiece)
  and we want to A/B it against Param-2 WITHOUT disturbing the verified
  ASR/TTS/Param-2 setup. So it runs in its OWN venv on its OWN port, and the
  main server can be pointed at it by changing ONE env var:

      # talk to Param-2 (default):
      PARAM_URL=http://localhost:8500  uvicorn server:app ...
      # talk to IndicTrans2 instead:
      PARAM_URL=http://localhost:8501  uvicorn server:app ...

  The API here is BYTE-FOR-BYTE the same as param_service.py:
      POST /translate  {"text": "...", "src": "hindi", "tgt": "tamil"}
                    -> {"translation": "...", "model_seconds": <float|null>}
      GET  /health  -> {"ok": true, "loaded": <bool>}
  so server.py's translate_text() needs ZERO changes to switch backends.

Why IndicTrans2 should be much faster than Param-2:
  Param-2 is a 17B "thinking" LLM that generates up to ~1024 tokens of <think>
  reasoning before the answer. IndicTrans2 is a dedicated ~1B encoder-decoder MT
  model — text in, translation out, no reasoning. Order-of-magnitude smaller.

CRITICAL: IndicTrans2 is NOT plain HF generate. It REQUIRES IndicTransToolkit's
IndicProcessor to preprocess (script-normalize + prepend language-tag tokens)
and postprocess (script back). Skipping it gives garbage output — not optional.

Run this in ITS OWN venv (NOT the main one, NOT param_venv):
    python -m venv ~/indictrans_venv && source ~/indictrans_venv/bin/activate
    pip install torch transformers accelerate sentencepiece
    pip install git+https://github.com/VarunGumma/IndicTransToolkit.git
    export HF_HOME=/workspace/hf_cache          # persist the ~4GB download
    uvicorn indictrans_service:app --host 0.0.0.0 --port 8501

Test it directly (no main server needed):
    curl -s -X POST http://localhost:8501/translate \
      -H "Content-Type: application/json" \
      -d '{"text":"नमस्ते, आप कैसे हैं?","src":"hindi","tgt":"tamil"}'
"""
import os
import time

# Persist the IndicTrans2 download on the network volume (NOT ephemeral /root),
# same as Param-2 — otherwise a pod stop wipes it. Must be set BEFORE importing
# transformers. pod_setup.sh exports this already; setdefault covers a manual run.
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# We only need Indic -> Indic for this pipeline (Hindi -> Tamil etc.). One
# checkpoint. To trade quality for speed, swap to the distilled 320M:
#     "ai4bharat/indictrans2-indic-indic-dist-320M"
INDIC_INDIC_CKPT = "ai4bharat/indictrans2-indic-indic-1B"

# IndicTrans2 identifies languages by a "<script>_<Script>" code, NOT plain
# names. This maps the slugs server.py uses to those codes. These were verified
# correct against FLORES (incl. odia ory_Orya, assamese asm_Beng, urdu urd_Arab,
# punjabi pan_Guru) in the project notes.
LANG_CODES = {
    "english":   "eng_Latn",   # kept for completeness; this service is Indic<->Indic
    "hindi":     "hin_Deva",
    "marathi":   "mar_Deva",
    "tamil":     "tam_Taml",
    "telugu":    "tel_Telu",
    "malayalam": "mal_Mlym",
    "kannada":   "kan_Knda",
    "odia":      "ory_Orya",
    "bengali":   "ben_Beng",
    "urdu":      "urd_Arab",
    "assamese":  "asm_Beng",
    "gujarati":  "guj_Gujr",
    "punjabi":   "pan_Guru",
}

app = FastAPI(title="IndicTrans2 translation service")

_model = None
_tokenizer = None
_ip = None     # the IndicProcessor (preprocess/postprocess)


def _load():
    """Load IndicTrans2 (indic-indic) + the processor once, lazily, and cache."""
    global _model, _tokenizer, _ip
    if _model is None:
        print(f">>> Loading IndicTrans2 ({INDIC_INDIC_CKPT}) on {DEVICE} ...", flush=True)
        _ip = IndicProcessor(inference=True)
        _tokenizer = AutoTokenizer.from_pretrained(INDIC_INDIC_CKPT, trust_remote_code=True)
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            INDIC_INDIC_CKPT, trust_remote_code=True, low_cpu_mem_usage=True,
        ).to(DEVICE)
        _model.eval()
        print(">>> IndicTrans2 loaded. Ready.", flush=True)
    return _model, _tokenizer, _ip


def _translate(text: str, src: str, tgt: str) -> tuple[str, float | None]:
    """Translate src->tgt. Returns (translation, model_seconds).

    model_seconds is wall-clock time inside model.generate() (CUDA-synced), so
    the main server can split this step into model vs HTTP overhead — exactly
    like the Param-2 service reports it. None for the same-language no-op.
    """
    if src == tgt:
        return text, None
    if src not in LANG_CODES or tgt not in LANG_CODES:
        raise ValueError(f"Unsupported language pair: {src} -> {tgt}")

    model, tok, ip = _load()
    src_code, tgt_code = LANG_CODES[src], LANG_CODES[tgt]

    # 1. Preprocess: normalize + prepend the language-tag tokens (MANDATORY).
    batch = ip.preprocess_batch([text], src_lang=src_code, tgt_lang=tgt_code)
    # 2. Tokenize and move to the model's device.
    inputs = tok(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

    _t0 = time.perf_counter()
    # 3. Generate. beam search (num_beams=5) is what the official example uses.
    with torch.no_grad():
        out = model.generate(**inputs, max_length=256, num_beams=5, num_return_sequences=1)
    # CUDA kernels are async — sync before stopping the clock (no-op on CPU).
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_s = time.perf_counter() - _t0

    # 4. Decode, then postprocess (the toolkit undoes the tag tokens/normalization).
    decoded = tok.batch_decode(out, skip_special_tokens=True)
    translation = ip.postprocess_batch(decoded, lang=tgt_code)[0]
    print(f"[indictrans timing] {src}->{tgt}: {model_s:.3f}s", flush=True)
    return translation, model_s


class TranslateRequest(BaseModel):
    text: str
    src: str
    tgt: str


@app.post("/translate")
def translate_endpoint(req: TranslateRequest):
    translation, model_s = _translate(req.text, req.src, req.tgt)
    # Same response shape as param_service.py — server.py reads {translation, model_seconds}.
    return {"translation": translation, "model_seconds": model_s}


@app.get("/health")
def health():
    return {"ok": True, "loaded": _model is not None}


@app.on_event("startup")
def _eager_load():
    """Load at startup, not on first request — so the first /speak isn't slow and
    a load failure screams in the log immediately (mirrors param_service.py)."""
    try:
        _load()
    except Exception as e:
        print(f">>> !!! IndicTrans2 FAILED to load at startup: {type(e).__name__}: {e}",
              flush=True)


if __name__ == "__main__":
    # Tiny self-test (loads the model — only run where a GPU/CPU can hold it).
    for txt, s, t in [("नमस्ते, आप कैसे हैं?", "hindi", "tamil"),
                      ("मैं ठीक हूँ, धन्यवाद।", "hindi", "telugu")]:
        out, secs = _translate(txt, s, t)
        print(f"[{s}->{t}] {txt!r} -> {out!r}  ({secs:.3f}s)")
