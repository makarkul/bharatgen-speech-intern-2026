"""Param-2 translation as a STANDALONE service (its own process + venv).

Why separate? Param-2 needs transformers==4.52.3, but the main server's
Shrutam-2 + Sooktam-2 stack needs transformers==4.56.2. Those can't coexist in
one process, so Param-2 runs here on its own and the main server talks to it
over HTTP. The verified ASR/TTS stack stays completely untouched.

    main server (server.py)  --HTTP-->  this service  --> Param-2

Run this in ITS OWN venv (NOT the main one):
    python -m venv ~/param_venv && source ~/param_venv/bin/activate
    pip install "transformers==4.52.3" torch accelerate fastapi uvicorn
    uvicorn param_service:app --host 0.0.0.0 --port 8001

Then the main server reaches it at http://localhost:8001/translate.

Endpoint:
    POST /translate  {"text": "...", "src": "hindi", "tgt": "tamil"}
                  -> {"translation": "..."}
    GET  /health  -> {"ok": true, "loaded": <bool>}
"""
import re

import os

# Safety net: if HF_HOME isn't already set (e.g. running this standalone for
# debugging, not via pod_setup.sh), default the ~34GB Param-2 download to the
# PERSISTENT volume so it isn't wiped on pod stop. Must be set BEFORE importing
# transformers. pod_setup.sh sets HF_HOME=/workspace/hf_cache and we inherit it.
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "bharatgenai/Param2-17B-A2.4B-Thinking"

# Human-readable names for the prompt (an LLM wants names, not script codes).
LANG_NAMES = {
    "english": "English", "hindi": "Hindi", "marathi": "Marathi",
    "tamil": "Tamil", "telugu": "Telugu", "malayalam": "Malayalam",
    "kannada": "Kannada", "odia": "Odia", "bengali": "Bengali",
    "urdu": "Urdu", "assamese": "Assamese", "gujarati": "Gujarati",
    "punjabi": "Punjabi",
}

app = FastAPI(title="Param-2 translation service")

_model = None
_tokenizer = None


def _load():
    """Load Param-2 once (lazily) and cache it. The 17B load is slow."""
    global _model, _tokenizer
    if _model is None:
        print(f">>> Loading {MODEL_NAME} (17B — slow, needs a big GPU) ...", flush=True)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        _model.eval()
        print(">>> Param-2 loaded. Ready.", flush=True)
    return _model, _tokenizer


def _strip_thinking(text: str) -> str:
    """Remove the <think>...</think> reasoning block, keep only the answer.

    Handles: a complete <think>...</think> block; leading reasoning up to a
    stray closing </think>; AND an UNCLOSED/dangling <think> (generation got
    truncated mid-reasoning) — in that last case we drop from <think> to the end
    so we never return raw reasoning as if it were the translation.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # complete block
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)         # leading reasoning
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)           # dangling, unclosed
    # Strip leftover chat/end special tokens (e.g. <|im_end|>, <<|EOS|>>,
    # <|endoftext|>) — we decode with skip_special_tokens=False to see <think>,
    # so these survive and would otherwise be SPOKEN by the TTS.
    text = re.sub(r"<+\|[^<>]*\|>+", "", text)
    return text.strip()


def _translate(text: str, src: str, tgt: str) -> str:
    if src == tgt:
        return text
    if src not in LANG_NAMES or tgt not in LANG_NAMES:
        raise ValueError(f"Unsupported language pair: {src} -> {tgt}")

    model, tok = _load()
    conversation = [
        {"role": "system",
         "content": "You are a precise translator. Output ONLY the translated "
                    "text, with no explanations, notes, or quotation marks."},
        {"role": "user",
         "content": f"Translate the following {LANG_NAMES[src]} text into "
                    f"{LANG_NAMES[tgt]}:\n\n{text}"},
    ]
    inputs = tok.apply_chat_template(
        conversation=conversation, return_tensors="pt", add_generation_prompt=True,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs,
            # A "thinking" model spends tokens on <think> reasoning BEFORE the
            # answer; 300 was too tight (could truncate mid-think). 1024 leaves
            # room for reasoning + the translation.
            max_new_tokens=1024,
            do_sample=False,        # deterministic — the card's advice for reliable output
            # KV cache ON for usable latency. With cache OFF (as the card's
            # example showed) a 17B generating ~1k tokens recomputes attention
            # every step — minutes per call, and it'd blow the server's 180s
            # timeout. If Param-2's custom modeling ever errors with cache on,
            # revert this to use_cache=False.
            use_cache=True,
        )
    generated = out[0][inputs.shape[-1]:]
    raw = tok.decode(generated, skip_special_tokens=False)
    return _strip_thinking(raw)


class TranslateRequest(BaseModel):
    text: str
    src: str
    tgt: str


@app.post("/translate")
def translate_endpoint(req: TranslateRequest):
    return {"translation": _translate(req.text, req.src, req.tgt)}


@app.get("/health")
def health():
    return {"ok": True, "loaded": _model is not None}


@app.on_event("startup")
def _eager_load():
    """Load Param-2 at startup, NOT on the first request.

    Two reasons: (1) the first /speak would otherwise hang ~a minute while the
    17B model loads, with no feedback; (2) if loading fails (OOM, version clash,
    bad weights) we want it screaming in param_service.log at startup — not
    silently on the first user request. If it can't load, log loudly and let the
    service stay up so /health reports loaded=false.
    """
    try:
        _load()
    except Exception as e:
        print(f">>> !!! Param-2 FAILED to load at startup: {type(e).__name__}: {e}",
              flush=True)


if __name__ == "__main__":
    # Tiny self-test (loads the model — only run on the GPU pod).
    for txt, s, t in [("Hello, how are you?", "english", "hindi"),
                      ("मैं ठीक हूँ।", "hindi", "tamil")]:
        print(f"[{s}->{t}] {txt!r} -> {_translate(txt, s, t)!r}")
