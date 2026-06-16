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
    """Remove the <think>...</think> reasoning block, keep only the answer."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
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
            inputs, max_new_tokens=300, do_sample=False, use_cache=False,
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


if __name__ == "__main__":
    # Tiny self-test (loads the model — only run on the GPU pod).
    for txt, s, t in [("Hello, how are you?", "english", "hindi"),
                      ("मैं ठीक हूँ।", "hindi", "tamil")]:
        print(f"[{s}->{t}] {txt!r} -> {_translate(txt, s, t)!r}")
