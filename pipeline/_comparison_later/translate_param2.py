"""Param-2 translation — the BharatGen LLM alternative to IndicTrans2.

This is the SAME interface as translate.py — `translate(text, src, tgt)` — but
backed by BharatGen's Param2-17B-A2.4B-Thinking instead of IndicTrans2. That
lets compare.py swap one for the other and translate identical sentences.

Honest caveats baked into this code:
  * Param-2 is a general chat LLM. Translation is NOT a listed feature — we
    PROMPT it to translate. So we must instruct it firmly ("output ONLY the
    translation, nothing else") or it adds commentary.
  * It's a "THINKING" model: it emits a <think>...</think> reasoning block
    before the answer. We decode WITH special tokens so we can see and then
    STRIP that block, keeping only the final translation.
  * We use do_sample=False — the model card's own advice for "stable,
    repeatable, reliable" output, which is what translation wants.

Needs transformers==4.52.3 (per the model card) and a big GPU (H100). The
17B weights are far too large for a 3090 alongside Shrutam + Sooktam.

Install:
    pip install "transformers==4.52.3" torch accelerate

Test on its own:
    python translate_param2.py
"""
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "bharatgenai/Param2-17B-A2.4B-Thinking"

# Human-readable language names for the prompt (an LLM wants names, not codes).
LANG_NAMES = {
    "english": "English", "hindi": "Hindi", "marathi": "Marathi",
    "tamil": "Tamil", "telugu": "Telugu", "malayalam": "Malayalam",
    "kannada": "Kannada", "odia": "Odia", "bengali": "Bengali",
    "urdu": "Urdu", "assamese": "Assamese", "gujarati": "Gujarati",
    "punjabi": "Punjabi",
}

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
    return _model, _tokenizer


def _strip_thinking(text: str) -> str:
    """Remove the <think>...</think> reasoning block, keep only the answer."""
    # Drop a complete <think>...</think> block, or anything up to a closing </think>.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate `text` from src_lang to tgt_lang (slugs like 'hindi').

    Same signature as translate.py so the two are drop-in interchangeable.
    """
    if src_lang == tgt_lang:
        return text
    if src_lang not in LANG_NAMES or tgt_lang not in LANG_NAMES:
        raise ValueError(f"Unsupported language pair: {src_lang} -> {tgt_lang}")

    model, tok = _load()
    src_name, tgt_name = LANG_NAMES[src_lang], LANG_NAMES[tgt_lang]

    # Firm instruction so the LLM translates and nothing else. The thinking model
    # will still reason in <think>; we strip that afterward.
    conversation = [
        {"role": "system",
         "content": "You are a precise translator. Output ONLY the translated "
                    "text, with no explanations, notes, or quotation marks."},
        {"role": "user",
         "content": f"Translate the following {src_name} text into {tgt_name}:\n\n{text}"},
    ]
    inputs = tok.apply_chat_template(
        conversation=conversation, return_tensors="pt", add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=300,
            do_sample=False,          # deterministic — the card's advice for reliable output
            use_cache=False,
        )
    # Decode ONLY the newly generated tokens, keeping special tokens so we can
    # see and strip the <think> block.
    generated = out[0][inputs.shape[-1]:]
    raw = tok.decode(generated, skip_special_tokens=False)
    return _strip_thinking(raw)


if __name__ == "__main__":
    tests = [
        ("Hello, how are you?", "english", "hindi"),
        ("मैं ठीक हूँ, धन्यवाद।", "hindi", "english"),
        ("मैं ठीक हूँ, धन्यवाद।", "hindi", "tamil"),
    ]
    for txt, s, t in tests:
        print(f"[{s} -> {t}] {txt!r}\n    -> {translate(txt, s, t)!r}\n")
