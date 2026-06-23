"""IndicTrans2 translation — the middle step of the speech-to-speech pipeline.

The pipeline today is:   speech --(Shrutam-2)--> text --(Sooktam-2)--> speech
all in ONE language. Adding translation makes it a real translator:

    speech in X  --(Shrutam-2)-->  text in X
                 --(IndicTrans2)-> text in Y       <-- THIS FILE
                 --(Sooktam-2)-->  speech in Y

Why IndicTrans2 (and not BharatGen's Param-1/Param-2)?
  * Param-1 is Hindi<->English ONLY — can't do the 12 languages Shrutam handles.
  * Param-2 covers 22 languages but is a 17B chat LLM that doesn't even LIST
    translation as a feature, and would hog the GPU next to Shrutam + Sooktam.
  * IndicTrans2 is a DEDICATED translation model: tiny (distilled 200M/320M),
    fast, and built for exactly English<->Indic and Indic<->Indic.

IndicTrans2 needs a preprocessing step (the IndicProcessor) that normalizes
text and adds language-tag tokens BEFORE the model sees it, then postprocesses
the output. That's what IndicTransToolkit gives us. Without it the model
mistranslates — it's not optional.

Install (on the RunPod pod):
    pip install transformers torch IndicTransToolkit

Test this file on its own (no server needed):
    python translate.py
"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# Run on GPU if the pod has one (it does — 3090/H100), else fall back to CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# IndicTrans2 ships SEPARATE checkpoints per direction. We load all three so the
# pipeline can go any-language -> any-language. Total is ~720M params (<2 GB) —
# fits easily alongside Shrutam + Sooktam. The keys are our own internal names.
CKPTS = {
    "en-indic":    "ai4bharat/indictrans2-en-indic-dist-200M",   # English -> Indian language
    "indic-en":    "ai4bharat/indictrans2-indic-en-dist-200M",   # Indian language -> English
    "indic-indic": "ai4bharat/indictrans2-indic-indic-dist-320M", # Indian <-> Indian
}

# IndicTrans2 identifies languages by a "<script>_<Script>" code, NOT plain names.
# This maps the language slugs your server.py already uses (PROMPTS keys) to those
# codes, so the rest of the pipeline keeps speaking in "hindi"/"tamil"/... .
LANG_CODES = {
    "english":   "eng_Latn",
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

# Lazily-loaded (model, tokenizer) per direction, plus the shared preprocessor.
# We load a checkpoint only the first time a direction is actually used, so
# importing this module is cheap and we don't pay for directions we never call.
_models: dict[str, tuple] = {}
_ip: IndicProcessor | None = None


def _pick_direction(src_lang: str, tgt_lang: str) -> str:
    """Choose which checkpoint handles a given src->tgt pair."""
    if src_lang == "english":
        return "en-indic"
    if tgt_lang == "english":
        return "indic-en"
    return "indic-indic"      # both sides are Indian languages


def _get(direction: str) -> tuple:
    """Load (and cache) the model + tokenizer for a direction on first use."""
    global _ip
    if _ip is None:
        _ip = IndicProcessor(inference=True)
    if direction not in _models:
        ckpt = CKPTS[direction]
        print(f">>> Loading IndicTrans2 [{direction}] ({ckpt}) ...", flush=True)
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt, trust_remote_code=True, low_cpu_mem_usage=True,
        ).to(DEVICE)
        model.eval()
        _models[direction] = (model, tok)
    return _models[direction]


def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate `text` from src_lang to tgt_lang (both are slugs like 'hindi').

    No-op if the languages match (e.g. hindi->hindi) — nothing to translate.
    """
    if src_lang == tgt_lang:
        return text
    if src_lang not in LANG_CODES or tgt_lang not in LANG_CODES:
        raise ValueError(f"Unsupported language pair: {src_lang} -> {tgt_lang}")

    src_code, tgt_code = LANG_CODES[src_lang], LANG_CODES[tgt_lang]
    model, tok = _get(_pick_direction(src_lang, tgt_lang))

    # 1. Preprocess: normalize + prepend the language-tag tokens IndicTrans2 needs.
    batch = _ip.preprocess_batch([text], src_lang=src_code, tgt_lang=tgt_code)
    # 2. Tokenize and move to the model's device.
    inputs = tok(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
    # 3. Generate. beam search (num_beams=5) is what the official example uses.
    with torch.no_grad():
        out = model.generate(**inputs, max_length=256, num_beams=5, num_return_sequences=1)
    # 4. Decode, then postprocess (the toolkit undoes the tag tokens / normalization).
    decoded = tok.batch_decode(out, skip_special_tokens=True)
    return _ip.postprocess_batch(decoded, lang=tgt_code)[0]


if __name__ == "__main__":
    # Quick self-test covering all three directions.
    tests = [
        ("Hello, how are you?", "english", "hindi"),
        ("मैं ठीक हूँ, धन्यवाद।", "hindi", "english"),
        ("मैं ठीक हूँ, धन्यवाद।", "hindi", "tamil"),
    ]
    for txt, s, t in tests:
        print(f"[{s} -> {t}] {txt!r}\n    -> {translate(txt, s, t)!r}\n")
