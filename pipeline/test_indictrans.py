"""Standalone smoke test for IndicTrans2 — run BEFORE wiring the service.

Confirms two things, with NO server and NO main pipeline involved:
  1. CORRECTNESS — IndicTrans2 + the IndicProcessor actually produce sensible
     Indic->Indic translations (the processor step is mandatory; this proves it
     works end to end).
  2. LATENCY — how long each translation's model.generate() takes, so you can
     compare against Param-2's [param timing] lines in param_service.log.

Run in the IndicTrans2 venv (see indictrans_service.py header for setup):
    source ~/indictrans_venv/bin/activate
    export HF_HOME=/workspace/hf_cache
    python test_indictrans.py

It imports _translate from the service module, so it exercises the EXACT code
path the service uses — if this passes, the service will translate identically.
"""
import time

# Reuse the real service logic so the test can't drift from what the service does.
from indictrans_service import _translate, _load, INDIC_INDIC_CKPT, DEVICE

# Generic sentences only (no personal info). A spread of source languages and
# target scripts so we exercise several language-tag paths through the processor.
CASES = [
    ("नमस्ते, आप कैसे हैं?",                 "hindi",   "tamil"),
    ("मुझे यह जगह बहुत पसंद है।",            "hindi",   "telugu"),
    ("आज मौसम बहुत अच्छा है।",               "hindi",   "kannada"),
    ("எனக்கு உதவி தேவை.",                   "tamil",   "hindi"),
    ("ఈ పుస్తకం చాలా బాగుంది.",              "telugu",  "marathi"),
]


def main():
    print(f">>> IndicTrans2 smoke test — checkpoint={INDIC_INDIC_CKPT}, device={DEVICE}")

    # Time the one-off load separately from per-call latency.
    _t0 = time.perf_counter()
    _load()
    print(f">>> Model load: {time.perf_counter() - _t0:.1f}s\n")

    times = []
    for txt, src, tgt in CASES:
        out, secs = _translate(txt, src, tgt)
        times.append(secs)
        print(f"[{src} -> {tgt}]")
        print(f"    in : {txt}")
        print(f"    out: {out}")
        print(f"    model time: {secs:.3f}s\n")

    if times:
        print(f">>> Per-call model time: "
              f"min={min(times):.3f}s  max={max(times):.3f}s  "
              f"avg={sum(times) / len(times):.3f}s")
        print(">>> Compare these to the [param timing] lines in param_service.log.")


if __name__ == "__main__":
    main()
