"""Compare IndicTrans2 vs Param-2 on the same sentences — quality + speed.

Runs an identical set of (text, src, tgt) cases through BOTH translators and
prints them side by side with per-call timing, so you can judge for yourself
which to put in the pipeline. This is the "build both, compare on real audio"
step — decide with evidence, not vendor claims.

Run on the H100 pod (Param-2 needs the big GPU):
    pip install transformers torch accelerate IndicTransToolkit
    python compare.py

To compare on YOUR OWN audio instead of the canned sentences below: first run
your clips through Shrutam-2 to get transcripts, then paste those transcripts
into CASES as the source text (the real pipeline input).
"""
import time

import translate as it2            # IndicTrans2 module
import translate_param2 as p2      # Param-2 module

# (source text, source lang slug, target lang slug). Swap in real Shrutam-2
# transcripts here to compare on actual pipeline data.
CASES = [
    ("Hello, how are you doing today?",            "english", "hindi"),
    ("मैं ठीक हूँ, आपका बहुत धन्यवाद।",              "hindi",   "english"),
    ("मुझे भारतीय भाषाओं में अनुवाद करना पसंद है।",  "hindi",   "tamil"),
    ("The weather is nice and the train is on time.", "english", "marathi"),
]


def _timed(fn, *a):
    """Run fn, returning (result_or_error_string, seconds)."""
    t0 = time.perf_counter()
    try:
        result = fn(*a)
    except Exception as e:                      # one engine failing shouldn't abort the run
        result = f"<ERROR: {type(e).__name__}: {e}>"
    return result, time.perf_counter() - t0


def main():
    for text, src, tgt in CASES:
        print("=" * 70)
        print(f"[{src} -> {tgt}]  {text}")
        print("-" * 70)
        it_out, it_s = _timed(it2.translate, text, src, tgt)
        p2_out, p2_s = _timed(p2.translate, text, src, tgt)
        print(f"  IndicTrans2 ({it_s:5.2f}s): {it_out}")
        print(f"  Param-2     ({p2_s:5.2f}s): {p2_out}")
        print()


if __name__ == "__main__":
    main()
