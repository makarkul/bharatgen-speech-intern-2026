#!/usr/bin/env python
"""
Step 4 grid: same sentence x 3 languages x 3 reference speakers = 9 samples.

Loads the model once, then loops. Setup notes in run_sooktam2_local.py.
Output -> week4/samples/grid_<lang>_spk<N>.wav

Each language gets ONE neutral target sentence written natively in that
language (not a transliteration), long enough (~10-14s) to judge voice
similarity. For each language we clone it with reference clips 1, 2, 3 —
three speakers verified distinct by fetch_tts_refs.py. Holding the sentence
fixed within a language isolates the speaker variable: differences you hear
across spk1/2/3 are the voice transfer, not the text.
"""

import sys
import time
from pathlib import Path

SOOKTAM_SRC = Path.home() / "models" / "sooktam2" / "src"
sys.path.insert(0, str(SOOKTAM_SRC))

from huggingface_hub import hf_hub_download

REF_ROOT = Path.home() / "datasets" / "tts_refs"
OUT_DIR = Path(__file__).parent / "samples"
OUT_DIR.mkdir(exist_ok=True)

# One fixed, neutral target sentence per language (native, not transliterated).
TARGET = {
    "hindi": "हमारे शहर में हर साल सर्दियों की शुरुआत में एक बड़ा मेला लगता है, "
             "जहाँ दूर-दूर से लोग आते हैं और तरह-तरह के व्यंजन और हस्तकला का आनंद लेते हैं।",
    "marathi": "आमच्या गावात दरवर्षी हिवाळ्याच्या सुरुवातीला एक मोठी जत्रा भरते, "
               "जिथे दूरदूरून लोक येतात आणि विविध पदार्थ आणि हस्तकलेचा आनंद घेतात.",
    "tamil": "எங்கள் ஊரில் ஒவ்வொரு ஆண்டும் குளிர்காலத்தின் தொடக்கத்தில் ஒரு பெரிய திருவிழா நடைபெறும், "
             "அங்கு தொலைதூரத்திலிருந்து மக்கள் வந்து பலவகையான உணவுகளையும் கைவினைப் பொருட்களையும் ரசிக்கிறார்கள்.",
}

SPEAKERS = ["clip_001", "clip_002", "clip_003"]


def ref_for(lang: str, clip: str):
    tsv = REF_ROOT / lang / "transcripts.tsv"
    for line in tsv.read_text(encoding="utf-8").splitlines()[1:]:
        fname, text, _ = line.split("\t")
        if fname == f"{clip}.wav":
            return REF_ROOT / lang / "clips" / fname, text
    raise ValueError(f"{clip} not found in {tsv}")


ckpt = hf_hub_download("bharatgenai/sooktam2", "model_1250000.pt")
vocab = hf_hub_download("bharatgenai/sooktam2", "vocab.txt")

from f5_tts.api import F5TTS

t0 = time.time()
model = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab)
print(f"model loaded in {time.time() - t0:.0f}s\n")

for lang, gen_text in TARGET.items():
    for n, clip in enumerate(SPEAKERS, start=1):
        ref_file, ref_text = ref_for(lang, clip)
        out_name = f"grid_{lang}_spk{n}.wav"
        t0 = time.time()
        wav, sr, _ = model.infer(
            ref_file=str(ref_file),
            ref_text=ref_text,
            gen_text=gen_text,
            tokenizer="cls",
            cls_language=lang,
            file_wave=str(OUT_DIR / out_name),
        )
        gen_time = time.time() - t0
        audio_len = len(wav) / sr
        flag = "  <-- SHORT, check for collapse" if audio_len < 5 else ""
        print(f"{out_name}: {audio_len:5.1f}s audio in {gen_time:3.0f}s (RTF {gen_time / audio_len:.1f}){flag}")

print("\ndone ->", OUT_DIR)
