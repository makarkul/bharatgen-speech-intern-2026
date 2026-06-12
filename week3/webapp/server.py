"""Shrutam-2 + Sooktam-2 speech-to-speech web demo — backend.

The loop:  record audio  ->  Shrutam-2 (speech->text)  ->  Sooktam-2 (text->speech)
           ->  play the synthesized audio back in the browser.

ONE endpoint does the whole thing:
    POST /speak  (audio file + language + voice)  ->  {"text": "...", "audio": "<base64 wav>"}

Right now BOTH models are STUBS (fake text, fake audio) so the entire
record -> transcribe -> synthesize -> play loop can be built and tested
WITHOUT a GPU, on a laptop. The real models plug in at the two spots marked
>>> SWAP HERE <<<  below (done later, on a RunPod GPU pod).

Run (stub mode, no GPU):
    pip install fastapi uvicorn python-multipart
    uvicorn server:app --reload --port 8000

Then open  http://localhost:8000/  in a browser (see README).
"""
import base64
import io
import math
import os
import struct
import sys
import tempfile
import wave
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="Shrutam-2 + Sooktam-2 demo")

# ---------------------------------------------------------------------------
# Model loading. The SAME file runs two ways:
#   * Laptop (no model repos)  -> STUB_MODE = True  -> fake text + fake beep.
#   * Pod (repos present)      -> STUB_MODE = False -> real Shrutam + Sooktam.
# setup.sh clones the repos and prints the env vars to export. If those repos
# exist, we load both models ONCE here at import (NOT per request — the Shrutam
# checkpoint is ~5GB and the Sooktam load is slow).
# ---------------------------------------------------------------------------
SHRUTAM_DIR = Path(os.environ.get("SHRUTAM_DIR", Path.home() / "models" / "Shrutam-2"))
SOOKTAM_DIR = Path(os.environ.get("SOOKTAM_DIR", Path.home() / "models" / "sooktam2"))

STUB_MODE = not (SHRUTAM_DIR.exists() and SOOKTAM_DIR.exists())

_shrutam = None   # the imported inference_script module
_sooktam = None   # the F5TTS instance

if not STUB_MODE:
    print(">>> Real-model mode. Loading Sooktam-2 and Shrutam-2 (once) ...", flush=True)

    # --- Sooktam-2 (TTS) first: it only needs its src/ on sys.path. ---
    sys.path.insert(0, str(SOOKTAM_DIR / "src"))
    from huggingface_hub import hf_hub_download
    from f5_tts.api import F5TTS
    _ckpt = hf_hub_download("bharatgenai/sooktam2", "model_1250000.pt")
    _vocab = hf_hub_download("bharatgenai/sooktam2", "vocab.txt")
    # MUST pass ckpt_file/vocab_file explicitly — empty makes F5TTS download the
    # ORIGINAL English F5-TTS weights instead of Sooktam-2.
    _sooktam = F5TTS(model="F5TTS_v1_Base", ckpt_file=_ckpt, vocab_file=_vocab)
    print(">>> Sooktam-2 loaded.", flush=True)

    # --- Shrutam-2 (ASR) last: it has RELATIVE paths + an unguarded sample run
    # at import time, so it must be imported with the cwd INSIDE its repo. ---
    sys.path.insert(0, str(SHRUTAM_DIR))
    os.chdir(SHRUTAM_DIR)
    import inference_script as _shrutam   # runs a warmup sample at import — expected
    print(">>> Shrutam-2 loaded. Ready.", flush=True)
else:
    print(">>> STUB MODE (model repos not found) — fake text + fake beep.", flush=True)

# The page and this server may be two separate origins (e.g. the page is local
# and the backend is on a RunPod pod), so the browser blocks the request unless
# we say it's allowed. For a demo, allow everything.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shrutam-2's 12 supported languages, keyed slug -> natural-language prompt.
# Wording follows the pattern the Week 3 eval verified to work ("...to Hindi text.").
PROMPTS = {
    "hindi":     "Transcribe speech to Hindi text.",
    "marathi":   "Transcribe speech to Marathi text.",
    "tamil":     "Transcribe speech to Tamil text.",
    "telugu":    "Transcribe speech to Telugu text.",
    "malayalam": "Transcribe speech to Malayalam text.",
    "kannada":   "Transcribe speech to Kannada text.",
    "odia":      "Transcribe speech to Odia text.",
    "bengali":   "Transcribe speech to Bengali text.",
    "urdu":      "Transcribe speech to Urdu text.",
    "assamese":  "Transcribe speech to Assamese text.",
    "gujarati":  "Transcribe speech to Gujarati text.",
    "punjabi":   "Transcribe speech to Punjabi text.",
}

# Sooktam-2 is a VOICE-CLONING TTS: to synthesize, it needs a short reference
# clip + that clip's transcript, and it speaks the new text in that voice. So we
# ship a few preset voices. These are clean 16kHz mono Hindi clips (~9s each)
# from the tts_refs benchmark set, with their verified transcripts. The frontend
# reads this dict (via GET /voices) to populate its voice picker.
#
# ref_file paths are resolved relative to THIS file's folder (VOICES_DIR below),
# so the server works no matter what directory uvicorn is launched from.
VOICES_DIR = Path(__file__).parent / "voices"

VOICES = {
    "voice_a": {
        "label": "Voice A",
        "ref_file": "voice_a.wav",
        "ref_text": "मन की बात कार्यक्रम आकाशवाणी द्रदर्शन समाचार प्रधानमंत्री कार्यालय तथा सूचना और प्रसारण मंत्रालय के यूट्यूब चैनलों पर भी सीधा प्रसारित होगा",
        "language": "hindi",   # language the reference clip is spoken in
    },
    "voice_b": {
        "label": "Voice B",
        "ref_file": "voice_b.wav",
        "ref_text": "आकाशवाणी से मैच का आंखों देखा हाल ढाई बजे से राजधानी और एम रेनबो तथा डीटीएच हिंदी पर उपलब्ध रहेगा",
        "language": "hindi",
    },
}


def transcribe(wav_path: str, language: str) -> str:
    """Turn an audio file into text (Shrutam-2)."""
    if STUB_MODE:
        size_kb = Path(wav_path).stat().st_size / 1024
        return (
            f"[STUB] Pretend this is the {language} transcript. "
            f"Received {size_kb:.0f} KB of audio — the real Shrutam-2 goes here."
        )
    # inference() takes a file path + prompt and returns a ONE-ELEMENT list
    # like ['transcript'] — unwrap to the bare string.
    prompt = PROMPTS[language]
    result = _shrutam.inference(wav_path, prompt)
    text = result[0] if isinstance(result, (list, tuple)) else result
    return str(text).strip()


def synthesize(text: str, voice: str) -> bytes:
    """Turn text into speech (Sooktam-2). Returns WAV bytes."""
    if STUB_MODE:
        # Fake beep so the browser's audio path is testable without a GPU.
        return _fake_beep_wav(duration_s=0.6, freq_hz=440, sample_rate=16000)
    v = VOICES[voice]
    # VOICES_DIR is absolute (__file__-based), so Shrutam's os.chdir doesn't
    # break this path. Output is 24 kHz; we encode the float array to WAV bytes.
    wav, sr, _ = _sooktam.infer(
        ref_file=str(VOICES_DIR / v["ref_file"]),
        ref_text=v["ref_text"],
        gen_text=text,
        tokenizer="cls",
        cls_language=v["language"],
    )
    return _wav_to_bytes(wav, sr)


def _wav_to_bytes(wav, sample_rate: int) -> bytes:
    """Encode a float audio array (range ~[-1, 1]) to 16-bit mono WAV bytes."""
    import numpy as np

    samples = np.asarray(wav, dtype=np.float32).flatten()
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2")  # little-endian signed 16-bit
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _fake_beep_wav(duration_s: float, freq_hz: int, sample_rate: int) -> bytes:
    """Generate a short sine-tone WAV entirely in memory (stub audio, no deps)."""
    n_samples = int(duration_s * sample_rate)
    amplitude = 16000  # well under the 32767 max for 16-bit, so it's not harsh
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)        # mono
        wav.setsampwidth(2)        # 16-bit
        wav.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            sample = int(amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate))
            frames += struct.pack("<h", sample)  # little-endian signed 16-bit
        wav.writeframes(bytes(frames))
    return buf.getvalue()


@app.post("/speak")
async def speak_endpoint(
    audio: UploadFile = File(...),
    language: str = Form("hindi"),
    voice: str = Form("voice_a"),
):
    """The full loop: receive recorded audio -> transcribe -> synthesize -> return both.

    Returns {"text": <transcript>, "audio": <base64-encoded WAV>}. The frontend
    shows the text and plays the audio. We base64 the audio so text + audio fit
    in one JSON response (one round-trip instead of two).
    """
    # Shrutam-2's inference() takes a FILE PATH, so we write the upload to disk.
    suffix = Path(audio.filename or "rec.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        text = transcribe(tmp_path, language)      # step 1: speech -> text
        wav_bytes = synthesize(text, voice)         # step 2: text -> speech
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "text": text,
        "audio": base64.b64encode(wav_bytes).decode("ascii"),
    }


@app.get("/voices")
async def voices_endpoint():
    """List the preset TTS voices, so the frontend can build its voice picker."""
    return {key: {"label": v["label"]} for key, v in VOICES.items()}


@app.get("/")
async def index():
    """Serve the page itself, so you can just visit http://localhost:8000/."""
    return FileResponse(Path(__file__).parent / "index.html")
