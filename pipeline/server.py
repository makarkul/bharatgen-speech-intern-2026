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
import subprocess
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

# ---------------------------------------------------------------------------
# Translation (the middle step that makes this a TRANSLATOR, not just ASR->TTS).
# Param-2 needs transformers==4.52.3, which CLASHES with the 4.56.2 this stack
# pins for Shrutam/Sooktam — so Param-2 runs as a SEPARATE service (its own venv
# + process; see param_service.py) and we call it over HTTP. That keeps the
# verified ASR/TTS stack untouched. PARAM_URL points at that service.
#
# We don't need any heavy deps here — just `requests` to POST to the service.
# If the service is down/unreachable, cross-language requests surface a clear
# error; same-language requests never call it.
# ---------------------------------------------------------------------------
PARAM_URL = os.environ.get("PARAM_URL", "http://localhost:8001") + "/translate"
print(f">>> Translator: Param-2 service at {PARAM_URL}", flush=True)

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


def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate text from src_lang to tgt_lang (slugs like 'hindi', 'tamil').

    Calls the separate Param-2 service over HTTP (see param_service.py). No-op
    when the languages match. In STUB_MODE we tag the text instead of pretending
    — keeps the laptop loop testable without the service running.
    """
    if src_lang == tgt_lang:
        return text
    if STUB_MODE:
        return f"[STUB translate {src_lang}->{tgt_lang}] {text}"
    # Real mode: ask the Param-2 service. Generous timeout — the 17B model is slow,
    # and a cold first request also pays the one-time model load.
    import requests
    resp = requests.post(
        PARAM_URL, json={"text": text, "src": src_lang, "tgt": tgt_lang}, timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["translation"]


def synthesize(text: str, voice: str, language: str,
               ref_file: str | None = None, ref_text: str | None = None,
               ref_language: str | None = None) -> bytes:
    """Turn `text` into speech (Sooktam-2). Returns WAV bytes.

    `text` is the text to SPEAK and `language` is ITS language — i.e. the target
    language of the pipeline. cls_language follows `text`, not the reference clip.

    Sooktam clones a reference voice. Normally that's a preset (VOICES[voice]).
    But if ref_file + ref_text are passed (the "use my own voice" mode), we clone
    those instead — the caller hands us the user's own recording + its transcript.
    With translation on, the reference clip may be in a DIFFERENT language than
    `text` (you spoke Hindi, we speak Tamil back in your voice); `ref_language`
    just documents the clip's language — Sooktam reads `ref_text` directly and
    tokenizes the generated `text` with cls_language=`language`.
    """
    if STUB_MODE:
        # Fake beep so the browser's audio path is testable without a GPU.
        return _fake_beep_wav(duration_s=0.6, freq_hz=440, sample_rate=16000)

    if ref_file is not None:
        # "Use my voice": clone the user's own clip. cls_language follows `text`
        # (the language we GENERATE), which may differ from the reference clip's
        # language when translation is on — that's fine, ref_text matches the clip.
        clone_ref_file, clone_ref_text, cls_language = ref_file, ref_text or "", language
    else:
        v = VOICES[voice]
        # VOICES_DIR is absolute (__file__-based), so Shrutam's os.chdir doesn't
        # break this path.
        clone_ref_file, clone_ref_text, cls_language = (
            str(VOICES_DIR / v["ref_file"]), v["ref_text"], v["language"],
        )

    # Output is 24 kHz; we encode the float array to WAV bytes.
    wav, sr, _ = _sooktam.infer(
        ref_file=clone_ref_file,
        ref_text=clone_ref_text,
        gen_text=text,
        tokenizer="cls",
        cls_language=cls_language,
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


def _to_wav_16k_mono(src_path: str, dst_path: str) -> None:
    """Transcode any audio (webm/ogg/mp4/wav/...) to 16 kHz mono 16-bit WAV.

    Browsers hand us Opus-in-webm, which soundfile can't decode. ffmpeg can read
    essentially anything, so we normalize here before the models touch the file.
    """
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000",
         "-f", "wav", dst_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to convert the uploaded audio:\n"
            + result.stderr.decode("utf-8", "replace")[-500:]
        )


@app.post("/speak")
async def speak_endpoint(
    audio: UploadFile = File(...),
    language: str = Form("hindi"),
    target_language: str = Form(""),
    voice: str = Form("voice_a"),
):
    """The full loop: audio -> transcribe -> translate -> synthesize -> return all.

    `language` is the SPOKEN (source) language; `target_language` is what to
    translate to and speak back. If target_language is empty or equals language,
    it's the old same-language behavior (no translation).

    Returns {"text": <source transcript>, "translation": <target text>,
    "audio": <base64 WAV>}. We base64 the audio so text + audio fit in one JSON
    response (one round-trip instead of two).
    """
    tgt = target_language or language       # empty target -> same language (no translation)

    # Browsers record as .webm (Opus), which soundfile can't read. Save the raw
    # upload, then transcode to 16 kHz mono WAV with ffmpeg so the models (which
    # expect WAV via soundfile) can load it. Resampling to 16 kHz also matches
    # what Shrutam-2 was trained on.
    suffix = Path(audio.filename or "rec.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        raw_path = tmp.name
    wav_path = raw_path + ".wav"

    try:
        _to_wav_16k_mono(raw_path, wav_path)            # step 0: any format -> 16k mono WAV
        text = transcribe(wav_path, language)           # step 1: speech -> text (source lang)
        translation = translate_text(text, language, tgt)  # step 2: source -> target text
        # step 3: synthesize the TRANSLATED text, in the TARGET language.
        if voice == "mine":
            # "Use my voice": clone the user's own recording. The reference clip
            # is their SOURCE-language audio, so its transcript is `text` (source),
            # NOT the translation — the ref_text must match the ref audio. Sooktam
            # then speaks the target-language `translation` in that cloned voice.
            wav_bytes = synthesize(translation, voice, tgt,
                                   ref_file=wav_path, ref_text=text,
                                   ref_language=language)
        else:
            wav_bytes = synthesize(translation, voice, tgt)   # preset voice
    finally:
        Path(raw_path).unlink(missing_ok=True)
        Path(wav_path).unlink(missing_ok=True)

    return {
        "text": text,
        "translation": translation,
        "audio": base64.b64encode(wav_bytes).decode("ascii"),
    }


@app.get("/voices")
async def voices_endpoint():
    """List the TTS voices for the frontend's picker: the presets plus a special
    'mine' option that clones whatever the user just recorded."""
    voices = {key: {"label": v["label"]} for key, v in VOICES.items()}
    voices["mine"] = {"label": "Use my voice"}
    return voices


@app.get("/")
async def index():
    """Serve the page itself, so you can just visit http://localhost:8000/."""
    return FileResponse(Path(__file__).parent / "index.html")
