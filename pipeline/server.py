"""Speech-translation voice bot — main backend (ASR + TTS; translation is a service).

The loop:  record audio  ->  Shrutam-2 (speech->text, source lang)
           ->  translate (Param-2 service over HTTP)  ->  text in target lang
           ->  Sooktam-2 (text->speech, target lang)  ->  play it back.

ONE endpoint does the whole thing:
    POST /speak  (audio + language + target_language + voice)
              -> {"text": <source>, "translation": <target>, "audio": <base64 wav>}

Two run modes, chosen automatically by whether the model repos exist:
  * Laptop (no repos)  -> STUB_MODE: fake transcript + beep, and translation is
    tagged (the Param-2 service isn't running). Lets the whole UI loop be tested
    on a laptop with NO GPU.
  * GPU pod (repos present) -> real Shrutam + Sooktam loaded once at import;
    translation calls the separate Param-2 service (see param_service.py).

Run (stub mode, no GPU):
    pip install fastapi uvicorn python-multipart requests
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

app = FastAPI(title="Speech-translation voice bot (Shrutam-2 + Param-2 + Sooktam-2)")

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
PARAM_URL = os.environ.get("PARAM_URL", "http://localhost:8500") + "/translate"
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

# Sooktam-2 is a VOICE-CLONING TTS: it clones a reference clip's voice/accent and
# speaks the new text in it. Cloning a HINDI reference while generating (say) Tamil
# sounds off — a Hindi speaker's accent on Tamil. So we keep ONE reference clip PER
# LANGUAGE and pick the one matching the TARGET (output) language, so the accent
# matches the text. Clips are clean ~9s native recordings (tts_refs benchmark set),
# each paired with its EXACT transcript (must match the audio or cloning degrades).
#
# Only 3 languages have native clips so far (hindi, marathi, tamil). The other 9
# Shrutam languages fall back to the Hindi clip (functional, just Hindi-accented)
# until native clips are added — drop a ref_<lang>.wav + transcript here to upgrade.
#
# ref paths resolve relative to THIS file's folder, so cwd (Shrutam's os.chdir)
# doesn't matter.
VOICES_DIR = Path(__file__).parent / "voices"

LANG_VOICES = {
    "hindi": {
        "ref_file": "ref_hindi.wav",
        "ref_text": "इस बीच कल शिमला से वीडियो कांफ्रेसिंग के माध्यम से संगठनात्मक जिला नूरपुर के अन्य पिछड़ा वर्ग मोर्चा की वर्घुअल रैली को भी मुख्यमंत्री ने संबोधित किया",
    },
    "marathi": {
        "ref_file": "ref_marathi.wav",
        "ref_text": "हे चक्रीवादळ ज्या ठिकाणांडून जाणार आहे त्या सर्व ठिकाणी अन्न पिण्याचं पाणी औषधं आणि तिर अत्यावशक सुविधा पोचवण्याचे निर्देश केंद्रीय सचीव पी के सिन्हा यांनी संबधित यंत्रणांना दिले आहेत",
    },
    "tamil": {
        "ref_file": "ref_tamil.wav",
        "ref_text": "உலக ககாதார அமைப்பின் தலைமை இயக்குநர் திரு டெட்ரோஸ் அதானோ கெப்ரீசஸ் வெளியிட்டுள்ள அறிக்கையில் பல நாடுகளில் கொரோனா தொற்று பரவியிருப்பதால் அச்சம் ஏற்பட்டுள்ளதாக தெரிவித்துள்ளார்",
    },
}

# Languages with no native clip yet fall back to this one's reference.
VOICE_FALLBACK = "hindi"


def _voice_for(language: str) -> dict:
    """Pick the reference clip whose language matches the target (or fall back)."""
    return LANG_VOICES.get(language, LANG_VOICES[VOICE_FALLBACK])


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
    # Real mode: ask the Param-2 service. Generous timeout — the 17B model is slow.
    # (It pre-loads at its own startup, so we shouldn't pay the load here.)
    import requests
    try:
        resp = requests.post(
            PARAM_URL, json={"text": text, "src": src_lang, "tgt": tgt_lang}, timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["translation"]
    except requests.exceptions.RequestException as e:
        # Service down/unreachable/slow: give a clear message instead of a raw 500,
        # so the UI shows something actionable and the cause is obvious in the log.
        print(f">>> Param-2 service error ({PARAM_URL}): {e}", flush=True)
        raise RuntimeError(
            f"Translation service unavailable ({src_lang}->{tgt_lang}). "
            f"Is param_service running on {PARAM_URL}?"
        ) from e


def synthesize(text: str, language: str) -> bytes:
    """Turn `text` into speech (Sooktam-2). Returns WAV bytes.

    `text` is the text to SPEAK and `language` is ITS language (the pipeline's
    TARGET language). We pick the reference clip for that language so the voice's
    accent matches the text, and set cls_language to it. Falls back to the Hindi
    clip for languages without a native reference yet.
    """
    if STUB_MODE:
        # Fake beep so the browser's audio path is testable without a GPU.
        return _fake_beep_wav(duration_s=0.6, freq_hz=440, sample_rate=16000)

    v = _voice_for(language)
    # VOICES_DIR is absolute (__file__-based), so Shrutam's os.chdir doesn't break
    # this path. cls_language follows the GENERATED text's language (= `language`).
    wav, sr, _ = _sooktam.infer(
        ref_file=str(VOICES_DIR / v["ref_file"]),
        ref_text=v["ref_text"],
        gen_text=text,
        tokenizer="cls",
        cls_language=language,
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
):
    """The full loop: audio -> transcribe -> translate -> synthesize -> return all.

    `language` is the SPOKEN (source) language; `target_language` is what to
    translate to and speak back. If target_language is empty or equals language,
    it's same-language behavior (no translation). The output voice is chosen
    automatically to match the target language (see _voice_for).

    Returns {"text": <source transcript>, "translation": <target text>,
    "audio": <base64 WAV>}. We base64 the audio so text + audio fit in one JSON
    response (one round-trip instead of two).
    """
    tgt = target_language or language       # empty target -> same language (no translation)

    # Sooktam-2 only speaks Indian languages (the PROMPTS set). English (and any
    # other non-Indic target) has no Sooktam voice, so guard here rather than let
    # it crash deep in the TTS. The frontend already only offers these targets.
    if tgt not in PROMPTS:
        return {"text": "", "translation": "",
                "error": f"'{tgt}' is not a supported speech output language "
                         f"(Sooktam-2 speaks: {', '.join(PROMPTS)})."}

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
        # step 3: synthesize the TRANSLATED text, in the TARGET language (the voice
        # is auto-picked to match tgt inside synthesize()).
        wav_bytes = synthesize(translation, tgt)
    finally:
        Path(raw_path).unlink(missing_ok=True)
        Path(wav_path).unlink(missing_ok=True)

    return {
        "text": text,
        "translation": translation,
        "audio": base64.b64encode(wav_bytes).decode("ascii"),
    }


@app.get("/")
async def index():
    """Serve the page itself, so you can just visit http://localhost:8000/."""
    return FileResponse(Path(__file__).parent / "index.html")
