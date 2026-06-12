# Shrutam-2 + Sooktam-2 — speech-to-speech web demo

Record your voice → **Shrutam-2** transcribes it → **Sooktam-2** speaks the text
back in a preset voice. Same language in and out (no translation).

The **same `server.py` runs two ways**:
- **Laptop / no GPU** → *stub mode*: fake transcript + a short beep, so the whole
  record → text → audio → playback loop is testable for free.
- **GPU pod** → *real mode*: loads both models once at startup and runs them.
  It switches automatically based on whether the model repos exist.

## Files
- `server.py` — FastAPI backend. `POST /speak` (audio+language+voice → text+audio),
  `GET /voices`.
- `index.html` — the page (records mic, sends audio, shows text, plays the reply).
- `voices/` — preset reference clips + their transcripts live in `VOICES` in
  `server.py` (Sooktam clones the reference voice).
- `requirements.txt` / `setup.sh` — pod-side deps and one-time setup.

## Run locally (stub mode, no GPU)
```bash
source ../../.venv/bin/activate     # the repo venv (has fastapi/uvicorn)
uvicorn server:app --reload --port 8000
```
Open **http://localhost:8000/**. Pick a language + voice, record, and you'll get
stub text and hear a beep — proof the loop works end to end.

> Mic needs a "secure context": `localhost` counts. Served by IP, browsers need HTTPS.

## Run on a RunPod GPU pod (real models)

1. **Launch a pod** with a PyTorch template; expose **HTTP port 8000**. For
   debugging use an RTX 3090 (24 GB) — enough VRAM for both models. For
   real-time testing, an H100 SXM. The same code runs on both.
2. **Get the code + run setup** (in the pod's web terminal):
   ```bash
   git clone <this-repo-url>
   cd bharatgen-speech-intern-2026/week3/webapp
   bash setup.sh        # installs deps, clones both model repos, patches them, caches weights
   ```
   `setup.sh` is idempotent — safe to re-run.
3. **Start the server** (setup.sh prints these exact lines with real paths):
   ```bash
   export SHRUTAM_DIR=$HOME/models/Shrutam-2
   export SOOKTAM_DIR=$HOME/models/sooktam2
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```
   On first import Shrutam-2 runs a warmup sample and prints a transcript — expected.
4. **Point the page at the pod**: edit `API_BASE` in `index.html` to the pod's
   proxy URL (RunPod shows it, like `https://<pod-id>-8000.proxy.runpod.net`),
   then open `index.html` locally in your browser.

### Patches setup.sh applies (needed on every fresh clone)
- **Shrutam-2** — `mmap=True` on the checkpoint load (avoids a system-RAM OOM
  when the 5 GB checkpoint + the LLM load at once).
- **Sooktam-2** — windows the "robotic head" boundary search to the first ~1s,
  or long generations randomly collapse to ~1s of audio.

### If Sooktam breaks under transformers 4.56.2
Shrutam pins `transformers==4.56.2` and Sooktam (an F5-TTS fork) runs on top of
it in one process. If Sooktam misbehaves at runtime, that's the signal to split
the two models into separate processes/services — not done yet to keep it simple.
