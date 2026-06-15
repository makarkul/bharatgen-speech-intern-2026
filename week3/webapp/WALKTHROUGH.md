# Speech-to-Speech Web App — Full Walkthrough

A record of what this project is, how it was built, and every real-world problem
solved along the way. Written so it can be understood from scratch and re-explained
in your own words (e.g. for an intern report or a demo).

---

## 1. What it does

A **speech-to-speech web app**. The loop:

1. You open a web page and click a mic button.
2. You speak (in one of 12 Indian languages).
3. **Shrutam-2** (BharatGen's speech-recognition / ASR model) turns your speech into **text**.
4. **Sooktam-2** (BharatGen's text-to-speech / TTS model) turns that text **back into speech**, in a chosen voice.
5. The page shows the transcript and plays the synthesized audio.

Same language in and out — no translation. It's a demonstration of the two
models working together end to end.

---

## 2. The architecture (mental model)

```
   YOUR LAPTOP                                RENTED GPU (RunPod pod)
   ───────────────────                        ──────────────────────────────
   Browser running index.html                 FastAPI server (server.py)
     • pick language + voice                     • POST /speak   (the whole loop)
     • record mic  ──── audio (webm) ────►       • GET  /voices  (list preset voices)
     • show transcript                           │
     • play audio  ◄─── {text, audio} ───────────┤
                                                  ├─ ffmpeg: webm → 16kHz WAV
                                                  ├─ Shrutam-2: audio → text
                                                  └─ Sooktam-2: text → audio
```

Two separate machines:

- **Frontend** = `index.html`, runs in your browser on your laptop. Pure HTML +
  JavaScript. It records the mic, POSTs the audio to the backend, displays the
  returned text, and plays the returned audio.
- **Backend** = `server.py`, a small **FastAPI** server. It runs *on the GPU pod*
  because the models need a big GPU. It loads both models **once** at startup and
  serves requests.
- They talk over a **public URL** that RunPod assigns to the pod
  (`https://<pod-id>-8000.proxy.runpod.net`).

### Key files
| File | Role |
|------|------|
| `index.html` | The web page: mic recording, sends audio, shows text, plays reply. |
| `server.py` | FastAPI backend. `/speak` runs the full pipeline; `/voices` lists voices. |
| `voices/` | Two clean Hindi reference clips Sooktam clones (`voice_a.wav`, `voice_b.wav`). |
| `pod_setup.sh` | One command that rebuilds the entire GPU environment after a restart. |
| `requirements.txt` | Python deps (note: superseded in practice by `pod_setup.sh` — see §6). |

---

## 3. Why two models, and a crucial detail about each

- **Shrutam-2 (ASR).** Not a standard HuggingFace model — it ships its own
  `inference_script.py` that you drive directly. You put its folder on Python's
  path, `cd` into it, `import inference_script`, and call
  `inference(wav_path, prompt)`. The "prompt" picks the language
  (e.g. "Transcribe speech to Hindi text."). It returns a one-element list, so
  we unwrap `[0]`.

- **Sooktam-2 (TTS).** This is the one with the important twist: it's a
  **voice-cloning** TTS (an F5-TTS fork). It does **not** just speak text in a
  default voice. To synthesize, it needs:
  - a **reference audio clip** (3–10 seconds of clean speech), AND
  - the **transcript** of that reference clip,
  - plus the **target text** to speak and the **language**.

  It then speaks the new text *in the voice of the reference clip*. That's why
  we bundle "preset voices": each is a clean reference WAV + its known transcript.
  We picked two ~9-second Hindi clips from the benchmark set.

---

## 4. How it was built — Phase A (laptop, free, with "stubs")

A **stub** is a fake stand-in for the real thing. We wrote `server.py` so that:

- If the model folders **aren't present** (your laptop) → `STUB_MODE = True`:
  `transcribe()` returns fake text, `synthesize()` returns a short beep.
- If they **are present** (the GPU pod) → it loads and runs the real models.

This let us build and test the **entire app** — recording, uploading, the
round-trip, showing text, playing audio — **on the laptop, with no GPU, for
free**. We confirmed the full loop worked (record → fake text → beep plays).

The payoff of this design: **the same `server.py` runs in both places**. Going
from "fake" to "real" is just a matter of the model files existing on the pod —
no separate code path to maintain.

We also built the **voice picker** dropdown (populated from `/voices`) and the
**audio player** that decodes and plays the returned audio.

---

## 5. How it was built — Phase B (real models on a rented GPU)

Your laptop's GPU has only 4 GB of VRAM — Shrutam's LLM needs far more — so the
real models can't run locally. We rented a cloud GPU on **RunPod**.

This phase is where the real engineering happened. We hit a chain of problems;
each is a genuine, reusable lesson:

### 5.1 Picking the GPU
Chose an **RTX PRO 4500 (32 GB VRAM, ~$0.74/hr)**. We measured actual usage once
running: Shrutam ~8 GB, Sooktam ~7 GB peak — together ~10 GB, comfortably inside
32 GB. (Lesson: measure, don't guess — we'd worried it might not fit, and it had
tons of room.)

### 5.2 Getting code + models onto the pod
- The webapp code lives on **GitHub**, so the pod just `git clone`s it. (Gotcha:
  the webapp was on the `week3` branch, and `git clone` checks out the *default*
  branch — had to `git switch week3`.)
- The model **weights** (~17 GB total) download from HuggingFace.

### 5.3 Problem: the disk filled up
The pod had two ~30 GB disks. Two things ate the space:
- **git-LFS keeps a duplicate** of every large file inside the hidden `.git`
  folder — so the weights existed *twice*.
- We were downloading onto a disk with a quota smaller than it appeared.

**Fix:** delete the `.git` folders after cloning (the models load from the
working files, not git history — freed ~16 GB), and re-download the two big
weight files cleanly. git-LFS had **silently truncated** them (e.g. 4.9 GB
instead of 5.16 GB), so we re-fetched with `hf_hub_download` and **verified the
exact byte counts** matched.

### 5.4 Problem: the GPU was too new for the default PyTorch
The RTX PRO 4500 is a **Blackwell** chip (architecture "sm_120"). The PyTorch
that came pre-installed on the pod was built before Blackwell existed, so it had
**no GPU code for this chip** — every real operation crashed with *"no kernel
image is available for execution on the device."* Confusingly, `cuda` reported
"available" — but availability doesn't mean compatible kernels exist.

**Fix:** install the newer **cu128** PyTorch build (torch 2.11.0+cu128), which
includes Blackwell kernels. We confirmed `sm_120` appeared in the supported
arch list and that a real matrix multiply ran on the GPU. (Gotcha: `pip install`
said "already satisfied" and skipped — had to use `--force-reinstall`. Another:
install torch + torchvision + torchaudio *together from the same index* so their
versions match.)

### 5.5 Problem: the new PyTorch broke audio loading
The newer **torchaudio dropped its built-in `.load()`** function — it now wants a
separate `torchcodec` library, which itself wanted a CUDA-13 library that wasn't
there (a rabbit hole). Both models use `torchaudio.load()` to read audio.

**Fix:** instead of chasing torchcodec, we **patched both models to load audio
with `soundfile`** instead — a small, dependency-free swap that produces the same
data the models expected.

### 5.6 Problem: the browser's audio format
Browsers record audio as **`.webm`** (Opus codec). `soundfile` can't read webm —
so when a *real* recording arrived, the server crashed with "Format not
recognised" (HTTP 500). (Side effect: the 500 response dropped its CORS headers,
which made the browser *also* complain about CORS — a red herring; the real cause
was the format.)

**Fix:** added a step in `/speak` that runs each upload through **ffmpeg** to
convert it to a clean **16 kHz mono WAV** before the models touch it. This also
matches the sample rate Shrutam was trained on. Needed `apt-get install ffmpeg`
on the pod.

### 5.7 Problem: the browser couldn't reach the server
The server listened on **port 8000**, but RunPod only exposes ports you *declare*.
The pod had only declared 22 (SSH) and 8888 (Jupyter). So the browser got "page
not found."

**Fix:** edited the pod to **expose HTTP port 8000**. This restarts the pod
(wiping installed packages, but the models on `/workspace` survive). After the
restart we replayed the environment with one script (next point).

### 5.8 Making it repeatable: `pod_setup.sh`
Because a pod restart wipes the installed software, we captured **every fix** into
one script: install ffmpeg → install the cu128 PyTorch + all Python deps →
re-apply the 4 source patches → start the server. After any restart, one command
(`bash pod_setup.sh`) rebuilds everything in ~10 minutes.

---

## 6. The exact working stack (for reference)

- **GPU:** NVIDIA RTX PRO 4500 (Blackwell, sm_120, 32 GB)
- **PyTorch:** torch 2.11.0+cu128 (+ matching torchvision/torchaudio)
- **transformers** 4.56.2 (pinned by Shrutam)
- **ffmpeg** (for webm → WAV)
- **4 source patches:** Shrutam `mmap=True` on checkpoint load; Sooktam
  "robotic-head" cut windowed; `torchaudio.load → soundfile` in *both* models.
- Models at `/workspace/models/{Shrutam-2,sooktam2}`; HF cache at `/root/hf_cache`.
- Server: `uvicorn server:app --host 0.0.0.0 --port 8000`.

> Note: `requirements.txt` and the older `setup.sh` still reflect earlier
> (pre-Blackwell) assumptions. **`pod_setup.sh` is the source of truth** for a
> working environment.

---

## 7. How to run it again from scratch

A terminated pod deletes everything (including `/workspace`). To bring it back up:

1. Launch a new RunPod pod with a Blackwell GPU (e.g. RTX PRO 4500). **Expose
   HTTP ports `8888,8000` at creation** (avoids a later restart).
2. In the pod terminal:
   ```bash
   cd /workspace
   git clone https://github.com/makarkul/bharatgen-speech-intern-2026.git
   cd bharatgen-speech-intern-2026 && git switch week3
   ```
3. Download the model weights (delete `.git` dirs after cloning each model repo;
   re-fetch the big files with `hf_hub_download` and verify byte sizes — see §5.3).
   *(Future improvement: fold this into `pod_setup.sh` for a true one-command setup.)*
4. Replay the environment + start the server:
   ```bash
   bash /workspace/bharatgen-speech-intern-2026/week3/webapp/pod_setup.sh
   ```
5. Wait for `Uvicorn running on http://0.0.0.0:8000` in `/workspace/server.log`.
6. In `index.html` on your laptop, set `API_BASE` to the new pod's proxy URL
   (`https://<pod-id>-8000.proxy.runpod.net`), open the page, and test.

**Always stop or terminate the pod when done — the GPU bills per hour.**

---

## 8. One-sentence summary

> "I built a web app where you speak into the browser, a speech-recognition model
> (Shrutam-2) transcribes it, and a text-to-speech model (Sooktam-2) speaks it
> back in a chosen voice — with a FastAPI backend running both models on a rented
> GPU. Along the way I solved real deployment issues: GPU/driver (Blackwell)
> compatibility, disk limits and corrupted downloads, an audio library that
> dropped its loader, browser audio-format conversion, and network port exposure."
