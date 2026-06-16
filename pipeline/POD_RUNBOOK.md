# Fresh Pod Runbook — Speech Translation Pipeline

Step-by-step to bring the pipeline up on a **brand-new** RunPod pod with a
persistent network volume. Written for "from scratch" — nothing is assumed to
already exist on the pod.

> First run downloads ~35GB+ of weights (Shrutam 5GB + Sooktam 5GB + Param-2
> ~34GB) onto the volume. Slow ONCE; reused every session after.

---

## 0. One-time: the network volume (DONE)

- Network volume created: **`distinct_olive_platypus`, 100GB, region Japan (AP-JP-1)**.
- Costs ~$7/mo max (pro-rated; delete it to stop charges). Persists across pod
  stop/terminate — that's what saves the re-download.

## 1. Create the pod (in the SAME region as the volume)

| Setting | Value | Why |
|---|---|---|
| **Region** | **Japan / AP-JP-1** | MUST match the volume's region or it won't attach |
| **Network volume** | `distinct_olive_platypus` | mounts at `/workspace` |
| **GPU** | **H100 SXM (80GB)** | fits all 3 models (Shrutam ~8 + Sooktam ~7 + Param-2 17B) |
| **Template** | a PyTorch template | gives Python + CUDA base |
| **Expose HTTP port** | **8000** | ⚠️ set at CREATION — RunPod only proxies ports declared up front. Adding it later restarts the pod (wipes ephemeral disk). |

Port 8001 (Param service) does NOT need exposing — the main server reaches it
internally via localhost.

## 2. Get the code (in the pod's web terminal)

```bash
cd /workspace
git clone https://github.com/<your-repo>/bharatgen-speech-intern-2026.git
cd bharatgen-speech-intern-2026
git checkout week3        # or whichever branch has pipeline/
```

(If the repo is already on the volume from a past session, just `git pull`.)

## 3. Bring everything up — one command

```bash
bash /workspace/bharatgen-speech-intern-2026/pipeline/pod_setup.sh
```

This (idempotently):
1. Downloads + verifies the ASR/TTS weights (Shrutam, Sooktam) onto `/workspace`.
2. Installs ffmpeg + probes the GPU, installing cu128 torch only if needed.
3. Installs main-stack deps (transformers 4.56.2) + applies the 4 source patches.
4. Starts the MAIN server (ASR+TTS) on :8000.
5. Creates the Param-2 venv (transformers 4.52.3) + starts the Param service on :8001.
6. Polls until Param-2 reports loaded — prints **PASS** or **WARN**.

`HF_HOME=/workspace/hf_cache` so the big Param-2 download lands on the volume
(persists), NOT ephemeral disk.

## 4. Watch it come up

```bash
tail -f /workspace/server.log /workspace/param_service.log
```
- Main server ready: `Uvicorn running on http://0.0.0.0:8000`
- Param-2 ready: `>>> Param-2 loaded. Ready.` (or a loud FAIL line if it can't)

## 5. Sanity-check the translator directly

```bash
curl -s http://localhost:8001/health          # {"ok":true,"loaded":true}
curl -s -X POST http://localhost:8001/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"नमस्ते, आप कैसे हैं?","src":"hindi","tgt":"tamil"}'
```

## 6. Use the web app

- Find the pod's :8000 proxy URL (RunPod shows it, like
  `https://<pod-id>-8000.proxy.runpod.net`).
- Edit `pipeline/index.html` → set `API_BASE` to that URL.
- Open `index.html` locally in a browser. Pick "Speak in" + "Translate to",
  record, and you should hear the translated reply.

## 7. WHEN DONE — stop the GPU

```bash
# stopping the pod stops the big GPU cost. The volume (and all weights) persist.
```
- **Stop/terminate the pod** in the RunPod console → GPU billing stops.
- **Leave the volume** → weights stay; next session skips the 35GB download.
- The GPU (several $/hr) is the cost to watch; the volume is pennies/day.

---

## Things most likely to go wrong (first real run)

1. **Param-2 fails to load** (transformers 4.52.3 + trust_remote_code on this
   GPU). Check `param_service.log`. This is genuinely untested — expect to debug.
2. **OOM** if somehow on a smaller GPU — H100 80GB should be fine for all three.
3. **`<think>` leakage** in translations — `param_service.py` strips it; verify
   the output is clean, adjust `_strip_thinking` if not.
4. **Weights truncated** on download — `pod_setup.sh` verifies exact byte sizes
   for Shrutam/Sooktam; Param-2 goes through hf_hub_download (integrity-checked).
