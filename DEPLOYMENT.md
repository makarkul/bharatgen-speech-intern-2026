# DEPLOYMENT — Duplicate the Speech-to-Speech Translation Pipeline

This is a **complete, from-scratch** guide to standing up the BharatGen
speech-translation pipeline on your own RunPod pod. Follow it top to bottom and
you will have the same working web app: speak in one Indian language, hear it
spoken back in another.

It is written for someone who has **never seen this project**. Every step is
spelled out — including the small ones (which port to expose, where to paste the
HuggingFace token, what to click in the RunPod console). Inline notes
(`# why:`) explain *why* each thing is done the way it is, because most of these
choices encode a bug we already hit and fixed. Don't skip them — they are the
difference between "works first try" and "spend an afternoon debugging."

> **Scope:** this document covers the `pipeline/` folder only (the sibling of the
> `week*/` folders) — the integrated speech-translation pipeline. The weekly
> learning notebooks are not part of this. The **default** translator is
> **Param-2**; a faster ~1B **IndicTrans2** alternative is selectable with a single
> environment variable — see
> [§6 · Switching the translator](#switching-the-translator-param-2-or-indictrans2).

---

## 0. What this pipeline actually is

It chains **three models** into a cross-language voice translator:

```
  speech in (language X)
     │
     ▼
  Shrutam-2   (ASR: speech → text)   →  text in X
     │
     ▼
  Param-2     (translation)          →  text in Y     ← the translation step
     │
     ▼
  Sooktam-2   (TTS: text → speech)   →  speech in Y
     │
     ▼
  play it back in the browser
```

All three are BharatGen models pulled from HuggingFace. The user records audio in
the browser, it goes to the pod, runs through the chain, and a synthesized reply
in the target language plays back.

### Why TWO services (this is the central design decision)

Param-2 needs a **different version of `transformers`** than the ASR/TTS stack,
and the two versions **cannot live in the same Python process**:

| Service | Port | transformers | Holds |
|---|---|---|---|
| **main server** | `8000` | `4.56.2` | Shrutam-2 (ASR) + Sooktam-2 (TTS) |
| **Param-2 service** | `8500` | `4.52.3` | Param-2 (the translator) |

So they run as **two separate processes, each in its own Python venv**, and the
main server calls Param-2 **over HTTP** (`PARAM_URL`). The HTTP hop costs ~9 ms
(measured) — effectively free — and it keeps the verified ASR/TTS stack
completely untouched.

> **About Param-2's latency:** Param-2 is a 17B "thinking" LLM — it generates up
> to ~1024 tokens of `<think>` reasoning before the answer, so the translate step
> dominates the total time and is **highly variable** (median ~7s, p90 ~28s). The
> service strips the `<think>` block before returning. See
> [Appendix D](#appendix-d--measured-performance-what-to-expect) for the full
> profile, and [Appendix C](#appendix-c--the-models-size-storage--precision) for
> model sizes/precision.

---

## 1. Prerequisites (gather these BEFORE creating the pod)

1. **A RunPod account** with credit. The GPU is the real cost (several $/hr);
   the storage volume is pennies/day.
2. **A HuggingFace account + access token** (`hf_...`).
   - Create one at <https://huggingface.co/settings/tokens> (a *read* token is enough).
   - **Param-2 is GATED** — you must click "Agree and access" on its model page
     while logged in, or the download 401s:
     <https://huggingface.co/bharatgenai/Param2-17B-A2.4B-Thinking>
   - The ASR/TTS models (Shrutam-2, sooktam2) are **not** gated — no token needed
     for those, only for Param-2.
3. **The repo URL:** `https://github.com/makarkul/bharatgen-speech-intern-2026.git`
   The pipeline lives on the **`week3`** branch (that's where `pipeline/` is).

---

## 2. One-time: create a persistent network volume

> **Why a network volume:** the first run downloads **~44GB+ of weights**
> (Shrutam ~5GB + Sooktam ~5GB + Param-2 ~34GB). Pod *ephemeral* disk is **wiped
> on every stop/restart**. A **network volume** persists across pod
> stop/terminate, so you pay the giant download **once** and every later session
> reuses it. It also persists the built Python venvs, so a restart costs ~30s
> instead of a ~10-minute reinstall.

In the RunPod console:

1. Go to **Storage → Network Volumes → + New Network Volume**.
2. **Size:** `100 GB` (fits all three models with headroom; ~$7/mo max, pro-rated).
3. **Region:** pick a region that has **H100 80GB** GPUs — **remember it**.
   ⚠️ The pod you create later **must be in this same region** or the volume
   won't attach.
4. Name it something memorable (e.g. `speech-pipeline-vol`).

The volume mounts at **`/workspace`** inside the pod. Everything persistent in
this guide lives under `/workspace`.

---

## 3. Create the pod (in the SAME region as the volume)

In the RunPod console: **Pods → + Deploy**.

| Setting | Value | Why |
|---|---|---|
| **Region** | **same as your volume** | MUST match or the volume won't attach |
| **Network volume** | select the volume from step 2 | mounts at `/workspace` |
| **GPU** | **H100 SXM (80GB)** | Param-2 is a 17B model — it needs a big GPU. All three models together fit comfortably in 80GB. Smaller cards (24GB) will OOM on Param-2. |
| **Template** | any **PyTorch** template | gives a Python + CUDA base image |
| **Expose HTTP port** | **`8000`** | ⚠️ **set this AT CREATION** — see the box below |

> ### ⚠️ Exposing port 8000 — do this at pod CREATION, not later
>
> RunPod only proxies HTTP ports that are **declared up front**, before the pod
> starts. There's an **"Expose HTTP Ports"** field for exactly this. Where to
> find it:
>
> - **At deploy time:** in the deploy form, under the template's settings
>   (sometimes behind an **"Edit Template"** / advanced toggle), find
>   **"Expose HTTP Ports (max 10)"** and add **`8000`**.
> - **On a pod you just created** (the common case): open the pod, click
>   **"Edit Pod"**, find the **"Expose HTTP ports (max 10)"** field, and add
>   **`,8000`** to whatever is already there. The PyTorch template usually
>   pre-fills `8888` (Jupyter) — so it becomes `8888,8000`. **Comma-separate**
>   the ports; leave `8888` alone, just append `,8000`. Click **Save**.
>
> *(The `22` you'll see under "Expose TCP ports" is SSH, also from the template —
> leave it as-is. You only need `8000` under HTTP ports.)*
>
> If you forget and try to add it afterward, RunPod **restarts the pod to apply
> it**, which **wipes the ephemeral disk**. (Your `/workspace` volume survives,
> but you'll re-pay the apt installs and re-launch.) So set `8000` now.
>
> **Only `8000` needs exposing.** The Param-2 service on `8500` is reached
> *internally* by the main server over `localhost` — it never faces the internet,
> so do **not** expose it.
>
> **Never use port `8001`** — RunPod's own nginx squats on it inside the pod.
> That's exactly why the Param-2 service uses `8500`.

Once the pod is **Running**, open its **web terminal** (the "Connect" →
"Start Web Terminal", or SSH if you set that up). All commands below run **in the
pod's terminal**.

---

## 4. Put your HuggingFace token on the pod (one time)

> **Why a file, not typing it each run:** `pod_setup.sh` reads the token from
> `/workspace/.hf_token`. Because it's on the volume, you set it **once** and
> every future restart picks it up automatically. It's never committed to git.

In the pod terminal:

```bash
echo 'HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx' > /workspace/.hf_token
```

Replace `hf_xxx...` with your real token from step 1.

> Param-2 is gated, so this token is **required** — without it the Param-2 weight
> download 401s. (The ASR/TTS models aren't gated, so if you somehow only ran the
> same-language path with no translation, the token wouldn't be needed — but
> translation is the whole point.)

---

## 5. Get the code onto the volume

```bash
cd /workspace
git clone https://github.com/makarkul/bharatgen-speech-intern-2026.git
cd bharatgen-speech-intern-2026
git checkout week3        # the branch that has pipeline/
```

> **Why clone into `/workspace`:** so the repo (and the venvs/logs the setup
> script writes next to it) live on the persistent volume. If you'd already
> cloned it in a past session, just `cd /workspace/bharatgen-speech-intern-2026
> && git pull` instead.

---

## 6. Bring the whole pipeline up — ONE command

```bash
bash /workspace/bharatgen-speech-intern-2026/pipeline/pod_setup.sh
```

That's it. This single, **idempotent** script does everything and brings up
**Param-2** as the translator (the default — no flag needed).

### What `pod_setup.sh` does, step by step (and why)

It is heavily commented inline (search it for `[GOTCHA #N]`), but here's the map:

1. **Sets persistent paths.** `HF_HOME=/workspace/hf_cache` so the big HF
   downloads (including Param-2's ~34GB) land on the **volume**, not ephemeral
   disk. Models go to `/workspace/models/`, venvs to `/workspace/{main,param}_venv`.
   *(Why: same reasoning as the network volume — survive restarts.)*

2. **Sources your token** from `/workspace/.hf_token` (softly — it doesn't abort
   at this point if missing, since the ASR/TTS stack needs no token). It then
   *requires* the token on **every** Param-2 run — both the one-time venv build
   **and** every later relaunch — because the Param-2 step has a hard
   `${HF_TOKEN:?...}` guard. So if you blank/remove the token after building, a
   restart will still stop at that guard with a clear "Param-2 is gated" message.

3. **`[0/8]` Downloads + verifies the ASR/TTS weights** (Shrutam-2, Sooktam-2) if
   missing. *(Why the care: the pod template has **no git-lfs**, so a plain
   `git clone` **silently truncates** the big `.pt` files. The script installs
   git-lfs, then **re-fetches the two big checkpoints via `hf_hub_download`** —
   which is integrity-checked — and **asserts their exact byte sizes**. Skips
   entirely if they're already the right size. Param-2's weights are fetched
   later, when its venv is built, also via the integrity-checked HF path.)*

4. **`[1/8]` Installs system packages** (`ffmpeg`, `curl`, `psmisc`).
   *(Why ffmpeg: the browser records WebM/Opus audio, which `soundfile` cannot
   decode; ffmpeg transcodes it to 16kHz mono WAV. These live on ephemeral disk
   so they reinstall every run — the script handles the transient apt-lock at
   boot with a retry, then verifies ffmpeg is actually present.)*

5. **`[2/8]` Picks the right CUDA torch wheel** by reading the GPU's compute
   capability from `nvidia-smi`. Defaults to **cu124** (the H100/Hopper target);
   uses **cu128** only for a positively-detected Blackwell (sm_120+).
   *(Why: an H100 needs cu124; a too-new card needs cu128. Getting this wrong →
   "no kernel image" / silent CPU fallback.)*

6. **`[3/8]` Builds the MAIN venv** at `/workspace/main_venv` (transformers
   `4.56.2` + all ASR/TTS deps), then **force-reinstalls the matched cu124 torch
   trio (torch + torchvision + torchaudio) LAST**.
   *(Why force-reinstall, and why last: `vocos` (a TTS dep) silently drags
   torch/torchvision/torchaudio in from PyPI. Two ways that bites: (1) a
   mismatched **torchvision** → "operator torchvision::nms does not exist"; and
   (2) a PyPI-nightly **torchaudio** built for the wrong CUDA → it wants
   `libcudart.so.13`, which a cu124 pod doesn't have, so `import torchaudio`
   dies with `OSError: libcudart.so.13: cannot open shared object file` and the
   main server crashes at startup. `--force-reinstall` of the matched cu124 trio
   at the end overwrites whatever PyPI left. A `.ready` sentinel + an import
   "canary" (which imports `torchvision.ops` AND `torchaudio` to catch both
   failures here, not at server start) make the ~10-min build skippable on
   restart. There's also a **torchaudio self-heal**: if the canary's `import
   torchaudio` fails on an existing venv, the script force-reinstalls just
   torchaudio in place — recovering in seconds instead of a full rebuild.)*

7. **`[4/8]` Applies 4 source patches** to the model repos (idempotent):
   - **A** — Shrutam `torch.load(..., mmap=True)` so staging the 5GB checkpoint
     doesn't OOM system RAM.
   - **C** — Shrutam `torchaudio.load` → `soundfile` (**load-bearing**; without
     it the server crashes at import).
   - **B** — Sooktam "robotic head" fix (windows the boundary search to the first
     ~1s so a 13s output isn't catastrophically cut to ~1s).
   - **D** — Sooktam `torchaudio.load` → `soundfile` (**load-bearing**).
   - **E** — strips Shrutam's unguarded import-time demo line.
   *(Why patch at all: these are upstream model-repo quirks; the script edits the
   copies on `/workspace` and re-applies on every run. If a load-bearing patch's
   target line vanished (repo changed upstream), it **aborts loudly** instead of
   degrading to a mystery crash.)*

8. **`[5/8]` Starts the MAIN server** on `:8000` from `main_venv`, pointed at the
   Param-2 service via `PARAM_URL=http://localhost:8500`.

9. **`[6/8]` Builds + starts the Param-2 service.** It builds `param_venv`
   (`transformers==4.52.3` + accelerate + fastapi/uvicorn), then starts
   `param_service.py` on `:8500`, which **pre-loads the 17B model** at startup
   (slow once; so the first real request isn't slow and a load failure screams in
   the log immediately).

10. **`[7/8]` & `[8/8]` Waits for readiness** and prints **PASS**/**WARN** for
    each service. *(Why poll: the OLD script never checked — a dead `:8000`
    looked "successful." Now a crash-on-import is caught in seconds.)*

> First run is **slow once** (the ~34GB Param-2 download + the ~10-min main_venv
> build + the 17B model load). Every restart after is **~30s** because the venvs
> and weights persist on `/workspace`. The script is safe to re-run anytime — it
> skips what's already done.

### Switching the translator: Param-2 or IndicTrans2

The pipeline ships with **two interchangeable translators**, and which one runs is
chosen by a single environment variable — **`TRANSLATOR`** — on the bring-up
command. Nothing else changes: the main server, `server.py`, and the ASR/TTS stack
are identical either way.

| Command | Translator | Service port | HF token | Weights |
|---|---|---|---|---|
| `bash pipeline/pod_setup.sh`  *(= `TRANSLATOR=param …`)* | **Param-2** (17B — the default) | `8500` | **required** (gated) | ~34 GB |
| `TRANSLATOR=indictrans bash pipeline/pod_setup.sh` | **IndicTrans2** (~1B) | `8501` | **none** (public) | ~4 GB |

```bash
# Param-2 — the default (these two are identical):
bash /workspace/bharatgen-speech-intern-2026/pipeline/pod_setup.sh
TRANSLATOR=param bash /workspace/bharatgen-speech-intern-2026/pipeline/pod_setup.sh

# IndicTrans2 — the fast opt-in alternative:
TRANSLATOR=indictrans bash /workspace/bharatgen-speech-intern-2026/pipeline/pod_setup.sh
```

**What the flag actually does** (so it isn't magic): `pod_setup.sh` reads
`TRANSLATOR`, then in step `[5/8]` points the main server at the matching port via
`PARAM_URL` — `http://localhost:8500` for Param-2, `:8501` for IndicTrans2 — and in
`[6/8]` builds + starts that translator's service. The main server doesn't care
which one: both services expose the **identical** `POST /translate` API
(`{text,src,tgt}` → `{translation, model_seconds}`), so `server.py` needs **zero**
changes. *(Heads-up: the env var stays named `PARAM_URL` even when it points at
IndicTrans2 — it's just the variable `server.py` reads; the name is historical,
don't let it confuse you.)*

**Confirm which one is live** — the script announces it at the top and bottom of
its run:

```
>>> Pipeline bring-up. Translator = indictrans
...
>>> Done. Translator=indictrans (port 8501).
```

…and you can hit the service directly (mind the port — `8500` vs `8501`):

```bash
curl -s http://localhost:8500/health    # Param-2     → {"ok":true,"loaded":true}
curl -s http://localhost:8501/health    # IndicTrans2 → {"ok":true,"loaded":true}
```

**To switch when the pipeline is already up:** there is no live toggle — just
**re-run** `pod_setup.sh` with the other `TRANSLATOR` value. Step `[5/8]` kills the
running main server (`pkill -f "uvicorn server:app"`) and relaunches it pointed at
the new translator (starting that translator's service if it isn't already up).
~30s when everything's already built. *(The previously-running translator stays up
on its own port, harmlessly — the main server just stops calling it. Reclaim its
VRAM with `pkill -f "param_service:app"` or `pkill -f "indictrans_service:app"`.)*

> **# why the first switch can be slow — translator weights download at SERVICE
> start, not in `[0/8]`.** Step `[0/8]` fetches only the ASR/TTS checkpoints
> (Shrutam + Sooktam). Each translator's weights come down the **first time its
> service loads them**, into `HF_HOME=/workspace/hf_cache` — visible in the
> translator log, **not** the `[0/8]` setup output. So the first switch to a
> translator with an empty cache pays its download: **~34 GB for Param-2** (gated;
> token required; needs the disk headroom) or **~4 GB for IndicTrans2** (public).
> Once cached on the volume, every later switch is instant. *(Log location: Param-2
> and an already-built IndicTrans2 relaunch log to `/workspace/translator.log`;
> IndicTrans2's very first build is started by `indictrans_setup.sh`, which logs to
> `/workspace/indictrans_service.log`.)*

**Which to pick:**

- **Param-2** — the project's primary / reference translator: a 17B "thinking"
  LLM. **Slow and highly variable** (profiled translate median ~7s, p90 ~28s; ~60s
  cold start) but it is the reference model. Needs the gated HF token and an
  H100 80GB.
- **IndicTrans2** — a dedicated ~1B translation model. On the same pod and corpus
  it is **~20× faster on the translate step and ~3× faster end-to-end**
  (cross-language total median **4.1s vs 13.3s**), with **near-constant** latency
  (profiled translate ~0.4–0.5s warm, p90 ~0.7s — no blow-ups) and **no ~60s cold
  start** (first request ~3s total). Public (no token), ~4 GB. Full head-to-head:
  `pipeline/profiling_results_indictrans/REPORT.md` vs
  `pipeline/profiling_results/REPORT.md`. *(Speed is settled; a translation
  **quality** comparison vs Param-2 is the open question — IndicTrans2 already wins
  decisively on latency.)*

**Per-direction gotchas:**

| Switching to | Watch for |
|---|---|
| **Param-2** | `HF_TOKEN` must be in `/workspace/.hf_token` — the script hard-aborts at `[6/8]` with "Param-2 is gated" without it. First load re-downloads ~34 GB if the cache was cleared (needs the free space), and it needs an H100 80GB or it OOMs. |
| **IndicTrans2** | First use builds `indictrans_venv` by delegating to `indictrans_setup.sh` (one-time, a few minutes). No token, no gate. If that venv ever breaks: `rm -rf /workspace/indictrans_venv && bash pipeline/indictrans_setup.sh`. |

---

## 7. Watch it come up

```bash
tail -f /workspace/server.log /workspace/translator.log
```

Ready signals:
- **Main server:** `Uvicorn running on http://0.0.0.0:8000` and
  `>>> Shrutam-2 loaded. Ready.`
- **Param-2 service:** `>>> Param-2 loaded. Ready.` (or a loud FAIL line if it
  can't load).

`Ctrl-C` to stop tailing (it does **not** stop the servers — they run via `nohup`
in the background).

---

## 8. Sanity-check the translator directly (before touching the browser)

This confirms Param-2 works in isolation, with no ASR/TTS involved:

```bash
curl -s http://localhost:8500/health
# expect: {"ok":true,"loaded":true}

curl -s -X POST http://localhost:8500/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"नमस्ते, आप कैसे हैं?","src":"hindi","tgt":"tamil"}'
# expect: {"translation":"<Tamil text>","model_seconds":<float>}
```

If `loaded` is `false` or the translate errors, check
`/workspace/translator.log` — almost always a missing/invalid HF token or the
model gate not accepted (revisit steps 1 and 4), or an OOM if you're not on an
H100 80GB.

---

## 9. Use the web app

1. In the RunPod console, find the pod's **`:8000` proxy URL**. RunPod shows it
   under the pod's "Connect" panel — it looks like:
   `https://<pod-id>-8000.proxy.runpod.net`
2. **Open that URL directly in your browser.** The main server *serves the page
   itself*, so you don't edit any file — just visit the proxy URL.
   > **Why no config edit:** `index.html` uses `API_BASE = ""` (same origin), so
   > the page calls `/speak` on whatever host served it — the proxy URL on the
   > pod.
3. Pick **"Speak in"** and **"Translate to"**, click **Start recording**, speak,
   click **Stop**. After a moment you'll see the transcript, the translation, and
   hear the synthesized reply. A per-stage **timing breakdown** appears below.

> **Microphone needs HTTPS:** browsers only allow mic access on a secure origin.
> The RunPod `https://...proxy.runpod.net` URL satisfies this. (Opening a raw
> `http://<ip>:8000` would block the mic.)
>
> **First request is a cold start (~60s).** Param-2 pays a one-time warmup on the
> very first translate after a boot. Do one throwaway recording first if you're
> demoing live; every request after is warm.

---

## 10. WHEN DONE — stop the GPU to stop the bill

```text
In the RunPod console: STOP (or terminate) the pod.
```

- **Stop/terminate the pod** → the expensive GPU billing stops.
- **Leave the network volume** → all weights + venvs persist; the next session
  skips the giant download and the ~10-min build (restart ≈ 30s).
- The GPU (several $/hr) is the cost to watch. The volume is pennies/day.

To resume later: start a pod again **in the same region, with the same volume and
port 8000 exposed**, then just re-run `bash .../pipeline/pod_setup.sh` (it'll skip
everything already on the volume and relaunch in ~30s).

---

## Appendix A — Files in `pipeline/`

| File | Purpose |
|---|---|
| `pod_setup.sh` | **The one command.** Brings up the whole pipeline (Param-2 by default); encodes every fix as `[GOTCHA #N]`. |
| `server.py` | Main FastAPI backend: ASR + translate-over-HTTP + TTS. One endpoint, `POST /speak`. |
| `param_service.py` | Standalone Param-2 translation service (`POST /translate`). Own venv (`transformers==4.52.3`). |
| `index.html` | Browser frontend: "Speak in → Translate to", records, shows transcript + translation, plays the reply, shows timings. |
| `voices/` | Per-language reference voices for Sooktam-2 (see Appendix B). |
| `profile_pipeline.py` | Batch latency profiler — POSTs a benchmark corpus through `/speak`, writes `profiling_results/REPORT.md`. |
| `profiling_results/` | The committed latency report (REPORT.md + raw CSV + environment.json). |
| `requirements.txt` | The main ASR/TTS stack's pinned deps (installed into `main_venv`). |
| `setup.sh` | **Older** single-service setup (home dir, no venvs). Superseded by `pod_setup.sh`; kept for reference. |

> **Footnote — the IndicTrans2 alternative translator.** The repo also contains
> `indictrans_setup.sh`, `indictrans_service.py`, and a `_comparison_later/`
> harness — the faster ~1B IndicTrans2 MT model, which runs as an alternative
> translator on its own venv/port (`:8501`). Param-2 is the default; to run the
> pipeline on IndicTrans2 instead, pass `TRANSLATOR=indictrans` to `pod_setup.sh`
> — full instructions in
> [§6 · Switching the translator](#switching-the-translator-param-2-or-indictrans2).

---

## Appendix B — The reference voices (how the output voice is chosen)

Sooktam-2 is a **voice-cloning** TTS: it clones a short reference clip's
voice/accent and speaks the new text in it. Cloning a **Hindi** reference while
generating **Tamil** sounds wrong (a Hindi accent on Tamil), so the server keeps
**one reference clip per language** and auto-picks the one matching the **target
(output) language**.

- Native reference clips exist for **hindi, marathi, tamil** (committed under
  `pipeline/voices/ref_<lang>.wav`, each paired with its exact transcript in
  `server.py`'s `LANG_VOICES`).
- The other 9 supported languages **fall back to the Hindi clip** (functional,
  just Hindi-accented) until native clips are added.
- **To add a native voice:** drop a clean ~9s `ref_<lang>.wav` into
  `pipeline/voices/` and add its entry (with the **exact** transcript of what's
  said) to `LANG_VOICES` in `server.py`. The transcript must match the audio or
  cloning degrades.

**Supported languages (all 12, Indic only):** hindi, marathi, tamil, telugu,
malayalam, kannada, odia, bengali, urdu, assamese, gujarati, punjabi.
*(Sooktam-2 speaks Indic languages only — English and other non-Indic targets are
rejected with a clear error, since there's no Sooktam voice for them.)*

---

## Appendix C — The models (size, storage & precision)

What's actually running in the pipeline, and how big each piece is. Sizes drive
the GPU/volume requirements; the precision column explains why a "small" model
can still take a lot of disk (FP32 is 4 bytes/param, BF16 is 2).

| Model | Role | Parameters | On-disk size | Precision |
|---|---|---|---|---|
| **Shrutam-2** | ASR (speech → text) | ~1.2B (Conformer encoder + 8-expert MoE + ~1.1B Llama decoder) | ~5 GB | Mixed (BF16 decoder, FP32 encoder) |
| **Param-2** | Translation (LLM) | 17B total / 2.4B active (MoE) | ~34 GB | BF16 |
| **IndicTrans2** *(opt-in)* | Translation (MT) | ~1B | ~4–5 GB | FP32 |
| **Sooktam-2** | TTS (text → speech) | 336M | ~5 GB | FP32 |

Notes (verified against the code and the model checkpoints):

- **Param-2** (`bharatgenai/Param2-17B-A2.4B-Thinking`) is the only true BF16
  model. The name encodes it: **17B total** parameters but only **2.4B active**
  per token (it's a Mixture-of-Experts model). At BF16 that's ~34 GB on disk
  (17B × 2 bytes), and `param_service.py` loads it with
  `torch_dtype=torch.bfloat16`. This is the heavy piece — it's why the pipeline
  needs an H100 80GB.
- **Sooktam-2** (`bharatgenai/sooktam2`) was measured exactly by loading the
  checkpoint: **336,071,814 params, all FP32**. That's why a 336M model is ~5 GB
  on disk — FP32 weights plus the checkpoint also carries EMA + optimizer state.
- **Shrutam-2** (`bharatgenai/Shrutam-2`) is **mixed precision** — its Llama
  decoder config is `bfloat16`, but the Conformer encoder/MoE projector load in
  FP32. Its ~5 GB checkpoint / ~8 GB GPU footprint matches that.
- **IndicTrans2** (`ai4bharat/indictrans2-indic-indic-1B`) loads with **no
  dtype override**, so it defaults to **FP32**. At ~1B params it's tiny next to
  Param-2 — that's exactly why it exists as the fast opt-in alternative. *(Only
  relevant if you opt into `TRANSLATOR=indictrans`; see Appendix A.)*

> **Why these add up to needing 80GB:** loaded together on the GPU, Param-2 (the
> 17B BF16 model, ~34 GB) dominates; Shrutam-2 (~8 GB) and Sooktam-2 (a few GB)
> sit alongside it. An H100 80GB holds all three with comfortable headroom.

---

## Appendix D — Measured performance (what to expect)

From `pipeline/profiling_results/REPORT.md` (24 warm runs, one H100, Param-2
translator). These are single-request latencies, not throughput:

| Stage | Median | p90 | % of total |
|---|---|---|---|
| Transcode (ffmpeg) | 0.1s | 0.1s | 1% |
| ASR (Shrutam-2) | 0.56s | 0.9s | 5% |
| **Translate (Param-2)** | **6.88s** | **20.9s** | **67%** |
| TTS (Sooktam-2) | 2.73s | 3.5s | 27% |
| **TOTAL** | **10.25s** | **27s** | 100% |

Key takeaways:
- **Param-2 translation dominates** (~67% of latency) and is **highly variable** —
  6.8s to 35s on similar inputs, because it's a "thinking" model that spends a
  variable number of reasoning tokens before answering. Input length does *not*
  predict latency. Plan for the p90, not just the median, in a live demo.
- The two-service HTTP hop is **~9 ms** — the architecture costs nothing; the cost
  is the 17B model.
- **Cold start ~60s:** the very first request after a boot pays a one-time warmup.
  Send one throwaway request first for a live demo. *(Param-2 pre-loads at service
  startup, so once the log says "Ready," requests are warm.)*
- A **same-language** request skips translation entirely → ~3.3s total.

To re-run the profiler on your own pod (both services must be up). The benchmark
corpus **ships with the repo** at `pipeline/benchmark_data/`, so point `--data`
at it — the script's default (`~/datasets`) is empty on a fresh pod and would
exit with "No clips found":

```bash
/workspace/main_venv/bin/python \
  /workspace/bharatgen-speech-intern-2026/pipeline/profile_pipeline.py \
  --data /workspace/bharatgen-speech-intern-2026/pipeline/benchmark_data
```

---

## Appendix E — Troubleshooting (the things most likely to bite)

| Symptom | Cause / fix |
|---|---|
| `uvicorn: command not found` / dead `:8000` after restart | Old failure mode (bare install into ephemeral python). The current `pod_setup.sh` uses `/workspace/main_venv` — make sure you're running the committed version. |
| Param-2 `loaded:false` / 401 on download | HF token missing/invalid, or you didn't click "Agree and access" on the Param-2 model page. Fix `/workspace/.hf_token` (step 4) and accept the gate (step 1). |
| **Param-2 OOM** | You're on a GPU smaller than ~40GB. Param-2 (17B) needs a big GPU — use **H100 80GB**. |
| `operator torchvision::nms does not exist` | torch/torchvision mismatch (vocos dragged in a bad torch). `pod_setup.sh` force-reinstalls the matched cu124 trio LAST and has an import canary; if you see this, delete `/workspace/main_venv/.ready` and re-run to rebuild. |
| `OSError: libcudart.so.13: cannot open shared object file` (main server "died during load") | A PyPI-nightly **torchaudio** built for CUDA 13 landed in `main_venv`, but a cu124 pod has no `libcudart.so.13`. `pod_setup.sh` force-reinstalls the cu124 torchaudio and self-heals on re-run; just **re-run `pod_setup.sh`** (the self-heal fixes it in seconds, no full rebuild). |
| Weights truncated / size mismatch | No git-lfs → silent truncation. `pod_setup.sh` re-fetches via `hf_hub_download` and asserts exact byte sizes; re-run it. |
| `fuser: command not found` | `psmisc` is ephemeral. The scripts use `pkill` (always present) instead — you shouldn't hit this with the current scripts. |
| `/speak` fails decoding the recording | ffmpeg missing (it transcodes the browser's WebM/Opus). `pod_setup.sh` verifies ffmpeg is installed; re-run it. |
| Microphone blocked in browser | You opened a non-HTTPS URL. Use the `https://...proxy.runpod.net` URL. |
| `<think>` text spoken aloud | Param-2 reasoning leaked into the output. `param_service.py` strips it via `_strip_thinking`; if output is still dirty, tighten that function. |

---

## Appendix F — Quick reference (cheat sheet)

```bash
# ── First time on a fresh pod (after creating volume + pod with :8000 exposed) ──
echo 'HF_TOKEN=hf_xxx' > /workspace/.hf_token            # your gated Param-2 token
cd /workspace
git clone https://github.com/makarkul/bharatgen-speech-intern-2026.git
cd bharatgen-speech-intern-2026 && git checkout week3
bash pipeline/pod_setup.sh                               # brings up Param-2 (the default)
# ...or run the fast IndicTrans2 translator instead (no token needed):
TRANSLATOR=indictrans bash pipeline/pod_setup.sh

# ── Watch it come up ──
tail -f /workspace/server.log /workspace/translator.log

# ── Sanity check the Param-2 service ──
curl -s http://localhost:8500/health                     # expect {"ok":true,"loaded":true}

# ── After a pod restart (volume intact) — just re-run; ~30s ──
bash /workspace/bharatgen-speech-intern-2026/pipeline/pod_setup.sh

# ── Open the app ──
# Browse to  https://<pod-id>-8000.proxy.runpod.net  (no file edits needed)
```

**Persistent locations (all on the `/workspace` volume, survive restarts):**
- `/workspace/hf_cache` — HF downloads (`HF_HOME`), including Param-2's ~34GB
- `/workspace/models/` — Shrutam-2, sooktam2 checkpoints
- `/workspace/main_venv`, `/workspace/param_venv` — the two venvs
- `/workspace/.hf_token` — your HF token
- `/workspace/server.log`, `/workspace/translator.log` — service logs
