# SETUP

One-time setup before Week 0 Day 1. Budget ~2 hours.

## 1. Hugging Face

1. Create an HF account at https://huggingface.co.
2. Accept terms on the gated model pages:
   - https://huggingface.co/bharatgenai/Shrutam-2
   - https://huggingface.co/bharatgenai/sooktam2
   - https://huggingface.co/ai4bharat/indictrans2-indic-en-1B
   - https://huggingface.co/ai4bharat/indictrans2-en-indic-1B
3. Generate an access token (Settings → Access Tokens → "read" scope is enough).
4. On the pod: `huggingface-cli login` and paste the token. Or `export HF_TOKEN=hf_...` in `~/.bashrc`.

## 2. GPU environment — RunPod

- Create a RunPod account; spin up a **Pod** with a **persistent volume** attached (80–100 GB recommended).
- GPU template: **RTX 4090** (24 GB, cheaper) or **A40** (48 GB, headroom for Week 6 multi-model).
- Record the pod ID, volume ID, and GPU type in `EXPERIMENTS.md` (created in Week 7). Benchmarks across weeks are only comparable on the same hardware.

## 3. HF cache on persistent volume — critical

The default HF cache (`~/.cache/huggingface`) lives on the container's ephemeral storage and is wiped on every pod restart. Without redirecting it, you re-download IndicTrans2-1B + Shrutam-2 + Sooktam-2 on every cold start (~30 min wasted each time).

```bash
echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
source ~/.bashrc
mkdir -p "$HF_HOME"
```

Replace `/workspace` with wherever your persistent volume is actually mounted. Verify by loading a tiny model and confirming files land on the volume, not on `/`.

## 4. System tooling

```bash
apt-get update && apt-get install -y ffmpeg git
```

`ffmpeg` is a hidden dependency of `librosa`, `torchaudio`, and `soundfile`. Installing it now prevents a Day-1 audio-loading bug that looks unrelated.

## 5. Python environment

Python 3.10. Use `uv` (preferred) or `conda`:

```bash
# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10 .venv
source .venv/bin/activate
```

Then in Week 0 install only what each day needs (numpy, matplotlib, jupyter, then torch + transformers). Freeze a `requirements.txt` per week so the environment is reproducible.

## 6. Editor

VS Code with Python + Jupyter extensions, or `jupyter lab` over an SSH tunnel.

## 7. Pre-reading (≤3 hours, async, no notes)

- AI4Bharat blog on IndicTTS/IndicASR
- Skim the Shrutam-2 and Sooktam-2 model cards on Hugging Face

Goal: vocabulary exposure, not retention.

## Verification checklist

Before starting Week 0 Day 1, you should be able to:

- [ ] `python -c "import torch; print(torch.cuda.is_available())"` prints `True`
- [ ] `echo $HF_HOME` prints the path on your persistent volume
- [ ] `huggingface-cli whoami` prints your username
- [ ] `ffmpeg -version` works
- [ ] `jupyter lab` launches
