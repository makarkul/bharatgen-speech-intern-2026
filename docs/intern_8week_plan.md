# 8-Week Intern Plan: BharatGen Speech Translation Pipeline

**Intern profile:** 2nd-year CS undergrad, comfortable in Python, no prior AI/ML.
**Structure:** Week 0 is a 5-working-day Python+ML bootstrap. Weeks 1–7 are the working project.
**End state:** Working `STT → Translate → TTS` demo using BharatGen Shrutam-2 + AI4Bharat IndicTrans2 + BharatGen Sooktam-2, with a handoff package that the next intern can pick up cold.

---

## Guiding principles

1. **Build first, theorize second.** He doesn't need to derive attention to call `model.generate()`. Theory is introduced *just-in-time* when it explains something he's already seen break.
2. **One model per week, one deliverable per week.** No skipping. No "I'll write the README later."
3. **The inference stack is taught by profiling, not lecturing.** Memory and latency measurements are part of every weekly deliverable from Week 3 onward.
4. **Limitations are not afterthoughts.** Every model exercise ends with "where does this fail?" written down.

---

## Week 0 — Python + ML Bootstrap (5 working days)

**Goal:** Close the gap between "knows Python from coursework" and "can productively use PyTorch, Hugging Face, and Jupyter without panic." End-state: he can load a pre-trained HF model, run inference, and tell you what a tensor, a tokenizer, and a model are in his own words.

**Before Day 1 (async, ~2 hours total over a weekend):**
- HF account + accept terms on `bharatgenai/Shrutam-2`, `bharatgenai/sooktam2`, `ai4bharat/indictrans2-indic-en-1B` and `en-indic-1B`. Generate HF access token.
- GPU access: **RunPod Pod with persistent volume**. Recommend RTX 4090 (24 GB, cheaper) or A40 (48 GB, headroom for Week 6 multi-model). Note pod ID, volume ID, GPU type — documented in repo README so benchmarks across weeks are comparable.
- **Critical**: set `export HF_HOME=/workspace/.cache/huggingface` (or wherever your persistent volume is mounted) and add to `~/.bashrc`. Default HF cache is on container ephemeral storage and gets wiped on pod restart — without this you re-download IndicTrans2-1B + Shrutam-2 + Sooktam-2 every cold start.
- Repo created: `bharatgen-speech-intern-2026` with `week0/` through `week7/` folders.
- Tooling: Python 3.10, `uv` or `conda`, VS Code + Python + Jupyter extensions, `ffmpeg`, `jupyter lab`.
- Pre-reading (≤3 hours): AI4Bharat blog on IndicTTS/IndicASR + skim the Shrutam-2 and Sooktam-2 model cards. No notes; goal is vocabulary exposure.

**Day 1 — numpy / Jupyter / venv hygiene**
- numpy: arrays, `.shape`, `.dtype`, broadcasting rules.
- Jupyter: cells, kernels, kernel state, `%time`, `%matplotlib inline`.
- 10 small numpy exercises — predict the output `.shape` *before* running each cell.
- Create a fresh venv, install numpy/matplotlib/jupyter, freeze a `requirements.txt`.

> **Resources:** [SciPy Lectures: NumPy](https://lectures.scientific-python.org/intro/numpy/index.html) (2h, primary) · [Real Python: Jupyter Intro](https://realpython.com/jupyter-notebook-introduction/) (30m) · [Astral uv: Getting Started](https://docs.astral.sh/uv/getting-started/) (20m)

**Day 2 — matplotlib + reading library docs**
- `fig, ax = plt.subplots()`. `plot()`, `scatter()`, `hist()`, labels and titles.
- Plot a sine wave, a 1000-sample Gaussian histogram, a 2-variable scatter. Each labeled.
- Read the librosa "Core IO and DSP" page end-to-end. Try one example from each section. (Reading docs is *the* skill; we practice it on a library he'll use in Week 1.)

> **Resources:** [Matplotlib: Pyplot tutorial](https://matplotlib.org/stable/tutorials/pyplot.html) (45m) · [librosa: Tutorial](https://librosa.org/doc/latest/tutorial.html) (1h)

**Day 3 — ML concepts at 30,000 feet (concepts only, no code)**
- *What is a "model"?* A function with learned parameters. Input → math → output. We never train one in 7 weeks; we only run pre-trained ones.
- *What is a tensor (vs. numpy array)?* Same data structure, but lives on GPU and tracks gradients (we won't use gradients). For us: numpy array on a graphics card.
- *What is a tokenizer?* Converts text → list of integers (tokens). Each model has its own. "नमस्ते" might be 1 token or 5 depending on the tokenizer.
- *Training vs. inference.* We do inference only. Knowing what we're NOT doing scopes the work.
- *GPU vs. CPU intuition.* GPU = thousands of small parallel cores good at matrix multiplications. Neural networks are millions of matmuls → GPU is 10–100× faster.

> **Resources:** [Stephen Wolfram: What Is ChatGPT Doing?](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/) (1.5h, primary — read only through the "Inside ChatGPT" section; stop there) · [HF Tokenizer Summary](https://huggingface.co/docs/transformers/tokenizer_summary) (20m) · [NVIDIA: CPU vs GPU](https://blogs.nvidia.com/blog/whats-the-difference-between-a-cpu-and-a-gpu/) (15m)

**Day 4 — Neural network intuition**
- Watch 3Blue1Brown chapters 1–3: "But what IS a Neural Network?", "Gradient descent", "What does backpropagation really do?". ~60 min total. **Watch, don't pause to implement.**
- Write a 1-page reflection in his own words: *What is a neural network? What does training mean? Why do we need data?* **This is the bootstrap diagnostic** — a vague reflection means Week 1 needs to slow down.
- Read the first 1/3 of Jay Alammar's "Illustrated Transformer" for a one-page-equivalent primer on attention and transformers.

> **Resources:** [3Blue1Brown: Neural Networks Ch 1-3](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (60m, primary) · [Jay Alammar: Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (30m, first 1/3 only)

**Day 5 — First contact with PyTorch + Hugging Face**
- `pip install torch`. Create `torch.tensor([1,2,3])`. Move `.to('cuda')`. Print `.shape`, `.dtype`, `.device`.
- Matmul two random 1000×1000 tensors on CPU vs. GPU; time both. (This makes Day 3's "why GPU" concrete.)
- Run HF distilbert via `pipeline()` — three lines, just see "model goes brrr" work. No `AutoModel` yet; that's Week 2.
- First commit: `week0_bootstrap.ipynb` to the repo.

> **Resources:** [PyTorch: Tensors tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) (30m — **stop here, do not continue to Autograd**) · [HF LLM Course Ch 1.3](https://huggingface.co/learn/llm-course/chapter1/3) (45m)

**Deliverable:** `week0_bootstrap.ipynb` covering all Day 1–5 exercises + `reflection.md` (the 1-page Day 4 reflection).

**Anti-patterns:**
- Implementing a neural network from scratch in numpy. (Conceptual coverage, not building.)
- Reading PyTorch tutorials past the "Tensors" section. (Autograd / training tutorials muddle the mental model — we don't train.)
- Stressing over 3Blue1Brown's gradient-descent algebra. (Watch for intuition; math doesn't gate Week 1.)

**Bootstrap diagnostic:** At end of Week 0, the mentor reads `reflection.md` and skims the notebook. If the reflection is vague or the notebook is sparse, slow Week 1 down by half a day to firm up Day 3 concepts. If specific and complete, proceed at planned pace.

---

## Week 1 — Audio and tensors

**Goal:** He understands what audio looks like to a model, and he can move tensors around without panicking.

> **Primary resource:** [HF Audio Course Unit 1: Working with audio data](https://huggingface.co/learn/audio-course/chapter1/introduction) (2h) — this unit is the curriculum for the week.

**Concepts (lean):** sampling rate, mono vs stereo, waveform vs spectrogram, mel scale, what a tensor is, CPU vs GPU placement. No backprop, no loss functions, no architectures.

**Hands-on:**
- Record 5 audio clips on his phone in different languages, load each with `librosa` and `torchaudio`.
- Plot waveform and mel-spectrogram side by side. Zoom in on silence vs speech regions.
- Resample 44.1 kHz → 16 kHz. Listen to both. Convince himself why ASR models want 16 kHz.
- Tiny PyTorch warm-up: create a random tensor, move it to GPU, do a matmul, print shape and dtype.

**Deliverable:** `week1_audio_basics.ipynb` with plots, his own annotations, and a 1-page note `audio_cheatsheet.md` for the next intern.

**Anti-patterns:** Reading three Stanford CS224S lectures. Trying to "understand" Fourier transforms before plotting one.

---

## Week 2 — Hugging Face mental model

**Goal:** He can load *any* HF model, run inference, and explain what a tokenizer, a config, and a checkpoint are.

> **Primary resources:** [HF LLM Course Ch 2: Using Transformers](https://huggingface.co/learn/llm-course/chapter2/1) (2-3h, primary — AutoModel/Tokenizer/Processor + pipeline internals) · [HF Audio Course Unit 2](https://huggingface.co/learn/audio-course/chapter2/introduction) (1h) · [HF Audio Course Unit 3: Transformer architectures for audio](https://huggingface.co/learn/audio-course/chapter3/introduction) (supplementary).

**Concepts (lean):** `AutoModel` / `AutoTokenizer` / `AutoProcessor`, what `trust_remote_code=True` actually means (and the security implication), `from_pretrained` caching, `model.eval()` vs `model.train()`, `torch.no_grad()`.

**Hands-on:**
- Run `distilbert-base-uncased-finetuned-sst-2-english` on 10 sentences. Print the logits, softmax them manually.
- Run `openai/whisper-tiny` on his Week 1 audio clips. Get the transcription out.
- Inspect what got downloaded into `~/.cache/huggingface/`. Open the `config.json`. Open the tokenizer files. He should be able to *describe* what each file does.

**Deliverable:** `how_to_run_an_hf_model.md` — a tutorial written *for the next intern*, not for himself. Code snippets, screenshots of cache dir, explanation of `pipeline()` vs manual loading.

**Anti-patterns:** Fine-tuning. Pip-installing 14 things. Touching `accelerate` or `bitsandbytes` yet.

---

## Week 3 — Shrutam-2 (ASR)

**Goal:** BharatGen's ASR runs reliably on his machine, he can quantify how good it is and where it breaks.

> **Primary resources:** [HF Audio Course Unit 5: ASR](https://huggingface.co/learn/audio-course/chapter5/introduction) (2-3h) · [jiwer library](https://github.com/jitsi/jiwer) (for WER computation, 15m) · [Whisper paper](https://arxiv.org/abs/2212.04356) (skim, supplementary).

**Concepts (lean):** What ASR is, what WER (word error rate) and CER are, why code-mixing is hard, beam search vs greedy decoding (1 paragraph each, no implementation).

**Hands-on:**
- Get `bharatgenai/Shrutam-2` running on 10 Hindi clips, 10 Tamil clips, 5 code-mixed Hindi-English clips. Use real clips — YouTube interviews, podcast snippets, his own recordings — not just the model card examples.
- Compute WER against ground-truth transcripts using `jiwer`.
- Run `openai/whisper-large-v3` on the same clips. Compare WER per language.
- **Profiling (introduce here):** measure end-to-end latency, GPU memory peak (`torch.cuda.max_memory_allocated()`), and tokens/sec. Tabulate.

**Deliverable:** `week3_shrutam2_eval.md` with: WER table (Shrutam-2 vs Whisper per language), latency/memory table, 5 transcription failures with his hypothesis on *why* each failed.

**Anti-patterns:** Trying to "improve" the model. Confusing audio length with model latency. Reporting numbers without describing the hardware they were measured on.

---

## Week 4 — Sooktam-2 (TTS)

**Goal:** TTS pipeline runs, he understands reference-guided voice cloning, and he has opinions on output quality.

> **Primary resource:** [HF Audio Course Unit 6: From text to speech](https://huggingface.co/learn/audio-course/chapter6/introduction) (2h) — TTS architectures, vocoders, evaluation.

**Concepts (lean):** Why TTS is harder than ASR for evaluation (no single ground truth), what "reference-guided" means, vocoder vs acoustic model (1-paragraph distinction).

**Hands-on:**
- Run `bharatgenai/sooktam2` with `trust_remote_code=True`. Supply a reference WAV (3–10 sec, clean) + reference transcript + target text.
- Generate the same sentence in Hindi, Marathi, Tamil with three different reference speakers. Listen carefully.
- Try edge cases: very long target text, English loanwords inside Hindi, numbers ("2026", "₹450"), code-mixed input. Document what breaks.
- Compare with `ai4bharat/indic-parler-tts` on the same Hindi sentences — different paradigm (description-conditioned vs reference-conditioned), worth contrasting.
- Same profiling table as Week 3: latency, GPU memory, real-time factor (RTF = audio duration / generation time).

**Deliverable:** `week4_sooktam2_eval.md` with audio samples committed (or linked), failure cases, latency/RTF table, and a short "Sooktam-2 vs Indic-Parler-TTS" comparison.

**Anti-patterns:** Subjective "this sounds good" without samples. Forgetting that the reference clip's quality bounds the output quality.

---

## Week 5 — Translation (IndicTrans2)

**Goal:** Text-to-text translation between Indic languages and English working cleanly, plus a clear mental model of the encoder-decoder pattern.

> **Primary resources:** [HF LLM Course Ch 7.4: Translation](https://huggingface.co/learn/llm-course/chapter7/4) (1.5h) · [Jay Alammar: Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (full read this time, 1h — encoder/decoder/seq2seq detail) · [IndicTrans2 paper](https://arxiv.org/abs/2305.16307) (skim Sections 1-3).

**Concepts (lean):** NMT vs ASR/TTS (both are seq2seq but operate on different modalities), why encoder-decoder beats encoder-only for translation, beam search (now implement it conceptually — set `num_beams=5` and see what changes).

**Hands-on:**
- Run `ai4bharat/indictrans2-indic-en-1B` and `ai4bharat/indictrans2-en-indic-1B` on 30 sample sentences across 4 languages.
- Try domain edge cases: technical/legal Hindi, very casual WhatsApp-style text, sentences with numbers and dates, code-mixed input.
- Bidirectional sanity check: translate Hindi→English→Hindi, compare to original. Where does meaning drift?
- Same profiling: latency, GPU memory, tokens/sec.

**Deliverable:** `week5_indictrans2_eval.md` + a small CLI: `translate.py --src hi --tgt en --text "..."`.

**Anti-patterns:** Treating BLEU as the only metric. Forgetting that translation quality degrades with sentence length.

---

## Week 6 — Pipeline integration

**Goal:** The three models are stitched into one working `STT → Translate → TTS` demo.

> **Primary resources:** [HF Audio Course Unit 7: Putting it all together](https://huggingface.co/learn/audio-course/chapter7/introduction) (2h — literally the speech-to-speech pipeline structure) · [Gradio Quickstart](https://www.gradio.app/guides/quickstart) (45m) · [PyTorch: CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html) (30m, GPU memory management).

**Concepts (lean):** Pipeline architecture, where state lives (audio buffers, intermediate text), failure modes that only appear when models compose.

**Hands-on:**
- Wire Shrutam-2 + IndicTrans2 + Sooktam-2 together. Input: a Hindi audio clip. Output: an English audio clip (or Hindi → Tamil, etc.).
- Build a simple Gradio UI: file upload or mic in, language selector for source and target, audio player for output.
- Memory-plan: which models stay on GPU? Can all three fit? If not, what's the eviction strategy? This is where the inference stack becomes real.
- Handle the obvious failure modes: empty transcription, translation timeout, very long inputs. Don't try to handle every edge — pick three and handle them well.

**Deliverable:** Working Gradio demo + `architecture.md` with a block diagram (he draws it) showing data flow, model placements, memory budget, and known failure modes.

**Anti-patterns:** Streaming/realtime. WebSocket servers. Containerizing prematurely. The goal is an honest demo a human can click through, not production infra.

---

## Week 7 — Profile, document, hand off

**Goal:** Everything is captured well enough that the next intern starts at Week 6, not Week 0.

> **Primary resources:** [PyTorch: Profiler recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) (30m) · [Diátaxis framework](https://diataxis.fr/) (30m — structures the README/SETUP/EXPERIMENTS/NEXT_STEPS split).

**Hands-on:**
- **End-to-end profiling:** total pipeline latency for a 10-second Hindi → English audio task. Break it down: ASR ms, translate ms, TTS ms, overhead ms. He should be able to point to where the time goes and why.
- **Memory profiling:** peak GPU memory for each model loaded individually and concurrently. Implication for which GPU tier we actually need.
- **Limitations document:** a single `LIMITATIONS.md` covering: language coverage gaps, code-mixing failures, accent sensitivity (Sooktam-2 is reference-bounded), translation degradation patterns, latency bottlenecks, models that don't fit on a single 16 GB GPU.
- **Handoff package:**
  - `README.md` at repo root — "How to run this in 30 minutes from scratch."
  - `SETUP.md` — env, GPU, HF tokens, model downloads.
  - `EXPERIMENTS.md` — what was tried, what worked, what didn't, what to try next.
  - `NEXT_STEPS.md` — three concrete projects the next intern could pick up (e.g., streaming ASR, on-device quantization, fine-tuning Sooktam-2 on a new speaker).
- **30-minute walkthrough video** (Loom or similar) — him demoing the pipeline and pointing at the repo structure.

**Deliverable:** the complete repo, tagged `v1.0-intern-handoff`.

---

## Weekly cadence

| When | What | Duration |
|---|---|---|
| Monday morning | He posts the week's plan in a thread; you react with corrections | async, 15 min |
| Wed mid-week | Quick blocker check — Slack or 10-min call | 10 min |
| Friday EOD | He commits the week's deliverable + writes a 5-bullet progress note | async |
| Friday next-day | You review the commit, leave PR-style comments, schedule any Monday correction | 30 min |

No daily standups. He's learning to work asynchronously, and you're not a manager.

---

## Things to deliberately *not* do

- No fine-tuning. Not even LoRA. Inference-only for 7 weeks.
- No training-from-scratch toy models. He doesn't need a CIFAR detour.
- No deep transformer math. If he asks, point him at 3blue1brown's videos *outside* work hours.
- No vLLM / TensorRT-LLM / quantization. Those are interesting but Week 8+ territory.
- No real-time streaming. The demo is request-response.
- No Docker until Week 6, and only if he asks.

---

## What "inference stack understanding" looks like by Week 7

If by the end he can answer these without hedging, the plan worked:

1. For a 10-second Hindi audio file end-to-end translated to English speech, where does the time actually go? Give it to me in milliseconds per stage.
2. Why can't we run all three models on a 16 GB GPU simultaneously without something giving?
3. What's the difference between how Shrutam-2 and Whisper handle code-mixed Hindi-English, based on your measurements?
4. Sooktam-2 is reference-guided. What does that mean for output quality, and what does it mean for inference latency compared to a description-conditioned model like Indic Parler-TTS?
5. If I gave you a 30-minute audio file, what breaks and why?

Those are the questions worth designing the whole 7 weeks around.
