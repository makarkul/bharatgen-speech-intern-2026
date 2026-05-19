# BharatGen Speech Intern — Tracker

Source of truth: [`bharatgen_intern_tracker.xlsx`](bharatgen_intern_tracker.xlsx). This file is **generated** by `scripts/render_tracker.py`; edit the xlsx and rerun.

## Status summary

| Metric | Count |
| --- | --- |
| Total items | 98 |
| ⬜ Not Started | 69 |
| ✅ Done | 29 |
| % Complete | 29.6% |

## Project setup

| Field | Value |
| --- | --- |
| Intern name | <fill in> |
| Mentor name | <fill in> |
| Start date (Week 0 Day 1) | <YYYY-MM-DD> |
| End date (computed) |  |
| Repo URL | <fill in> |
| GPU environment | RunPod (RTX 4090 or A40, persistent volume) |
| HF token configured? | No |

## Full tracker (98 items)

GitHub renders CSV files as a sortable, searchable table — open [`tracker.csv`](tracker.csv).

## Per-week status

| Week | Done | % | 🟡 In Progress | 🔴 Blocked | ⬜ Not Started |
| --- | --- | --- | --- | --- | --- |
| Week 0 | 27/28 | 96% | 0 | 0 | 1 |
| Week 1 | 2/11 | 18% | 0 | 0 | 9 |
| Week 2 | 0/11 | 0% | 0 | 0 | 11 |
| Week 3 | 0/9 | 0% | 0 | 0 | 9 |
| Week 4 | 0/11 | 0% | 0 | 0 | 11 |
| Week 5 | 0/10 | 0% | 0 | 0 | 10 |
| Week 6 | 0/10 | 0% | 0 | 0 | 10 |
| Week 7 | 0/8 | 0% | 0 | 0 | 8 |

## Weekly deliverables

| Week | Deliverable | Description | Why it matters | Status | Evidence / Link |
| --- | --- | --- | --- | --- | --- |
| Week 0 | week0_bootstrap.ipynb + reflection.md | Notebook covering numpy, matplotlib, first PyTorch tensor, first HF pipeline() call. reflection.md = 1-page 'what is a NN' in his own words. | Diagnostic for whether bootstrap landed. Vague reflection -> slow Week 1 down. | Done | https://github.com/makarkul/bharatgen-speech-intern-2026/tree/claude/setup-intern-project-FkRbX/week0 |
| Week 1 | week1_audio_basics.ipynb + audio_cheatsheet.md | Audio plots + 1-page cheatsheet on sampling rate, mono/stereo, resampling, mel-spectrogram. | Cheatsheet seeds the handoff package. | Not Started |  |
| Week 2 | how_to_run_an_hf_model.md | Tutorial: AutoModel/Tokenizer/Processor, cache, eval mode, no_grad. Code + screenshots. | Writing-for-others test of understanding. | Not Started |  |
| Week 3 | week3_shrutam2_eval.md | WER table (Shrutam-2 vs Whisper, per language), profiling table, 5 failure cases with hypotheses. | First quantitative comparison. Establishes the rigor bar. | Not Started |  |
| Week 4 | week4_sooktam2_eval.md + audio samples | Markdown + audio. Indic Parler comparison. Edge-case failures documented. | Quality claims backed by listenable artifacts. | Not Started |  |
| Week 5 | week5_indictrans2_eval.md + translate.py CLI | Translation eval + reusable CLI consumed by Week 6 pipeline. | First reusable artifact. | Not Started |  |
| Week 6 | Gradio demo + architecture.md | Click-through STT -> Translate -> TTS demo. Hand-drawn architecture diagram with memory budget and failure modes. | The actual goal. Drawing tests understanding. | Not Started |  |
| Week 7 | Tagged repo v1.0-intern-handoff | Complete handoff: README, SETUP, EXPERIMENTS, NEXT_STEPS, LIMITATIONS, 30-min Loom. | Next intern starts at Week 6, not Week 0. | Not Started |  |

## Weekly cadence

| When | Activity | Duration | Format | Owner | Rationale |
| --- | --- | --- | --- | --- | --- |
| Monday morning | Intern posts the week's plan | 15 min | Async (Slack) | Intern | Forces intent-setting before the week starts. Monday corrections are cheap; mid-week corrections cost a day. |
| Wednesday | Blocker check (optional) | 10 min | Slack or call | Either | Mid-week safety net. Skip if not blocked. Prevents 'silently stuck Tue-Thu' pattern. |
| Friday EOD | Commit week's deliverable + 5-bullet progress note | Async | Git + Slack | Intern | Forced reflection. The 5 bullets become EXPERIMENTS.md timeline in Week 7. |
| Friday next day | Mentor reviews commits | 30 min | GitHub PR comments | Mentor | Async review. Schedules any Monday correction. |
| End of each week | Update Tracker status + evidence links | 5 min | This spreadsheet | Intern | Single source of truth for progress. |
| End of Week 0 | Mentor reads intern's reflection.md | 10 min | GitHub | Mentor | Bootstrap diagnostic. Vague reflection -> slow Week 1. Specific reflection -> proceed at planned pace. |

## Week-7 success criteria

| # | Question | What a good answer looks like | Intern's answer (Week 7) |
| --- | --- | --- | --- |
| 1 | For a 10-second Hindi audio file end-to-end translated to English speech, where does the time actually go? Give it to me in milliseconds per stage. | Stage-by-stage breakdown: ASR ms, translation ms, TTS ms, overhead/copies ms, total ms. Hardware noted. Points at the dominant stage and explains why. |  |
| 2 | Why can't we run all three models on a 16 GB GPU simultaneously without something giving? | Numbers: per-model peak GPU memory. Aggregate. Names the dominant consumer. Discusses fp16/bf16, offload, lazy load as mitigations. |  |
| 3 | What's the difference between how Shrutam-2 and Whisper handle code-mixed Hindi-English, based on your measurements? | WER numbers on the same code-mixed test set. Qualitative observation: where each tokenizer breaks. Speculation about training-data distribution. |  |
| 4 | Sooktam-2 is reference-guided. What does that mean for output quality, and what does it mean for inference latency compared to a description-conditioned model like Indic Parler-TTS? | Quality: bounded by reference clip quality. Latency: extra reference encoding pass, RTF numbers for both. Trade-off articulated. |  |
| 5 | If I gave you a 30-minute audio file, what breaks and why? | Memory growth (KV cache or buffer). Context limits. Chunking strategy needed. Error compounding across chunks for translation stage. |  |

## Anti-patterns

| Week | Don't do this | Why it's tempting | Why to avoid |
| --- | --- | --- | --- |
| Week 0 | Implementing a neural network from scratch in numpy | Feels like 'doing prerequisites properly'. | Week 0 is conceptual coverage, not building. Implementation can be a post-internship project. |
| Week 0 | Reading PyTorch tutorials past 'Tensors' | PyTorch.org's tutorials look authoritative. | Autograd / training tutorials are training-stack content. We don't train; reading them muddles the model. |
| Week 0 | Stressing over 3Blue1Brown's math | Math fluency feels prerequisite. | Watch for intuition, not algebra. Math doesn't gate Week 1. |
| Week 1 | Reading Stanford CS224S lectures front-to-back | Feels like 'doing prerequisites properly'. | Three weeks disappear into lectures with zero code shipped. |
| Week 1 | Trying to derive Fourier transforms before plotting one | First-principles instinct. | FFT math is not on the critical path to running BharatGen models. |
| Week 2 | Fine-tuning anything (LoRA, full, PEFT) | Most-talked-about thing in ML. | Half-trained checkpoints with no documentation. Inference depth > training breadth here. |
| Week 2 | Pulling in accelerate / bitsandbytes / vLLM | Sounds professional. | Adds debugging surface without solving a current problem. |
| Week 3 | Trying to 'improve' Shrutam-2 | WER is bad on some clips - instinct is to fix it. | Improvement is meaningless before measurement is solid. |
| Week 3 | Reporting numbers without hardware/batch context | Easy oversight in a hurry. | A number without context is noise. |
| Week 4 | Claiming TTS quality without samples | 'Sounds good' feels honest. | Untestable claims become folk wisdom. |
| Week 4 | Blaming the model when the reference clip is bad | Garbage in is harder to spot than garbage out. | Wastes weeks debugging the wrong thing. |
| Week 5 | Using BLEU as the only translation metric | It's the metric in every paper. | BLEU is brittle for morphologically rich Indic languages. |
| Week 6 | Streaming / realtime / WebSockets | Sounds impressive. | Streaming is its own multi-week project. |
| Week 6 | Docker / Kubernetes containerization | 'Production-ready' instinct. | Containerization without a deployment target is theater. |
| Week 7 | Leaving docs for the last day | Always tempting. | Doc-only-in-week-7 misses ~30% of what was done. |

## Glossary

| Category | Term | Plain-English definition | First appears |
| --- | --- | --- | --- |
| Audio | Sampling rate | How many audio measurements are taken per second. 16,000/sec (16 kHz) is the speech-model standard; CD audio is 44,100/sec. | Week 1 |
| Audio | Waveform | The raw audio signal as a 1-D array of amplitude values over time. What you see in Audacity. | Week 1 |
| Audio | Spectrogram | A 2-D image showing how energy at different frequencies changes over time. Models eat this, not the raw waveform. | Week 1 |
| Audio | Mel-spectrogram | A spectrogram with frequencies rescaled to the mel scale (closer to how humans hear). Standard input for speech models. | Week 1 |
| Audio | Mono / Stereo | Mono = 1 channel. Stereo = 2 channels. Speech models almost always want mono. | Week 1 |
| PyTorch | Tensor | An n-dimensional array of numbers (like a numpy array) that can live on a GPU and support gradient tracking. For our purposes: numpy array on a graphics card. | Week 0 |
| PyTorch | Device | Where a tensor lives: 'cpu' or 'cuda' (GPU). Operations between tensors on different devices fail. | Week 0 |
| PyTorch | dtype | The numeric type of tensor elements: float32 (default), float16, bfloat16, int8, etc. Affects memory and precision. | Week 0 |
| PyTorch | Broadcasting | Rules that let operations between tensors of different shapes work (e.g., adding (3,1) and (1,4) produces (3,4)). | Week 0 |
| PyTorch | torch.no_grad() | A context manager that turns off gradient tracking. Use during inference: halves memory, speeds up. | Week 2 |
| PyTorch | model.eval() | Switches a model from training mode to inference mode. Affects layers like dropout and batchnorm. | Week 2 |
| ML | Model | A function with learned parameters. Input -> math -> output. For us, always pre-trained — we never train. | Week 0 |
| ML | Parameters / Weights | The learned numbers inside a model. A '2.9B parameter model' has 2.9 billion of these. | Week 0 |
| ML | Inference | Running a trained model on new inputs to get outputs. The only thing we do for 7 weeks. | Week 0 |
| ML | Training | Adjusting model parameters using labeled data. We don't do this. | Week 0 |
| ML | Forward pass | Running input through the model to produce output. The thing that happens during inference. | Week 0 |
| NLP | Token | An integer that represents a chunk of text. Could be a word, a subword, a character, or even a byte. | Week 0 |
| NLP | Tokenizer | Converts text -> list of tokens (integers). Each model ships with its own tokenizer. | Week 0 |
| NLP | Vocabulary / vocab size | The set of all possible tokens the model knows. vocab_size is how many distinct tokens exist. | Week 2 |
| NLP | Embedding | A vector of numbers representing something (a token, a speaker, an image). Similar things -> similar vectors. | Week 4 |
| NLP | Logits | The raw numeric output of a model, one per vocab token. Higher = model thinks this token is more likely. | Week 2 |
| NLP | Softmax | Converts logits into probabilities (each between 0 and 1, summing to 1). Normalization with exponentials. | Week 2 |
| NLP | Hidden size | The width of the model's internal representations (e.g., 768 = each token is represented by a 768-number vector inside the model). | Week 2 |
| NLP | Layer | One stage of the model's processing. A 12-layer transformer applies its core operation 12 times. | Week 2 |
| NLP | Attention | The operation that lets each token in a sequence 'look at' (weight) every other token. The core of transformers. | Week 0 (primer) |
| NLP | Encoder | Part of a model that reads the whole input and builds a representation of it. | Week 5 |
| NLP | Decoder | Part of a model that generates output token by token, often conditioned on an encoder's representation. | Week 5 |
| NLP | Seq2seq | A model whose input is a sequence and output is a sequence. ASR, TTS, and translation are all seq2seq. | Week 5 |
| NLP | Autoregressive | Generates output one token at a time, each new token depends on previously generated tokens. | Week 5 |
| NLP | Greedy decoding | At each generation step, pick the single highest-probability token. Fast, sometimes suboptimal. | Week 3 |
| NLP | Beam search | At each step, keep the top-k partial sequences. Slower but usually better output. | Week 3 |
| NLP | KV cache | Stored intermediate states from previous generation steps. Speeds up autoregressive decoding; eats GPU memory linearly with output length. | Week 7 |
| Speech | ASR / STT | Audio -> text. ASR (Automatic Speech Recognition) and STT (Speech-to-Text) mean the same thing. | Week 3 |
| Speech | TTS | Text -> audio. Text-to-Speech. | Week 4 |
| Speech | NMT | Neural Machine Translation. Text in one language -> text in another, done by a neural network. | Week 5 |
| Speech | Acoustic model | In TTS, the part that turns text into a mel-spectrogram. | Week 4 |
| Speech | Vocoder | In TTS, the part that turns a mel-spectrogram into an audible waveform. | Week 4 |
| Speech | Reference-guided / voice cloning | TTS conditioned on a reference audio clip: the output mimics the reference speaker. | Week 4 |
| Speech | Description-conditioned TTS | TTS conditioned on a text description ('a calm female voice with low pitch') rather than a reference clip. Indic Parler-TTS works this way. | Week 4 |
| Speech | WER | Word Error Rate. (insertions + deletions + substitutions) / total reference words. Lower is better. | Week 3 |
| Speech | CER | Character Error Rate. Same as WER but at character level. More forgiving for morphologically rich languages. | Week 3 |
| Speech | RTF | Real-Time Factor. generation_time / audio_duration. RTF < 1 means faster than real-time. | Week 4 |
| HF | Model card | The README on a Hugging Face model page. Documents usage, training data, intended use, limitations. | Week 0 |
| HF | from_pretrained() | The HF method to download and load a model or tokenizer by ID. Caches in ~/.cache/huggingface/. | Week 2 |
| HF | AutoModel / AutoTokenizer / AutoProcessor | Generic loaders that figure out the right model/tokenizer class from the model ID. | Week 2 |
| HF | pipeline() | The highest-level HF interface. Three lines: load, call, get output. Hides AutoModel + AutoTokenizer + preprocessing. | Week 0 |
| HF | trust_remote_code=True | Allows the model repo to execute its own Python code when loaded. Required by some models (Sooktam-2). Treat untrusted repos with caution. | Week 2 |
| Engineering | Latency | Wall-clock time from input to output. Lower is better. | Week 3 |
| Engineering | Throughput | How much work per unit time (audio-seconds processed per inference-second, or tokens/sec). | Week 3 |
| Engineering | Peak GPU memory | The maximum GPU memory used during an operation. Measured with torch.cuda.max_memory_allocated(). | Week 3 |

## Learning resources

| Phase | Topic | Priority | Resource | Type | Time | URL |
| --- | --- | --- | --- | --- | --- | --- |
| Pre-start | RunPod basics | Primary | RunPod Pods Overview + Quickstart | Docs | 30 min | https://docs.runpod.io/pods/overview |
| Pre-start | BharatGen context | Primary | AI4Bharat blog (skim IndicASR/IndicTTS posts) | Blog | 1 hour | https://ai4bharat.iitm.ac.in/blog |
| Week 0 Day 1 | numpy | Primary | SciPy Lectures: NumPy | Course | 2 hours | https://lectures.scientific-python.org/intro/numpy/index.html |
| Week 0 Day 1 | numpy | Supplementary | NumPy Absolute Beginners guide | Docs | 45 min | https://numpy.org/doc/stable/user/absolute_beginners.html |
| Week 0 Day 1 | Jupyter | Primary | Real Python: Jupyter Notebook Introduction | Article | 30 min | https://realpython.com/jupyter-notebook-introduction/ |
| Week 0 Day 1 | venv / uv | Primary | Astral uv: Getting Started | Docs | 20 min | https://docs.astral.sh/uv/getting-started/ |
| Week 0 Day 2 | matplotlib | Primary | Matplotlib: Pyplot tutorial | Docs | 45 min | https://matplotlib.org/stable/tutorials/pyplot.html |
| Week 0 Day 2 | librosa (reading practice) | Primary | librosa: Tutorial | Docs | 1 hour | https://librosa.org/doc/latest/tutorial.html |
| Week 0 Day 3 | What is a 'model' | Primary | Stephen Wolfram: What Is ChatGPT Doing? (first third only — through 'Inside ChatGPT') | Essay | 1.5 hours | https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/ |
| Week 0 Day 3 | Tokenizer concept | Primary | HF Tokenizers: Summary | Docs | 20 min | https://huggingface.co/docs/transformers/tokenizer_summary |
| Week 0 Day 3 | GPU vs CPU | Primary | NVIDIA blog: CPU vs GPU | Article | 15 min | https://blogs.nvidia.com/blog/whats-the-difference-between-a-cpu-and-a-gpu/ |
| Week 0 Day 4 | Neural network intuition | Primary | 3Blue1Brown: Neural Networks (Ch 1-3) | Video | 60 min | https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi |
| Week 0 Day 4 | Transformer primer | Primary | Jay Alammar: The Illustrated Transformer (first 1/3 only) | Article | 30 min | https://jalammar.github.io/illustrated-transformer/ |
| Week 0 Day 5 | PyTorch tensors | Primary | PyTorch: Tensors tutorial (STOP after this — don't continue to Autograd) | Tutorial | 30 min | https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html |
| Week 0 Day 5 | HF pipeline() | Primary | HF LLM Course Ch 1.3: Transformers, what can they do? | Course | 45 min | https://huggingface.co/learn/llm-course/chapter1/3 |
| Week 1 | Audio data fundamentals | Primary | HF Audio Course Unit 1: Working with audio data | Course | 2 hours | https://huggingface.co/learn/audio-course/chapter1/introduction |
| Week 1 | librosa examples | Supplementary | librosa: Audio examples gallery | Docs | browse | https://librosa.org/doc/latest/auto_examples/index.html |
| Week 2 | Using HF Transformers | Primary | HF LLM Course Ch 2: Using Transformers (AutoModel, AutoTokenizer, pipeline internals) | Course | 2-3 hours | https://huggingface.co/learn/llm-course/chapter2/1 |
| Week 2 | Audio pipelines | Primary | HF Audio Course Unit 2: A gentle intro to audio applications | Course | 1 hour | https://huggingface.co/learn/audio-course/chapter2/introduction |
| Week 2 | Transformer architectures for audio | Supplementary | HF Audio Course Unit 3: Transformer architectures for audio | Course | 1 hour | https://huggingface.co/learn/audio-course/chapter3/introduction |
| Week 3 | ASR / WER | Primary | HF Audio Course Unit 5: Automatic Speech Recognition | Course | 2-3 hours | https://huggingface.co/learn/audio-course/chapter5/introduction |
| Week 3 | WER computation | Primary | jiwer library README | Docs | 15 min | https://github.com/jitsi/jiwer |
| Week 3 | Whisper context | Supplementary | OpenAI Whisper paper (skim) | Paper | 30 min | https://arxiv.org/abs/2212.04356 |
| Week 4 | TTS fundamentals | Primary | HF Audio Course Unit 6: From text to speech | Course | 2 hours | https://huggingface.co/learn/audio-course/chapter6/introduction |
| Week 5 | Translation / seq2seq | Primary | HF LLM Course Ch 7.4: Translation | Course | 1.5 hours | https://huggingface.co/learn/llm-course/chapter7/4 |
| Week 5 | Encoder-decoder deep dive | Primary | Jay Alammar: The Illustrated Transformer (full read this time) | Article | 1 hour | https://jalammar.github.io/illustrated-transformer/ |
| Week 5 | IndicTrans2 context | Supplementary | IndicTrans2 paper (skim Sections 1-3) | Paper | 30 min | https://arxiv.org/abs/2305.16307 |
| Week 6 | Speech-to-speech pipeline | Primary | HF Audio Course Unit 7: Putting it all together (speech-to-speech translation chapter) | Course | 2 hours | https://huggingface.co/learn/audio-course/chapter7/introduction |
| Week 6 | Gradio UI | Primary | Gradio Quickstart | Docs | 45 min | https://www.gradio.app/guides/quickstart |
| Week 6 | GPU memory management | Primary | PyTorch: CUDA semantics | Docs | 30 min | https://pytorch.org/docs/stable/notes/cuda.html |
| Week 7 | PyTorch profiling | Primary | PyTorch: Profiler recipe | Tutorial | 30 min | https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html |
| Week 7 | Technical writing structure | Primary | Diátaxis framework (4 doc types: tutorials, how-tos, reference, explanations) | Reading | 30 min | https://diataxis.fr/ |
