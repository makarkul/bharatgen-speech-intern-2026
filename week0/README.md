# Week 0 — Python + ML Bootstrap (5 working days)

**Goal:** Close the gap between "knows Python from coursework" and "can productively use PyTorch, Hugging Face, and Jupyter without panic." End-state: you can load a pre-trained HF model, run inference, and explain what a tensor, a tokenizer, and a model are in your own words.

**Pre-requisite:** Finish [`../SETUP.md`](../SETUP.md) before Day 1.

## Day 1 — numpy / Jupyter / venv hygiene

- numpy: arrays, `.shape`, `.dtype`, broadcasting.
- Jupyter: cells, kernels, kernel state, `%time`, `%matplotlib inline`.
- 10 small numpy exercises — **predict `.shape` before running each cell.**
- Fresh venv, `pip install numpy matplotlib jupyter`, freeze `requirements.txt`.

Resources: [SciPy Lectures: NumPy](https://lectures.scientific-python.org/intro/numpy/index.html) (2h) · [Real Python: Jupyter Intro](https://realpython.com/jupyter-notebook-introduction/) (30m) · [Astral uv](https://docs.astral.sh/uv/getting-started/) (20m).

## Day 2 — matplotlib + reading library docs

- `fig, ax = plt.subplots()`. `plot()`, `scatter()`, `hist()`, labels, titles.
- Make three plots: sine wave, 1000-sample Gaussian histogram, 2-variable scatter. Each labeled.
- Read the librosa "Tutorial" page end-to-end. Run one example from each section.

Resources: [Matplotlib Pyplot tutorial](https://matplotlib.org/stable/tutorials/pyplot.html) · [librosa Tutorial](https://librosa.org/doc/latest/tutorial.html).

## Day 3 — ML concepts at 30,000 feet (no code)

In your own words, answer these in `reflection.md` (Day 4 will expand it):

- What is a "model"? (function with learned parameters; we never train, only run pre-trained)
- What is a tensor vs. a numpy array? (numpy array on a GPU)
- What is a tokenizer? (text → list of integers; per-model)
- Training vs. inference. (we do inference only)
- GPU vs. CPU intuition. (1000s of parallel cores → matmuls 10–100× faster)

Resources: [Stephen Wolfram: What Is ChatGPT Doing?](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/) (read through "Inside ChatGPT", then stop) · [HF Tokenizer Summary](https://huggingface.co/docs/transformers/tokenizer_summary) · [NVIDIA: CPU vs GPU](https://blogs.nvidia.com/blog/whats-the-difference-between-a-cpu-and-a-gpu/).

## Day 4 — Neural network intuition

- Watch 3Blue1Brown chapters 1–3 (~60 min). **Watch, don't pause to implement.**
- Write a 1-page `reflection.md`: *What is a neural network? What does training mean? Why do we need data?* — **this is the bootstrap diagnostic.**
- Read the first 1/3 of Jay Alammar's "Illustrated Transformer."

Resources: [3Blue1Brown NN Ch 1-3](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) · [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (first 1/3).

## Day 5 — First contact with PyTorch + Hugging Face

- `pip install torch`. `torch.tensor([1,2,3])`. `.to('cuda')`. Print `.shape`, `.dtype`, `.device`.
- Matmul two random 1000×1000 tensors on CPU vs. GPU; time both. Day 3 "why GPU" becomes concrete.
- Run HF distilbert via `pipeline()` — three lines. Just see the model work. No `AutoModel` yet.
- First commit: `week0_bootstrap.ipynb` + `reflection.md`.

Resources: [PyTorch Tensors tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) — **stop here, do NOT continue to Autograd** · [HF LLM Course Ch 1.3](https://huggingface.co/learn/llm-course/chapter1/3).

## Deliverable

- `week0/week0_bootstrap.ipynb` — all Day 1–5 exercises
- `week0/reflection.md` — 1-page Day 4 reflection

## Anti-patterns

- Implementing a NN from scratch in numpy. (Conceptual coverage, not building.)
- Reading PyTorch tutorials past "Tensors." (Autograd/training tutorials muddle the model — we don't train.)
- Stressing over 3Blue1Brown's gradient-descent algebra. (Watch for intuition; math doesn't gate Week 1.)

## Diagnostic

End of Week 0: mentor reads `reflection.md` and skims the notebook. Vague reflection → slow Week 1 by half a day. Specific and complete → proceed at planned pace.
