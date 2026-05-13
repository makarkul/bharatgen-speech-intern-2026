# Day 3 — What is a "Model" in ML?

**Format:** Read this, then answer the five questions at the bottom in your own words.  
**No code today.** The concepts here are the vocabulary you need before touching PyTorch on Day 5.

---

## 1. What is a "model"?

A model is a **function**. That's it.

```
f(input) → output
```

You give it something — audio, text, an image — and it gives you something back. The "magic" is that instead of a human writing the rules inside that function, the rules (called **parameters** or **weights**) were found automatically by training on a lot of examples.

Once training is done, the parameters are frozen. The model is a fixed function. You just call it.

```
speech_to_text_model(audio_clip) → "नमस्ते, आप कैसे हैं?"

translation_model("नमस्ते, आप कैसे हैं?") → "Hello, how are you?"

text_to_speech_model("Hello, how are you?") → audio_clip
```

> **Key point for this internship:** We never train a model. We only *run* pre-trained ones.  
> Training is what happened before the model was posted on Hugging Face. Our job starts after that.

Why does this matter? Because "training" is where most of the scary math lives — backpropagation, loss functions, gradient descent. None of that is your problem this summer. You are a *user* of a function, not the person who built it.

---

## 2. What's inside a model? (Just enough to not be confused)

A neural network model is a chain of matrix multiplications with some non-linearities sprinkled in.

Concretely, for a large language/speech model:

- **Input** is turned into numbers (a tensor — more on this below).
- Those numbers pass through dozens or hundreds of **layers**, each doing some math.
- **Output** comes out the other end as numbers, which are decoded back into text or audio.

The model's *parameters* (weights) are the numbers that get multiplied at each layer. A model like Whisper (the ASR model you'll compare against Shrutam-2) has ~1.5 billion such numbers. They were learned. You just load them from a file.

You do not need to understand *what* those 1.5B numbers encode. You need to understand the interface:

```
model(input_tensor) → output_tensor
```

---

## 3. Tensor vs. numpy array

You already know numpy arrays. A **tensor** is the same idea, but:

1. It can live on a **GPU** (extremely fast for matrix math).
2. It can optionally track gradients for training (we turn this off with `torch.no_grad()`).

For us, a tensor is just a numpy array on a graphics card. Same shape, same dtype concept, different home.

```
numpy array: lives on CPU RAM
tensor:       lives on GPU VRAM (or CPU RAM if you haven't moved it)
```

You'll move tensors to the GPU with `.to('cuda')` and it'll feel exactly like that.

---

## 4. What is a tokenizer?

Models don't read text. They read **integers**.

A tokenizer is the converter:

```
text → list of integers (tokens)
```

Each model has its own tokenizer with its own vocabulary. The same word might map to a different integer in different models.

"नमस्ते" might be:
- One token in a model trained on a lot of Hindi.
- Four tokens in a model mostly trained on English.
- Something in between in IndicTrans2.

This matters because:
- Longer token lists = more computation.
- If a language has few tokens in the vocabulary, the model tends to perform worse on it.

You'll see this play out concretely when you run Shrutam-2 on code-mixed Hindi-English in Week 3.

---

## 5. Training vs. inference

| | Training | Inference |
|---|---|---|
| What happens | Parameters are updated based on errors | Parameters are fixed; just run the function |
| Who does it | The team that built the model (once, expensively) | You (fast, cheap per call) |
| What you need | GPU, data, weeks | GPU, the model file, an input |
| Gradients | Yes, needed to update weights | No, you turn them off (`torch.no_grad()`) |

**We do inference only.** Every week.

---

## 6. GPU vs. CPU intuition

A CPU has a few powerful cores (4–16 typically) that can each do complex work fast.

A GPU has **thousands of small cores** that are mediocre individually but blazing fast in parallel.

Neural networks are mostly matrix multiplications. Matrix multiplication maps perfectly onto thousands of parallel cores. Result: a 1000×1000 matmul runs **10–100× faster** on a GPU than a CPU.

On Day 5 you'll time this yourself. The numbers will be more convincing than this explanation.

For memory sizing: models store their parameters in GPU memory (VRAM). Shrutam-2, IndicTrans2, and Sooktam-2 together need ~20–30 GB of VRAM to all sit on the GPU at once. That's why your RunPod has a 24 GB or 48 GB card.

---

## Reading

Do all three before your Day 4 reflection.

1. **[Stephen Wolfram: What Is ChatGPT Doing?](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/)** — Read only through the "Inside ChatGPT" section, then stop. (~1.5 hours.) This is the primary read. If you only do one, do this one.
2. **[HF Tokenizer Summary](https://huggingface.co/docs/transformers/tokenizer_summary)** — (~20 min.) Gives you the vocabulary for Week 2 and beyond.
3. **[NVIDIA: CPU vs GPU](https://blogs.nvidia.com/blog/whats-the-difference-between-a-cpu-and-a-gpu/)** — (~15 min.) Short and concrete.

---

## Checkpoint: five questions for your reflection.md

Answer these in your own words when you write `reflection.md` (Day 4 or end of today). One paragraph each is enough.

1. What is a "model"? Explain it as if the person you're talking to has never touched ML.
2. What's the difference between a tensor and a numpy array? (One sentence is fine.)
3. What is a tokenizer, and why does each model have its own?
4. What is the difference between training and inference? Which one do we do, and why does it simplify our job?
5. Why is a GPU faster than a CPU for running neural networks?

> **Grading bar:** If you can answer all five without looking at this file, you're ready for Day 5.  
> If any answer still sounds like you're reciting a definition rather than explaining a thing you understand, slow down and re-read — Day 5 will feel chaotic otherwise.
