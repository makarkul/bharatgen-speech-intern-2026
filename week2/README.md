# Week 2 — Hugging Face mental model

**Goal:** Load *any* HF model, run inference, and explain what a tokenizer, a config, and a checkpoint are.

**Primary resources:** [HF LLM Course Ch 2: Using Transformers](https://huggingface.co/learn/llm-course/chapter2/1) (2-3h) · [HF Audio Course Unit 2](https://huggingface.co/learn/audio-course/chapter2/introduction) (1h) · [HF Audio Course Unit 3](https://huggingface.co/learn/audio-course/chapter3/introduction) (supplementary).

## Concepts (lean)

- `AutoModel` / `AutoTokenizer` / `AutoProcessor`
- `trust_remote_code=True` — what it actually does (executes Python from the model repo) and the security implication. Sooktam-2 in Week 4 needs this.
- `from_pretrained` caching — where things land on disk
- `model.eval()` vs `model.train()`
- `torch.no_grad()` — halves memory, speeds up inference
- Tokens → logits → softmax → probabilities

## Hands-on

- Run `distilbert-base-uncased-finetuned-sst-2-english` on 10 sentences manually (not `pipeline()`). Print logits; apply `torch.softmax` yourself. Compare to `pipeline()` output.
- Run `openai/whisper-tiny` on your Week 1 audio clips. Get transcriptions out. (This becomes the Week 3 baseline.)
- Inspect `$HF_HOME/hub/`. Open a `config.json`. Open the tokenizer files. Write in plain English what `hidden_size`, `num_hidden_layers`, `vocab_size` mean.

## Deliverable

- `week2/how_to_run_an_hf_model.md` — tutorial **written for the next intern**. Code snippets, screenshots of the cache dir, explanation of `pipeline()` vs manual loading.

## Anti-patterns

- Fine-tuning anything (no LoRA, no PEFT, no full fine-tune).
- Pip-installing `accelerate` / `bitsandbytes` / `vLLM`. Premature.
