# Speech-to-Speech Translation Pipeline

The **full** BharatGen speech pipeline — the overall project goal, not a single
week's task. It chains all three components into a cross-language translator:

```
speech in (lang X)
   -> Shrutam-2   (ASR)   -> text in X
   -> Param-2 service     -> text in Y      <-- the translation step
   -> Sooktam-2   (TTS)   -> speech in Y
```

This differs from the Week 3 demo in `week3/webapp/`, which was speech -> text
-> speech in the **same** language (no translation). That folder is kept intact
as the Week 3 deliverable; this folder is the integrated pipeline built on top
of it.

## Two-service architecture (why)

Param-2 needs `transformers==4.52.3`; the Shrutam-2 + Sooktam-2 stack needs
`4.56.2`. They can't coexist in one process, so they run as **two services**:

```
main server :8000            param service :8500
(transformers 4.56.2)        (transformers 4.52.3, own venv)
Shrutam-2 + Sooktam-2  --HTTP-->  Param-2
```

The main server calls the param service via `PARAM_URL` (default
`http://localhost:8500`). This keeps the verified ASR/TTS stack untouched.

## Files

| File | Purpose |
|------|---------|
| `server.py`        | Main FastAPI backend: ASR + translate-over-HTTP + TTS. `/speak` takes `target_language`, returns `{text, translation, audio}`. |
| `param_service.py` | Standalone Param-2 translation service (`POST /translate`). Runs in its own venv. |
| `index.html`       | Browser frontend: "Speak in → Translate to" pickers, shows transcript + translation, plays the reply. |
| `voices/`          | Preset reference voices for Sooktam-2. |
| `pod_setup.sh`     | One command: brings up BOTH services on a RunPod pod. |
| `_comparison_later/` | IndicTrans2 + comparison harness, parked until we revisit Param-2 vs IndicTrans2. |

## Status

- [x] ASR (Shrutam-2) + TTS (Sooktam-2) working (inherited from week3).
- [x] Translation wired in (Param-2 service); `/speak` + frontend updated.
- [x] Stub-mode loop verified locally (laptop, no GPU).
- [ ] Run both services on the pod; verify the full translation loop with real models.
- [ ] (later) Compare Param-2 vs IndicTrans2 using `_comparison_later/`.

## Running it

**On the pod:** `bash pod_setup.sh` brings up both services. Then point
`index.html`'s `API_BASE` at the pod's :8000 proxy URL.

**Locally (stub mode, no GPU):** `uvicorn server:app --port 8000` — same-language
works; cross-language shows a `[STUB translate ...]` tag (the real Param service
isn't running). Good enough to test the UI loop.
