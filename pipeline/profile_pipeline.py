"""Batch profiler for the speech-translation pipeline — produces a report for review.

Sends a FIXED corpus of benchmark clips through the LIVE /speak endpoint (so it
profiles the REAL path: ffmpeg transcode -> Shrutam ASR -> translate (Param-2 OR IndicTrans2)
-> Sooktam TTS, including HTTP), collects the per-stage `timings` the server
returns, and writes:

  * results.csv  — one row per run (raw data; the drill-down behind the summary)
  * REPORT.md    — environment header + summary tables (median/p90 per stage,
                   % of total, cold-vs-warm, per-target breakdown)

WHY over HTTP and not by importing the models: this measures exactly what a real
user request pays, end to end, on the same running server — nothing is faked or
bypassed. WHY a fixed corpus: reproducible (re-runnable, same clips) and varied
(short/long clips, multiple language pairs), so the numbers are representative
rather than cherry-picked.

Run ON THE POD, with both services already up (server :8000 + a translator):
    # Param-2 backend (server launched with TRANSLATOR=param):
    python pipeline/profile_pipeline.py --translator param
    # IndicTrans2 backend (server launched with TRANSLATOR=indictrans):
    python pipeline/profile_pipeline.py --translator indictrans \
        --out pipeline/profiling_results_indictrans

Optional flags:
    --server      URL of the main server      (default http://localhost:8000)
    --data        benchmark dataset root       (default ~/datasets)
    --out         output directory             (default pipeline/profiling_results)
    --limit       max clips per source language (for a quick smoke run)
    --translator  which backend the server runs: param | indictrans (default param).
                  MUST match how the server was launched — it only sets the report
                  labels; the profiler can't auto-detect which translator it hit.
"""
import argparse
import csv
import datetime
import json
import statistics
import subprocess
import sys
from pathlib import Path

import requests

# --- The profiling matrix --------------------------------------------------
# Source language -> list of TARGET languages to translate+speak into. We pick
# a few REPRESENTATIVE targets rather than all pairs:
#   * native-voice targets (hindi/marathi/tamil have native reference clips)
#     show real cross-language cost with a matched accent;
#   * a same-language target (src == tgt) is the CONTROL: translate is a no-op,
#     so it isolates the ASR+TTS cost and shows exactly how much translate adds.
# Source clips come from datasets/shrutilipi/<lang>/ (known source language, so
# we hand the ASR the right prompt; transcripts.tsv gives the clip duration).
MATRIX = {
    "hindi": ["tamil", "marathi", "hindi"],   # last one = same-language control
    "tamil": ["hindi", "tamil"],              # last one = same-language control
}

# Which translator the pipeline was running for this profile. The profiler talks
# to the main server (:8000), which can be pointed at EITHER backend, so it can't
# tell which one it hit — you pass --translator to say so, and every label in the
# report/environment.json follows from here. Pick the one matching how the server
# was launched (TRANSLATOR=param|indictrans in pod_setup.sh). This is what keeps a
# report honest: get it wrong and the numbers are right but the name is a lie.
TRANSLATORS = {
    "param": {
        "label": "Param-2",
        "model": "bharatgenai/Param2-17B-A2.4B-Thinking",
        "venv_note": "the Param-2 service runs transformers 4.52.3 in its own venv",
    },
    "indictrans": {
        "label": "IndicTrans2",
        "model": "ai4bharat/indictrans2-indic-indic-1B",
        "venv_note": "the IndicTrans2 service runs transformers 4.56.2 in its own venv",
    },
}


def discover_clips(data_root: Path, limit: int | None):
    """Yield (clip_path, source_language, duration_s) for every corpus clip."""
    clips = []
    for src_lang in MATRIX:
        clip_dir = data_root / "shrutilipi" / src_lang / "clips"
        tsv = data_root / "shrutilipi" / src_lang / "transcripts.tsv"
        # Map filename -> duration from the transcript TSV (it has a duration col).
        durations = {}
        if tsv.exists():
            with open(tsv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    try:
                        durations[row["filename"]] = float(row["duration"])
                    except (KeyError, ValueError):
                        pass
        wavs = sorted(clip_dir.glob("*.wav"))
        if limit:
            wavs = wavs[:limit]
        for wav in wavs:
            clips.append((wav, src_lang, durations.get(wav.name)))
    return clips


def collect_env(server_url: str, tr: dict) -> dict:
    """Capture environment metadata so the numbers are falsifiable/reproducible."""
    def _run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=15).stdout.strip()
        except Exception:
            return ""

    gpu = _run(["nvidia-smi", "--query-gpu=name,memory.total",
                "--format=csv,noheader"]) or "unknown (nvidia-smi failed)"
    commit = _run(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
    # transformers version is per-venv; report the main server's (the param
    # service runs a different one — noted in the report text).
    try:
        import transformers
        tfm = transformers.__version__
    except Exception:
        tfm = "unknown"
    return {
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "gpu": gpu,
        "git_commit": commit,
        "transformers_main_server": tfm,
        "server_url": server_url,
        "asr_model": "bharatgenai/Shrutam-2",
        "tts_model": "bharatgenai/sooktam2",
        "translate_model": tr["model"],
        "translator_label": tr["label"],
    }


def run_one(server_url: str, clip: Path, src: str, tgt: str) -> dict | None:
    """POST one clip through /speak and return its timing row (or None on error)."""
    try:
        with open(clip, "rb") as f:
            resp = requests.post(
                server_url + "/speak",
                files={"audio": (clip.name, f, "audio/wav")},
                data={"language": src, "target_language": tgt},
                timeout=300,
            )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  !! {clip.name} {src}->{tgt}: {type(e).__name__}: {e}", flush=True)
        return None
    if data.get("error"):
        print(f"  !! {clip.name} {src}->{tgt}: server error: {data['error']}", flush=True)
        return None
    t = data.get("timings")
    if not t:
        print(f"  !! {clip.name} {src}->{tgt}: response had no timings", flush=True)
        return None
    return {
        "clip": clip.name, "src": src, "tgt": tgt,
        "same_lang": src == tgt,
        "ffmpeg_s": t.get("ffmpeg_s"),
        "asr_s": t.get("asr_s"),
        "translate_s": t.get("translate_s"),
        "translate_model_s": t.get("translate_model_s"),
        "translate_http_s": t.get("translate_http_s"),
        "tts_s": t.get("tts_s"),
        "total_s": t.get("total_s"),
        "src_chars": len(data.get("text") or ""),
        "tgt_chars": len(data.get("translation") or ""),
    }


# --- Summary stats ---------------------------------------------------------
STAGES = ["ffmpeg_s", "asr_s", "translate_s", "tts_s", "total_s"]


def _pctl(values, p):
    """p-th percentile (0..100) by nearest-rank — no numpy dependency."""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * (len(s) - 1))))
    return s[k]


def summarize(rows, predicate=None):
    """median / p90 / max for each stage over rows (optionally filtered)."""
    sel = [r for r in rows if predicate is None or predicate(r)]
    out = {"n": len(sel)}
    for stage in STAGES:
        vals = [r[stage] for r in sel if r.get(stage) is not None]
        out[stage] = {
            "median": round(statistics.median(vals), 2) if vals else None,
            "p90": round(_pctl(vals, 90), 2) if vals else None,
            "max": round(max(vals), 2) if vals else None,
        }
    return out


def md_stage_table(summary, translate_label: str = "Translate") -> str:
    """One summary block -> a markdown table with a % of total column."""
    total_med = summary["total_s"]["median"] or 0
    lines = [
        f"_n = {summary['n']} runs_",
        "",
        "| Stage | Median (s) | p90 (s) | Max (s) | % of total (median) |",
        "|---|---|---|---|---|",
    ]
    labels = {"ffmpeg_s": "Transcode (ffmpeg)", "asr_s": "ASR (Shrutam-2)",
              "translate_s": f"Translate ({translate_label})", "tts_s": "TTS (Sooktam-2)",
              "total_s": "**TOTAL**"}
    for stage in STAGES:
        s = summary[stage]
        med = s["median"]
        # `med is not None` (not truthiness) so a legit 0.0 shows "0%", not "—".
        pct = f"{med / total_med * 100:.0f}%" if (med is not None and total_med) else "—"
        if stage == "total_s":
            pct = "100%"
        lines.append(f"| {labels[stage]} | {med} | {s['p90']} | {s['max']} | {pct} |")
    return "\n".join(lines)


def build_report(env, rows, cold_row, out_dir: Path, tr: dict):
    label = tr["label"]
    L = []
    L.append(f"# Speech-Translation Pipeline — Latency Profile ({label} translator)\n")
    L.append(f"Generated: **{env['date_utc']}**  ·  commit `{env['git_commit']}`\n")

    L.append("## Environment\n")
    L.append("| Field | Value |")
    L.append("|---|---|")
    L.append(f"| GPU | {env['gpu']} |")
    L.append(f"| Date (UTC) | {env['date_utc']} |")
    L.append(f"| Git commit | `{env['git_commit']}` |")
    L.append(f"| ASR model | `{env['asr_model']}` |")
    L.append(f"| Translate model | `{env['translate_model']}` |")
    L.append(f"| TTS model | `{env['tts_model']}` |")
    L.append(f"| transformers (main server) | {env['transformers_main_server']} "
             f"({tr['venv_note']}) |")
    L.append("")

    L.append("## How to read this\n")
    L.append("The pipeline runs four stages **sequentially**, so total latency is "
             "their sum. Each clip from the benchmark corpus "
             "(`datasets/shrutilipi/{hindi,tamil}`) was POSTed through the live "
             "`/speak` endpoint, so these are real end-to-end request times "
             f"(including the HTTP hop to the {label} translate service), not "
             "isolated micro-benchmarks. We report **median** (typical) and "
             "**p90 / max** (worst-case, which matters for a live demo) rather "
             "than a mean, because a single slow run shouldn't define the story.\n")

    # Cold start, called out separately so it doesn't skew the warm medians.
    L.append("## Cold start\n")
    if cold_row:
        L.append("The server was freshly booted for this run, so the first request "
                 "pays one-time GPU/CUDA warmup. It is **excluded from the summary "
                 "below** and shown here on its own:\n")
        L.append(f"- First request `{cold_row['src']}->{cold_row['tgt']}` "
                 f"({cold_row['clip']}): **{cold_row['total_s']}s total** "
                 f"(asr {cold_row['asr_s']}s · translate {cold_row['translate_s']}s "
                 f"· tts {cold_row['tts_s']}s)\n")
    else:
        L.append("Both models are loaded at **server startup** (not on first "
                 "request), so once the server is up every request is warm — there "
                 "is no per-request model-load cost. All runs below are warm.\n")

    L.append("## Summary — all warm runs\n")
    all_summary = summarize(rows)
    L.append(md_stage_table(all_summary, label))
    L.append("")

    # The control: same-language runs (translate is a no-op) vs cross-language.
    L.append("## Cross-language vs same-language (the cost translation adds)\n")
    L.append("Same-language runs skip translation (it's a no-op), so comparing "
             f"them to cross-language runs isolates how much the {label} step "
             "actually costs.\n")
    L.append("### Cross-language runs (translate active)\n")
    L.append(md_stage_table(summarize(rows, lambda r: not r["same_lang"]), label))
    L.append("")
    L.append("### Same-language runs (translate skipped — control)\n")
    L.append(md_stage_table(summarize(rows, lambda r: r["same_lang"]), label))
    L.append("")

    # Translate internals: model compute vs HTTP overhead (cross-language only).
    xrows = [r for r in rows if not r["same_lang"] and r.get("translate_model_s")]
    if xrows:
        model_vals = [r["translate_model_s"] for r in xrows]
        http_vals = [r["translate_http_s"] for r in xrows if r.get("translate_http_s") is not None]
        L.append("## Inside the translate step (cross-language runs)\n")
        L.append(f"Splitting translate into model compute (reported by the {label} "
                 "service) vs HTTP/network overhead shows whether the bottleneck "
                 "is the translation model or the service hop.\n")
        L.append("| Component | Median (s) | p90 (s) |")
        L.append("|---|---|---|")
        L.append(f"| Model generation | {round(statistics.median(model_vals), 2)} "
                 f"| {round(_pctl(model_vals, 90), 2)} |")
        if http_vals:
            L.append(f"| HTTP overhead | {round(statistics.median(http_vals), 3)} "
                     f"| {round(_pctl(http_vals, 90), 3)} |")
        L.append("")

    # Per-target breakdown.
    L.append("## Per target language\n")
    L.append("| Target | n | Median total (s) | p90 total (s) |")
    L.append("|---|---|---|---|")
    for tgt in sorted({r["tgt"] for r in rows}):
        s = summarize(rows, lambda r: r["tgt"] == tgt)
        L.append(f"| {tgt} | {s['n']} | {s['total_s']['median']} | {s['total_s']['p90']} |")
    L.append("")

    L.append("## Raw data\n")
    L.append("Per-run measurements are in [`results.csv`](results.csv) "
             f"({len(rows)} warm runs). Each row is one `/speak` request with "
             "every stage time, input duration, and output length, so any number "
             "above can be traced back to its source runs.\n")

    (out_dir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--data", default=str(Path.home() / "datasets"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "profiling_results"))
    ap.add_argument("--limit", type=int, default=None,
                    help="max clips per source language (quick smoke run)")
    ap.add_argument("--translator", choices=sorted(TRANSLATORS), default="param",
                    help="which translator the server is running, so the report "
                         "is labeled correctly (param | indictrans). MUST match how "
                         "the server was launched (TRANSLATOR=... in pod_setup.sh) — "
                         "the profiler can't detect it, and a wrong value mislabels "
                         "an otherwise-correct report.")
    ap.add_argument("--cold-start", action="store_true",
                    help="ONLY pass this if the server was JUST booted and has "
                         "served no requests yet. It pulls the first run aside as "
                         "a cold-start sample. Otherwise every run is warm (models "
                         "load at server startup, so a used server has no cold run).")
    args = ap.parse_args()

    server = args.server.rstrip("/")
    data_root = Path(args.data).expanduser()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = TRANSLATORS[args.translator]

    # Fail fast if the server isn't up — better than 50 timeouts.
    try:
        requests.get(server + "/", timeout=10)
    except Exception as e:
        sys.exit(f"Cannot reach server at {server} ({e}). "
                 "Is server.py running on the pod?")

    clips = discover_clips(data_root, args.limit)
    if not clips:
        sys.exit(f"No clips found under {data_root}/shrutilipi/. Check --data.")

    env = collect_env(server, tr)
    print(f">>> Translator label: {tr['label']} ({tr['model']})", flush=True)
    print(f">>> Environment: {env['gpu']} | commit {env['git_commit']}", flush=True)
    print(f">>> {len(clips)} clips, targets per matrix -> profiling...", flush=True)

    # Build the full work list (clip x target), then run sequentially. The FIRST
    # run is the cold-start sample — kept separate so it doesn't skew warm stats.
    work = []
    for clip, src, dur in clips:
        for tgt in MATRIX[src]:
            work.append((clip, src, tgt, dur))

    rows = []
    cold_row = None
    for i, (clip, src, tgt, dur) in enumerate(work):
        # Only treat run #1 as cold when the user explicitly asserts a fresh boot
        # (--cold-start). Models load at server STARTUP, so on an already-used
        # server there is no cold run and every request is warm.
        is_cold_slot = args.cold_start and i == 0
        tag = "[cold]" if is_cold_slot else f"[{i + 1}/{len(work)}]"
        print(f"  {tag} {clip.name} {src}->{tgt} ...", flush=True)
        row = run_one(server, clip, src, tgt)
        if row is None:
            continue
        row["clip_duration_s"] = dur
        if is_cold_slot:
            cold_row = row          # cold-start sample, reported separately
        else:
            rows.append(row)        # warm runs feed the summary

    if not rows:
        sys.exit("No successful warm runs — nothing to report. See errors above.")

    # Write raw CSV (every run, including the cold one, flagged).
    fieldnames = ["clip", "src", "tgt", "same_lang", "cold", "clip_duration_s",
                  "ffmpeg_s", "asr_s", "translate_s", "translate_model_s",
                  "translate_http_s", "tts_s", "total_s", "src_chars", "tgt_chars"]
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        if cold_row:
            w.writerow({**cold_row, "cold": True})
        for r in rows:
            w.writerow({**r, "cold": False})

    build_report(env, rows, cold_row, out_dir, tr)
    # Stamp env alongside, for full reproducibility.
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    print(f"\n>>> Done. {len(rows)} warm runs.", flush=True)
    print(f">>> Wrote {out_dir}/REPORT.md, results.csv, environment.json", flush=True)


if __name__ == "__main__":
    main()
