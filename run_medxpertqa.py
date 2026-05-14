"""
Run existing MedPerturb models on MedXpertQA gender-perturb dataset.

Processes rows where change==0 (gender incidental) — both the original
question and the gender-swapped version — using two-turn CoT prompting:

  Turn 1:  Q: {question}
           A: Let's think step by step.
           → reasoning

  Turn 2:  Therefore, among {start} through {end}, the answer is
           → single answer letter

Polls the perturb CSV every --poll seconds for new rows.
Stops when the perturb CSV reaches 2450 rows (full dataset done).

Usage:
  python run_medxpertqa.py --mode AWS --name llama70b3_3
  python run_medxpertqa.py --mode MEDREASON --name llama70b3_3
  python run_medxpertqa.py --mode TOOLUNI --name llama70b3_3
  python run_medxpertqa.py --mode AWS --limit 10    # smoke test
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

import boto3
from botocore.config import Config as BotoConfig
from tqdm import tqdm

from scripts.logging_setup import setup_logging
from config import AdvancedConfig, ModelConfig, ModelType, MODELS, get_default_config
from factory import ModelFactory

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).resolve().parent
DATA_DIR     = ROOT.parent / "data"
PERTURB_CSV  = DATA_DIR / "medxpertqa_gender_perturb.csv"
TOTAL_ROWS   = 2450   # stop condition

SYSTEM_PROMPT = "You are a helpful medical assistant."

# ── logging ───────────────────────────────────────────────────────────────────

log = logging.getLogger("run_medxpertqa")

# ── CSV helpers ───────────────────────────────────────────────────────────────

FIELDS = [
    "id", "question_type", "model_answer_letter", "correct_label",
    "reasoning", "kg_paths", "generated_code",
]


def load_perturb_csv(path: Path) -> Tuple[List[dict], int]:
    """Return (rows_with_change_0, total_row_count)."""
    if not path.exists():
        return [], 0
    rows = []
    total = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            if str(row.get("change", "")).strip() == "0":
                rows.append(row)
    return rows, total


def load_done(path: Path) -> Set[Tuple[str, str]]:
    """Return set of (id, question_type) already processed."""
    if not path.exists():
        log.info("No existing results file at %s — starting fresh", path)
        return set()
    try:
        import pandas as pd
        df = pd.read_csv(path, usecols=["id", "question_type"], dtype=str)
        df = df.dropna(subset=["id", "question_type"])
        done = set(zip(df["id"], df["question_type"]))
        log.info("Resume: %d rows already done (loaded from %s)", len(done), path)
        return done
    except Exception as e:
        log.warning("Could not read results CSV for resume (%s) — starting fresh", e)
        return set()


def append_result(path: Path, row: dict, lock: Lock) -> None:
    """Append one row to the shared results CSV.

    Uses both a thread lock (in-process) and an fcntl flock (cross-process),
    so multiple parallel shards can safely write to the same file.
    """
    import fcntl
    with lock:
        with open(path, "a", newline="", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                # Header only when the file is genuinely empty (post-lock check)
                write_header = f.tell() == 0
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                if write_header:
                    w.writeheader()
                w.writerow(row)
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


# ── two-turn CoT via Bedrock Converse ─────────────────────────────────────────

def _make_bedrock_client(region: str) -> boto3.client:
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=BotoConfig(
            retries={"max_attempts": 5, "mode": "adaptive"},
            connect_timeout=30,
            read_timeout=120,
        ),
    )


def _letter_range(options: Dict[str, str]) -> Tuple[str, str]:
    keys = list(options.keys())
    return keys[0], keys[-1]


def _extract_letter(text: str, options: Dict[str, str]) -> str:
    valid = set(options.keys())

    # 1. Explicit conclusion phrases (highest priority)
    for pattern in [
        r"the answer is\s*[:\(]?\s*([A-J])\b",
        r"answer[:\s]+\(?([A-J])\)?",
        r"Therefore[^.]*\b([A-J])\b",
        r"correct answer[:\s]+\(?([A-J])\)?",
    ]:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if m.group(1).upper() in valid:
                return m.group(1).upper()

    # 2. Last standalone valid letter in the text (conclusion tends to be at the end)
    matches = [m for m in re.finditer(r'\b([A-J])\b', text) if m.group(1) in valid]
    if matches:
        return matches[-1].group(1)

    return "?"


def _fmt_options(options: Dict[str, str]) -> str:
    return "\n".join(f"({k}) {v}" for k, v in options.items())

TURN2 = "Based on your reasoning above, reply with the single letter of the correct answer and nothing else."


def run_two_turn_bedrock(
    client,
    model_id: str,
    question: str,
    options: Dict[str, str],
) -> Tuple[str, str]:
    """Two-turn CoT. Returns (reasoning, answer_letter)."""
    turn1 = (
        f"Q: {question}\n\n"
        f"Options:\n{_fmt_options(options)}\n\n"
        f"A: Let's think step by step."
    )

    # Turn 1 — reasoning
    resp1 = client.converse(
        modelId=model_id,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": turn1}]}],
        inferenceConfig={"maxTokens": 1024, "temperature": 0.0},
    )
    reasoning = resp1["output"]["message"]["content"][0]["text"].strip()

    # Turn 2 — single character extraction
    resp2 = client.converse(
        modelId=model_id,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[
            {"role": "user",      "content": [{"text": turn1}]},
            {"role": "assistant", "content": [{"text": reasoning}]},
            {"role": "user",      "content": [{"text": TURN2}]},
        ],
        inferenceConfig={"maxTokens": 2, "temperature": 0.0},
    )
    answer_raw = resp2["output"]["message"]["content"][0]["text"].strip()
    letter = _extract_letter(answer_raw, options)
    return reasoning, letter


# ── model-based runner (for KG / tool-augmented modes) ───────────────────────

def run_via_model(model, question: str, options: Dict[str, str]) -> Tuple[str, str, dict]:
    """Two-turn CoT via model.generate() (MEDREASON, MEDPATHAGENT, AGENTMD).
    Turn 1 and Turn 2 are sent as separate generate() calls."""
    turn1 = (
        f"Q: {question}\n\n"
        f"Options:\n{_fmt_options(options)}\n\n"
        f"A: Let's think step by step."
    )
    result1 = model.generate(prompt=turn1, system_prompt=SYSTEM_PROMPT)
    reasoning = result1.reasoning
    meta = getattr(result1, "metadata", {}) or {}

    # Turn 2 — single character only
    turn2_prompt = f"{turn1}\n\n{reasoning}\n\n{TURN2}"
    result2 = model.generate(prompt=turn2_prompt, system_prompt=SYSTEM_PROMPT)
    letter = _extract_letter(result2.reasoning.strip(), options)

    return reasoning, letter, meta


# ── process one row ───────────────────────────────────────────────────────────

def process_row(
    perturb_row: dict,
    question_type: str,       # "baseline" | "gender_swap"
    model_name: str,
    mode: str,
    bedrock_client,
    bedrock_model_id: Optional[str],
    augment_model,            # MEDREASON / MEDPATHAGENT / AGENTMD / TOOLUNI model, or None
    output_path: Path,
    lock: Lock,
    log_dir: Optional[Path] = None,   # if set, write per-row detail JSON for tool models
) -> dict:
    qid     = perturb_row["id"]
    label   = perturb_row["original_label"]
    options = json.loads(perturb_row["gender_swap_options"]) if question_type == "gender_swap" \
              else _rebuild_options(perturb_row["original_question"])

    if question_type == "baseline":
        question = perturb_row["original_question"]
    else:
        question = perturb_row["gender_swap_perturbed"]

    kg_paths = ""
    generated_code = ""

    if augment_model is not None:
        reasoning, letter, meta = run_via_model(augment_model, question, options)
        kg_paths = meta.get("kg_paths", "")
        rounds = meta.get("agentmd_rounds", [])
        generated_code = "\n---\n".join(
            f"[round {r['round']}] {r['exec_output']}"
            for r in rounds if r.get("exec_output")
        )
        # ── per-row tool log ──────────────────────────────────────────────────
        if log_dir is not None and mode in ("TOOLUNI", "AGENTMD"):
            _save_tool_log(log_dir, qid, question_type, mode, meta, lock)
    else:
        reasoning, letter = run_two_turn_bedrock(bedrock_client, bedrock_model_id, question, options)

    return {
        "id":                  qid,
        "question_type":       question_type,
        "model_answer_letter": letter,
        "correct_label":       label,
        "reasoning":           reasoning,
        "kg_paths":            kg_paths,
        "generated_code":      generated_code,
    }


def _save_tool_log(
    log_dir: Path,
    qid: str,
    question_type: str,
    mode: str,
    meta: dict,
    lock: Lock,
) -> None:
    """Write a detailed JSON tool log to logs/medxpertqa/<model_tag>/detailed_<id>_<qtype>.json."""
    if mode == "TOOLUNI":
        detail = {
            "id":            qid,
            "question_type": question_type,
            "tu_rounds":     meta.get("tu_rounds", []),      # [{step, plan, tool_context, status}]
            "tool_context":  meta.get("tool_context", ""),
            "trace_manage":  meta.get("trace_manage", ""),
            "trace_visit":   meta.get("trace_visit", ""),
            "trace_resource": meta.get("trace_resource", ""),
            "final_prompt":  meta.get("final_prompt", ""),   # full augmented Turn-1 prompt
        }
    else:  # AGENTMD
        detail = {
            "id":                  qid,
            "question_type":       question_type,
            "tool_id":             meta.get("tool_id", ""),
            "tool_title":          meta.get("tool_title", ""),
            "calculator_result":   meta.get("calculator_result", ""),
            "agentmd_n_rounds":    meta.get("agentmd_n_rounds", 0),
            "agentmd_code_executed":  meta.get("agentmd_code_executed", False),
            "agentmd_exec_had_error": meta.get("agentmd_exec_had_error", False),
            "agentmd_rounds":      meta.get("agentmd_rounds", []),  # [{round, tool_id, tool_title, code, exec_output, error}]
            "final_prompt":        meta.get("final_prompt", ""),    # prompt + calculator block sent to answer LLM
        }

    fname = log_dir / f"detailed_{qid}_{question_type}.json"
    with lock:
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2, default=str)


def _rebuild_options(question_text: str) -> Dict[str, str]:
    """Parse options from the embedded 'Answer Choices: (A) … (B) …' in question text."""
    opts: Dict[str, str] = {}
    for m in re.finditer(r'\(([A-J])\)\s*([^(]+)', question_text):
        opts[m.group(1)] = m.group(2).strip()
    return opts or {k: "" for k in "ABCDEFGHIJ"}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      default="AWS",
                        choices=["AWS", "TXAGENT_BEDROCK", "MEDREASON", "MEDPATHAGENT", "AGENTMD", "TOOLUNI"])
    parser.add_argument("--name",      default=None)
    parser.add_argument("--kg-path",   default="/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/kg.csv")
    parser.add_argument("--emb-path",  default="/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/node_embeddings_sapbert.pt")
    parser.add_argument("--poll",      type=int, default=60, help="Seconds between CSV polls")
    parser.add_argument("--limit",     type=int, default=None, help="Max questions (smoke test)")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume and reprocess all rows.")
    parser.add_argument("--perturb-csv", default=str(PERTURB_CSV))
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker threads (default 1). "
                             "Safe to increase for AGENTMD/TOOLUNI since Bedrock calls are I/O-bound.")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Fraction [0,1) of change=0 rows to start from (default 0.0).")
    parser.add_argument("--end",   type=float, default=1.0,
                        help="Fraction (0,1] of change=0 rows to end at (default 1.0).")
    args = parser.parse_args()

    setup_logging()

    # ── resolve model name ────────────────────────────────────────────────────
    default_name = {
        "AWS": "llama70b3_3", "TXAGENT_BEDROCK": "llama70b3_3",
        "MEDREASON": "llama70b3_3", "MEDPATHAGENT": "llama70b3_3",
        "AGENTMD": "llama70b3_3", "TOOLUNI": "llama70b3_3",
    }[args.mode]
    name = args.name or default_name

    # ── output path ───────────────────────────────────────────────────────────
    model_tag = f"{args.mode.lower()}_{name}"
    output_dir = ROOT / "output" / "medxpertqa" / model_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "results.csv"
    lock = Lock()

    # Tool-log directory (only written for TOOLUNI / AGENTMD)
    log_dir: Optional[Path] = None
    if args.mode in ("TOOLUNI", "AGENTMD"):
        log_dir = ROOT / "logs" / "medxpertqa" / model_tag
        log_dir.mkdir(parents=True, exist_ok=True)
        log.info("Tool logs → %s", log_dir)

    log.info("Mode=%s  Model=%s", args.mode, name)
    log.info("Output CSV → %s", output_csv)

    # ── build model / client ──────────────────────────────────────────────────
    cfg = get_default_config(name=name, mode=args.mode)
    mc  = cfg.model_configs[0]

    if args.kg_path:  mc.kg_path = args.kg_path
    if args.emb_path: mc.node_embeddings_path = args.emb_path

    bedrock_client   = None
    bedrock_model_id = None
    augment_model    = None

    if args.mode in ("MEDREASON", "MEDPATHAGENT", "AGENTMD", "TOOLUNI"):
        log.info("Loading %s model (may take a moment)...", args.mode)
        augment_model = ModelFactory.create_model(mc, cfg)
    else:
        bedrock_model_id = MODELS.get(name, name)
        region = os.environ.get("AWS_REGION", "us-east-1")
        bedrock_client = _make_bedrock_client(region)
        log.info("Bedrock client ready  model_id=%s", bedrock_model_id)

    perturb_path = Path(args.perturb_csv)
    processed: Set[Tuple[str, str]] = set() if args.no_resume else load_done(output_csv)
    log.info("Already processed: %d rows", len(processed))

    processed_count = 0

    is_sliced = (args.start > 0.0) or (args.end < 1.0)
    if is_sliced:
        log.info("Slice: rows [%.3f, %.3f) of the change=0 set", args.start, args.end)

    while True:
        change0_rows, total_in_perturb = load_perturb_csv(perturb_path)
        log.info("Perturb CSV: %d total rows, %d with change=0", total_in_perturb, len(change0_rows))

        # Apply slice [start, end) of the change=0 row list.
        # All shards write to one results.csv (fcntl-locked) so partitions are
        # disjoint and resume skips already-processed rows.
        if is_sliced:
            n = len(change0_rows)
            lo = int(args.start * n)
            hi = int(args.end   * n)
            change0_rows = change0_rows[lo:hi]
            log.info("After slice [%d, %d): %d rows for this job", lo, hi, len(change0_rows))

        # Build work queue: (perturb_row, question_type) not yet processed
        queue = []
        for row in change0_rows:
            for qtype in ("baseline", "gender_swap"):
                key = (row["id"], qtype)
                if key not in processed:
                    queue.append((row, qtype))

        if args.limit:
            queue = queue[:args.limit - processed_count]

        if queue:
            pbar = tqdm(total=len(queue), desc=f"[{model_tag}]", unit="q")

            def _process_one(item):
                perturb_row, qtype = item
                qid = perturb_row["id"]
                try:
                    result = process_row(
                        perturb_row, qtype, name, args.mode,
                        bedrock_client, bedrock_model_id, augment_model,
                        output_csv, lock,
                        log_dir=log_dir,
                    )
                    append_result(output_csv, result, lock)
                    with lock:
                        processed.add((qid, qtype))
                    tqdm.write(
                        f"  {qid} [{qtype}] → {result['model_answer_letter']} "
                        f"(gold={result['correct_label']})"
                    )
                except Exception as exc:
                    tqdm.write(f"  ERROR {qid} [{qtype}]: {exc}")
                    append_result(output_csv, {
                        "id": qid, "question_type": qtype,
                        "model_answer_letter": "?", "correct_label": perturb_row["original_label"],
                        "reasoning": f"ERROR: {exc}", "kg_paths": "", "generated_code": "",
                    }, lock)
                    with lock:
                        processed.add((qid, qtype))
                finally:
                    pbar.update(1)

            n_workers = max(1, args.workers)
            if n_workers == 1:
                for item in queue:
                    _process_one(item)
                    time.sleep(0.3)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = [ex.submit(_process_one, item) for item in queue]
                    concurrent.futures.wait(futures)

            with lock:
                processed_count += len(queue)
            pbar.close()

        # ── stop condition ────────────────────────────────────────────────────
        if total_in_perturb >= TOTAL_ROWS:
            log.info("Perturb CSV complete (%d rows). Finishing.", total_in_perturb)
            break

        if args.limit and processed_count >= args.limit:
            log.info("Limit reached (%d). Stopping.", args.limit)
            break

        # When sliced (parallel run), the perturb CSV is fixed — don't poll.
        if is_sliced and not queue:
            log.info("Slice empty — finishing.")
            break

        log.info("Perturb CSV not complete yet (%d/%d). Polling again in %ds...",
                 total_in_perturb, TOTAL_ROWS, args.poll)
        time.sleep(args.poll)

    log.info("Done. Results → %s", output_csv)


if __name__ == "__main__":
    main()
