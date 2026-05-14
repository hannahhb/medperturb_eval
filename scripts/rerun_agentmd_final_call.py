"""
Rerun only the final LLM call (Step 3) of AgentMD for rows up to Text-1789,
using the already-computed `generated_code` (exec outputs) stored in the CSV.

This skips tool selection and code execution entirely — it just rebuilds the
augmented prompt (question + options + calculator result) and calls the LLM,
then overwrites model_answer_letter, correct_label, reasoning in the CSV while
preserving generated_code and all other columns.

Rows with empty generated_code are skipped (no computation result available).

Usage:
    python rerun_agentmd_final_call.py [--workers N] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, Tuple

import boto3
from botocore.config import Config as BotoConfig

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "output/medxpertqa/agentmd_llama70b3_3/results.csv"
PERTURB_CSV = ROOT.parent / "data" / "medxpertqa_gender_perturb.csv"

MODEL_ID    = "us.meta.llama3-3-70b-instruct-v1:0"
REGION      = os.environ.get("AWS_REGION", "us-east-1")
SYSTEM_PROMPT = "You are a helpful medical assistant."
MAX_ID      = 1789   # inclusive upper bound on Text-N

FIELDS = ["id", "question_type", "model_answer_letter", "correct_label",
          "reasoning", "kg_paths", "generated_code"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rerun_agentmd")


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=REGION,
        config=BotoConfig(
            retries={"max_attempts": 5, "mode": "adaptive"},
            connect_timeout=30,
            read_timeout=120,
        ),
    )


def _extract_letter(text: str, options: Dict[str, str]) -> str:
    valid = set(options.keys())
    for pattern in [
        r"the answer is\s*[:\(]?\s*([A-J])\b",
        r"answer[:\s]+\(?([A-J])\)?",
        r"Therefore[^.]*\b([A-J])\b",
        r"correct answer[:\s]+\(?([A-J])\)?",
    ]:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if m.group(1).upper() in valid:
                return m.group(1).upper()
    matches = [m for m in re.finditer(r'\b([A-J])\b', text) if m.group(1) in valid]
    if matches:
        return matches[-1].group(1)
    return "?"


def _fmt_options(options: Dict[str, str]) -> str:
    return "\n".join(f"({k}) {v}" for k, v in options.items())


TURN2 = "Based on your reasoning above, reply with the single letter of the correct answer and nothing else."


def _rebuild_options(question_text: str) -> Dict[str, str]:
    """Parse inline '(A) ... (B) ...' options from the original question string."""
    opts = {}
    for m in re.finditer(r'\(([A-J])\)\s*([^(]+)', question_text):
        opts[m.group(1)] = m.group(2).strip()
    return opts


def _llm_call(client, prompt: str, max_tokens: int = 1024) -> str:
    for attempt in range(3):
        try:
            resp = client.converse(
                modelId=MODEL_ID,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
            )
            return resp["output"]["message"]["content"][0]["text"].strip()
        except Exception as exc:
            log.warning("Bedrock attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2 ** attempt)
    raise RuntimeError("Bedrock call failed after 3 retries")


def _run_final_call(
    client,
    question: str,
    options: Dict[str, str],
    calculator_result: str,
) -> Tuple[str, str]:
    """
    Reconstruct the augmented prompt (same as _answer_with_calculator) and run
    the two-turn CoT:
      Turn 1: question + options + calculator result → reasoning
      Turn 2: reasoning → single letter
    Returns (reasoning, letter).
    """
    turn1_base = (
        f"Q: {question}\n\n"
        f"Options:\n{_fmt_options(options)}\n\n"
        f"A: Let's think step by step."
    )
    calc_block = f"Risk Calculator Result:\n{calculator_result.strip()[:1000]}"
    augmented = f"{turn1_base}\n\n{calc_block}"

    reasoning = _llm_call(client, augmented, max_tokens=1024)

    turn2_prompt = f"{augmented}\n\n{reasoning}\n\n{TURN2}"
    letter_raw = _llm_call(client, turn2_prompt, max_tokens=4)
    letter = _extract_letter(letter_raw, options)

    return reasoning, letter


# ── load perturb CSV for question text + options ──────────────────────────────

def load_perturb_index(path: Path) -> Dict[Tuple[str, str], dict]:
    """Return {(id, question_type): perturb_row} for quick lookup."""
    index = {}
    if not path.exists():
        raise FileNotFoundError(f"Perturb CSV not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("change", "")).strip() != "0":
                continue
            qid = row["id"]
            index[(qid, "baseline")]    = row
            index[(qid, "gender_swap")] = row
    return index


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",  type=int, default=4,
                        help="Parallel worker threads (default 4)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print what would be done without calling the LLM or writing")
    args = parser.parse_args()

    # ── load existing results ─────────────────────────────────────────────────
    log.info("Loading results CSV: %s", RESULTS_CSV)
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    log.info("Total rows in CSV: %d", len(all_rows))

    # ── filter to rows with id <= MAX_ID ─────────────────────────────────────
    target_rows = []
    for row in all_rows:
        try:
            n = int(row["id"].replace("Text-", ""))
        except ValueError:
            continue
        if n <= MAX_ID:
            target_rows.append(row)
    log.info("Rows with id <= Text-%d: %d", MAX_ID, len(target_rows))

    # ── skip rows with no generated_code ─────────────────────────────────────
    to_rerun = [r for r in target_rows if r.get("generated_code", "").strip()]
    skipped  = len(target_rows) - len(to_rerun)
    log.info("To rerun: %d  (skipping %d with empty generated_code)", len(to_rerun), skipped)

    if args.dry_run:
        log.info("Loading perturb CSV for dry-run prompt preview: %s", PERTURB_CSV)
        perturb = load_perturb_index(PERTURB_CSV)
        print(f"\nDRY RUN — showing full prompt for first 3 rows to rerun ({len(to_rerun)} total)\n")
        for r in to_rerun[:3]:
            qid   = r["id"]
            qtype = r["question_type"]
            perturb_row = perturb.get((qid, qtype))
            if perturb_row is None:
                print(f"[{qid}/{qtype}] — no perturb row found\n")
                continue

            if qtype == "gender_swap":
                question = perturb_row["gender_swap_perturbed"]
                try:
                    options = json.loads(perturb_row["gender_swap_options"])
                except Exception:
                    options = _rebuild_options(question)
            else:
                question = perturb_row["original_question"]
                options  = _rebuild_options(question)

            raw_code = r["generated_code"]
            parts = re.split(r"\n---\n", raw_code)
            exec_outputs = []
            for part in parts:
                cleaned = re.sub(r"^\[round\s*\d+\]\s*", "", part.strip())
                if cleaned:
                    exec_outputs.append(cleaned)
            calculator_result = "\n\n".join(exec_outputs) or raw_code

            turn1_base = (
                f"Q: {question}\n\n"
                f"Options:\n{_fmt_options(options)}\n\n"
                f"A: Let's think step by step."
            )
            calc_block = f"Risk Calculator Result:\n{calculator_result.strip()[:1000]}"
            augmented  = f"{turn1_base}\n\n{calc_block}"

            sep = "=" * 72
            print(f"{sep}")
            print(f"  {qid}  |  {qtype}")
            print(f"{sep}")
            print(augmented)
            print()
        return

    # ── load perturb index ────────────────────────────────────────────────────
    log.info("Loading perturb CSV: %s", PERTURB_CSV)
    perturb = load_perturb_index(PERTURB_CSV)

    # ── build an index into all_rows for in-place update ─────────────────────
    # Key: (id, question_type) → list of indices (there can be duplicates; update all)
    row_index: Dict[Tuple[str, str], list] = {}
    for i, row in enumerate(all_rows):
        key = (row["id"], row["question_type"])
        row_index.setdefault(key, []).append(i)

    lock = threading.Lock()
    done_count = [0]
    error_count = [0]

    # One Bedrock client per thread
    _thread_local = threading.local()

    def get_client():
        if not hasattr(_thread_local, "client"):
            _thread_local.client = _make_client()
        return _thread_local.client

    def rerun_one(row: dict) -> None:
        qid   = row["id"]
        qtype = row["question_type"]
        key   = (qid, qtype)

        perturb_row = perturb.get(key)
        if perturb_row is None:
            log.warning("No perturb row for %s/%s — skipping", qid, qtype)
            with lock:
                error_count[0] += 1
            return

        # Reconstruct question and options
        if qtype == "gender_swap":
            question = perturb_row["gender_swap_perturbed"]
            try:
                options = json.loads(perturb_row["gender_swap_options"])
            except Exception:
                options = _rebuild_options(question)
        else:
            question = perturb_row["original_question"]
            options  = _rebuild_options(question)

        if not options:
            log.warning("Could not parse options for %s/%s — skipping", qid, qtype)
            with lock:
                error_count[0] += 1
            return

        label = perturb_row["original_label"]

        # Calculator result = strip the [round N] prefixes, join exec outputs
        raw_code = row["generated_code"]
        # Format: "[round 0] <output>\n---\n[round 1] <output>"
        parts = re.split(r"\n---\n", raw_code)
        exec_outputs = []
        for part in parts:
            # Strip leading "[round N] " tag
            cleaned = re.sub(r"^\[round\s*\d+\]\s*", "", part.strip())
            if cleaned:
                exec_outputs.append(cleaned)
        calculator_result = "\n\n".join(exec_outputs) or raw_code

        try:
            client = get_client()
            reasoning, letter = _run_final_call(client, question, options, calculator_result)
        except Exception as exc:
            log.error("LLM call failed for %s/%s: %s", qid, qtype, exc)
            with lock:
                error_count[0] += 1
            return

        # Update all matching rows in all_rows in-place
        with lock:
            for idx in row_index.get(key, []):
                all_rows[idx]["model_answer_letter"] = letter
                all_rows[idx]["correct_label"]       = label
                all_rows[idx]["reasoning"]            = reasoning
            done_count[0] += 1
            if done_count[0] % 50 == 0:
                log.info("Progress: %d / %d", done_count[0], len(to_rerun))

    # ── run ───────────────────────────────────────────────────────────────────
    log.info("Starting with %d worker(s)…", args.workers)
    if args.workers == 1:
        for row in to_rerun:
            rerun_one(row)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(rerun_one, row) for row in to_rerun]
            concurrent.futures.wait(futures)

    log.info("Done. Reran %d rows, %d errors.", done_count[0], error_count[0])

    # ── write back the full CSV ───────────────────────────────────────────────
    log.info("Writing updated CSV to %s", RESULTS_CSV)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    log.info("CSV written (%d rows total).", len(all_rows))


if __name__ == "__main__":
    main()
