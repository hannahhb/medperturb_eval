"""
Run models on CureBench validation set (phase 1, 459 MCQ questions).

Two-turn, no CoT:
  Turn 1: "You are a medical expert… {augmentation?} {question} {options}"
           → augmented models prepend KG/tool/calculator context
  Turn 2: Extract single letter from Turn 1 response (max_tokens=2)

Modes:
  AWS          — plain Bedrock Llama (baseline)
  MEDREASON    — KG-augmented reasoning
  MEDPATHAGENT — combined KG + FDA tool retrieval
  TOOLUNI      — FDA tool retrieval (ToolUniverse)
  AGENTMD      — MedCPT risk calculator + code execution

Resume is always on: rows already present in results.csv are skipped.

--seed-csv PATH
  Pre-populate results from an existing CSV (e.g. a prior MedReason run).
  Rows whose id already appears in the seed are written directly to the output
  CSV and skipped during inference.  Expected columns: id, choice, correct_answer.
  Use this to reuse existing MedReason predictions without re-running.

Usage:
  python run_curebench.py --mode AWS
  python run_curebench.py --mode MEDREASON --seed-csv /path/to/result.csv
  python run_curebench.py --mode TOOLUNI --workers 4
  python run_curebench.py --mode AGENTMD
  python run_curebench.py --mode MEDPATHAGENT \\
      --medreason-csv output/medreason_llama70b3_3/oncqa_askdocs/run_0/results.csv
  python run_curebench.py --mode AWS --limit 5   # smoke test
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be",
                        category=UserWarning)

import boto3
from botocore.config import Config as BotoConfig

from scripts.logging_setup import setup_logging
from config import MODELS, get_default_config
from factory import ModelFactory
from prompts import (
    CUREBENCH_SYSTEM_PROMPT,
    CUREBENCH_EXTRACT_PROMPT,
    CUREBENCH_TOOLUNI_RETRIEVAL_QUERY,
    build_curebench_prompt,
)

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parent
DATA_DIR  = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data")
VAL_JSONL = DATA_DIR / "curebench_valset_phase1.jsonl"

log = logging.getLogger("run_curebench")

# ── output columns ────────────────────────────────────────────────────────────

FIELDS = ["id", "choice", "correct_answer", "correct", "reasoning",
          "kg_paths", "tool_context"]


# ── data loading ──────────────────────────────────────────────────────────────

def load_questions(path: Path) -> List[dict]:
    import jsonlines
    with jsonlines.open(path) as r:
        return list(r)


def load_done(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        done = set()
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("id"):
                    done.add(row["id"])
        log.info("Resume: %d rows already in %s", len(done), path)
        return done
    except Exception as e:
        log.warning("Could not read results CSV (%s) — starting fresh", e)
        return set()


def load_kg_paths_jsonl(path: str) -> Dict[str, str]:
    """
    Load KG paths from a MedPathAgent log.jsonl keyed by question id.
    Each line: {"id": ..., "paths": "entity -> rel -> entity\n...", ...}
    "nopaths" or empty strings are stored as "" so callers can filter them.
    Returns {id: paths_text}.
    """
    lookup: Dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj  = json.loads(line)
                qid  = (obj.get("id") or "").strip()
                raw  = (obj.get("paths") or "").strip()
                if qid:
                    lookup[qid] = "" if raw.lower() == "nopaths" else raw
        n_paths = sum(1 for v in lookup.values() if v)
        log.info("KG paths JSONL: %d entries, %d with actual paths  (%s)",
                 len(lookup), n_paths, path)
    except Exception as e:
        log.warning("Could not read KG paths JSONL %s (%s)", path, e)
    return lookup


def load_kg_paths_csv(path: str) -> Dict[str, str]:
    """
    Load KG paths from a MedReason results CSV that has 'id' and 'kg_paths' columns.
    Empty / "nopaths" values are stored as "" so callers can filter them out.
    Returns {id: paths_text}.
    """
    lookup: Dict[str, str] = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qid = (row.get("id") or "").strip()
                raw = (row.get("kg_paths") or "").strip()
                if qid:
                    lookup[qid] = "" if raw.lower() == "nopaths" else raw
        n_paths = sum(1 for v in lookup.values() if v)
        log.info("KG paths CSV: %d entries, %d with actual paths  (%s)",
                 len(lookup), n_paths, path)
    except Exception as e:
        log.warning("Could not read KG paths CSV %s (%s)", path, e)
    return lookup


def _load_tool_context_csv(path: str) -> Dict[str, str]:
    """Load id → tool_context from a ToolUni results CSV."""
    lookup: Dict[str, str] = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qid = (row.get("id") or "").strip()
                tc  = (row.get("tool_context") or "").strip()
                if qid:
                    lookup[qid] = tc
    except Exception as e:
        log.warning("Could not read ToolUni CSV %s: %s", path, e)
    return lookup


def load_raw_tool_outputs(log_dir: str) -> Dict[str, str]:
    """
    Load id → concatenated raw tool outputs from a ToolUni log directory.
    Each tool_<id>.json contains tu_rounds[*].tools[*].output (raw FDA text).
    This preserves details (e.g. "(Nighttime only)") that the summariser drops.
    Returns {id: raw_concatenated_output}.
    """
    import glob as _glob, json as _json
    lookup: Dict[str, str] = {}
    for fpath in _glob.glob(os.path.join(log_dir, "tool_*.json")):
        try:
            with open(fpath, encoding="utf-8") as f:
                data = _json.load(f)
            qid = (data.get("id") or "").strip()
            if not qid:
                # derive id from filename: tool_<id>.json
                qid = os.path.basename(fpath)[5:-5]
            outputs = []
            for rnd in data.get("tu_rounds", []):
                for t in rnd.get("tools", []):
                    out = (t.get("output") or "").strip()
                    if out:
                        outputs.append(out)
            lookup[qid] = "\n\n".join(outputs)
        except Exception as e:
            log.warning("Could not read ToolUni log %s: %s", fpath, e)
    n_with = sum(1 for v in lookup.values() if v)
    log.info("Raw ToolUni outputs: %d logs, %d with content  (%s)", len(lookup), n_with, log_dir)
    return lookup


def load_seed_csv(path: str) -> Dict[str, dict]:
    """
    Load an existing result CSV (e.g. prior MedReason run) keyed by id.
    Expected columns: id, choice, correct_answer (and optionally reasoning, kg_paths).
    Returns {id: row_dict}.
    """
    out: Dict[str, dict] = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qid = (row.get("id") or "").strip()
                if qid:
                    out[qid] = row
        log.info("Seed CSV: loaded %d rows from %s", len(out), path)
    except Exception as e:
        log.warning("Could not read seed CSV %s (%s)", path, e)
    return out


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def append_result(path: Path, row: dict, lock: Lock) -> None:
    with lock:
        with open(path, "a", newline="", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                write_header = f.tell() == 0
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                if write_header:
                    w.writeheader()
                w.writerow(row)
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


# ── letter extraction ─────────────────────────────────────────────────────────

def extract_letter(text: str, valid: set) -> str:
    """Extract a valid answer letter from model output."""
    # Explicit patterns first
    for pattern in [
        r"\bthe answer is\s*[:\(]?\s*([A-J])\b",
        r"\banswer[:\s]+\(?([A-J])\)?",
        r"^([A-J])[.:\)]",          # letter at start of line
        r"\b([A-J])\b",             # any standalone letter
    ]:
        for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            if m.group(1).upper() in valid:
                return m.group(1).upper()
    return "?"


# ── Bedrock client ────────────────────────────────────────────────────────────

def make_bedrock_client(region: str):
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=BotoConfig(
            retries={"max_attempts": 5, "mode": "adaptive"},
            connect_timeout=30,
            read_timeout=120,
        ),
    )


def bedrock_chat(client, model_id: str, messages: list,
                 max_tokens: int = 512, temperature: float = 0.0) -> str:
    resp = client.converse(
        modelId=model_id,
        system=[{"text": CUREBENCH_SYSTEM_PROMPT}],
        messages=messages,
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    return resp["output"]["message"]["content"][0]["text"].strip()


def run_two_turn_bedrock(
    client, model_id: str, question: str, options: dict, augmentation: str = ""
) -> Tuple[str, str]:
    """
    Two-turn, no CoT.
    Turn 1: answer prompt (+ optional augmentation prepended)
    Turn 2: extract single letter
    Returns (reasoning, letter).
    """
    valid = set(options.keys())
    turn1_text = build_curebench_prompt(question, options, augmentation)

    msgs = [{"role": "user", "content": [{"text": turn1_text}]}]
    reasoning = bedrock_chat(client, model_id, msgs, max_tokens=512)

    # Turn 2: extract letter
    msgs += [
        {"role": "assistant", "content": [{"text": reasoning}]},
        {"role": "user",      "content": [{"text": CUREBENCH_EXTRACT_PROMPT}]},
    ]
    raw2 = bedrock_chat(client, model_id, msgs, max_tokens=2)
    letter = extract_letter(raw2, valid) or extract_letter(reasoning, valid)
    return reasoning, letter


# ── augmented model runner ────────────────────────────────────────────────────

def run_via_model(
    model,
    bedrock_client,
    bedrock_model_id: str,
    question: str,
    options: dict,
    kg_paths: str = "",
    cached_tool_context: str = "",
) -> Tuple[str, str, dict]:
    """
    Turn 1: call model.generate(prompt=turn1_text, retrieval_query=question,
            kg_paths=kg_paths) to get augmented reasoning.
    Turn 2: plain Bedrock call to extract letter (cheap, no augmentation needed).
    Returns (reasoning, letter, metadata).
    """
    valid = set(options.keys())

    turn1_text = build_curebench_prompt(question, options)

    options_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    retrieval_query = f"Question: {question}. Options: {options_text}"

    result = model.generate(
        prompt=turn1_text,
        system_prompt=CUREBENCH_SYSTEM_PROMPT,
        retrieval_query=retrieval_query,
        kg_paths=kg_paths,
        cached_tool_context=cached_tool_context,
    )
    reasoning = result.reasoning
    meta = getattr(result, "metadata", {}) or {}

    # Turn 2: extract letter via plain Bedrock (no augmentation needed)
    msgs = [
        {"role": "user",      "content": [{"text": turn1_text}]},
        {"role": "assistant", "content": [{"text": reasoning}]},
        {"role": "user",      "content": [{"text": CUREBENCH_EXTRACT_PROMPT}]},
    ]
    raw2 = bedrock_chat(bedrock_client, bedrock_model_id, msgs, max_tokens=2)
    letter = extract_letter(raw2, valid) or extract_letter(reasoning, valid)

    return reasoning, letter, meta


# ── tool log ──────────────────────────────────────────────────────────────────

def save_tool_log(log_dir: Path, qid: str, mode: str, meta: dict, lock: Lock) -> None:
    """Write a per-question detail JSON to logs/curebench/<model_tag>/tool_<id>.json."""
    if not meta:
        return

    if meta.get("tooluni") or meta.get("medpathagent"):
        detail = {
            "id":           qid,
            "tu_rounds":    meta.get("tu_rounds", []),
            "tool_context": meta.get("tool_context", ""),
            "kg_paths":     meta.get("kg_paths", ""),
            "final_prompt": meta.get("final_prompt", ""),
        }
    elif meta.get("medreason"):
        detail = {
            "id":       qid,
            "kg_paths": meta.get("kg_paths", ""),
        }
    elif "tool_id" in meta:   # AgentMD
        detail = {
            "id":                     qid,
            "tool_id":                meta.get("tool_id", ""),
            "tool_title":             meta.get("tool_title", ""),
            "calculator_result":      meta.get("calculator_result", ""),
            "agentmd_n_rounds":       meta.get("agentmd_n_rounds", 0),
            "agentmd_code_executed":  meta.get("agentmd_code_executed", False),
            "agentmd_exec_had_error": meta.get("agentmd_exec_had_error", False),
            "agentmd_rounds":         meta.get("agentmd_rounds", []),
        }
    else:
        return   # nothing worth logging

    log_dir.mkdir(parents=True, exist_ok=True)
    fname = log_dir / f"tool_{qid}.json"
    with lock:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2, default=str)


# ── process one question ───────────────────────────────────────────────────────

def process_question(
    q: dict,
    mode: str,
    bedrock_client,
    bedrock_model_id: str,
    augment_model,                           # None for AWS baseline
    kg_paths: str = "",                      # pre-extracted KG paths for MEDPATHAGENT
    cached_tool_context: str = "",           # pre-loaded ToolUni tool_context
    log_dir: Optional[Path] = None,
    lock: Optional[Lock] = None,
) -> dict:
    qid          = q["id"]
    question     = q["question"]
    options      = q.get("options", {})
    correct      = q.get("correct_answer", "")

    tool_context = ""
    meta: dict   = {}

    if augment_model is not None:
        reasoning, letter, meta = run_via_model(
            augment_model, bedrock_client, bedrock_model_id, question, options,
            kg_paths=kg_paths,
            cached_tool_context=cached_tool_context,
        )
        kg_paths     = meta.get("kg_paths", kg_paths)
        tool_context = meta.get("tool_context", "")
    else:
        # Baseline: plain two-turn Bedrock, no augmentation
        reasoning, letter = run_two_turn_bedrock(
            bedrock_client, bedrock_model_id, question, options
        )

    # Write per-question tool log for augmented modes
    if log_dir is not None and lock is not None and meta:
        save_tool_log(log_dir, qid, mode, meta, lock)

    is_correct = 1 if letter == correct else 0

    return {
        "id":             qid,
        "choice":         letter,
        "correct_answer": correct,
        "correct":        is_correct,
        "reasoning":      reasoning,
        "kg_paths":       kg_paths,
        "tool_context":   tool_context,
    }


# ── seed row adapter ──────────────────────────────────────────────────────────

def adapt_seed_row(seed_row: dict, q: dict) -> dict:
    """
    Convert a seed CSV row (from e.g. old MedReason/MedPathAgent run) into
    our standard FIELDS dict.  The old CSV columns are:
      id, choice, prediction, reasoning
    We need to add correct_answer, correct, tool_context, kg_paths.
    """
    qid     = (seed_row.get("id") or q["id"]).strip()
    choice  = (seed_row.get("choice") or "").strip().upper()
    correct = q.get("correct_answer", "")
    return {
        "id":             qid,
        "choice":         choice,
        "correct_answer": correct,
        "correct":        1 if choice == correct else 0,
        "reasoning":      seed_row.get("reasoning", seed_row.get("prediction", "")),
        "kg_paths":       seed_row.get("kg_paths", ""),
        "tool_context":   seed_row.get("tool_context", ""),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run models on CureBench val set (MCQ, no CoT, two-turn letter extraction)"
    )
    parser.add_argument("--mode", default="AWS",
                        choices=["AWS", "MEDREASON", "MEDPATHAGENT", "TOOLUNI", "AGENTMD", "KGRANK"],
                        help="Model mode (default: AWS = plain Bedrock baseline)")
    parser.add_argument("--name", default=None,
                        help="Model key from MODELS dict (default: llama70b3_3)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max questions to process (for smoke testing)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker threads (default: 1)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Reprocess all rows (ignore existing results)")
    parser.add_argument("--seed-csv", default=None,
                        help="Pre-populate results from an existing CSV (skips inference for matching ids). "
                             "Useful for reusing prior MedReason results without re-running.")
    parser.add_argument("--kg-paths-jsonl", default=None,
                        help="Path to a MedPathAgent log.jsonl containing pre-extracted "
                             "KG paths (field: 'paths') keyed by question id. "
                             "Used for MEDPATHAGENT mode. "
                             "Default: MedPathAgent run0_val/log.jsonl")
    parser.add_argument("--baseline-csv", default=None,
                        help="Path to a baseline (AWS) results CSV. "
                             "In MEDPATHAGENT mode, questions with no KG paths are "
                             "copied directly from this CSV (no re-inference needed).")
    parser.add_argument("--medreason-csv", default=None,
                        help="Path to MedReason results CSV with 'id' and 'kg_paths' columns. "
                             "For MEDPATHAGENT: used as the KG paths source (preferred over "
                             "--kg-paths-jsonl for CureBench runs). "
                             "Example: output/curebench/medreason_llama70b3_3/results_0.csv")
    parser.add_argument("--mpa-mode", default="dual",
                        choices=["dual", "cached_dual", "paths_only"],
                        help="MedPathAgent retrieval mode (default: dual)")
    parser.add_argument("--tooluni-log-dir", default=None,
                        help="Directory of ToolUni log JSONs (tool_<id>.json). "
                             "MEDPATHAGENT loads raw tool outputs from here (preferred over "
                             "--tooluni-csv which only has the summarised text). "
                             "Example: logs/curebench/tooluni_llama70b3_3")
    parser.add_argument("--tooluni-csv", default=None,
                        help="Path to a ToolUni results CSV with 'id' and 'tool_context' columns. "
                             "Used by MEDPATHAGENT: cached ToolUni tool outputs are combined "
                             "with KG-guided novel tools before summarisation. "
                             "Example: output/curebench/tooluni_llama70b3_3/results.csv")
    parser.add_argument("--no-summarise", action="store_true",
                        help="Skip the tool-context summariser and pass raw FDA output "
                             "directly to the LLM. Preserves specific numbers, qualifiers, "
                             "and drug names that the summariser may compress away.")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Slice start [0,1] for parallel runs")
    parser.add_argument("--end", type=float, default=1.0,
                        help="Slice end [0,1] for parallel runs")
    parser.add_argument("--filter-ids", default=None,
                        help="Comma-separated question IDs to process. "
                             "All other questions are skipped. "
                             "Useful for targeted re-runs on failure cases.")
    args = parser.parse_args()

    setup_logging()

    name = args.name or "llama70b3_3"
    model_tag = f"{args.mode.lower()}_{name}"

    out_dir = ROOT / "output" / "curebench" / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "results.csv"
    lock    = Lock()

    log_dir: Optional[Path] = None
    if args.mode in ("TOOLUNI", "AGENTMD", "MEDPATHAGENT", "MEDREASON", "KGRANK"):
        log_dir = ROOT / "logs" / "curebench" / model_tag
        log_dir.mkdir(parents=True, exist_ok=True)

    log.info("Mode=%s  Model=%s  Output=%s", args.mode, name, out_csv)
    if log_dir:
        log.info("Tool logs → %s", log_dir)

    # ── build Bedrock client (always needed for Turn 2 letter extraction) ─────
    region           = os.environ.get("AWS_REGION", "us-east-2")
    bedrock_model_id = MODELS.get(name, name)
    bedrock_client   = make_bedrock_client(region)
    log.info("Bedrock client ready  model_id=%s", bedrock_model_id)

    # ── build augmented model (None for baseline AWS) ─────────────────────────
    augment_model = None
    if args.mode != "AWS":
        cfg = get_default_config(name=name, mode=args.mode)
        mc  = cfg.model_configs[0]

        if args.mode == "MEDPATHAGENT":
            mc.mpa_retrieval_mode = args.mpa_mode
            if args.medreason_csv:
                cfg.medreason_csv_path = args.medreason_csv
            if args.mpa_mode == "cached_dual":
                if not args.tooluni_log_dir:
                    raise ValueError("--tooluni-log-dir is required for cached_dual mode")
                cfg.tooluni_log_path = args.tooluni_log_dir

        log.info("Loading %s model...", args.mode)
        augment_model = ModelFactory.create_model(mc, cfg)
        log.info("%s model ready", args.mode)

        if args.no_summarise and hasattr(augment_model, "_tool_reasoner"):
            augment_model._tool_reasoner.summarise_tool_context = lambda text, **_: text
            log.info("--no-summarise: tool summariser disabled, raw FDA output passed to LLM")

    # ── load questions ────────────────────────────────────────────────────────
    questions = load_questions(VAL_JSONL)
    log.info("Loaded %d questions from %s", len(questions), VAL_JSONL)

    # Apply slice
    n = len(questions)
    s = int(round(n * args.start))
    e = int(round(n * args.end))
    questions = questions[s:e]
    log.info("Slice [%d:%d] → %d questions", s, e, len(questions))

    if args.limit:
        questions = questions[:args.limit]
        log.info("Limit: %d questions", len(questions))

    # ── seed from existing CSV ────────────────────────────────────────────────
    seed_rows: Dict[str, dict] = {}
    if args.seed_csv:
        seed_rows = load_seed_csv(args.seed_csv)

    # ── resume ────────────────────────────────────────────────────────────────
    done_ids: Set[str] = set() if args.no_resume else load_done(out_csv)

    # Write seed rows for any not already in output
    q_by_id = {q["id"]: q for q in questions}
    seeded = 0
    for qid, seed_row in seed_rows.items():
        if qid in done_ids:
            continue
        if qid not in q_by_id:
            continue
        row = adapt_seed_row(seed_row, q_by_id[qid])
        append_result(out_csv, row, lock)
        done_ids.add(qid)
        seeded += 1
    if seeded:
        log.info("Seeded %d rows from %s", seeded, args.seed_csv)

    # ── KG paths + cached ToolUni tool_context lookup (MEDPATHAGENT only) ──────
    kg_paths_lookup:    Dict[str, str] = {}
    tooluni_tc_lookup:  Dict[str, str] = {}
    if args.mode == "MEDPATHAGENT":
        if args.medreason_csv:
            kg_paths_lookup = load_kg_paths_csv(args.medreason_csv)
        else:
            jsonl_path = args.kg_paths_jsonl or str(
                Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4")
                / "MedPathAgent/output/run0_val/log.jsonl"
            )
            kg_paths_lookup = load_kg_paths_jsonl(jsonl_path)

        if args.tooluni_log_dir:
            # Preferred: load raw tool outputs from log JSONs (preserves all FDA detail)
            tooluni_tc_lookup = load_raw_tool_outputs(args.tooluni_log_dir)
        elif args.tooluni_csv:
            # Fallback: summarised tool_context from results CSV
            tooluni_tc_lookup = _load_tool_context_csv(args.tooluni_csv)
            log.info("Loaded ToolUni tool_context for %d rows", len(tooluni_tc_lookup))

    # Filter to remaining questions
    remaining = [q for q in questions if q["id"] not in done_ids]
    log.info("%d questions remaining after resume/seed", len(remaining))

    # --filter-ids: restrict to a specific set of question IDs
    if args.filter_ids:
        filter_set = {i.strip() for i in args.filter_ids.split(",") if i.strip()}
        remaining = [q for q in remaining if q["id"] in filter_set]
        log.info("--filter-ids: restricted to %d questions", len(remaining))

    # MEDPATHAGENT: skip rows with no KG paths entirely — caller will merge
    # results from another model (e.g. ToolUni) for those rows afterwards.
    if args.mode == "MEDPATHAGENT" and kg_paths_lookup:
        before = len(remaining)
        remaining = [q for q in remaining if kg_paths_lookup.get(q["id"], "").strip()]
        skipped = before - len(remaining)
        if skipped:
            log.info("MEDPATHAGENT: skipping %d no-paths rows (merge from another model later)",
                     skipped)

    if not remaining:
        log.info("Nothing to process.")
    else:
        # ── print summary ─────────────────────────────────────────────────────
        print(f"\n{'='*55}")
        print(f"  CureBench Eval  [{args.mode}]")
        print(f"  Model    : {name}  ({bedrock_model_id})")
        print(f"  Questions: {len(remaining)} remaining / {len(questions)} total")
        print(f"  Workers  : {args.workers}")
        print(f"  Output   : {out_csv}")
        print(f"{'='*55}\n")

        # ── process ───────────────────────────────────────────────────────────
        from tqdm import tqdm
        pbar = tqdm(total=len(remaining), desc=f"[{model_tag}]", unit="q")

        def _work(q: dict) -> None:
            qid = q["id"]
            try:
                result = process_question(
                    q, args.mode, bedrock_client, bedrock_model_id, augment_model,
                    kg_paths=kg_paths_lookup.get(qid, ""),
                    cached_tool_context=tooluni_tc_lookup.get(qid, ""),
                    log_dir=log_dir,
                    lock=lock,
                )
                append_result(out_csv, result, lock)
                tqdm.write(
                    f"  {qid} → {result['choice']} "
                    f"(gold={result['correct_answer']}, "
                    f"{'✓' if result['correct'] else '✗'})"
                )
            except Exception as exc:
                log.error("Error on %s: %s", qid, exc)
                append_result(out_csv, {
                    "id": qid, "choice": "?",
                    "correct_answer": q.get("correct_answer", ""),
                    "correct": 0, "reasoning": f"ERROR: {exc}",
                    "kg_paths": "", "tool_context": "",
                }, lock)
            finally:
                pbar.update(1)

        if args.workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(_work, q) for q in remaining]
                concurrent.futures.wait(futures)
        else:
            for q in remaining:
                _work(q)
                time.sleep(0.2)

        pbar.close()

    # ── final accuracy ────────────────────────────────────────────────────────
    try:
        import pandas as pd
        df = pd.read_csv(out_csv, dtype=str)
        df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
        n_total   = len(df)
        n_correct = int(df["correct"].sum())
        acc       = n_correct / n_total if n_total else 0.0
        print(f"\n{'='*55}")
        print(f"  DONE  —  {n_total} questions")
        print(f"  Accuracy : {acc:.4f}  ({n_correct}/{n_total})")
        print(f"  Results  : {out_csv}")
        print(f"{'='*55}\n")
    except Exception as e:
        log.warning("Could not compute final accuracy: %s", e)
        print(f"\nResults → {out_csv}")


if __name__ == "__main__":
    main()
