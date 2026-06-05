"""
Run models on MedXpertQA (original text dataset).

Loads /Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/medxpertqa_text.jsonl
and filters by --body-system and/or --medical-task before running.

  Turn 1:  Q: {question}
           A: Let's think step by step.
           → reasoning

  Turn 2:  Based on your reasoning above, reply with the single letter
           of the correct answer and nothing else.
           → single answer letter

Usage:
  python run_medxpertqa.py --mode AWS
  python run_medxpertqa.py --mode MEDREASON --body-system Cardiovascular
  python run_medxpertqa.py --mode TOOLUNI --medical-task Diagnosis
  python run_medxpertqa.py --mode AWS --body-system Nervous --medical-task Treatment
  python run_medxpertqa.py --mode AWS --limit 10    # smoke test

body_system choices : Cardiovascular, Digestive, Endocrine, Integumentary,
                      Lymphatic, Muscular, Nervous, Other / NA, Reproductive,
                      Respiratory, Skeletal, Urinary
medical_task choices: Basic Science, Diagnosis, Treatment
"""
from __future__ import annotations

import argparse
import concurrent.futures
import warnings
# Suppress harmless "leaked semaphore" warning from PyTorch/SentenceTransformer
# internal DataLoader workers that don't explicitly release OS semaphores at shutdown.
warnings.filterwarnings(
    "ignore",
    message="resource_tracker: There appear to be",
    category=UserWarning,
)
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
DATA_DIR     = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data")
MEDXPERT_JSONL = DATA_DIR / "medxpertqa_text.jsonl"

SYSTEM_PROMPT = "You are a helpful medical assistant."

# ── logging ───────────────────────────────────────────────────────────────────

log = logging.getLogger("run_medxpertqa")

# ── CSV helpers ───────────────────────────────────────────────────────────────

FIELDS = [
    "id", "body_system", "medical_task", "question_type",
    "model_answer_letter", "correct_label",
    "reasoning", "kg_paths", "generated_code",
]


def load_questions(
    path: Path,
    body_system: Optional[str] = None,
    medical_task: Optional[str] = None,
) -> List[dict]:
    """Load medxpertqa_text.jsonl, optionally filtered by body_system / medical_task."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if body_system and row.get("body_system", "") != body_system:
                continue
            if medical_task and row.get("medical_task", "") != medical_task:
                continue
            rows.append(row)
    return rows


def load_done(path: Path) -> Set[str]:
    """Return set of ids already processed."""
    if not path.exists():
        log.info("No existing results file at %s — starting fresh", path)
        return set()
    try:
        import pandas as pd
        df = pd.read_csv(path, usecols=["id"], dtype=str).dropna()
        done = set(df["id"])
        log.info("Resume: %d rows already done", len(done))
        return done
    except Exception as e:
        log.warning("Could not read results for resume (%s) — starting fresh", e)
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

    # 1. Explicit conclusion phrases — letter must be isolated (paren/space/end),
    #    not the start of a word like "choices" or "condition".
    for pattern in [
        r"the answer is\s*[:\(]?\s*([A-J])(?:\)|\.|\s|$)",
        r"correct answer[:\s]+\(?([A-J])\)?",
        r"Therefore[^.]*\b([A-J])\b",
        r"answer\s*:\s*\(?([A-J])\)?",          # "Answer: (H)" or "Answer: H"
        r"\*{1,2}([A-J])\*{1,2}",               # **H** or *H*
    ]:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if m.group(1).upper() in valid:
                return m.group(1).upper()

    # 2. Last standalone valid letter (uppercase only to avoid word-start false matches)
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

def run_via_model(
    model, bedrock_client, bedrock_model_id: str,
    question: str, options: Dict[str, str],
    mode: str = "",
) -> Tuple[str, str, dict]:
    """
    Two-turn CoT for augmented models (MEDREASON, MEDPATHAGENT, AGENTMD, TOOLUNI).

    Turn 1: model.generate() — retrieves augmentation, prepends to prompt, calls LLM
    Turn 2: direct Bedrock call — plain letter extraction (no augmentation needed)
    """
    turn1 = (
        f"Q: {question}\n\n"
        f"Options:\n{_fmt_options(options)}\n\n"
        f"A: Let's think step by step."
    )
    # Include options in retrieval query — MedPathAgent extracts entities from
    # "Question: … Options: …" and option text often contains drug/disease names
    retrieval_query = f"Question: {question}. Options: {_fmt_options(options)}"
    result1 = model.generate(
        prompt=turn1,
        system_prompt=SYSTEM_PROMPT,
        retrieval_query=retrieval_query,
        cot_mode=(mode == "MEDREASON"),   # entity-anchored CoT scaffold for MedReason
    )
    reasoning = result1.reasoning
    meta = getattr(result1, "metadata", {}) or {}

    # Turn 2: plain Bedrock call — single-letter extraction, no tool retrieval
    resp2 = bedrock_client.converse(
        modelId=bedrock_model_id,
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

    return reasoning, letter, meta


# ── process one row ───────────────────────────────────────────────────────────

def process_row(
    row: dict,               # question dict from medxpertqa_text.jsonl
    model_name: str,
    mode: str,
    bedrock_client,
    bedrock_model_id: Optional[str],
    augment_model,
    output_path: Path,
    lock: Lock,
    log_dir: Optional[Path] = None,
) -> dict:
    qid         = row["id"]
    question    = row["question"]
    options     = row["options"]
    label       = row["label"]
    body_system = row.get("body_system", "")
    medical_task= row.get("medical_task", "")
    question_type = row.get("question_type", "")

    kg_paths = ""
    generated_code = ""

    if augment_model is not None:
        reasoning, letter, meta = run_via_model(
            augment_model, bedrock_client, bedrock_model_id, question, options,
            mode=mode,
        )
        kg_paths = meta.get("kg_paths", "")
        rounds = meta.get("agentmd_rounds", [])
        generated_code = "\n---\n".join(
            f"[round {r['round']}] {r['exec_output']}"
            for r in rounds if r.get("exec_output")
        )
        if log_dir is not None and mode in ("TOOLUNI", "AGENTMD"):
            _save_tool_log(log_dir, qid, question_type, mode, meta, lock)
    else:
        reasoning, letter = run_two_turn_bedrock(bedrock_client, bedrock_model_id, question, options)

    return {
        "id":                  qid,
        "body_system":         body_system,
        "medical_task":        medical_task,
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


# ── two-pass tool planning helpers ───────────────────────────────────────────

def run_stage1_planning(questions: List[dict], model, plan_log: Path,
                        workers: int = 1) -> None:
    """
    Stage 1: plan_tools() + run_tools_detailed() for every question.
    Saves raw tool outputs to a JSONL log — no summarisation, no inference.
    Stage 2 loads these outputs and runs summarise + LLM.
    """
    import fcntl

    done_ids: Set[str] = set()
    if plan_log.exists():
        with open(plan_log, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    log.info("Stage 1: %d questions, %d already done", len(questions), len(done_ids))

    queue = [q for q in questions if q["id"] not in done_ids]
    if not queue:
        log.info("Stage 1: nothing to do.")
        return

    plan_log.parent.mkdir(parents=True, exist_ok=True)
    lock = Lock()
    pbar = tqdm(total=len(queue), desc="[stage1]", unit="q")

    def _run_one(row: dict) -> None:
        qid   = row["id"]
        query = f"Question: {row['question']}. Options: {_fmt_options(row['options'])}"
        try:
            tr   = model._tool_reasoner
            plan = tr.plan_tools(query)
            if plan:
                details, _ = tr.run_tools_detailed(plan, summarise=False, question=query)
                raw_outputs = [d["output"] for d in details if (d.get("output") or "").strip()]
                tool_calls  = [{"tool_name": d["tool_name"], "arguments": d["arguments"],
                                "output": (d.get("output") or "")[:2000],
                                "error": d.get("error")} for d in details]
            else:
                raw_outputs, tool_calls = [], []
            entry = {
                "id":           qid,
                "body_system":  row.get("body_system", ""),
                "medical_task": row.get("medical_task", ""),
                "plan":         plan,
                "tool_calls":   tool_calls,
                "raw_outputs":  raw_outputs,
            }
        except Exception as exc:
            log.warning("Stage 1 failed for %s: %s", qid, exc)
            entry = {"id": qid, "plan": [], "tool_calls": [], "raw_outputs": [],
                     "error": str(exc)}

        with lock:
            with open(plan_log, "a", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        tqdm.write(f"  {qid}: {len(entry['raw_outputs'])} tool outputs")
        pbar.update(1)

    if workers == 1:
        for row in queue:
            _run_one(row)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            concurrent.futures.wait([ex.submit(_run_one, row) for row in queue])

    pbar.close()
    log.info("Stage 1 done. Tool outputs → %s", plan_log)


def load_stage1_outputs(plan_log: Path) -> Dict[str, List[str]]:
    """Load id → raw_outputs list from a stage-1 JSONL log."""
    outputs: Dict[str, List[str]] = {}
    if not plan_log.exists():
        raise FileNotFoundError(f"Stage-1 log not found: {plan_log}. Run --stage 1 first.")
    with open(plan_log, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                outputs[entry["id"]] = entry.get("raw_outputs", [])
            except Exception:
                pass
    n_with = sum(1 for v in outputs.values() if v)
    log.info("Loaded stage-1 outputs for %d questions (%d with tool results) from %s",
             len(outputs), n_with, plan_log)
    return outputs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--mode", default="AWS",
                        choices=["AWS","TXAGENT","TXAGENT_BEDROCK","MEDREASON",
                                 "MEDPATHAGENT","AGENTMD","TOOLUNI","KGRANK"])
    parser.add_argument("--name",     default=None)
    parser.add_argument("--kg-path",  default=str(DATA_DIR / "kg.csv"))
    parser.add_argument("--emb-path", default=str(DATA_DIR / "node_embeddings_sapbert.pt"))
    parser.add_argument("--data",     default=str(MEDXPERT_JSONL),
                        help="Path to medxpertqa_text.jsonl")
    parser.add_argument("--body-system", default=None,
                        help="Filter to one body system. "
                             "Options: Cardiovascular, Digestive, Endocrine, "
                             "Integumentary, Lymphatic, Muscular, Nervous, "
                             "'Other / NA', Reproductive, Respiratory, Skeletal, Urinary")
    parser.add_argument("--medical-task", default=None,
                        help="Filter to one medical task. "
                             "Options: 'Basic Science', Diagnosis, Treatment")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Max questions to process (smoke test)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Reprocess all rows (ignore existing results)")
    parser.add_argument("--workers",   type=int, default=1,
                        help="Parallel worker threads")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Fraction [0,1) of questions to start from")
    parser.add_argument("--end",   type=float, default=1.0,
                        help="Fraction (0,1] of questions to end at")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2],
                        help="Two-pass tool retrieval: "
                             "0=single pass (default), "
                             "1=plan tools only (save plans to logs), "
                             "2=load plans from logs, execute tools + run inference")
    parser.add_argument("--plan-log", default=None,
                        help="Path to tool plan JSONL log. "
                             "Stage 1 writes here; stage 2 reads from here. "
                             "Defaults to logs/medxpertqa/<model_tag>/plans.jsonl")
    args = parser.parse_args()

    setup_logging()

    default_name = {
        "AWS":"llama70b3_3", "TXAGENT":"txagent_hf",
        "TXAGENT_BEDROCK":"llama70b3_3", "MEDREASON":"llama70b3_3",
        "MEDPATHAGENT":"llama70b3_3", "AGENTMD":"llama70b3_3",
        "TOOLUNI":"llama70b3_3", "KGRANK":"llama70b3_3",
    }[args.mode]
    name = args.name or default_name

    # ── output path — encode filters in directory name ────────────────────────
    filter_tag = ""
    if args.body_system:
        filter_tag += f"_bs-{args.body_system.replace(' ','_').replace('/','')}"
    if args.medical_task:
        filter_tag += f"_mt-{args.medical_task.replace(' ','_')}"
    model_tag  = f"{args.mode.lower()}_{name}{filter_tag}"
    output_dir = ROOT / "output" / "medxpertqa" / model_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "results.csv"
    lock = Lock()

    log_dir: Optional[Path] = None
    if args.mode in ("TOOLUNI", "AGENTMD", "TXAGENT"):
        log_dir = ROOT / "logs" / "medxpertqa" / model_tag
        log_dir.mkdir(parents=True, exist_ok=True)
        log.info("Tool logs → %s", log_dir)

    log.info("Mode=%s  Model=%s  body_system=%s  medical_task=%s",
             args.mode, name, args.body_system or "ALL", args.medical_task or "ALL")
    log.info("Output CSV → %s", output_csv)

    # ── build model / client ──────────────────────────────────────────────────
    cfg = get_default_config(name=name, mode=args.mode)
    mc  = cfg.model_configs[0]
    if args.kg_path:  mc.kg_path = args.kg_path
    if args.emb_path: mc.node_embeddings_path = args.emb_path

    bedrock_client   = None
    bedrock_model_id = None
    augment_model    = None

    # Always create a Bedrock client — augmented modes need it for Turn 2 (letter extraction)
    bedrock_model_id = mc.bedrock_model_id or MODELS.get(name, name)
    region = os.environ.get("AWS_REGION", "us-east-2")
    bedrock_client = _make_bedrock_client(region)
    log.info("Bedrock client ready  model_id=%s", bedrock_model_id)

    if args.mode in ("TXAGENT","MEDREASON","MEDPATHAGENT","AGENTMD","TOOLUNI","KGRANK"):
        log.info("Loading %s model...", args.mode)
        augment_model = ModelFactory.create_model(mc, cfg)
        log.info("%s model ready", args.mode)

    # ── load and filter questions ─────────────────────────────────────────────
    questions = load_questions(
        Path(args.data),
        body_system=args.body_system,
        medical_task=args.medical_task,
    )
    log.info("Loaded %d questions (body_system=%s  medical_task=%s)",
             len(questions), args.body_system or "ALL", args.medical_task or "ALL")

    # Slice
    if args.start > 0.0 or args.end < 1.0:
        n  = len(questions)
        lo = int(args.start * n)
        hi = int(args.end   * n)
        questions = questions[lo:hi]
        log.info("Slice [%d, %d): %d questions", lo, hi, len(questions))

    # ── plan log path ─────────────────────────────────────────────────────────
    plan_log = Path(args.plan_log) if args.plan_log else \
               ROOT / "logs" / "medxpertqa" / model_tag / "plans.jsonl"

    # ── stage 1: plan tools only ──────────────────────────────────────────────
    if args.stage == 1:
        if augment_model is None or not hasattr(augment_model, "_tool_reasoner"):
            log.error("Stage 1 requires a model with _tool_reasoner (e.g. TOOLUNI)")
            return
        run_stage1_planning(questions, augment_model, plan_log, workers=args.workers)
        log.info("Stage 1 complete. Run with --stage 2 to execute tools + inference.")
        return

    # ── load stage-1 tool outputs for stage 2 ────────────────────────────────
    stage1_outputs: Dict[str, List[str]] = {}
    if args.stage == 2:
        stage1_outputs = load_stage1_outputs(plan_log)

    # ── resume ────────────────────────────────────────────────────────────────
    done: Set[str] = set() if args.no_resume else load_done(output_csv)
    queue = [q for q in questions if q["id"] not in done]
    if args.stage == 2:
        missing = [q["id"] for q in queue if q["id"] not in stage1_outputs]
        if missing:
            log.warning("Stage 2: %d questions have no stage-1 output — skipping: %s",
                        len(missing), missing[:5])
        queue = [q for q in queue if q["id"] in stage1_outputs]
    if args.limit:
        queue = queue[:args.limit]
    log.info("%d questions to process (%d already done)", len(queue), len(done))

    if not queue:
        log.info("Nothing to do.")
        return

    # ── process ───────────────────────────────────────────────────────────────
    pbar = tqdm(total=len(queue), desc=f"[{model_tag}]", unit="q")

    def _process_one(row: dict) -> None:
        import gc
        qid = row["id"]

        # Stage 2: bypass plan+execute, run summarise+LLM directly
        if args.stage == 2 and qid in stage1_outputs:
            try:
                raw_outputs = stage1_outputs[qid]
                combined    = "\n\n".join(raw_outputs)
                question    = row["question"]
                options     = row["options"]
                query       = f"Question: {question}. Options: {_fmt_options(options)}"

                tr = augment_model._tool_reasoner if augment_model and \
                     hasattr(augment_model, "_tool_reasoner") else None

                # Summarise raw tool outputs
                tool_context = ""
                if combined.strip() and tr is not None:
                    tool_context = tr.summarise_tool_context(combined, question=query)

                # Build augmented Turn 1 prompt
                from prompts import TOOLUNI_AUGMENTATION_HEADER
                turn1 = (f"Q: {question}\n\nOptions:\n{_fmt_options(options)}\n\n"
                         f"A: Let's think step by step.")
                if tool_context.strip():
                    turn1 = f"{TOOLUNI_AUGMENTATION_HEADER}\n{tool_context[:2500]}\n\n{turn1}"

                # LLM reasoning (Turn 1)
                if augment_model is not None:
                    raw_reasoning = augment_model._llm.chat(
                        turn1, temperature=augment_model.model_config.temperature,
                        max_tokens=augment_model.model_config.max_tokens,
                        system_prompt=SYSTEM_PROMPT,
                    )
                else:
                    raw_reasoning, _ = run_two_turn_bedrock(
                        bedrock_client, bedrock_model_id, question, options)

                # Turn 2 — letter extraction
                resp2 = bedrock_client.converse(
                    modelId=bedrock_model_id,
                    system=[{"text": SYSTEM_PROMPT}],
                    messages=[
                        {"role": "user",      "content": [{"text": turn1}]},
                        {"role": "assistant", "content": [{"text": raw_reasoning}]},
                        {"role": "user",      "content": [{"text": TURN2}]},
                    ],
                    inferenceConfig={"maxTokens": 2, "temperature": 0.0},
                ) if bedrock_client else None

                if resp2:
                    answer_raw = resp2["output"]["message"]["content"][0]["text"].strip()
                    letter = _extract_letter(answer_raw, options)
                else:
                    letter = _extract_letter(raw_reasoning, options)

                result = {
                    "id": qid,
                    "body_system":         row.get("body_system",""),
                    "medical_task":        row.get("medical_task",""),
                    "question_type":       row.get("question_type",""),
                    "model_answer_letter": letter,
                    "correct_label":       row.get("label",""),
                    "reasoning":           raw_reasoning,
                    "kg_paths":            "",
                    "generated_code":      "",
                }
                append_result(output_csv, result, lock)
                tqdm.write(
                    f"  {qid} → {letter} (gold={row.get('label','')}, "
                    f"{'✓' if letter==row.get('label','') else '✗'})"
                )
            except Exception as exc:
                tqdm.write(f"  ERROR {qid}: {exc}")
                append_result(output_csv, {
                    "id": qid, "body_system": row.get("body_system",""),
                    "medical_task": row.get("medical_task",""),
                    "question_type": row.get("question_type",""),
                    "model_answer_letter": "?", "correct_label": row.get("label",""),
                    "reasoning": f"ERROR: {exc}", "kg_paths": "", "generated_code": "",
                }, lock)
            finally:
                pbar.update(1)
                gc.collect()
            return

        try:
            result = process_row(
                row, name, args.mode,
                bedrock_client, bedrock_model_id, augment_model,
                output_csv, lock, log_dir=log_dir,
            )
            append_result(output_csv, result, lock)
            tqdm.write(
                f"  {qid} → {result['model_answer_letter']} "
                f"(gold={result['correct_label']}, "
                f"{'✓' if result['model_answer_letter']==result['correct_label'] else '✗'})"
            )
        except Exception as exc:
            tqdm.write(f"  ERROR {qid}: {exc}")
            append_result(output_csv, {
                "id": qid,
                "body_system":   row.get("body_system",""),
                "medical_task":  row.get("medical_task",""),
                "question_type": row.get("question_type",""),
                "model_answer_letter": "?",
                "correct_label": row.get("label",""),
                "reasoning": f"ERROR: {exc}",
                "kg_paths": "", "generated_code": "",
            }, lock)
        finally:
            pbar.update(1)
            gc.collect()

    n_workers = max(1, args.workers)
    if n_workers == 1:
        for row in queue:
            _process_one(row)
            time.sleep(0.2)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            concurrent.futures.wait([ex.submit(_process_one, row) for row in queue])

    pbar.close()
    log.info("Done. Results → %s", output_csv)


if __name__ == "__main__":
    main()
