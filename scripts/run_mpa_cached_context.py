"""
Re-run MedPathAgent final-answer LLM call with a new backbone, reusing
pre-computed tool context + kg_paths.

Three source modes:

  --source-mode mpa      (default)
      Read mpa_tool_context + mpa_kg_paths from an existing MPA results CSV.
      No re-summarisation.

  --source-mode tooluni
      Read tu_tool_context from a ToolUni CSV + kg_paths from --medreason-csv.
      No novel tools. No re-summarisation.

  --source-mode hybrid   ← NEW
      Read tu_tool_context from a ToolUni CSV (backbone-specific)
      + mpa_tu_rounds novel tool outputs from --mpa-rounds-csv (e.g. llama70b run)
      + kg_paths from --medreason-csv.
      Combines TU context + novel raw outputs, re-summarises, then runs LLM.
      This is the correct MPA setup for backbones that cannot run plan_tools()
      reliably (llama8b, qwen235b), reusing novel tool outputs already retrieved
      by a stronger backbone.

Usage (hybrid — recommended for llama8b / qwen235b):
    python scripts/run_mpa_cached_context.py \\
        --source-mode hybrid \\
        --source-csv  output/tooluni/llama8b3_1/oncqa_askdocs/run_0/results.csv \\
        --mpa-rounds-csv output/medpathagent/llama70b3_3/oncqa_askdocs/run_0/results.csv \\
        --medreason-csv output/medreason/llama8b3_1/oncqa_askdocs/run_0/results.csv \\
        --name llama8b3_1 \\
        --out-dir output/medpathagent/llama8b3_1/oncqa_askdocs/run_0

Usage (MPA source — re-run with different backbone):
    python scripts/run_mpa_cached_context.py \\
        --source-csv  output/medpathagent/qwen235b/oncqa_askdocs/run_0/results.csv \\
        --name        llama8b3_1 \\
        --out-dir     output/medpathagent/llama8b3_1/oncqa_askdocs/run_0

Usage (ToolUni source — TU context + KG paths only, no novel tools):
    python scripts/run_mpa_cached_context.py \\
        --source-mode tooluni \\
        --source-csv  output/tooluni/llama8b3_1/oncqa_askdocs/run_0/results.csv \\
        --medreason-csv output/medreason/llama8b3_1/oncqa_askdocs/run_0/results.csv \\
        --name        llama8b3_1 \\
        --out-dir     output/medpathagent/llama8b3_1/oncqa_askdocs/run_0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import re
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

# ── repo root on sys.path ────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_TOOLUNI_REPO = REPO.parent / "tooluni_agent"
if str(_TOOLUNI_REPO) not in sys.path:
    sys.path.insert(0, str(_TOOLUNI_REPO))

from prompts import (
    SYSTEM_PROMPT,
    build_triage_prompt,
    TOOLUNI_AUGMENTATION_HEADER,
    KG_AUGMENTATION_HEADER,
)
from config import MODELS
from tooluni_agent.agent import BedrockLLM, ToolReasoner  # noqa: E402

# ── constants ────────────────────────────────────────────────────────────────

MEDPERTURB_CSV = REPO.parent / "data" / "medperturb_data.csv"
DATASETS       = {"oncqa", "askdocs"}
TASKS          = ["manage", "visit", "resource"]

_KG_PATHS_HEADER = KG_AUGMENTATION_HEADER
_write_lock      = threading.Lock()


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_response(text: str) -> dict[str, str]:
    out = {}
    for task in TASKS:
        m = re.search(rf"{task.upper()}[:\s]+([Yy][Ee][Ss]|[Nn][Oo])", text)
        out[task] = m.group(1).upper() if m else ""
    return out


def _is_empty(v) -> bool:
    return not v or str(v).strip().lower() in ("", "nan", "none", "[]")


def build_augmented_prompt(clinical_context: str, kg_paths: str,
                            tool_context: str) -> str:
    parts = []
    if not _is_empty(kg_paths):
        parts.append(f"{_KG_PATHS_HEADER}\n{str(kg_paths).strip()[:2000]}")
    if not _is_empty(tool_context):
        parts.append(f"{TOOLUNI_AUGMENTATION_HEADER}\n{str(tool_context).strip()[:2500]}")
    return build_triage_prompt(clinical_context,
                                augmentation="\n\n".join(parts))


def extract_novel_raw(mpa_tu_rounds_str: str) -> list[str]:
    """
    Parse mpa_tu_rounds JSON → list of non-empty raw tool outputs.
    Each entry is one tool's raw output string.
    """
    if _is_empty(mpa_tu_rounds_str):
        return []
    try:
        rounds = json.loads(str(mpa_tu_rounds_str))
    except Exception:
        return []
    outputs = []
    for rd in rounds:
        for t in rd.get("tools", []):
            out = str(t.get("output", "") or "").strip()
            if out:
                outputs.append(out)
    return outputs


def append_row(row: dict, out_dir: str) -> None:
    out_path = Path(out_dir) / "results.csv"
    df = pd.DataFrame([row])
    with _write_lock:
        if out_path.exists():
            df.to_csv(out_path, mode="a", header=False, index=False)
        else:
            df.to_csv(out_path, index=False)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", required=True,
                        help="ToolUni or MPA results CSV (see --source-mode)")
    parser.add_argument("--source-mode", default="mpa",
                        choices=["mpa", "tooluni", "hybrid"],
                        help="mpa | tooluni | hybrid")
    parser.add_argument("--mpa-rounds-csv", default=None,
                        help="[hybrid] MPA results CSV containing mpa_tu_rounds "
                             "(e.g. llama70b run). Novel tool outputs are reused.")
    parser.add_argument("--medreason-csv", default=None,
                        help="MedReason results CSV with kg_paths column "
                             "(required for tooluni and hybrid modes)")
    parser.add_argument("--name", default="llama8b3_1",
                        help="Target backbone model key from MODELS dict")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.source_mode in ("tooluni", "hybrid") and not args.medreason_csv:
        parser.error("--medreason-csv is required for tooluni and hybrid modes")
    if args.source_mode == "hybrid" and not args.mpa_rounds_csv:
        parser.error("--mpa-rounds-csv is required for hybrid mode")

    model_id = MODELS.get(args.name, args.name)
    region   = os.environ.get("AWS_REGION", "us-east-2")

    print(f"Backbone    : {args.name} → {model_id}")
    print(f"Source mode : {args.source_mode}")
    print(f"Source CSV  : {args.source_csv}")
    if args.mpa_rounds_csv:
        print(f"MPA rounds  : {args.mpa_rounds_csv}")
    if args.medreason_csv:
        print(f"MedReason   : {args.medreason_csv}")
    print(f"Output      : {args.out_dir}")

    llm = BedrockLLM(model_id=model_id, region=region, pool_size=args.workers)

    # ── tool reasoner for re-summarisation (hybrid only) ─────────────────────
    tool_reasoner = ToolReasoner(model_id=model_id, llm=llm) \
                    if args.source_mode == "hybrid" else None

    # ── load source CSV ──────────────────────────────────────────────────────
    src = pd.read_csv(args.source_csv)
    print(f"Source rows : {len(src)}")

    # ── load kg_paths from MedReason CSV ────────────────────────────────────
    kg_paths_lookup: dict[str, str] = {}
    if args.medreason_csv:
        mr = pd.read_csv(args.medreason_csv)
        if "kg_paths" in mr.columns and "row_id" in mr.columns:
            kg_paths_lookup = (
                mr[mr["kg_paths"].notna() & (mr["kg_paths"].str.strip() != "")]
                .set_index("row_id")["kg_paths"]
                .astype(str)
                .to_dict()
            )
        print(f"KG paths    : {len(kg_paths_lookup)} rows")

    # ── load mpa_tu_rounds from MPA rounds CSV (hybrid) ──────────────────────
    # keyed by (context_id, perturbation) → raw mpa_tu_rounds string
    mpa_rounds_lookup: dict[tuple, str] = {}
    if args.source_mode == "hybrid" and args.mpa_rounds_csv:
        mpa_ref = pd.read_csv(args.mpa_rounds_csv)
        if "mpa_tu_rounds" in mpa_ref.columns:
            for _, row in mpa_ref.iterrows():
                key = (str(row["context_id"]), str(row["perturbation"]))
                val = str(row.get("mpa_tu_rounds", "") or "")
                if not _is_empty(val):
                    mpa_rounds_lookup[key] = val
        print(f"MPA rounds  : {len(mpa_rounds_lookup)} rows with novel tools")

    # ── load clinical contexts ────────────────────────────────────────────────
    mp = pd.read_csv(MEDPERTURB_CSV)
    mp = mp[mp["dataset"].isin(DATASETS)]
    ctx_lookup = mp.set_index(["context_id", "perturbation"])["clinical_context"].to_dict()
    print(f"Clinical contexts: {len(ctx_lookup)} entries")

    # ── resume ────────────────────────────────────────────────────────────────
    out_path  = Path(args.out_dir) / "results.csv"
    done_ids: set[str] = set()
    if not args.no_resume and out_path.exists():
        done_df = pd.read_csv(out_path, header=None)
        # first col is always row_id regardless of whether file has a header
        col0 = done_df.iloc[:, 0].astype(str)
        # skip any header row that literally says "row_id"
        done_ids = set(col0[col0 != "row_id"])
        print(f"Resuming    : {len(done_ids)} rows already done")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    col_prefix = f"medpathagent_{args.name}"

    # ── per-row processing ────────────────────────────────────────────────────
    def process_row(row):
        rid          = str(row["row_id"])
        if rid in done_ids:
            return None
        context_id   = str(row["context_id"])
        perturbation = str(row["perturbation"])

        clinical_context = ctx_lookup.get((context_id, perturbation), "")
        if not clinical_context:
            try:
                clinical_context = ctx_lookup.get((int(context_id), perturbation), "")
            except Exception:
                pass
        if not clinical_context:
            print(f"  WARNING: no clinical_context for {rid}")

        # ── build tool_context per mode ───────────────────────────────────────
        if args.source_mode == "mpa":
            tool_context = str(row.get("mpa_tool_context", "") or "")
            kg_paths     = str(row.get("mpa_kg_paths", "") or "")
            context_src  = "mpa_cached"
            mpa_tu_rounds_out = str(row.get("mpa_tu_rounds", "") or "")

        elif args.source_mode == "tooluni":
            tool_context = str(row.get("tu_tool_context", "") or "")
            kg_paths     = kg_paths_lookup.get(rid, "")
            context_src  = "tooluni_cached"
            mpa_tu_rounds_out = ""

        else:  # hybrid
            tu_ctx   = str(row.get("tu_tool_context", "") or "")
            kg_paths = kg_paths_lookup.get(rid, "")
            key      = (context_id, perturbation)
            rounds_str = mpa_rounds_lookup.get(key, "")
            novel_outputs = extract_novel_raw(rounds_str)

            # Combine: TU text + novel raw outputs → re-summarise
            parts = []
            if not _is_empty(tu_ctx):
                parts.append(tu_ctx.strip())
            parts.extend(novel_outputs)
            combined_raw = "\n\n".join(parts)

            if combined_raw.strip():
                tool_context = tool_reasoner.summarise_tool_context(
                    combined_raw, question=clinical_context[:500]
                )
            else:
                tool_context = ""

            n_novel      = len(novel_outputs)
            context_src  = f"hybrid_tu+{n_novel}novel"
            mpa_tu_rounds_out = rounds_str   # preserve for downstream analysis

        prompt = build_augmented_prompt(clinical_context, kg_paths, tool_context)

        try:
            text = llm.chat(
                prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                system_prompt=SYSTEM_PROMPT,
            )
        except Exception as exc:
            text = f"ERROR: {exc}"

        parsed = parse_response(text or "")

        out_row = {
            "row_id":       rid,
            "context_id":   context_id,
            "perturbation": perturbation,
            "model":        col_prefix,
            "dataset":      row.get("dataset", ""),
            "gold_standard_manage":   row.get("gold_standard_manage", ""),
            "gold_standard_visit":    row.get("gold_standard_visit", ""),
            "gold_standard_resource": row.get("gold_standard_resource", ""),
            f"{col_prefix}_manage":            parsed.get("manage", ""),
            f"{col_prefix}_visit":             parsed.get("visit", ""),
            f"{col_prefix}_resource":          parsed.get("resource", ""),
            f"{col_prefix}_manage_reasoning":  text or "",
            "mpa_kg_paths":       kg_paths,
            "mpa_tool_context":   tool_context,
            "mpa_tu_rounds":      mpa_tu_rounds_out,
            "mpa_context_source": context_src,
        }
        for col in ["original_gender", "age", "gendered_condition"]:
            if col in row.index:
                out_row[col] = row.get(col, "")

        append_row(out_row, args.out_dir)
        return rid

    rows      = [row for _, row in src.iterrows()]
    remaining = [r for r in rows if str(r["row_id"]) not in done_ids]
    print(f"Processing  : {len(remaining)} rows with {args.workers} workers\n")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_row, r): r for r in remaining}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="MPA cached-context"):
            fut.result()

    final = pd.read_csv(out_path)
    print(f"\nDone. {len(final)} rows → {out_path}")
    print(final[[f"{col_prefix}_manage", f"{col_prefix}_visit",
                 f"{col_prefix}_resource"]].value_counts().head(10))


if __name__ == "__main__":
    main()
