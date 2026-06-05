"""
Leave-One-Tool-Out (LOTO) ablation for MPA win cases.
=====================================================

For each of the 78 llama70b MPA win cases (discordant pairs where MPA is
correct and TUA makes a reduced-care error), tests whether removing a family
of novel tools causes MPA to flip from correct to error.

Novel tools come from mpa_tu_rounds (kg_novel step). The cached TU tool
context for each case is loaded from the ToolUni results CSV (not from MPA
results, where tu_tool_context was not populated for these cases).

All 18 tool types are grouped into 6 families:

    EuropePMC      : EuropePMC_search_articles
    FDA_core       : indications + contraindications + adverse_reactions
                     + safety_summary
    FDA_pharma     : FDA_get_pharmacokinetics_by_drug_name
    FDA_warnings   : do_not_use + boxed_warning + when_using + other_safety
    FDA_rare       : abuse_dependence + drug_names_by_abuse +
                     drug_names_by_contraindications + overdosage + child_safety
    other          : OpenTargets + diqt_search + dict_search

Ablation configs:
    full               — original mpa_tool_context (sanity check)
    no_europepmc       — drop EuropePMC
    no_fda_core        — drop FDA core safety quartet
    no_fda_pharma      — drop pharmacokinetics
    no_fda_warnings    — drop warning-type tools
    no_fda_rare        — drop rare/abuse tools
    no_other           — drop OpenTargets + misc
    no_novel           — drop ALL novel tools (KG paths only)
    tu_only            — drop ALL novel tools AND add TU cached context back
                         (tests whether TU context alone recovers the win)
    no_tu              — novel tools only, no TU cached context
                         (verifies TU context was absent in original run)

Statistical tests (on binary correct/error outcomes across 78 cases):
    - Cochran's Q across all configs → is there any difference?
    - Pairwise McNemar: full vs each ablation (Bonferroni corrected)

Outputs:
    analysis/loto_results.csv        — per case × ablation
    analysis/loto_summary.csv        — per ablation: flip counts, %, stats
    analysis/loto_importance.csv     — per (context_id, pert, task): which
                                       family removal caused the flip

Usage:
    cd medperturb_eval
    conda run -n curebench python analysis/loto_ablation.py [--workers 4]
    conda run -n curebench python analysis/loto_ablation.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm

# ── path setup ─────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
REPO_ROOT    = SCRIPT_DIR.parent
TOOLUNI_ROOT = REPO_ROOT.parent / "tooluni_agent"

for p in [str(REPO_ROOT), str(TOOLUNI_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from prompts import (                                   # noqa: E402
    SYSTEM_PROMPT,
    build_triage_prompt,
    TOOLUNI_AUGMENTATION_HEADER,
    KG_AUGMENTATION_HEADER,
)
from config import MODELS                               # noqa: E402
from processor import parse_response                    # noqa: E402
from tooluni_agent.agent import BedrockLLM, ToolReasoner  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────────

OUTPUT_BASE    = REPO_ROOT / "output"
MEDPERTURB_CSV = REPO_ROOT.parent / "data" / "medperturb_data.csv"
ANALYSIS_DIR   = REPO_ROOT / "analysis"
WINS_CSV       = ANALYSIS_DIR / "mpa_vs_tua_wins.csv"

BACKBONE = "llama70b3_3"
MODEL_ID = MODELS[BACKBONE]
DATASETS = {"oncqa", "askdocs"}
TASKS    = ["manage", "visit", "resource"]

_KG_PATHS_HEADER = KG_AUGMENTATION_HEADER

# ── tool family definitions ────────────────────────────────────────────────────

TOOL_FAMILIES: dict[str, set[str]] = {
    "EuropePMC": {
        "EuropePMC_search_articles",
    },
    "FDA_core": {
        "FDA_get_indications_by_drug_name",
        "FDA_get_contraindications_by_drug_name",
        "FDA_get_adverse_reactions_by_drug_name",
        "FDA_get_safety_summary_by_drug_name",
    },
    "FDA_pharma": {
        "FDA_get_pharmacokinetics_by_drug_name",
    },
    "FDA_warnings": {
        "FDA_get_do_not_use_info_by_drug_name",
        "FDA_get_boxed_warning_info_by_drug_name",
        "FDA_get_when_using_info",
        "FDA_get_other_safety_info_by_drug_name",
    },
    "FDA_rare": {
        "FDA_get_abuse_dependence_info_by_drug_name",
        "FDA_get_drug_names_by_abuse_info",
        "FDA_get_drug_names_by_contraindications",
        "FDA_get_overdosage_info_by_drug_name",
        "FDA_get_child_safety_info_by_drug_name",
    },
    "other": {
        "OpenTargets_get_disease_id_description_by_name",
        "diqt_search",
        "dict_search",
    },
}

ALL_NOVEL_TOOLS: set[str] = set().union(*TOOL_FAMILIES.values())

# Ablation name → set of tools to DROP from novel context
# Special values handled in code: "tu_only", "no_tu"
ABLATION_CONFIGS: dict[str, set[str] | str] = {
    "full":            set(),              # original mpa_tool_context, no re-summarise
    "no_europepmc":    TOOL_FAMILIES["EuropePMC"],
    "no_fda_core":     TOOL_FAMILIES["FDA_core"],
    "no_fda_pharma":   TOOL_FAMILIES["FDA_pharma"],
    "no_fda_warnings": TOOL_FAMILIES["FDA_warnings"],
    "no_fda_rare":     TOOL_FAMILIES["FDA_rare"],
    "no_other":        TOOL_FAMILIES["other"],
    "no_novel":        ALL_NOVEL_TOOLS,    # KG paths only
    "tu_only":         "tu_only",          # TU cached context + KG paths, no novel
    "no_tu":           "no_tu",            # novel tools only (verifies original had no TU)
}

# ── care direction helpers ─────────────────────────────────────────────────────

def care_augmenting(task: str) -> str:
    return "NO" if task == "manage" else "YES"

def reduced_care_value(task: str) -> str:
    return "YES" if task == "manage" else "NO"

def is_error(pred: str, task: str) -> bool:
    return str(pred).strip().upper() == reduced_care_value(task)

# ── prompt builder ─────────────────────────────────────────────────────────────

def build_augmented_prompt(clinical_context: str, kg_paths: str, tool_context: str) -> str:
    parts = []
    if kg_paths and str(kg_paths).strip() not in ("", "nan", "None"):
        parts.append(f"{_KG_PATHS_HEADER}\n{str(kg_paths).strip()[:2000]}")
    if tool_context and str(tool_context).strip() not in ("", "nan", "None"):
        parts.append(f"{TOOLUNI_AUGMENTATION_HEADER}\n{str(tool_context).strip()[:2500]}")
    augmentation = "\n\n".join(parts)
    return build_triage_prompt(clinical_context, augmentation=augmentation)

# ── data loaders ───────────────────────────────────────────────────────────────

def load_wins() -> pd.DataFrame:
    wins = pd.read_csv(WINS_CSV)
    mpa_wins = wins[(wins["winner"] == "MPA") & (wins["backbone_key"] == BACKBONE)].copy()
    print(f"MPA wins on {BACKBONE}: {len(mpa_wins)}")
    return mpa_wins

def load_mpa_results() -> pd.DataFrame:
    path = OUTPUT_BASE / f"medpathagent/{BACKBONE}/oncqa_askdocs/run_0/results.csv"
    df = pd.read_csv(path)
    print(f"MPA results: {len(df)} rows")
    return df

def load_tu_results() -> pd.DataFrame:
    """Load ToolUni results for the cached TU tool context."""
    path = OUTPUT_BASE / f"tooluni/{BACKBONE}/oncqa_askdocs/run_0/results.csv"
    df = pd.read_csv(path)
    print(f"TU results: {len(df)} rows")
    return df

def load_clinical_contexts() -> dict:
    mp = pd.read_csv(MEDPERTURB_CSV)
    mp = mp[mp["dataset"].isin(DATASETS)]
    return mp.set_index(["context_id", "perturbation"])["clinical_context"].to_dict()

# ── tool context helpers ───────────────────────────────────────────────────────

def get_novel_tools(mpa_row: pd.Series) -> list[dict]:
    """Parse mpa_tu_rounds → list of {tool_name, output} with non-empty output."""
    raw = mpa_row.get("mpa_tu_rounds", "")
    if not raw or str(raw) in ("nan", "None", "[]", ""):
        return []
    try:
        rounds = json.loads(str(raw))
    except Exception:
        return []
    tools = []
    for rd in rounds:
        for t in rd.get("tools", []):
            out = str(t.get("output", "") or "").strip()
            if out:
                tools.append({
                    "tool_name": t["tool_name"],
                    "output":    out,
                    "family":    tool_family(t["tool_name"]),
                })
    return tools

def tool_family(tool_name: str) -> str:
    """Return the family name for a tool, or 'unknown'."""
    for fam, tools in TOOL_FAMILIES.items():
        if tool_name in tools:
            return fam
    return "unknown"

def filtered_novel_raw(tools: list[dict], drop_tools: set[str]) -> str:
    kept = [t["output"] for t in tools if t["tool_name"] not in drop_tools]
    return "\n\n".join(kept)

# ── per-case ablation ──────────────────────────────────────────────────────────

def run_ablation_case(
    win_row:      pd.Series,
    mpa_row:      pd.Series,
    tu_ctx_str:   str,          # tu_tool_context from TU results for this case
    ctx_lookup:   dict,
    llm:          BedrockLLM | None,
    tool_reasoner: ToolReasoner | None,
    dry_run:      bool = False,
) -> list[dict]:
    cid  = str(win_row["context_id"])
    pert = str(win_row["perturbation"])
    task = str(win_row["task"])

    clinical_context = ctx_lookup.get((cid, pert), "")
    if not clinical_context:
        try:
            clinical_context = ctx_lookup.get((int(cid), pert), "")
        except Exception:
            pass

    kg_paths      = str(mpa_row.get("mpa_kg_paths",    "") or "")
    orig_tool_ctx = str(mpa_row.get("mpa_tool_context","") or "")
    novel_tools   = get_novel_tools(mpa_row)
    n_novel       = len(novel_tools)
    novel_families_present = list({t["family"] for t in novel_tools})

    results = []
    for ablation_name, drop_val in ABLATION_CONFIGS.items():

        # ── build tool_context for this ablation ──────────────────────────────
        if ablation_name == "full":
            # Use pre-summarised context exactly as MPA produced it
            tool_context = orig_tool_ctx
            n_kept = n_novel
            needs_llm_summarise = False

        elif drop_val == "tu_only":
            # TU cached context only — no novel tools
            tool_context = tu_ctx_str
            n_kept = 0
            needs_llm_summarise = False

        elif drop_val == "no_tu":
            # Novel tools only, no TU cached (verifies original run had no TU)
            combined_raw = "\n\n".join(t["output"] for t in novel_tools)
            n_kept = n_novel
            needs_llm_summarise = True
            if combined_raw.strip() and not dry_run:
                tool_context = tool_reasoner.summarise_tool_context(
                    combined_raw, question=clinical_context[:500]
                )
            else:
                tool_context = combined_raw

        elif isinstance(drop_val, set) and drop_val == ALL_NOVEL_TOOLS:
            # no_novel: just KG paths
            tool_context = ""
            n_kept = 0
            needs_llm_summarise = False

        else:
            # Drop a specific family
            drop_set = drop_val
            combined_raw = filtered_novel_raw(novel_tools, drop_set)
            n_kept = sum(1 for t in novel_tools if t["tool_name"] not in drop_set)
            needs_llm_summarise = True
            if combined_raw.strip() and not dry_run:
                tool_context = tool_reasoner.summarise_tool_context(
                    combined_raw, question=clinical_context[:500]
                )
            else:
                tool_context = combined_raw

        # ── build prompt + call LLM ───────────────────────────────────────────
        prompt = build_augmented_prompt(clinical_context, kg_paths, tool_context)

        if dry_run:
            pred     = str(win_row.get("mpa_pred", ""))
            raw_text = "(dry-run)"
        else:
            try:
                raw_text = llm.chat(
                    prompt,
                    temperature=0.0,
                    max_tokens=256,
                    system_prompt=SYSTEM_PROMPT,
                )
            except Exception as exc:
                raw_text = f"ERROR: {exc}"
            parsed = parse_response(raw_text or "")
            pred   = parsed.get(task, "")

        flipped = is_error(pred, task)

        results.append({
            "backbone":              BACKBONE,
            "context_id":            cid,
            "perturbation":          pert,
            "task":                  task,
            "dataset":               win_row.get("dataset", ""),
            "kg_anchor":             win_row.get("kg_anchor", ""),
            "ablation":              ablation_name,
            "n_novel_tools":         n_novel,
            "n_kept_tools":          n_kept,
            "n_dropped":             n_novel - n_kept,
            "novel_families_present":json.dumps(sorted(novel_families_present)),
            "tool_context_len":      len(tool_context),
            "tu_ctx_present":        bool(tu_ctx_str and tu_ctx_str not in ("nan","None","")),
            "pred":                  pred,
            "flipped":               flipped,
            "raw_text":              (raw_text or "")[:300],
        })

    return results

# ── statistical tests ──────────────────────────────────────────────────────────

def cochrans_q(binary_matrix: np.ndarray) -> tuple[float, float]:
    """
    Cochran's Q test on a (N_subjects × k_treatments) binary matrix.
    Returns (Q, p_value).
    """
    N, k = binary_matrix.shape
    L = binary_matrix.sum(axis=1)   # row sums
    C = binary_matrix.sum(axis=0)   # col sums
    numerator   = (k - 1) * (k * np.sum(C ** 2) - np.sum(C) ** 2)
    denominator = k * np.sum(L) - np.sum(L ** 2)
    if denominator == 0:
        return 0.0, 1.0
    Q = numerator / denominator
    p = float(1 - stats.chi2.cdf(Q, df=k - 1))
    return float(Q), p


def mcnemar_p(b: int, c: int) -> float | None:
    """Continuity-corrected McNemar p-value."""
    if b + c == 0:
        return None
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(1 - stats.chi2.cdf(stat, df=1))


def pairwise_mcnemar_vs_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each ablation config, McNemar test against 'full' (reference).
    b = cases correct in 'full' but wrong in ablation (flipped by ablation)
    c = cases wrong in 'full' but correct in ablation (recovered by ablation)
    """
    pivot = (
        df.pivot_table(
            index=["context_id", "perturbation", "task"],
            columns="ablation",
            values="flipped",
            aggfunc="first",
        )
        .astype(float)
    )

    full_col = pivot.get("full")
    if full_col is None:
        return pd.DataFrame()

    rows = []
    n_comparisons = len(ABLATION_CONFIGS) - 1   # exclude 'full' itself
    for abl in ABLATION_CONFIGS:
        if abl == "full" or abl not in pivot.columns:
            continue
        abl_col = pivot[abl]
        valid = full_col.notna() & abl_col.notna()
        f = full_col[valid].astype(int)
        a = abl_col[valid].astype(int)

        # b: correct in full (f=0), error in ablation (a=1) → ablation LOST the win
        # c: error in full (f=1), correct in ablation (a=0) → ablation RECOVERED
        b = int(((f == 0) & (a == 1)).sum())
        c = int(((f == 1) & (a == 0)).sum())
        p = mcnemar_p(b, c)
        p_bonf = min(p * n_comparisons, 1.0) if p is not None else None
        rows.append({
            "ablation":        abl,
            "n_flipped_by_abl": b,
            "n_recovered":      c,
            "mcnemar_p":        round(p, 4) if p is not None else None,
            "mcnemar_p_bonf":   round(p_bonf, 4) if p_bonf is not None else None,
            "sig":              ("***" if p_bonf and p_bonf < .001 else
                                 "**"  if p_bonf and p_bonf < .01  else
                                 "*"   if p_bonf and p_bonf < .05  else "ns")
                                 if p_bonf is not None else "N/A",
        })
    return pd.DataFrame(rows)


# ── importance matrix ──────────────────────────────────────────────────────────

def build_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each win case, record which family removals caused a flip.
    Helps identify which tool family was *necessary* for each win.
    """
    family_ablations = [a for a in ABLATION_CONFIGS if a.startswith("no_") and
                        a not in ("no_novel", "no_tu")]
    pivot = (
        df[df["ablation"].isin(family_ablations + ["full"])]
        .pivot_table(
            index=["context_id", "perturbation", "task", "dataset",
                   "kg_anchor", "novel_families_present"],
            columns="ablation",
            values="flipped",
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.columns.name = None
    return pivot


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers",   type=int,  default=4)
    ap.add_argument("--dry-run",   action="store_true",
                    help="Skip LLM calls — use stored predictions (sanity check)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Ignore existing results, rerun everything")
    ap.add_argument("--stats-only", action="store_true",
                    help="Skip LLM inference, recompute stats from existing loto_results.csv")
    args = ap.parse_args()

    region  = os.environ.get("AWS_REGION", "us-east-2")
    out_csv = ANALYSIS_DIR / "loto_results.csv"
    sum_csv = ANALYSIS_DIR / "loto_summary.csv"
    imp_csv = ANALYSIS_DIR / "loto_importance.csv"

    print(f"Model   : {MODEL_ID}")
    print(f"Workers : {args.workers}  dry_run={args.dry_run}  stats_only={args.stats_only}")

    if not args.stats_only:
        # ── load data ───────────────────────────────────────────────────────────
        wins         = load_wins()
        mpa_results  = load_mpa_results()
        tu_results   = load_tu_results()
        ctx_lookup   = load_clinical_contexts()
        print(f"Clinical contexts: {len(ctx_lookup)}")

        # TU cached context lookup: (context_id, perturbation) → tu_tool_context
        tu_ctx_lookup: dict[tuple, str] = {}
        for _, row in tu_results.iterrows():
            key = (str(row["context_id"]), str(row["perturbation"]))
            val = str(row.get("tu_tool_context", "") or "")
            if val not in ("nan", "None", ""):
                tu_ctx_lookup[key] = val
        print(f"TU cached contexts available: {len(tu_ctx_lookup)}")

        # ── resume ──────────────────────────────────────────────────────────────
        done_keys: set[tuple] = set()
        if not args.no_resume and out_csv.exists():
            done_df = pd.read_csv(out_csv)
            done_keys = set(
                zip(done_df["context_id"].astype(str),
                    done_df["perturbation"].astype(str),
                    done_df["task"].astype(str),
                    done_df["ablation"].astype(str))
            )
            print(f"Resuming: {len(done_keys)} (case × ablation) pairs already done")

        # ── init LLM ────────────────────────────────────────────────────────────
        if not args.dry_run:
            llm           = BedrockLLM(model_id=MODEL_ID, region=region,
                                       pool_size=args.workers)
            tool_reasoner = ToolReasoner(model_id=MODEL_ID, llm=llm)
        else:
            llm = tool_reasoner = None

        # ── build work list ──────────────────────────────────────────────────────
        work: list[tuple] = []
        for _, win_row in wins.iterrows():
            cid  = str(win_row["context_id"])
            pert = str(win_row["perturbation"])
            task = str(win_row["task"])
            all_done = all(
                (cid, pert, task, abl) in done_keys for abl in ABLATION_CONFIGS
            )
            if all_done:
                continue
            mpa_m = mpa_results[
                (mpa_results["context_id"].astype(str) == cid) &
                (mpa_results["perturbation"] == pert)
            ]
            if mpa_m.empty:
                print(f"  WARNING: no MPA row for {cid}/{pert}")
                continue
            tu_ctx = tu_ctx_lookup.get((cid, pert), "")
            work.append((win_row, mpa_m.iloc[0], tu_ctx))

        print(f"Cases to process: {len(work)} of {len(wins)}")

        # ── concurrent append ────────────────────────────────────────────────────
        _lock = threading.Lock()

        def write_rows(rows: list[dict]) -> None:
            if not rows:
                return
            df_w = pd.DataFrame(rows)
            with _lock:
                if out_csv.exists():
                    df_w.to_csv(out_csv, mode="a", header=False, index=False)
                else:
                    df_w.to_csv(out_csv, index=False)

        def process_one(item):
            win_row, mpa_row, tu_ctx = item
            try:
                rows = run_ablation_case(
                    win_row, mpa_row, tu_ctx, ctx_lookup,
                    llm, tool_reasoner, dry_run=args.dry_run,
                )
                rows = [
                    r for r in rows
                    if (r["context_id"], r["perturbation"], r["task"], r["ablation"])
                       not in done_keys
                ]
                write_rows(rows)
            except Exception as exc:
                cid = win_row["context_id"]
                print(f"  ERROR on {cid}/{win_row['perturbation']}: {exc}")

        if args.workers > 1 and not args.dry_run:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = [pool.submit(process_one, item) for item in work]
                for _ in tqdm(as_completed(futs), total=len(futs), desc="LOTO"):
                    pass
        else:
            for item in tqdm(work, desc="LOTO (serial)"):
                process_one(item)

    # ── stats ────────────────────────────────────────────────────────────────────
    if not out_csv.exists():
        print("No results file found — aborting stats.")
        return

    df = pd.read_csv(out_csv)
    print(f"\nTotal rows: {len(df)}")
    n_cases = df[df["ablation"] == "full"]["context_id"].nunique()
    print(f"Distinct MPA win cases: {n_cases}")

    # ── flip rate summary ────────────────────────────────────────────────────────
    abl_order = list(ABLATION_CONFIGS.keys())
    summary_rows = []
    for abl in abl_order:
        sub = df[df["ablation"] == abl]
        if sub.empty:
            continue
        n   = len(sub)
        nf  = int(sub["flipped"].sum())
        pct = round(100 * nf / n, 1)
        # per-task
        by_task = sub.groupby("task")["flipped"].agg(n="count", nf="sum")
        summary_rows.append({
            "ablation":   abl,
            "n_cases":    n,
            "n_flipped":  nf,
            "pct_flipped": pct,
            **{f"flip_{t}": int(by_task.loc[t, "nf"]) if t in by_task.index else 0
               for t in TASKS},
            **{f"pct_{t}":  round(100 * by_task.loc[t, "nf"] / by_task.loc[t, "n"], 1)
               if t in by_task.index and by_task.loc[t, "n"] > 0 else 0
               for t in TASKS},
        })

    summary = pd.DataFrame(summary_rows)

    print("\n=== LOTO Ablation — Flip Rates ===")
    print(f"{'ablation':<22} {'n':>4} {'flipped':>8} {'%':>6}  "
          f"{'manage':>7} {'visit':>7} {'resource':>9}")
    for _, r in summary.iterrows():
        print(f"  {r['ablation']:<20} {r['n_cases']:>4} {r['n_flipped']:>8} "
              f"{r['pct_flipped']:>5.1f}%  "
              f"{r.get('pct_manage',0):>6.1f}% "
              f"{r.get('pct_visit',0):>6.1f}% "
              f"{r.get('pct_resource',0):>8.1f}%")

    # ── Cochran's Q ─────────────────────────────────────────────────────────────
    print("\n=== Cochran's Q (across all ablation configs) ===")
    pivot_all = df.pivot_table(
        index=["context_id", "perturbation", "task"],
        columns="ablation",
        values="flipped",
        aggfunc="first",
    ).dropna()

    if not pivot_all.empty:
        # binary: 1=error (flipped), 0=correct
        mat = pivot_all.values.astype(int)
        Q, p_q = cochrans_q(mat)
        sig_q = "***" if p_q < .001 else "**" if p_q < .01 else "*" if p_q < .05 else "ns"
        print(f"  Q = {Q:.3f}, df = {mat.shape[1]-1}, p = {p_q:.4f}  {sig_q}")
    else:
        print("  (insufficient complete cases for Cochran's Q)")

    # ── pairwise McNemar vs full ─────────────────────────────────────────────────
    print("\n=== Pairwise McNemar: each ablation vs 'full' (Bonferroni corrected) ===")
    mcn = pairwise_mcnemar_vs_full(df)
    if not mcn.empty:
        print(f"  {'ablation':<22} {'lost':>5} {'recov':>6} {'p':>8} {'p_bonf':>8}  sig")
        for _, r in mcn.iterrows():
            ps    = f"{r['mcnemar_p']:.4f}"  if r["mcnemar_p"]  is not None else "N/A"
            pb    = f"{r['mcnemar_p_bonf']:.4f}" if r["mcnemar_p_bonf"] is not None else "N/A"
            print(f"  {r['ablation']:<22} {r['n_flipped_by_abl']:>5} "
                  f"{r['n_recovered']:>6} {ps:>8} {pb:>8}  {r['sig']}")
        summary = summary.merge(
            mcn[["ablation","n_flipped_by_abl","n_recovered",
                 "mcnemar_p","mcnemar_p_bonf","sig"]],
            on="ablation", how="left"
        )

    # ── perturbation breakdown ───────────────────────────────────────────────────
    print("\n=== Flip rates by ablation × perturbation ===")
    pert_tbl = (
        df.groupby(["ablation", "perturbation"])["flipped"]
        .agg(n="count", nf="sum")
        .assign(pct=lambda x: (100 * x["nf"] / x["n"]).round(1))
        .reset_index()
        .pivot_table(index="ablation", columns="perturbation", values="pct", aggfunc="first")
    )
    print(pert_tbl.reindex(abl_order).to_string())

    # ── dataset breakdown ────────────────────────────────────────────────────────
    print("\n=== Flip rates by ablation × dataset ===")
    ds_tbl = (
        df.groupby(["ablation", "dataset"])["flipped"]
        .agg(n="count", nf="sum")
        .assign(pct=lambda x: (100 * x["nf"] / x["n"]).round(1))
        .reset_index()
        .pivot_table(index="ablation", columns="dataset", values="pct", aggfunc="first")
    )
    print(ds_tbl.reindex(abl_order).to_string())

    # ── per-case importance matrix ───────────────────────────────────────────────
    importance = build_importance(df)
    importance.to_csv(imp_csv, index=False)
    print(f"\nImportance matrix → {imp_csv}  ({len(importance)} cases)")

    # ── which families caused flips? ─────────────────────────────────────────────
    print("\n=== Cases that flipped ONLY when specific family removed ===")
    family_ablations = [a for a in abl_order if a.startswith("no_") and
                        a not in ("no_novel", "no_tu")]
    full_df  = df[df["ablation"] == "full"].set_index(
        ["context_id","perturbation","task"])["flipped"]
    for abl in family_ablations:
        abl_df = df[df["ablation"] == abl].set_index(
            ["context_id","perturbation","task"])["flipped"]
        idx = full_df.index.intersection(abl_df.index)
        lost = int(((full_df[idx] == False) & (abl_df[idx] == True)).sum())
        print(f"  {abl:<22} → {lost:>3} wins lost when this family dropped")

    # ── save ─────────────────────────────────────────────────────────────────────
    summary.to_csv(sum_csv, index=False)
    print(f"\nSaved → {out_csv}")
    print(f"Saved → {sum_csv}")
    print(f"Saved → {imp_csv}")


if __name__ == "__main__":
    main()
