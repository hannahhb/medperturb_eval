"""
Cochran's Q test comparing Baseline, MRA, TUA, MPA on binary outcomes.

Two metrics supported via --metric:

  rcer  (default)
        Label: gold_standard_{task} only.
        Cases: care-augmenting vignettes (gold = c+).
        Binary: 1 = correct (no reduced-care error), 0 = reduced-care error.

  flip
        No label filter — all vignettes included.
        Binary: 1 = stable (prediction same as baseline), 0 = flipped.
        Requires baseline prediction for same vignette.

Cochran's Q tests whether the k=4 methods differ significantly.
Post-hoc: pairwise McNemar with Bonferroni correction (6 pairs).

Usage:
    cd medperturb_eval
    python analysis/cochrans_q.py
    python analysis/cochrans_q.py --metric flip
    python analysis/cochrans_q.py --metric rcer --task visit
    python analysis/cochrans_q.py --metric flip --perturbation gender_swap
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parent.parent
OUTPUT_BASE    = REPO_ROOT / "output"
MEDPERTURB_CSV = REPO_ROOT.parent / "data" / "medperturb_data.csv"

DATASETS = {"oncqa", "askdocs"}
TASKS    = ["manage", "visit", "resource"]
PERTS    = ["summary", "gender_swap", "uncertain_tone"]

BACKBONE_ORDER  = ["llama70b3_3", "qwen235b", "llama8b3_1"]
BACKBONE_LABELS = {"llama70b3_3": "Llama 70B", "qwen235b": "Qwen 235B", "llama8b3_1": "Llama 8B"}

# method → output subdirectory
METHOD_DIRS = {
    "Baseline":     "baseline",
    "MRA":          "medreason",
    "TUA":          "tooluni",
    "MPA":          "medpathagent",
}
METHOD_ORDER = ["Baseline", "MRA", "TUA", "MPA"]

# ── care direction ─────────────────────────────────────────────────────────────

def care_augmenting(task):  return "NO"  if task == "manage" else "YES"
def reduced_care(task):     return "YES" if task == "manage" else "NO"

# ── gold standard label lookup ─────────────────────────────────────────────────

def build_gold_lookup():
    """(context_id, task) → 'YES'|'NO' using gold_standard only."""
    mp = pd.read_csv(MEDPERTURB_CSV)
    mp = mp[(mp["perturbation"] == "baseline") & mp["dataset"].isin(DATASETS)]
    lookup = {}
    for _, row in mp.iterrows():
        cid = str(row["context_id"])
        for task in TASKS:
            gold = row.get(f"gold_standard_{task}", "")
            if pd.notna(gold) and str(gold).strip():
                lookup[(cid, task)] = str(gold).strip().upper()
    return lookup

# Baseline predictions live in medperturb_data.csv for L70B only
BASELINE_PREFIX = {"llama70b3_3": "llama"}

# ── load predictions ───────────────────────────────────────────────────────────

def load_pred_lookup(method, backbone):
    """Returns {(context_id, perturbation, task): 'YES'|'NO'}."""

    # ── Baseline special handling ──────────────────────────────────────────────
    if method == "Baseline":
        if backbone in BASELINE_PREFIX:
            # predictions stored in medperturb_data.csv
            prefix = BASELINE_PREFIX[backbone]
            mp = pd.read_csv(MEDPERTURB_CSV)
            mp = mp[mp["dataset"].isin(DATASETS)]
            lookup = {}
            for _, row in mp.iterrows():
                cid  = str(row["context_id"])
                pert = str(row["perturbation"])
                for task in TASKS:
                    col = f"{prefix}_{task}"
                    if col in mp.columns and pd.notna(row[col]):
                        lookup[(cid, pert, task)] = str(row[col]).strip().upper()
            return lookup
        else:
            # predictions in separate baseline results CSV (e.g. llama8b)
            path = OUTPUT_BASE / f"baseline/{backbone}/oncqa_askdocs/run_0/results.csv"
            if not path.exists():
                return {}
            df = pd.read_csv(path)

    else:
        path = OUTPUT_BASE / f"{METHOD_DIRS[method]}/{backbone}/oncqa_askdocs/run_0/results.csv"
        if not path.exists():
            return {}
        df = pd.read_csv(path)

    # find prediction column prefix from dataframe
    # matches both "something_{backbone}_manage" and "{backbone}_manage"
    prefix = None
    for col in df.columns:
        if col.endswith("_manage") and backbone in col:
            prefix = col.replace("_manage", "")
            break
    if prefix is None:
        return {}
    lookup = {}
    for _, row in df.iterrows():
        cid  = str(row["context_id"])
        pert = str(row["perturbation"])
        for task in TASKS:
            col = f"{prefix}_{task}"
            if col in df.columns and pd.notna(row[col]):
                lookup[(cid, pert, task)] = str(row[col]).strip().upper()
    return lookup

# ── statistical tests ──────────────────────────────────────────────────────────

def cochrans_q(mat: np.ndarray):
    """Cochran's Q on (N_subjects × k_treatments) binary matrix. Returns (Q, p, df)."""
    N, k = mat.shape
    L = mat.sum(axis=1)
    C = mat.sum(axis=0)
    num = (k - 1) * (k * np.sum(C**2) - np.sum(C)**2)
    den = k * np.sum(L) - np.sum(L**2)
    if den == 0:
        return 0.0, 1.0, k - 1
    Q = num / den
    p = float(1 - stats.chi2.cdf(Q, df=k - 1))
    return float(Q), p, k - 1

def mcnemar_p(b, c):
    if b + c == 0:
        return None
    stat = (abs(b - c) - 1)**2 / (b + c)
    return float(1 - stats.chi2.cdf(stat, df=1))

def sig_label(p, bonf=False):
    if p is None:
        return "N/A"
    return ("***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns")

# ── main analysis ──────────────────────────────────────────────────────────────

def run_analysis(task_filter=None, pert_filter=None, backbone_filter=None,
                 metric="rcer"):
    gold_lookup = build_gold_lookup() if metric == "rcer" else {}
    if metric == "rcer":
        print(f"Gold labels: {len(gold_lookup)} (context_id, task) pairs")
    print(f"Metric: {metric.upper()}\n")

    backbones = [backbone_filter] if backbone_filter else BACKBONE_ORDER

    all_results = []

    for backbone in backbones:
        pred_lookups = {m: load_pred_lookup(m, backbone) for m in METHOD_ORDER}
        available = {m: len(v) for m, v in pred_lookups.items()}
        print(f"{BACKBONE_LABELS[backbone]}: predictions loaded — {available}")

        tasks = [task_filter] if task_filter else TASKS
        perts = [pert_filter] if pert_filter else PERTS

        for task in tasks:
            c_plus  = care_augmenting(task)
            c_minus = reduced_care(task)

            for pert in perts:

                if metric == "rcer":
                    # ── RCER: care-augmenting cases, correct vs reduced-care error ──
                    cases = {}
                    for (cid, t), label in gold_lookup.items():
                        if t != task or label != c_plus:
                            continue
                        row = {"backbone": backbone, "task": task, "pert": pert,
                               "context_id": cid}
                        complete = True
                        for m in METHOD_ORDER:
                            pred = pred_lookups[m].get((cid, pert, task), "")
                            if not pred:
                                complete = False
                                break
                            row[m] = 0 if pred == c_minus else 1
                        if complete:
                            cases[cid] = row
                    all_results.extend(cases.values())

                else:
                    # ── FLIP: all cases, stable vs flipped from baseline ───────────
                    # Need baseline predictions for the same vignettes
                    base_preds = {}
                    for m in METHOD_ORDER:
                        for (cid, p, t), pred in pred_lookups[m].items():
                            if p == "baseline" and t == task:
                                base_preds.setdefault(m, {})[cid] = pred

                    # Collect all context_ids with a baseline pred for every method
                    all_cids = set.intersection(
                        *[set(base_preds[m].keys()) for m in METHOD_ORDER
                          if m in base_preds]
                    ) if all(m in base_preds for m in METHOD_ORDER) else set()

                    cases = {}
                    for cid in all_cids:
                        row = {"backbone": backbone, "task": task, "pert": pert,
                               "context_id": cid}
                        complete = True
                        for m in METHOD_ORDER:
                            pred_pert = pred_lookups[m].get((cid, pert, task), "")
                            pred_base = base_preds.get(m, {}).get(cid, "")
                            if not pred_pert or not pred_base:
                                complete = False
                                break
                            # 1 = stable (no flip), 0 = flipped
                            row[m] = 1 if pred_pert == pred_base else 0
                        if complete:
                            cases[cid] = row
                    all_results.extend(cases.values())

    if not all_results:
        print("No complete cases found.")
        return

    df = pd.DataFrame(all_results)
    print(f"\nTotal complete cases (all 4 methods present): {len(df)}")

    # ── run Cochran's Q ────────────────────────────────────────────────────────

    def run_q(sub, label, metric=metric):
        if len(sub) < 4:
            return
        mat = sub[METHOD_ORDER].values.astype(int)
        Q, p, dof = cochrans_q(mat)
        sig = sig_label(p)
        print(f"\n{label}")
        print(f"  N={len(sub)}, Q={Q:.3f}, df={dof}, p={p:.4f}  {sig}")

        # method rate (correct for rcer, stable for flip)
        accs = {m: sub[m].mean() * 100 for m in METHOD_ORDER}
        rate_label = "Correct rate" if metric == "rcer" else "Stable rate"
        print(f"  {rate_label} (%) per method: " +
              "  ".join(f"{m}={accs[m]:.1f}" for m in METHOD_ORDER))

        # post-hoc pairwise McNemar (Bonferroni: 6 pairs)
        pairs = [(a, b) for i, a in enumerate(METHOD_ORDER)
                        for b in METHOD_ORDER[i+1:]]
        n_pairs = len(pairs)
        print(f"  Post-hoc McNemar (Bonferroni ×{n_pairs}):")
        print(f"    {'pair':<16} {'b':>4} {'c':>4} {'p':>8} {'p_bonf':>8}  sig")
        for a, b in pairs:
            col_a = sub[a].values.astype(int)
            col_b = sub[b].values.astype(int)
            # b_count: correct in a, wrong in b
            b_count = int(((col_a == 1) & (col_b == 0)).sum())
            c_count = int(((col_a == 0) & (col_b == 1)).sum())
            p_raw = mcnemar_p(b_count, c_count)
            p_bonf = min(p_raw * n_pairs, 1.0) if p_raw is not None else None
            ps  = f"{p_raw:.4f}"  if p_raw  is not None else "N/A"
            pb  = f"{p_bonf:.4f}" if p_bonf is not None else "N/A"
            sig = sig_label(p_bonf)
            print(f"    {a} vs {b:<10} {b_count:>4} {c_count:>4} {ps:>8} {pb:>8}  {sig}")

    # ── pooled (correlated rows — reported with caveat) ───────────────────────
    run_q(df, "=== POOLED (all backbones × tasks × perturbations) [rows correlated] ===")

    # ── per perturbation — PRIMARY independent tests ───────────────────────────
    # Each vignette appears exactly once per perturbation → rows are independent
    pert_ps: list[float] = []
    perts_to_run = PERTS if not pert_filter else [pert_filter]
    for pert in perts_to_run:
        sub = df[df["pert"] == pert]
        if len(sub) >= 4:
            mat = sub[METHOD_ORDER].values.astype(int)
            Q, p, dof = cochrans_q(mat)
            pert_ps.append(p)
        run_q(sub, f"--- pert={pert} (all backbones × tasks) [independent] ---")

    # ── Fisher's combined p-value across perturbation-level Q tests ───────────
    if len(pert_ps) >= 2:
        print("\n=== Fisher's combined p-value (per-perturbation Q tests) ===")
        print(f"  Combining {len(pert_ps)} independent p-values: "
              + ", ".join(f"{p:.4f}" for p in pert_ps))
        chi2_combined, p_combined = stats.combine_pvalues(pert_ps, method="fisher")
        df_combined = 2 * len(pert_ps)
        sig_combined = sig_label(p_combined)
        print(f"  Fisher χ²({df_combined}) = {chi2_combined:.3f},  "
              f"p = {p_combined:.4f}  {sig_combined}")
        print("  Interpretation: pooled evidence across perturbations without "
              "assuming row independence")

    # ── per backbone ───────────────────────────────────────────────────────────
    print("\n[Exploratory — backbone-stratified, no additional correction]")
    for backbone in backbones:
        sub = df[df["backbone"] == backbone]
        run_q(sub, f"--- {BACKBONE_LABELS[backbone]} (all tasks × perts) ---")

    # ── per task (pooled backbones) ────────────────────────────────────────────
    for task in (TASKS if not task_filter else [task_filter]):
        sub = df[df["task"] == task]
        run_q(sub, f"--- task={task} (all backbones × perts) ---")

    # ── per backbone × task ────────────────────────────────────────────────────
    print("\n=== Per backbone × task (pooled perts) ===")
    print(f"  {'backbone':<12} {'task':<10} {'N':>5} {'Q':>7} {'p':>8}  sig  | "
          + "  ".join(f"{m:>10}" for m in METHOD_ORDER))
    for backbone in backbones:
        for task in (TASKS if not task_filter else [task_filter]):
            sub = df[(df["backbone"] == backbone) & (df["task"] == task)]
            if len(sub) < 4:
                continue
            mat = sub[METHOD_ORDER].values.astype(int)
            Q, p, _ = cochrans_q(mat)
            sig = sig_label(p)
            accs = "  ".join(f"{sub[m].mean()*100:>9.1f}%" for m in METHOD_ORDER)
            print(f"  {BACKBONE_LABELS[backbone]:<12} {task:<10} {len(sub):>5} "
                  f"{Q:>7.2f} {p:>8.4f}  {sig:<4}| {accs}")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric",      choices=["rcer", "flip"], default="rcer",
                    help="rcer (default) or flip")
    ap.add_argument("--task",        choices=TASKS,            default=None)
    ap.add_argument("--perturbation",choices=PERTS,            default=None)
    ap.add_argument("--backbone",    choices=BACKBONE_ORDER,   default=None)
    args = ap.parse_args()
    run_analysis(
        task_filter=args.task,
        pert_filter=args.perturbation,
        backbone_filter=args.backbone,
        metric=args.metric,
    )

if __name__ == "__main__":
    main()
