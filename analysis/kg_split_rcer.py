"""
KG-split RCER and Flip Rate analysis for MedReasonAgent.

Splits vignettes into:
  KG-found    : at least one KG path retrieved by MedReasonAgent
  KG-not-found: no KG paths retrieved (falls back to baseline-style reasoning)

Group membership defined by aggregating kg_paths across all three backbones
(llama70b3_3, llama8b3_1, qwen235b). A vignette is KG-found if ANY backbone
retrieved paths for it (the KG lookup is backbone-independent — same graph,
same entities — so the group is consistent).

For each group, RCER is computed for:
  - MedReasonAgent (averaged across 3 backbones)
  - Baseline       (on the same rows, averaged across 3 backbones)

This allows us to check whether KG-not-found rows are intrinsically harder
(higher baseline RCER) or whether MedReason simply adds less on those cases.

Outputs:
  plots/rcer/kg_split_rcer.png
  analysis/kg_split_rcer.csv
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick

REPO_ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR       = REPO_ROOT.parent / "data"
OUTPUT_BASE    = REPO_ROOT / "output"
MEDPERTURB_CSV = DATA_DIR / "medperturb_data.csv"
PLOT_DIR       = REPO_ROOT / "plots" / "rcer"
ANALYSIS_DIR   = REPO_ROOT / "analysis"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {"oncqa", "askdocs"}
TASKS    = ["manage", "visit", "resource"]
PERTS    = ["summary", "gender_swap", "uncertain_tone"]
BACKBONES = ["llama70b3_3", "llama8b3_1", "qwen235b"]

PERT_LABELS = {"summary": "Sum.", "gender_swap": "GS", "uncertain_tone": "UT"}
TASK_LABELS = {"manage": "Manage", "visit": "Visit", "resource": "Resource"}

# Baseline prediction column per backbone (from medperturb_data.csv or separate CSV)
BASELINE_SRC = {
    "llama70b3_3": ("medperturb", "llama"),       # from medperturb_data.csv
    "llama8b3_1":  ("csv", "llama8b3_1"),
    "qwen235b":    ("csv", "qwen235b"),
}

# ── care direction ─────────────────────────────────────────────────────────────

def care_augmenting(task): return "NO"  if task == "manage" else "YES"
def reduced_care(task):    return "YES" if task == "manage" else "NO"
def is_empty(v):           return str(v).strip() in ("", "nan", "None", "[]")

# ── load gold labels ───────────────────────────────────────────────────────────

def load_gold_lookup():
    mp = pd.read_csv(MEDPERTURB_CSV)
    mp = mp[(mp["perturbation"] == "baseline") & mp["dataset"].isin(DATASETS)]
    lookup = {}
    for _, row in mp.iterrows():
        cid = str(row["context_id"])
        for task in TASKS:
            gold = row.get(f"gold_standard_{task}", "")
            if pd.notna(gold) and str(gold).strip() not in ("", "nan"):
                lookup[(cid, task)] = str(gold).strip().upper()
    return lookup

# ── load baseline predictions ──────────────────────────────────────────────────

def load_baseline_lookup(backbone):
    """Returns {(context_id, perturbation, task): pred}."""
    src, prefix = BASELINE_SRC[backbone]
    if src == "medperturb":
        mp = pd.read_csv(MEDPERTURB_CSV)
        mp = mp[mp["dataset"].isin(DATASETS)]
        lookup = {}
        for _, row in mp.iterrows():
            cid  = str(row["context_id"])
            pert = str(row["perturbation"])
            for task in TASKS:
                val = row.get(f"{prefix}_{task}", "")
                if pd.notna(val) and not is_empty(val):
                    lookup[(cid, pert, task)] = str(val).strip().upper()
        return lookup
    else:
        path = OUTPUT_BASE / f"baseline/{backbone}/oncqa_askdocs/run_0/results.csv"
        df   = pd.read_csv(path)
        lookup = {}
        for _, row in df.iterrows():
            cid  = str(row["context_id"])
            pert = str(row["perturbation"])
            for task in TASKS:
                col = f"{prefix}_{task}"
                if col in df.columns and pd.notna(row[col]):
                    lookup[(cid, pert, task)] = str(row[col]).strip().upper()
        return lookup

# ── load MedReason results ─────────────────────────────────────────────────────

def load_mra(backbone):
    path = OUTPUT_BASE / f"medreason/{backbone}/oncqa_askdocs/run_0/results.csv"
    df   = pd.read_csv(path)
    prefix = f"medreason_{backbone}"
    return df, prefix

# ── RCER computation ───────────────────────────────────────────────────────────

def compute_rcer(rows_df, pred_col, task, gold_lookup):
    """RCER on a subset dataframe, using gold_standard label."""
    c_plus  = care_augmenting(task)
    c_minus = reduced_care(task)
    if pred_col not in rows_df.columns:
        return None
    total_errors = 0
    total_care_aug = 0
    for _, row in rows_df.iterrows():
        cid  = str(row["context_id"])
        label = gold_lookup.get((cid, task), "")
        if label != c_plus:
            continue
        total_care_aug += 1
        pred = str(row.get(pred_col, "") or "").strip().upper()
        if pred == c_minus:
            total_errors += 1
    if total_care_aug == 0:
        return None
    return 100 * total_errors / total_care_aug

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    gold_lookup = load_gold_lookup()
    print(f"Gold labels: {len(gold_lookup)} pairs")

    # ── Step 1: determine KG-found / KG-not-found per context_id ─────────────
    # A context_id is KG-found if ANY backbone retrieved paths for it
    kg_cids: set[str] = set()
    for backbone in BACKBONES:
        df, _ = load_mra(backbone)
        has_kg = df["kg_paths"].notna() & (df["kg_paths"].astype(str).str.strip() != "") & \
                 (df["kg_paths"].astype(str) != "nan")
        kg_cids.update(df[has_kg]["context_id"].astype(str).unique())

    print(f"KG-found context_ids: {len(kg_cids)}")

    # ── Step 2: compute RCER + flip rate per backbone × task × pert × KG ────
    records = []
    for backbone in BACKBONES:
        df_mra, mra_prefix = load_mra(backbone)
        base_lookup = load_baseline_lookup(backbone)

        # baseline lookup for flip rate (baseline perturbation predictions)
        base_baseline_lookup = {}
        for (cid, pert, task), pred in base_lookup.items():
            if pert == "baseline":
                base_baseline_lookup[(cid, task)] = pred

        # MRA baseline predictions for flip rate
        mra_base = df_mra[df_mra["perturbation"] == "baseline"]

        for pert in PERTS:
            mra_pert = df_mra[df_mra["perturbation"] == pert].copy()
            mra_pert["has_kg"] = mra_pert["context_id"].astype(str).isin(kg_cids)

            for kg_group, subset in [("KG-found",     mra_pert[mra_pert["has_kg"]]),
                                      ("KG-not-found", mra_pert[~mra_pert["has_kg"]])]:
                if subset.empty:
                    continue
                for task in TASKS:
                    mra_col = f"{mra_prefix}_{task}"

                    # ── RCER ──────────────────────────────────────────────────
                    c_plus  = care_augmenting(task)
                    c_minus = reduced_care(task)
                    mra_errors = 0; base_errors = 0; rcer_ca = 0
                    for _, row in subset.iterrows():
                        cid   = str(row["context_id"])
                        label = gold_lookup.get((cid, task), "")
                        if label != c_plus:
                            continue
                        rcer_ca += 1
                        mra_pred  = str(row.get(mra_col, "") or "").strip().upper()
                        base_pred = base_lookup.get((cid, pert, task), "")
                        if mra_pred  == c_minus: mra_errors  += 1
                        if base_pred == c_minus: base_errors += 1

                    # ── Flip rate ──────────────────────────────────────────────
                    mra_flips = 0; mra_total = 0
                    base_flips = 0; base_total = 0
                    for _, row in subset.iterrows():
                        cid = str(row["context_id"])

                        # MRA flip: compare perturbed vs baseline prediction
                        mra_base_row = mra_base[mra_base["context_id"].astype(str) == cid]
                        if not mra_base_row.empty and mra_col in df_mra.columns:
                            pred_base = str(mra_base_row.iloc[0].get(mra_col, "") or "").upper()
                            pred_pert = str(row.get(mra_col, "") or "").upper()
                            if pred_base and pred_pert:
                                mra_total += 1
                                if pred_base != pred_pert:
                                    mra_flips += 1

                        # Baseline flip
                        pred_base_b = base_baseline_lookup.get((cid, task), "")
                        pred_pert_b = base_lookup.get((cid, pert, task), "")
                        if pred_base_b and pred_pert_b:
                            base_total += 1
                            if pred_base_b != pred_pert_b:
                                base_flips += 1

                    mra_flip  = 100 * mra_flips  / mra_total  if mra_total  > 0 else None
                    base_flip = 100 * base_flips / base_total if base_total > 0 else None

                    records.append({
                        "backbone":      backbone,
                        "perturbation":  pert,
                        "task":          task,
                        "kg_group":      kg_group,
                        "n_rows":        len(subset),
                        # store raw numerators + denominators for proper aggregation
                        "mra_rcer_num":  mra_errors,
                        "base_rcer_num": base_errors,
                        "rcer_denom":    rcer_ca,
                        "mra_flip_num":  mra_flips,
                        "mra_flip_den":  mra_total,
                        "base_flip_num": base_flips,
                        "base_flip_den": base_total,
                    })

    df_res = pd.DataFrame(records)

    # ── Step 3: aggregate across backbones (sum num/denom, compute rate) ─────
    grp = (df_res.groupby(["perturbation", "task", "kg_group"])
                 .agg(
                     mra_rcer_num=("mra_rcer_num",   "sum"),
                     base_rcer_num=("base_rcer_num",  "sum"),
                     rcer_denom=("rcer_denom",         "sum"),
                     mra_flip_num=("mra_flip_num",     "sum"),
                     mra_flip_den=("mra_flip_den",     "sum"),
                     base_flip_num=("base_flip_num",   "sum"),
                     base_flip_den=("base_flip_den",   "sum"),
                     n_rows=("n_rows",                  "sum"),
                 )
                 .reset_index())

    grp["mra_rcer"]  = 100 * grp["mra_rcer_num"]  / grp["rcer_denom"].replace(0, np.nan)
    grp["base_rcer"] = 100 * grp["base_rcer_num"] / grp["rcer_denom"].replace(0, np.nan)
    grp["mra_flip"]  = 100 * grp["mra_flip_num"]  / grp["mra_flip_den"].replace(0, np.nan)
    grp["base_flip"] = 100 * grp["base_flip_num"] / grp["base_flip_den"].replace(0, np.nan)
    agg = grp

    out_csv = ANALYSIS_DIR / "kg_split_rcer.csv"
    agg[["perturbation","task","kg_group","n_rows",
         "mra_rcer","base_rcer","mra_flip","base_flip"]].to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")
    print(agg[["perturbation","task","kg_group","mra_rcer","base_rcer",
               "mra_flip","base_flip"]].round(2).to_string(index=False))

    # ── Step 4: plot — style matching reference image ─────────────────────────
    # 1 row × 3 tasks, perturbations on x-axis, 4 grouped bars per pert

    # Colours
    C_BASE_KG  = "#9e9e9e"   # dark grey — Baseline (KG found rows)
    C_MRA_KG   = "#5c85d6"   # blue      — MedReason (KG found)
    C_NKG      = "#e8956d"   # orange    — No KG (max of baseline / MRA, shown as one bar)

    # 3-bar layout: Baseline(KG), MRA(KG), NoKG(max)
    bar_configs = [
        ("KG-found",     "base", C_BASE_KG, "Baseline (KG found rows)"),
        ("KG-found",     "mra",  C_MRA_KG,  "MedReason (KG found)"),
        ("KG-not-found", "max",  C_NKG,     "No KG (max of Baseline / MedReason)"),
    ]

    FONT = 25

    def make_kg_plot(metric_suffix, ylabel, title, outname):
        """metric_suffix: 'rcer' or 'flip'"""
        fig, axes = plt.subplots(1, len(TASKS), figsize=(22, 7), sharey=False)
        fig.patch.set_facecolor("white")

        n_perts  = len(PERTS)
        n_bars   = len(bar_configs)
        grp_w    = 0.75
        bar_w    = grp_w / n_bars
        grp_gap  = 1.0
        x_perts  = np.arange(n_perts) * grp_gap

        for ci, task in enumerate(TASKS):
            ax = axes[ci]
            ax.set_facecolor("white")

            for bi, (kg_group, method, color, _) in enumerate(bar_configs):
                heights = []
                for pert in PERTS:
                    sub = agg[(agg["task"] == task) &
                              (agg["perturbation"] == pert) &
                              (agg["kg_group"] == kg_group)]
                    if sub.empty:
                        heights.append(0.0)
                        continue
                    if method == "max":
                        # show the larger of baseline and MRA for no-KG group
                        mra_v  = sub[f"mra_{metric_suffix}"].values[0]
                        base_v = sub[f"base_{metric_suffix}"].values[0]
                        val = max(v for v in [mra_v, base_v] if pd.notna(v)) \
                              if any(pd.notna(v) for v in [mra_v, base_v]) else 0.0
                    else:
                        col_name = f"{'mra' if method == 'mra' else 'base'}_{metric_suffix}"
                        val = sub[col_name].values[0] \
                              if pd.notna(sub[col_name].values[0]) else 0.0
                    heights.append(val)

                offset = (bi - (n_bars - 1) / 2) * bar_w
                ax.bar(x_perts + offset, heights, bar_w * 0.92,
                       color=color, edgecolor="white", linewidth=0.5,
                       label=_ if ci == 0 else "")

            ax.set_title(TASK_LABELS[task], fontsize=FONT + 1, fontweight="bold")
            ax.set_xticks(x_perts)
            ax.set_xticklabels([PERT_LABELS[p] for p in PERTS], fontsize=FONT)
            ax.set_ylim(0, None)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.tick_params(axis="y", labelsize=FONT - 2)
            ax.set_ylabel(ylabel, fontsize=FONT)
            ax.grid(axis="y", alpha=0.25, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)

            for xp in x_perts[1:]:
                ax.axvline(xp - grp_gap / 2, color="#cccccc", lw=0.8, ls="--")

        handles = [mpatches.Patch(facecolor=c, label=lbl)
                   for _, _, c, lbl in bar_configs]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   fontsize=FONT - 2, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.18))

        plt.suptitle(title, fontsize=FONT, fontweight="bold", y=1.02)
        plt.tight_layout()
        out = PLOT_DIR / outname
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved → {out}")

    make_kg_plot(
        "flip", "Flip Rate (%)",
        "Flip Rate by KG Path Availability — MedReasonAgent vs Baseline\n"
        "Aggregated across Llama 70B, Llama 8B, Qwen 235B",
        "kg_split_flip.png",
    )
    make_kg_plot(
        "rcer", "RCER (%)",
        "RCER by KG Path Availability — MedReasonAgent vs Baseline\n"
        "Aggregated across Llama 70B, Llama 8B, Qwen 235B",
        "kg_split_rcer.png",
    )


if __name__ == "__main__":
    main()
