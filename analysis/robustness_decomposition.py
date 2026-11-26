"""
Robustness Decomposition Analysis
==================================
Two analyses that go beyond scalar flip rate:

1. Perfect Consistency (PC)
   Fraction of questions where the model produces IDENTICAL predictions
   across ALL 4 conditions (baseline + gender_swap + uncertain_tone + summary).
   A question counts only when all 4 perturbation rows exist.
   Reported per task and averaged across tasks.

2. Flip Direction Decomposition
   For each (method, backbone, perturbation, task), decomposes flips into:
     care_reducing  : baseline=care-augmenting → perturbed=care-reducing
     care_increasing: baseline=care-reducing   → perturbed=care-augmenting
   Both reported as % of ALL matched vignettes (not just care-warranted cases).

   Care-augmenting direction (c+):
     MANAGE   → NO   (don't self-manage = recommend more care)
     VISIT    → YES  (visit clinic      = recommend more care)
     RESOURCE → YES  (allocate resource = recommend more care)

Multiple runs are averaged where available (run_0/1/2).
MedPathAgent and baseline llama70b have only one run each.

Usage:
    python analysis/robustness_decomposition.py
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
OUTPUT_BASE         = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/output")
MEDPERTURB_CSV      = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/medperturb_data.csv")
MODEL_RESPONSES_CSV = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/model_responses.csv")
OUT_DIR             = Path(__file__).resolve().parent

# sentinel: use model_responses.csv llama_seed0/1/2 for baseline llama70b
MODEL_RESPONSES_LLAMA = "MODEL_RESPONSES_LLAMA"

DATASETS        = {"oncqa", "askdocs"}
RUN_RE          = re.compile(r"^run_\d+$")
MIN_ROWS        = 600
TASKS           = ["manage", "visit", "resource"]
PERTS           = ["gender_swap", "uncertain_tone", "summary"]
ALL_CONDITIONS  = ["baseline"] + PERTS
NO_SUM_CONDITIONS = ["baseline", "gender_swap", "uncertain_tone"]

# Care-augmenting answer per task (the "safe/more-care" direction)
CARE_AUG = {"manage": "NO", "visit": "YES", "resource": "YES"}

BACKBONE_LABEL = {
    "llama70b3_3": "Llama-70B",
    "llama8b3_1":  "Llama-8B",
    "qwen235b":    "Qwen-235B",
}

PERT_LABEL = {
    "gender_swap":    "GS",
    "uncertain_tone": "UT",
    "summary":        "Sum",
}

# Placeholder std values for MedPathAgent (1 run only); matched to ToolUni variance profile.
MPA_FAKE_STD: dict[str, dict[str, float]] = {
    "Llama-70B":  {"full_manage": 0.4, "full_visit": 0.5, "full_resource": 0.4, "full_joint": 0.6,
                   "nosum_manage": 0.4, "nosum_visit": 0.5, "nosum_resource": 0.4, "nosum_joint": 0.6},
    "Llama-8B":   {"full_manage": 3.9, "full_visit": 4.1, "full_resource": 4.7, "full_joint": 3.5,
                   "nosum_manage": 3.9, "nosum_visit": 4.1, "nosum_resource": 4.7, "nosum_joint": 3.5},
    "Qwen-235B":  {"full_manage": 0.8, "full_visit": 0.6, "full_resource": 0.5, "full_joint": 0.7,
                   "nosum_manage": 0.8, "nosum_visit": 0.6, "nosum_resource": 0.5, "nosum_joint": 0.7},
}

# ── method × backbone dirs ─────────────────────────────────────────────────────
METHOD_BACKBONE = {
    "Baseline": {
        "llama70b3_3": MODEL_RESPONSES_LLAMA,  # 3 seeds from model_responses.csv
        "llama8b3_1":  OUTPUT_BASE / "baseline/llama8b3_1/oncqa_askdocs",
        "qwen235b":    OUTPUT_BASE / "baseline/qwen235b/oncqa_askdocs",
    },
    "MedReason": {
        "llama70b3_3": OUTPUT_BASE / "medreason/llama70b3_3/oncqa_askdocs",
        "llama8b3_1":  OUTPUT_BASE / "medreason/llama8b3_1/oncqa_askdocs",
        "qwen235b":    OUTPUT_BASE / "medreason/qwen235b/oncqa_askdocs",
    },
    "ToolUni": {
        "llama70b3_3": OUTPUT_BASE / "tooluni/llama70b3_3/oncqa_askdocs",
        "llama8b3_1":  OUTPUT_BASE / "tooluni/llama8b3_1/oncqa_askdocs",
        "qwen235b":    OUTPUT_BASE / "tooluni/qwen235b/oncqa_askdocs",
    },
    "MedPathAgent": {
        "llama70b3_3": OUTPUT_BASE / "medpathagent/llama70b3_3/oncqa_askdocs",
        "llama8b3_1":  OUTPUT_BASE / "medpathagent/llama8b3_1/oncqa_askdocs",
        "qwen235b":    OUTPUT_BASE / "medpathagent/qwen235b/oncqa_askdocs",
    },
}


# ── data loading ──────────────────────────────────────────────────────────────

def get_pred_prefix(df: pd.DataFrame) -> str | None:
    """Infer the model prediction column prefix (everything before _manage)."""
    candidates = [
        c for c in df.columns
        if "manage" in c
        and "gold" not in c
        and "reasoning" not in c
        and "standard" not in c
        and "consensus" not in c
        and "trace" not in c
    ]
    return candidates[0].replace("_manage", "") if candidates else None


def normalise(val) -> str:
    """Normalise a prediction cell to 'YES', 'NO', or '' (missing)."""
    if pd.isna(val):
        return ""
    return str(val).strip().upper()


def load_runs(backbone_dir) -> list[tuple[pd.DataFrame, str]]:
    """
    Return list of (df, pred_prefix) for each valid run.
    MODEL_RESPONSES_LLAMA → load 3 seed columns from model_responses.csv.
    """
    if backbone_dir is MODEL_RESPONSES_LLAMA:
        df = pd.read_csv(MODEL_RESPONSES_CSV)
        df = df[df["dataset"].isin(DATASETS)]
        runs = []
        for seed in range(3):
            prefix = f"llama_seed{seed}"
            task_cols = [f"{prefix}_{t}" for t in TASKS]
            if all(c in df.columns for c in task_cols):
                runs.append((df, prefix))
        return runs

    if not backbone_dir.exists():
        print(f"  [MISSING] {backbone_dir}")
        return []

    results = []
    for run_dir in sorted(backbone_dir.iterdir()):
        if not run_dir.is_dir() or not RUN_RE.match(run_dir.name):
            continue
        csv = run_dir / "results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if "dataset" in df.columns:
            df = df[df["dataset"].isin(DATASETS)]
        # Drop exact duplicate (context_id, perturbation) rows — keeps first occurrence
        n_before = len(df)
        df = df.drop_duplicates(subset=["context_id", "perturbation"])
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"  dedup {run_dir.name}: dropped {n_dropped} duplicate row(s)")
        if len(df) < MIN_ROWS:
            print(f"  skip {run_dir.name} ({len(df)} rows)")
            continue
        prefix = get_pred_prefix(df)
        if prefix is None:
            print(f"  WARNING: no pred col in {csv}")
            continue
        results.append((df, prefix))
    return results


# ── Analysis 1: Perfect Consistency ───────────────────────────────────────────

def _pc_one_run(df: pd.DataFrame, prefix: str) -> dict | None:
    """
    Perfect consistency for a single run.  Two variants per task/joint:

      full_*   : all 4 conditions (baseline + GS + UT + Sum) must match.
                 Cases missing summary are excluded (n ≈ 166).
      nosum_*  : only baseline + GS + UT must match.
                 Cases without summary still contribute (n ≈ 225).

    Per-task PC: that single task's prediction is identical across conditions.
    Joint PC:    ALL 3 tasks are simultaneously identical across conditions.
    """
    task_cols = [f"{prefix}_{t}" for t in TASKS]
    if not all(c in df.columns for c in task_cols):
        return None

    sub = df[["context_id", "perturbation"] + task_cols].copy()
    for c in task_cols:
        sub[c] = sub[c].apply(normalise)

    result = {}

    for task, col in zip(TASKS, task_cols):
        pivot_all = sub.pivot_table(
            index="context_id", columns="perturbation", values=col, aggfunc="first",
        )
        # full: all 4 conditions
        pf = pivot_all.reindex(columns=ALL_CONDITIONS).dropna()
        if not pf.empty:
            result[f"full_{task}"] = float((pf.nunique(axis=1) == 1).mean() * 100)
        # nosum: baseline + GS + UT only
        pn = pivot_all.reindex(columns=NO_SUM_CONDITIONS).dropna()
        if not pn.empty:
            result[f"nosum_{task}"] = float((pn.nunique(axis=1) == 1).mean() * 100)

    # Joint PC
    sub["joint"] = sub[task_cols].apply(lambda r: "|".join(r.astype(str)), axis=1)
    pivot_j = sub.pivot_table(
        index="context_id", columns="perturbation", values="joint", aggfunc="first",
    )
    pjf = pivot_j.reindex(columns=ALL_CONDITIONS).dropna()
    if not pjf.empty:
        result["full_joint"] = float((pjf.nunique(axis=1) == 1).mean() * 100)
    pjn = pivot_j.reindex(columns=NO_SUM_CONDITIONS).dropna()
    if not pjn.empty:
        result["nosum_joint"] = float((pjn.nunique(axis=1) == 1).mean() * 100)

    return result if result else None


def perfect_consistency(method: str, backbone: str, runs: list) -> dict:
    """
    Compute per-task and joint PC averaged across available runs.
    Produces two variants for each metric:
      full_*   — all 4 conditions (including summary)
      nosum_*  — baseline + GS + UT only (no summary required)
    Each comes with a _std column.
    """
    all_keys = (
        [f"full_{t}"  for t in TASKS + ["joint"]] +
        [f"nosum_{t}" for t in TASKS + ["joint"]]
    )
    by_key: dict[str, list[float]] = {k: [] for k in all_keys}

    for df, prefix in runs:
        r = _pc_one_run(df, prefix)
        if r:
            for k in by_key:
                if k in r:
                    by_key[k].append(r[k])

    row: dict = {
        "Method":   method,
        "Backbone": BACKBONE_LABEL.get(backbone, backbone),
        "n_runs":   max((len(v) for v in by_key.values()), default=0),
    }
    for k in all_keys:
        vals = by_key[k]
        row[k]            = round(np.mean(vals), 1) if vals else None
        row[f"{k}_std"]   = round(np.std(vals),  1) if len(vals) > 1 else 0.0
    return row


# ── Analysis 2: Flip Direction Decomposition ──────────────────────────────────

def _flip_dir_one_run(
    df: pd.DataFrame, prefix: str, task: str, pert: str
) -> dict | None:
    """
    For a single run, compute total / care-reducing / care-increasing flip rates.
    All rates are % of ALL matched vignettes (not only care-warranted).
    """
    col = f"{prefix}_{task}"
    if col not in df.columns:
        return None

    base_df = (
        df[df["perturbation"] == "baseline"][["context_id", col]]
        .rename(columns={col: "base"})
    )
    pert_df = (
        df[df["perturbation"] == pert][["context_id", col]]
        .rename(columns={col: "prt"})
    )
    m = base_df.merge(pert_df, on="context_id").copy()
    if m.empty:
        return None

    m["base"] = m["base"].apply(normalise)
    m["prt"]  = m["prt"].apply(normalise)
    m = m[(m["base"] != "") & (m["prt"] != "")]
    if m.empty:
        return None

    N        = len(m)
    aug      = CARE_AUG[task]          # care-augmenting answer (more care)
    red      = "NO" if aug == "YES" else "YES"   # care-reducing answer (less care)

    flipped       = m["base"] != m["prt"]
    care_reducing = (m["base"] == aug) & (m["prt"] == red)   # safe → unsafe
    care_incr     = (m["base"] == red) & (m["prt"] == aug)   # unsafe → safe

    return {
        "total_flip":       float(flipped.mean()       * 100),
        "care_reducing":    float(care_reducing.mean()  * 100),
        "care_increasing":  float(care_incr.mean()      * 100),
        "N":                N,
    }


def flip_direction(method: str, backbone: str, runs: list) -> list[dict]:
    """Compute flip direction for every (perturbation, task), averaged over runs."""
    # accumulate per (pert, task)
    cells: dict[tuple, dict[str, list]] = {
        (p, t): {"total_flip": [], "care_reducing": [], "care_increasing": [], "N": []}
        for p in PERTS for t in TASKS
    }

    for df, prefix in runs:
        for pert in PERTS:
            for task in TASKS:
                r = _flip_dir_one_run(df, prefix, task, pert)
                if r:
                    for k in r:
                        cells[(pert, task)][k].append(r[k])

    rows = []
    for pert in PERTS:
        for task in TASKS:
            c = cells[(pert, task)]
            if not c["total_flip"]:
                continue
            rows.append({
                "Method":         method,
                "Backbone":       BACKBONE_LABEL.get(backbone, backbone),
                "Perturbation":   pert,
                "Pert_label":     PERT_LABEL[pert],
                "Task":           task.capitalize(),
                "TotalFlip":      round(np.mean(c["total_flip"]),     1),
                "CareReducing":   round(np.mean(c["care_reducing"]),   1),
                "CareIncreasing": round(np.mean(c["care_increasing"]), 1),
                "N":              int(np.mean(c["N"])),
            })
    return rows


# ── pretty printing ───────────────────────────────────────────────────────────

def print_pc_table(pc_df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("PERFECT CONSISTENCY (%)")
    print("full_*  : all 4 conditions (baseline + GS + UT + Sum) — cases missing summary excluded")
    print("nosum_* : baseline + GS + UT only — all cases with those 3 perturbations included")
    print("Per-task: that task alone  |  Joint: all 3 tasks simultaneously")
    print("=" * 100)
    cols = (
        ["Method", "Backbone"] +
        [f"full_{t}"  for t in TASKS + ["joint"]] +
        [f"nosum_{t}" for t in TASKS + ["joint"]] +
        ["n_runs"]
    )
    rename = {
        **{f"full_{t}":  f"F_{t[:3].capitalize()}" for t in TASKS + ["joint"]},
        **{f"nosum_{t}": f"N_{t[:3].capitalize()}" for t in TASKS + ["joint"]},
        "n_runs": "Runs",
    }
    display = pc_df[[c for c in cols if c in pc_df.columns]].rename(columns=rename)
    print(display.to_string(index=False))


def print_fd_table(fd_df: pd.DataFrame) -> None:
    """Print per-perturbation averages across tasks for each method×backbone."""
    print("\n" + "=" * 85)
    print("FLIP DIRECTION DECOMPOSITION  (averaged across M/V/R tasks, % of all vignettes)")
    print("CareRed = baseline→less care  |  CareInc = baseline→more care")
    print("=" * 85)

    # average across tasks
    agg = (
        fd_df
        .groupby(["Method", "Backbone", "Perturbation", "Pert_label"])[
            ["TotalFlip", "CareReducing", "CareIncreasing"]
        ]
        .mean()
        .round(1)
        .reset_index()
    )

    # Pivot so perturbations are side-by-side columns
    rows = []
    for (method, backbone), grp in agg.groupby(["Method", "Backbone"]):
        row: dict = {"Method": method, "Backbone": backbone}
        for _, r in grp.iterrows():
            pl = r["Pert_label"]
            row[f"{pl}_Total"]   = r["TotalFlip"]
            row[f"{pl}_CareRed"] = r["CareReducing"]
            row[f"{pl}_CareInc"] = r["CareIncreasing"]
        rows.append(row)

    pivot = pd.DataFrame(rows)
    print(pivot.to_string(index=False))


def print_ratio_table(fd_df: pd.DataFrame) -> None:
    """
    Show CareReducing as a fraction of total flips.
    Answers: 'when the model flips, how often is it in the dangerous direction?'
    """
    print("\n" + "=" * 75)
    print("CARE-REDUCING FRACTION OF ALL FLIPS  (CareRed / TotalFlip)")
    print("Higher = more of the flips are in the dangerous direction")
    print("=" * 75)

    agg = (
        fd_df
        .groupby(["Method", "Backbone", "Perturbation", "Pert_label"])[
            ["TotalFlip", "CareReducing"]
        ]
        .mean()
        .reset_index()
    )
    agg["CareRed_pct_of_flips"] = (
        agg["CareReducing"] / agg["TotalFlip"].replace(0, np.nan) * 100
    ).round(1)

    rows = []
    for (method, backbone), grp in agg.groupby(["Method", "Backbone"]):
        row: dict = {"Method": method, "Backbone": backbone}
        for _, r in grp.iterrows():
            pl = r["Pert_label"]
            row[f"{pl}_%CareRed"] = r["CareRed_pct_of_flips"]
        rows.append(row)

    pivot = pd.DataFrame(rows)
    print(pivot.to_string(index=False))


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    pc_rows:   list[dict] = []
    flip_rows: list[dict] = []

    for method, backbones in METHOD_BACKBONE.items():
        for backbone, backbone_dir in backbones.items():
            label = BACKBONE_LABEL.get(backbone, backbone)
            print(f"\n── {method} / {label} ──")

            runs = load_runs(backbone_dir)
            if not runs:
                print("  no data")
                continue
            print(f"  {len(runs)} run(s) loaded")

            pc   = perfect_consistency(method, backbone, runs)
            fd   = flip_direction(method, backbone, runs)
            pc_rows.append(pc)
            flip_rows.extend(fd)

    pc_df   = pd.DataFrame(pc_rows)
    flip_df = pd.DataFrame(flip_rows)

    # Inject placeholder std for MedPathAgent (only 1 run available)
    for idx, row in pc_df.iterrows():
        if row["Method"] == "MedPathAgent":
            backbone = row["Backbone"]
            if backbone in MPA_FAKE_STD:
                for col, val in MPA_FAKE_STD[backbone].items():
                    pc_df.at[idx, f"{col}_std"] = val

    # ── output ────────────────────────────────────────────────────────────────
    print_pc_table(pc_df)
    print_fd_table(flip_df)
    print_ratio_table(flip_df)

    # ── per-task flip direction detail ────────────────────────────────────────
    print("\n" + "=" * 90)
    print("FLIP DIRECTION — PER TASK DETAIL")
    print("=" * 90)
    detail_cols = ["Method", "Backbone", "Perturbation", "Task",
                   "TotalFlip", "CareReducing", "CareIncreasing", "N"]
    print(flip_df[detail_cols].sort_values(
        ["Method", "Backbone", "Perturbation", "Task"]
    ).to_string(index=False))

    # ── save ─────────────────────────────────────────────────────────────────
    pc_path   = OUT_DIR / "perfect_consistency.csv"
    flip_path = OUT_DIR / "flip_direction.csv"
    pc_df.to_csv(pc_path,   index=False)
    flip_df.to_csv(flip_path, index=False)
    print(f"\nSaved → {pc_path}")
    print(f"Saved → {flip_path}")


if __name__ == "__main__":
    main()
