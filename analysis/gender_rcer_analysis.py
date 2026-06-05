"""
Gender-stratified Reduced Care Error Rate (RCER) analysis.

Splits gender_swap (and other) perturbation rows by the *original* patient gender
(original_gender column: M / F / Unknown) to examine whether RCER differs
between male and female patients.

Outputs:
  - analysis/gender_rcer_results.csv  — per method × backbone × task × perturbation × gender
  - plots/rcer/rcer_gender_stratified.png
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────────

OUTPUT_BASE    = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/output")
MEDPERTURB_CSV = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/medperturb_data.csv")
PLOT_DIR       = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/plots/rcer")
OUT_CSV        = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/analysis/gender_rcer_results.csv")

DATASETS = {"oncqa", "askdocs"}
TASKS    = ["manage", "visit", "resource"]
PERTS    = ["summary", "gender_swap", "uncertain_tone"]

PERT_LABELS = {
    "summary":        "Sum.",
    "gender_swap":    "GS",
    "uncertain_tone": "UT",
}

NAME_MAP = {
    "llama70b3_3": "Llama 70B",
    "llama8b3_1":  "Llama 8B",
    "qwen235b":    "Qwen 235B",
}

METHOD_SHORT = {
    "Baseline":     "Baseline",
    "MedReason":    "MRA",
    "ToolUni":      "TUA",
    "MedPathAgent": "MPA",
}

MEDPERTURB_BASELINE_COLS = {
    "llama70b3_3": "llama",
    "qwen235b":    "qwen",
}

METHOD_CSV_PATHS = {
    "Baseline": {
        "llama70b3_3": None,
        "llama8b3_1":  OUTPUT_BASE / "baseline/llama8b3_1/oncqa_askdocs/run_0/results.csv",
        "qwen235b":    OUTPUT_BASE / "baseline/qwen235b/oncqa_askdocs/run_0/results.csv",
    },
    "MedReason": {
        "llama70b3_3": OUTPUT_BASE / "medreason/llama70b3_3/oncqa_askdocs/run_0/results.csv",
        "llama8b3_1":  OUTPUT_BASE / "medreason/llama8b3_1/oncqa_askdocs/run_0/results.csv",
        "qwen235b":    OUTPUT_BASE / "medreason/qwen235b/oncqa_askdocs/run_0/results.csv",
    },
    "ToolUni": {
        "llama70b3_3": OUTPUT_BASE / "tooluni/llama70b3_3/oncqa_askdocs/run_0/results.csv",
        "llama8b3_1":  OUTPUT_BASE / "tooluni/llama8b3_1/oncqa_askdocs/run_0/results.csv",
        "qwen235b":    OUTPUT_BASE / "tooluni/qwen235b/oncqa_askdocs/run_0/results.csv",
    },
    "MedPathAgent": {
        "llama70b3_3": OUTPUT_BASE / "medpathagent/llama70b3_3/oncqa_askdocs/run_0/results.csv",
        "llama8b3_1":  OUTPUT_BASE / "medpathagent/llama8b3_1/oncqa_askdocs/run_0/results.csv",
        "qwen235b":    OUTPUT_BASE / "medpathagent/qwen235b/oncqa_askdocs/run_0/results.csv",
    },
}

FONT = 16
GENDER_COLORS = {"M": "#4C72B0", "F": "#DD8452"}
METHOD_ORDER   = ["Baseline", "MedReason", "ToolUni", "MedPathAgent"]
BACKBONE_ORDER = ["Llama 70B", "Qwen 235B", "Llama 8B"]

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── load medperturb metadata (gender labels) ───────────────────────────────────

mp_meta = pd.read_csv(MEDPERTURB_CSV)
mp_meta = mp_meta[mp_meta["dataset"].isin(DATASETS)]

# Build lookup: (context_id, perturbation) → original_gender
gender_lookup = (
    mp_meta.set_index(["context_id", "perturbation"])["original_gender"]
    .to_dict()
)

# Clinician consensus lookup: (context_id, task) → label
clin_lookup: dict[tuple, str] = {}
for task in TASKS:
    col = f"clinician_consensus_{task}"
    if col in mp_meta.columns:
        base_rows = mp_meta[mp_meta["perturbation"] == "baseline"]
        for _, row in base_rows.iterrows():
            val = row.get(col, "")
            if pd.notna(val) and str(val).strip():
                clin_lookup[(str(row["context_id"]), task)] = str(val).strip().upper()


def get_gold(df: pd.DataFrame, context_id, task: str) -> str:
    """Return clinician label if available, else gold_standard."""
    clin = clin_lookup.get((int(context_id), task), "")
    if clin in ("YES", "NO"):
        return clin
    col = f"gold_standard_{task}"
    rows = df[df["context_id"] == context_id]
    if not rows.empty and col in rows.columns:
        return str(rows.iloc[0][col]).strip().upper()
    return ""


# ── helpers ───────────────────────────────────────────────────────────────────

def pred_col(df: pd.DataFrame, task: str) -> str | None:
    candidates = [c for c in df.columns
                  if task in c and "gold" not in c and "reasoning" not in c
                  and "standard" not in c and "consensus" not in c]
    return candidates[0] if candidates else None


def load_df(method: str, backbone: str, path) -> tuple[pd.DataFrame, str] | None:
    if path is None and method == "Baseline" and backbone in MEDPERTURB_BASELINE_COLS:
        prefix = MEDPERTURB_BASELINE_COLS[backbone]
        df = pd.read_csv(MEDPERTURB_CSV)
        df = df[df["dataset"].isin(DATASETS)]
        if f"{prefix}_manage" not in df.columns:
            return None
        return (df, prefix)
    if path is None or not Path(path).exists():
        print(f"  MISSING: {method} {backbone}: {path}")
        return None
    df = pd.read_csv(path)
    if "dataset" in df.columns:
        df = df[df["dataset"].isin(DATASETS)]
    if len(df) < 500:
        print(f"  skip {method} {backbone} ({len(df)} rows)")
        return None
    col = pred_col(df, "manage")
    if col is None:
        return None
    prefix = col.replace("_manage", "")
    return (df, prefix)


def care_augmenting(task: str) -> str:
    """Return the care-augmenting label for a task."""
    return "NO" if task == "manage" else "YES"


def compute_rcer_by_gender(df: pd.DataFrame, prefix: str, task: str, perturbation: str
                            ) -> dict[str, float | None]:
    """
    Compute RCER separately for M, F, and Unknown patients.
    RCER = (clinician=c+ AND pert=c-) / (clinician=c+)
    """
    col = f"{prefix}_{task}"
    c_plus = care_augmenting(task)
    c_minus = "YES" if c_plus == "NO" else "NO"

    pert_rows = df[df["perturbation"] == perturbation][["context_id", col]].copy()
    if pert_rows.empty or col not in df.columns:
        return {"M": None, "F": None, "Unknown": None, "All": None}

    results: dict[str, float | None] = {}
    genders = ["M", "F", "Unknown", "All"]

    for gender in genders:
        subset = pert_rows.copy()

        # attach gender
        def _get_gender(cid):
            for key in [cid, str(cid)]:
                v = gender_lookup.get((key, perturbation))
                if v:
                    return v
            return "Unknown"

        subset["gender"] = subset["context_id"].apply(_get_gender)
        if gender != "All":
            subset = subset[subset["gender"] == gender]

        if subset.empty:
            results[gender] = None
            continue

        # attach clinician/gold label
        gold_col = f"gold_standard_{task}"

        def _get_label(cid):
            for key in [str(cid), cid]:
                v = clin_lookup.get((key, task))
                if v in ("YES", "NO"):
                    return v
            rows = df.loc[df["context_id"] == cid, gold_col]
            return str(rows.values[0]).strip().upper() if not rows.empty else ""

        subset["clin_label"] = subset["context_id"].apply(_get_label)

        care_aug = subset[subset["clin_label"] == c_plus]
        if care_aug.empty:
            results[gender] = None
            continue

        errors = (care_aug[col].str.upper() == c_minus).sum()
        results[gender] = errors / len(care_aug) * 100

    return results


# ── main computation ───────────────────────────────────────────────────────────

all_rows = []

for method, bb_paths in METHOD_CSV_PATHS.items():
    print(f"\n=== {method} ===")
    for backbone, path in bb_paths.items():
        result = load_df(method, backbone, path)
        if result is None:
            continue
        df, prefix = result
        bb_label = NAME_MAP.get(backbone, backbone)
        print(f"  {backbone}: prefix={prefix}, rows={len(df)}")

        for task in TASKS:
            for pert in PERTS:
                gender_rcer = compute_rcer_by_gender(df, prefix, task, pert)
                for gender, val in gender_rcer.items():
                    all_rows.append({
                        "method":      method,
                        "method_short": METHOD_SHORT.get(method, method),
                        "backbone":    bb_label,
                        "task":        task,
                        "perturbation": pert,
                        "gender":      gender,
                        "rcer":        val,
                    })

df_results = pd.DataFrame(all_rows)
df_results.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV}")

# ── summary table ──────────────────────────────────────────────────────────────

print("\n=== Gender-stratified RCER (avg across backbones+perts, M vs F) ===")
mf = df_results[df_results["gender"].isin(["M", "F"])]
pivot = (mf.groupby(["method_short", "task", "gender"])["rcer"]
           .mean()
           .round(1)
           .reset_index()
           .pivot_table(index=["method_short", "gender"], columns="task", values="rcer"))
print(pivot.to_string())

print("\n=== Gender gap (F - M) by method×task ===")
gap = (mf.groupby(["method_short", "task", "gender"])["rcer"]
         .mean()
         .unstack("gender"))
gap["F-M"] = gap["F"] - gap["M"]
print(gap[["M", "F", "F-M"]].round(2).to_string())

# ── plot: gender-stratified RCER (gender_swap perturbation only) ───────────────
#   1 row × 3 tasks, M vs F bars grouped by method

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

gs_data = df_results[
    (df_results["perturbation"] == "gender_swap") &
    (df_results["gender"].isin(["M", "F"]))
]
agg_gs = gs_data.groupby(["method_short", "task", "gender"])["rcer"].mean().reset_index()

method_order_short = [METHOD_SHORT[m] for m in METHOD_ORDER if METHOD_SHORT[m] in df_results["method_short"].unique()]
x = np.arange(len(method_order_short))
width = 0.35

for ax, task in zip(axes, TASKS):
    sub = agg_gs[agg_gs["task"] == task]

    for gi, gender in enumerate(["M", "F"]):
        vals = []
        for m_short in method_order_short:
            row = sub[(sub["method_short"] == m_short) & (sub["gender"] == gender)]
            vals.append(row["rcer"].values[0] if not row.empty and pd.notna(row["rcer"].values[0]) else 0.0)
        offset = (gi - 0.5) * width
        ax.bar(x + offset, vals, width,
               color=GENDER_COLORS[gender], alpha=0.85,
               label=gender, edgecolor="white", linewidth=0.8)

    ax.set_title(task.capitalize(), fontsize=FONT + 1, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(method_order_short, fontsize=FONT - 1)
    ax.set_ylabel("RCER (%)", fontsize=FONT)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.tick_params(axis="y", labelsize=FONT - 2)
    ax.set_ylim(0, None)
    ax.grid(axis="y", alpha=0.3)

gender_handles = [
    mpatches.Patch(facecolor=GENDER_COLORS["M"], label="Male (original)"),
    mpatches.Patch(facecolor=GENDER_COLORS["F"], label="Female (original)"),
]
fig.legend(handles=gender_handles, fontsize=FONT - 1,
           loc="upper right", bbox_to_anchor=(0.99, 0.92),
           ncol=1, framealpha=0.9)

plt.suptitle("Gender-Stratified RCER — Gender Swap Perturbation",
             fontsize=FONT + 2, fontweight="bold", y=1.02)
plt.tight_layout()
out_plot = PLOT_DIR / "rcer_gender_stratified_gs.png"
plt.savefig(out_plot, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out_plot}")

# ── plot 2: all perturbations, M vs F, gender_swap only at method=aggregate ───
#   rows = tasks, cols = perturbations; show M vs F for each method (averaged across backbones)

all_pert_mf = df_results[df_results["gender"].isin(["M", "F"])]
agg_all = all_pert_mf.groupby(["method_short", "task", "perturbation", "gender"])["rcer"].mean().reset_index()

fig, axes = plt.subplots(len(TASKS), len(PERTS),
                          figsize=(5 * len(PERTS), 4.5 * len(TASKS)),
                          sharey="row")

for ri, task in enumerate(TASKS):
    for ci, pert in enumerate(PERTS):
        ax = axes[ri][ci]
        sub = agg_all[(agg_all["task"] == task) & (agg_all["perturbation"] == pert)]

        for gi, gender in enumerate(["M", "F"]):
            vals = []
            for m_short in method_order_short:
                row = sub[(sub["method_short"] == m_short) & (sub["gender"] == gender)]
                vals.append(row["rcer"].values[0] if not row.empty and pd.notna(row["rcer"].values[0]) else 0.0)
            offset = (gi - 0.5) * width
            ax.bar(x + offset, vals, width,
                   color=GENDER_COLORS[gender], alpha=0.85,
                   label=gender if (ri == 0 and ci == 0) else "",
                   edgecolor="white", linewidth=0.8)

        if ri == 0:
            ax.set_title(PERT_LABELS[pert], fontsize=FONT, fontweight="bold")
        if ci == 0:
            ax.set_ylabel(f"{task.capitalize()}\nRCER (%)", fontsize=FONT - 1)
        ax.set_xticks(x)
        ax.set_xticklabels(method_order_short, fontsize=FONT - 2, rotation=30, ha="right")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.tick_params(axis="y", labelsize=FONT - 3)
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)

fig.legend(handles=gender_handles, fontsize=FONT - 1,
           loc="upper right", bbox_to_anchor=(0.99, 0.97),
           ncol=1, framealpha=0.9)
plt.tight_layout()
out_plot2 = PLOT_DIR / "rcer_gender_stratified_all.png"
plt.savefig(out_plot2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_plot2}")
