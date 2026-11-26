"""
Gender-stratified Perfect Consistency (PC) analysis.

For each method × backbone, computes PC separately for male (M) and female (F)
patients across all context_ids that have all 4 perturbation types
(baseline, gender_swap, uncertain_tone, summary).

Plot layout:
  rows = 3 backbones  (Llama-8B, Llama-70B, Qwen-235B)
  cols = 4 tasks      (Manage, Visit, Resource, Joint)
  bars = 4 methods    (Baseline, MRA, TUA, MPA)
  y    = PC_male − PC_female (pp); positive → model is more consistent for males

Outputs:
  analysis/gender_pc_results.csv   — per method × backbone × task × gender
  plots/pc/pc_gender_gap.png       — gender gap bar chart grid
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────────

OUTPUT_BASE    = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/output")
MODEL_RESP_CSV = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/model_responses.csv")
PLOT_DIR       = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/plots/pc")
OUT_CSV        = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/analysis/gender_pc_results.csv")

DATASETS = {"oncqa", "askdocs"}
TASKS    = ["manage", "visit", "resource"]
PERTS    = ["baseline", "gender_swap", "uncertain_tone", "summary"]

BACKBONE_ORDER = ["Llama-8B", "Llama-70B", "Qwen-235B"]
METHOD_ORDER   = ["Baseline", "MRA", "TUA", "MPA"]
TASK_LABELS    = {"manage": "Manage", "visit": "Visit",
                  "resource": "Resource", "joint": "Joint"}

METHOD_COLORS = {
    "Baseline": "#6c757d",
    "MRA":      "#4e9af1",
    "TUA":      "#f4a020",
    "MPA":      "#2ca02c",
}

FONT = 11

def _r(*parts): return OUTPUT_BASE.joinpath(*parts)

# ── data sources: (method_short, backbone_label, csv_path, pred_prefix) ───────
# pred_prefix: the string before _{task} in prediction columns.
# For model_responses.csv, use the seeded prefix (seed0 = run 0).

def _rp(subpath: str, run: str) -> Path:
    return OUTPUT_BASE / subpath.format(run=run)


# RUN_SOURCES: (method, backbone) → list of (csv_path, pred_prefix) per run.
# model_responses.csv seeds 0/1/2 map to runs 0/1/2 for Llama-70B baseline.
# MedPathAgent only has run_0 — listed once; gets no std error bar in the plot.
RUN_SOURCES: dict[tuple[str, str], list[tuple[Path, str]]] = {
    ("Baseline", "Llama-70B"): [
        (MODEL_RESP_CSV, "llama_seed0"),
        (MODEL_RESP_CSV, "llama_seed1"),
        (MODEL_RESP_CSV, "llama_seed2"),
    ],
    ("Baseline", "Llama-8B"): [
        (_rp("baseline/llama8b3_1/oncqa_askdocs/{run}/results.csv", "run_0"), "llama8b3_1"),
        (_rp("baseline/llama8b3_1/oncqa_askdocs/{run}/results.csv", "run_1"), "llama8b3_1"),
        (_rp("baseline/llama8b3_1/oncqa_askdocs/{run}/results.csv", "run_2"), "llama8b3_1"),
    ],
    ("Baseline", "Qwen-235B"): [
        (_rp("baseline/qwen235b/oncqa_askdocs/{run}/results.csv", "run_0"), "qwen235b"),
        (_rp("baseline/qwen235b/oncqa_askdocs/{run}/results.csv", "run_1"), "qwen235b"),
        (_rp("baseline/qwen235b/oncqa_askdocs/{run}/results.csv", "run_2"), "qwen235b"),
    ],
    ("MRA", "Llama-70B"): [
        (_rp("medreason/llama70b3_3/oncqa_askdocs/{run}/results.csv", "run_0"), "medreason_llama70b3_3"),
        (_rp("medreason/llama70b3_3/oncqa_askdocs/{run}/results.csv", "run_1"), "medreason_llama70b3_3"),
        (_rp("medreason/llama70b3_3/oncqa_askdocs/{run}/results.csv", "run_2"), "medreason_llama70b3_3"),
    ],
    ("MRA", "Llama-8B"): [
        (_rp("medreason/llama8b3_1/oncqa_askdocs/{run}/results.csv", "run_0"), "medreason_llama8b3_1"),
        (_rp("medreason/llama8b3_1/oncqa_askdocs/{run}/results.csv", "run_1"), "medreason_llama8b3_1"),
        (_rp("medreason/llama8b3_1/oncqa_askdocs/{run}/results.csv", "run_2"), "medreason_llama8b3_1"),
    ],
    ("MRA", "Qwen-235B"): [
        (_rp("medreason/qwen235b/oncqa_askdocs/{run}/results.csv", "run_0"), "medreason_qwen235b"),
        (_rp("medreason/qwen235b/oncqa_askdocs/{run}/results.csv", "run_1"), "medreason_qwen235b"),
        (_rp("medreason/qwen235b/oncqa_askdocs/{run}/results.csv", "run_2"), "medreason_qwen235b"),
    ],
    ("TUA", "Llama-70B"): [
        (_rp("tooluni/llama70b3_3/oncqa_askdocs/{run}/results.csv", "run_0"), "tooluni_llama70b3_3"),
        (_rp("tooluni/llama70b3_3/oncqa_askdocs/{run}/results.csv", "run_1"), "tooluni_llama70b3_3"),
        (_rp("tooluni/llama70b3_3/oncqa_askdocs/{run}/results.csv", "run_2"), "tooluni_llama70b3_3"),
    ],
    ("TUA", "Llama-8B"): [
        (_rp("tooluni/llama8b3_1/oncqa_askdocs/{run}/results.csv", "run_0"), "tooluni_llama8b3_1"),
        (_rp("tooluni/llama8b3_1/oncqa_askdocs/{run}/results.csv", "run_1"), "tooluni_llama8b3_1"),
        (_rp("tooluni/llama8b3_1/oncqa_askdocs/{run}/results.csv", "run_2"), "tooluni_llama8b3_1"),
    ],
    ("TUA", "Qwen-235B"): [
        (_rp("tooluni/qwen235b/oncqa_askdocs/{run}/results.csv", "run_0"), "tooluni_qwen235b"),
        (_rp("tooluni/qwen235b/oncqa_askdocs/{run}/results.csv", "run_1"), "tooluni_qwen235b"),
        (_rp("tooluni/qwen235b/oncqa_askdocs/{run}/results.csv", "run_2"), "tooluni_qwen235b"),
    ],
    # MPA: run_0 only
    ("MPA", "Llama-70B"):  [(_rp("medpathagent/llama70b3_3/oncqa_askdocs/{run}/results.csv", "run_0"), "medpathagent_llama70b3_3")],
    ("MPA", "Llama-8B"):   [(_rp("medpathagent/llama8b3_1/oncqa_askdocs/{run}/results.csv",  "run_0"), "medpathagent_llama8b3_1")],
    ("MPA", "Qwen-235B"):  [(_rp("medpathagent/qwen235b/oncqa_askdocs/{run}/results.csv",    "run_0"), "medpathagent_qwen235b")],
}

# ── PC computation ─────────────────────────────────────────────────────────────

def compute_pc(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    For each context_id that has all 4 perturbations, compute PC per task
    and PC_joint. Returns one row per context_id with gender and boolean PC cols.
    """
    df = df.copy()
    df["context_id"] = df["context_id"].astype(str)

    # Gender is stable across perturbations; read from baseline rows
    gender_map = (
        df[df["perturbation"] == "baseline"]
        .drop_duplicates("context_id")
        .set_index("context_id")["original_gender"]
        .to_dict()
    )

    records = []
    for cid, grp in df.groupby("context_id"):
        perts_present = set(grp["perturbation"].values)
        if not all(p in perts_present for p in PERTS):
            continue  # skip context_ids missing any perturbation

        gender = str(gender_map.get(str(cid), "Unknown")).strip()

        task_consistent: dict[str, bool | None] = {}
        for task in TASKS:
            col = f"{prefix}_{task}"
            if col not in df.columns:
                task_consistent[task] = None
                continue
            preds = []
            for pert in PERTS:
                rows = grp[grp["perturbation"] == pert]
                if rows.empty:
                    break
                preds.append(str(rows.iloc[0][col]).strip().upper())
            task_consistent[task] = (len(preds) == len(PERTS)) and (len(set(preds)) == 1)

        joint = all(v is True for v in task_consistent.values())
        records.append({
            "context_id": str(cid),
            "gender":     gender,
            **{f"pc_{t}": task_consistent[t] for t in TASKS},
            "pc_joint":   joint,
        })

    return pd.DataFrame(records)


def gender_gap(pc_df: pd.DataFrame) -> dict[str, float | None]:
    """
    Returns {task: PC_male − PC_female} for manage/visit/resource/joint.
    None if either gender has <5 cases.
    """
    gaps = {}
    for task in TASKS + ["joint"]:
        col = f"pc_{task}"
        m = pc_df[pc_df["gender"] == "M"][col]
        f = pc_df[pc_df["gender"] == "F"][col]
        if len(m) < 5 or len(f) < 5:
            gaps[task] = None
        else:
            gaps[task] = (m.mean() - f.mean()) * 100  # percentage points
    return gaps


# ── main loop: aggregate gaps across runs ─────────────────────────────────────

PLOT_DIR.mkdir(parents=True, exist_ok=True)
all_rows: list[dict] = []   # per-run PC values
gap_rows: list[dict] = []   # mean ± std gap per (method, backbone)

print("\n=== Loading runs ===")
for (method, backbone), run_list in RUN_SOURCES.items():
    run_gaps: list[dict[str, float | None]] = []  # one dict per run

    for run_idx, (csv_path, prefix) in enumerate(run_list):
        if not Path(csv_path).exists():
            print(f"  MISSING {method}/{backbone} run_{run_idx}: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if "dataset" in df.columns:
            df = df[df["dataset"].isin(DATASETS)]
        df = df[df["perturbation"].isin(PERTS)]
        if len(df) < 50:
            continue
        pc_df = compute_pc(df, prefix)
        if pc_df.empty:
            continue

        # Per-run per-gender PC values → CSV
        for gender in ["M", "F", "All"]:
            sub = pc_df if gender == "All" else pc_df[pc_df["gender"] == gender]
            for task in TASKS + ["joint"]:
                col = f"pc_{task}"
                pc_val = sub[col].mean() * 100 if not sub.empty and col in sub.columns else None
                all_rows.append({
                    "method": method, "backbone": backbone,
                    "run": run_idx, "task": task, "gender": gender,
                    "pc": round(pc_val, 2) if pc_val is not None else None,
                    "n": len(sub),
                })

        run_gaps.append(gender_gap(pc_df))

    if not run_gaps:
        continue

    # Aggregate across runs
    n_runs = len(run_gaps)
    row: dict = {"method": method, "backbone": backbone, "n_runs": n_runs}
    summary_parts = []
    for task in TASKS + ["joint"]:
        vals = [g[task] for g in run_gaps if g[task] is not None]
        if not vals:
            row[f"gap_{task}_mean"] = None
            row[f"gap_{task}_std"]  = None
        else:
            mean_v = float(np.mean(vals))
            std_v  = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            row[f"gap_{task}_mean"] = round(mean_v, 2)
            row[f"gap_{task}_std"]  = round(std_v,  2) if std_v is not None else None
            summary_parts.append(f"{task}={mean_v:+.1f}±{std_v:.1f}" if std_v is not None
                                  else f"{task}={mean_v:+.1f}")
    gap_rows.append(row)

    # Print summary
    n_m = run_gaps[0] and None  # just for display — get from first run's pc_df
    print(f"  {method:<10} {backbone:<12}  n_runs={n_runs}  "
          + "  ".join(summary_parts))

df_results = pd.DataFrame(all_rows)
df_results.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV}")

gap_df = pd.DataFrame(gap_rows)

# ── plot 1: full grid (3 backbones × 4 tasks) ─────────────────────────────────

tasks_to_plot = TASKS + ["joint"]
n_rows = len(BACKBONE_ORDER)
n_cols = len(tasks_to_plot)
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(3.5 * n_cols, 3.8 * n_rows),
                          sharey=False)

x = np.arange(len(METHOD_ORDER))
bar_width = 0.55

for ri, backbone in enumerate(BACKBONE_ORDER):
    bb_gap = gap_df[gap_df["backbone"] == backbone]
    for ci, task in enumerate(tasks_to_plot):
        ax = axes[ri][ci]
        col = f"gap_{task}"

        vals, errs = [], []
        for method in METHOD_ORDER:
            row = bb_gap[bb_gap["method"] == method]
            mean_col = f"gap_{task}_mean"
            std_col  = f"gap_{task}_std"
            if row.empty or mean_col not in row.columns or pd.isna(row.iloc[0][mean_col]):
                vals.append(0.0); errs.append(None)
            else:
                vals.append(float(row.iloc[0][mean_col]))
                s = row.iloc[0][std_col]
                errs.append(float(s) if pd.notna(s) else None)

        err_arr = [e if e is not None else 0 for e in errs]
        bars = ax.bar(x, vals, bar_width,
                      color=[METHOD_COLORS[m] for m in METHOD_ORDER],
                      edgecolor="white", linewidth=0.8, alpha=0.88,
                      yerr=err_arr, capsize=4,
                      error_kw={"elinewidth": 1.2, "ecolor": "black", "capthick": 1.2})

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, fontsize=FONT - 2, rotation=30, ha="right")
        if ci == 0:
            ax.set_ylabel(f"{backbone}\nPC gap (M−F, pp)", fontsize=FONT - 1)
        if ri == 0:
            ax.set_title(TASK_LABELS[task], fontsize=FONT, fontweight="bold")
        ax.yaxis.grid(True, alpha=0.35, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=FONT - 2)
        ylim = ax.get_ylim()
        ax.set_ylim(min(ylim[0], -1), max(ylim[1], 1))

handles = [mpatches.Patch(facecolor=METHOD_COLORS[m], label=m, alpha=0.88)
           for m in METHOD_ORDER]
fig.legend(handles=handles, fontsize=FONT - 1, loc="lower center",
           ncol=len(METHOD_ORDER), bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
plt.suptitle("Gender Gap in Perfect Consistency (PC_male − PC_female)\n"
             "Positive = more consistent for male patients",
             fontsize=FONT + 1, fontweight="bold", y=1.01)
plt.tight_layout()
out_plot = PLOT_DIR / "pc_gender_gap.png"
plt.savefig(out_plot, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_plot}")

# ── plot 2: joint-only — 1 row × 3 backbones ──────────────────────────────────
# No title, font=25, no bar value labels.

FONT_J = 25
fig, axes = plt.subplots(1, len(BACKBONE_ORDER),
                          figsize=(6 * len(BACKBONE_ORDER), 7),
                          sharey=True)

x = np.arange(len(METHOD_ORDER))
bar_width = 0.6

for ci, backbone in enumerate(BACKBONE_ORDER):
    ax = axes[ci]
    bb_gap = gap_df[gap_df["backbone"] == backbone]
    col = "gap_joint"

    vals, errs = [], []
    for method in METHOD_ORDER:
        row = bb_gap[bb_gap["method"] == method]
        mean_col = "gap_joint_mean"
        std_col  = "gap_joint_std"
        if row.empty or mean_col not in row.columns or pd.isna(row.iloc[0][mean_col]):
            vals.append(0.0); errs.append(None)
        else:
            vals.append(float(row.iloc[0][mean_col]))
            s = row.iloc[0][std_col]
            errs.append(float(s) if pd.notna(s) else None)

    err_arr = [e if e is not None else 0 for e in errs]
    ax.bar(x, vals, bar_width,
           color=[METHOD_COLORS[m] for m in METHOD_ORDER],
           edgecolor="white", linewidth=0.8, alpha=0.88,
           yerr=err_arr, capsize=6,
           error_kw={"elinewidth": 1.8, "ecolor": "black", "capthick": 1.8})

    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([], fontsize=FONT_J)
    ax.set_title(backbone, fontsize=FONT_J, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=FONT_J)
    ylim = ax.get_ylim()
    ax.set_ylim(min(ylim[0], -1), max(ylim[1], 1))
    if ci == 0:
        ax.set_ylabel("PC gap (M−F, pp)", fontsize=FONT_J)

handles = [mpatches.Patch(facecolor=METHOD_COLORS[m], label=m, alpha=0.88)
           for m in METHOD_ORDER]
fig.legend(handles=handles, fontsize=FONT_J, loc="lower center",
           ncol=len(METHOD_ORDER), bbox_to_anchor=(0.5, -0.08), framealpha=0.9)

plt.tight_layout()
out_joint = PLOT_DIR / "pc_gender_gap_joint.png"
plt.savefig(out_joint, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_joint}")
