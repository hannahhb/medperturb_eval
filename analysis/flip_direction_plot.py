"""
Flip Direction Decomposition — stacked bar figure.

Per (method × backbone × perturbation), averaged across tasks and runs:
  - YES → NO or other  (baseline=YES, perturbed≠YES) — care-reducing  [red]
  - NO  → YES or other (baseline=NO,  perturbed≠NO)  — care-augmenting [blue]
  - Stable             (baseline == perturbed)                          [light grey]

Layout: 3 rows (backbone) × 3 columns (perturbation), 4 bars per panel (methods).
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
OUTPUT_BASE         = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/output")
MODEL_RESPONSES_CSV = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/model_responses.csv")
OUT_FIG             = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/analysis/flip_direction_decomp.png")

DATASETS  = {"oncqa", "askdocs"}
TASKS     = ["manage", "visit", "resource"]
PERTS     = ["summary", "gender_swap", "uncertain_tone"]
PERT_LABELS = {
    "summary":       "Summary",
    "gender_swap":   "Gender Swap",
    "uncertain_tone":"Uncertain Tone",
}
RUN_RE    = re.compile(r"^run_\d+$")
MIN_ROWS  = 600

# sentinel for Baseline Llama-70B (loaded from model_responses.csv)
_MR = "MODEL_RESPONSES"

METHOD_BACKBONE_DIRS = {
    "Baseline": {
        "Llama-8B":  OUTPUT_BASE / "baseline" / "llama8b3_1" / "oncqa_askdocs",
        "Llama-70B": _MR,
        "Qwen-235B": OUTPUT_BASE / "baseline" / "qwen235b"   / "oncqa_askdocs",
    },
    "MedReasonAgent": {
        "Llama-8B":  OUTPUT_BASE / "medreason" / "llama8b3_1"  / "oncqa_askdocs",
        "Llama-70B": OUTPUT_BASE / "medreason" / "llama70b3_3" / "oncqa_askdocs",
        "Qwen-235B": OUTPUT_BASE / "medreason" / "qwen235b"    / "oncqa_askdocs",
    },
    "ToolUniAgent": {
        "Llama-8B":  OUTPUT_BASE / "tooluni" / "llama8b3_1"  / "oncqa_askdocs",
        "Llama-70B": OUTPUT_BASE / "tooluni" / "llama70b3_3" / "oncqa_askdocs",
        "Qwen-235B": OUTPUT_BASE / "tooluni" / "qwen235b"    / "oncqa_askdocs",
    },
    "MedPathAgent": {
        "Llama-8B":  OUTPUT_BASE / "medpathagent" / "llama8b3_1"  / "oncqa_askdocs",
        "Llama-70B": OUTPUT_BASE / "medpathagent" / "llama70b3_3" / "oncqa_askdocs",
        "Qwen-235B": OUTPUT_BASE / "medpathagent" / "qwen235b"    / "oncqa_askdocs",
    },
}

METHODS   = ["Baseline", "MedReasonAgent", "ToolUniAgent", "MedPathAgent"]
BACKBONES = ["Llama-8B", "Llama-70B", "Qwen-235B"]
METHOD_SHORT = {
    "Baseline":       "Base",
    "MedReasonAgent": "MRA",
    "ToolUniAgent":   "TUA",
    "MedPathAgent":   "MPA",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def get_pred_prefix(df: pd.DataFrame) -> str | None:
    candidates = [
        c for c in df.columns
        if "manage" in c
        and "gold" not in c
        and "reasoning" not in c
        and "standard" not in c
        and "consensus" not in c
    ]
    return candidates[0].replace("_manage", "") if candidates else None


def load_runs(backbone_dir) -> list[tuple[pd.DataFrame, str]]:
    if backbone_dir is _MR or backbone_dir == _MR:
        df = pd.read_csv(MODEL_RESPONSES_CSV)
        df = df[df["dataset"].isin(DATASETS)]
        df = df[df["perturbation"].isin(PERTS + ["baseline"])]
        runs = []
        for seed in range(3):
            prefix = f"llama_seed{seed}"
            if all(f"{prefix}_{t}" in df.columns for t in TASKS):
                runs.append((df, prefix))
        return runs

    p = Path(backbone_dir)
    if not p.exists():
        return []
    runs = []
    for run_dir in sorted(p.iterdir()):
        if not run_dir.is_dir() or not RUN_RE.match(run_dir.name):
            continue
        csv = run_dir / "results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if "dataset" in df.columns:
            df = df[df["dataset"].isin(DATASETS)]
        if len(df) < MIN_ROWS:
            continue
        prefix = get_pred_prefix(df)
        if prefix is None:
            continue
        runs.append((df, prefix))
    return runs


def direction_fracs(df: pd.DataFrame, prefix: str, task: str, pert: str) -> dict | None:
    """
    Returns fractions (%) of:
      stable          — baseline pred == perturbed pred
      yes_to_other    — baseline YES, perturbed ≠ YES  (care-reducing)
      no_to_other     — baseline NO,  perturbed ≠ NO   (care-augmenting)
    Rows where baseline is UNKNOWN are excluded from total (handled separately
    in the 'stable' count — UNKNOWN→UNKNOWN is still stable).
    """
    col = f"{prefix}_{task}"
    if col not in df.columns:
        return None
    base_df = df[df["perturbation"] == "baseline"][["context_id", col]].rename(columns={col: "base"})
    pert_df = df[df["perturbation"] == pert   ][["context_id", col]].rename(columns={col: "pert"})
    m = base_df.merge(pert_df, on="context_id").dropna(subset=["base", "pert"])
    if m.empty:
        return None
    total = len(m)
    stable       = int((m["base"] == m["pert"]).sum())
    yes_to_other = int(((m["base"] == "YES") & (m["pert"] != "YES")).sum())
    no_to_other  = int(((m["base"] == "NO")  & (m["pert"] != "NO" )).sum())
    return {
        "stable":        stable       / total * 100,
        "yes_to_other":  yes_to_other / total * 100,
        "no_to_other":   no_to_other  / total * 100,
    }


# ── collect ───────────────────────────────────────────────────────────────────
# data[method][backbone][pert] = {"stable", "yes_to_other", "no_to_other"}
data: dict = {m: {b: {} for b in BACKBONES} for m in METHODS}

for method, bb_map in METHOD_BACKBONE_DIRS.items():
    for backbone, backbone_dir in bb_map.items():
        runs = load_runs(backbone_dir)
        if not runs:
            print(f"  SKIP {method}/{backbone}: no runs")
            continue
        for pert in PERTS:
            accum = {"stable": [], "yes_to_other": [], "no_to_other": []}
            for df, prefix in runs:
                for task in TASKS:
                    r = direction_fracs(df, prefix, task, pert)
                    if r:
                        for k in accum:
                            accum[k].append(r[k])
            if accum["stable"]:
                data[method][backbone][pert] = {k: float(np.mean(v)) for k, v in accum.items()}


# ── plot ─────────────────────────────────────────────────────────────────────
C_STABLE   = "#d0d0d0"   # light grey
C_YES_NO   = "#c0392b"   # red  — YES→NO or other
C_NO_YES   = "#2980b9"   # blue — NO→YES or other

fig, axes = plt.subplots(
    nrows=3, ncols=3,
    figsize=(13, 10),
    sharey=True,
    constrained_layout=True,
)

bar_width = 0.6
x_pos     = np.arange(len(METHODS))

for row_i, backbone in enumerate(BACKBONES):
    for col_i, pert in enumerate(PERTS):
        ax = axes[row_i][col_i]

        stable_vals   = []
        yes_no_vals   = []
        no_yes_vals   = []
        missing       = []

        for method in METHODS:
            d = data[method][backbone].get(pert)
            if d is None:
                stable_vals.append(np.nan)
                yes_no_vals.append(np.nan)
                no_yes_vals.append(np.nan)
                missing.append(True)
            else:
                stable_vals.append(d["stable"])
                yes_no_vals.append(d["yes_to_other"])
                no_yes_vals.append(d["no_to_other"])
                missing.append(False)

        # stacked bars: stable (bottom) → YES→NO (middle) → NO→YES (top)
        bars_s = ax.bar(x_pos, stable_vals, bar_width, color=C_STABLE,  label="Stable")
        bars_y = ax.bar(x_pos, yes_no_vals, bar_width, bottom=stable_vals,
                        color=C_YES_NO, label="YES→NO or other")
        bottom2 = [s + y for s, y in zip(stable_vals, yes_no_vals)]
        bars_n = ax.bar(x_pos, no_yes_vals, bar_width, bottom=bottom2,
                        color=C_NO_YES, label="NO→YES or other")

        # mark missing panels
        for xi, miss in enumerate(missing):
            if miss:
                ax.text(xi, 50, "N/A", ha="center", va="center",
                        fontsize=9, color="grey", style="italic")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([METHOD_SHORT[m] for m in METHODS], fontsize=10)
        ax.set_ylim(0, 100)
        ax.yaxis.set_tick_params(labelsize=9)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)

        # column header (perturbation) on top row only
        if row_i == 0:
            ax.set_title(PERT_LABELS[pert], fontsize=12, fontweight="bold", pad=8)

        # row label (backbone) on leftmost column only
        if col_i == 0:
            ax.set_ylabel(f"{backbone}\n% vignettes", fontsize=10)

# shared legend
legend_patches = [
    mpatches.Patch(color=C_STABLE,  label="Stable"),
    mpatches.Patch(color=C_YES_NO,  label="YES→NO or other  (care-reducing)"),
    mpatches.Patch(color=C_NO_YES,  label="NO→YES or other  (care-augmenting)"),
]
fig.legend(
    handles=legend_patches,
    loc="lower center",
    ncol=3,
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(0.5, -0.03),
)

fig.suptitle(
    "Flip Direction Decomposition by Backbone and Perturbation Type\n"
    "(averaged across triage tasks and runs)",
    fontsize=13, fontweight="bold",
)

plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_FIG}")
plt.show()
