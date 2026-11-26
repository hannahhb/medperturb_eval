"""
Flip Direction Decomposition — diverging bar figure, one plot per backbone.

Positive side (above zero, care-reducing):
  YES→NO    [red]    stacked below
  YES→other [purple] stacked on top

Negative side (below zero, care-augmenting):
  NO→YES    [blue]   stacked above (closest to zero)
  NO→other  [purple] stacked further below

"other" = any prediction that is neither YES nor NO (e.g. UNKNOWN / parse failure).

Style mirrors medperturb_robustness.py.
Averaged across runs where multiple runs exist.
Outputs three figures: one per backbone (Llama-8B, Llama-70B, Qwen-235B).
"""

from __future__ import annotations

import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# ── paths ─────────────────────────────────────────────────────────────────────
OUTPUT_BASE         = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/output")
MODEL_RESPONSES_CSV = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/model_responses.csv")
OUT_DIR             = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/analysis")

DATASETS    = {"oncqa", "askdocs"}
TASKS       = ["manage", "visit", "resource"]
TASK_LABELS = {"manage": "Manage", "visit": "Visit", "resource": "Resource"}
PERTS       = ["summary", "gender_swap", "uncertain_tone"]
PERT_LABELS = {"summary": "Sum.", "gender_swap": "GS", "uncertain_tone": "UT"}
RUN_RE      = re.compile(r"^run_\d+$")
MIN_ROWS    = 600

# ── visual style ──────────────────────────────────────────────────────────────
FONT = 14

METHOD_ORDER = ["Baseline", "MedReason", "ToolUni", "MedPathAgent"]
METHOD_SHORT = {"Baseline": "Base", "MedReason": "MRA",
                "ToolUni": "TUA",  "MedPathAgent": "MPA"}

# Colour = direction/category; hatch = method
C_YES_NO  = "#c0392b"   # red    — YES → NO
C_YES_OTH = "#8e44ad"   # purple — YES → other
C_NO_YES  = "#2980b9"   # blue   — NO  → YES
C_NO_OTH  = "#8e44ad"   # purple — NO  → other  (same hue as C_YES_OTH)

METHOD_HATCH = {
    "Baseline":     "",
    "MedReason":    "//",
    "ToolUni":      "\\\\",
    "MedPathAgent": "xx",
}

# Care-reducing direction is task-dependent. For Visit/Resource, YES = recommend
# care, so departing YES (YES→NO/other) reduces care → plotted UP. For Manage,
# YES = "patient can self-manage" (i.e. LESS care), so departing NO (NO→YES/other)
# is the care-reducing direction. Manage polarity is therefore inverted.
TASK_INVERTED = {"manage": True, "visit": False, "resource": False}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    FONT + 1,
    "axes.labelsize":    FONT,
    "xtick.labelsize":   FONT,
    "ytick.labelsize":   FONT,
    "legend.fontsize":   FONT - 1,
})

# ── data loading ──────────────────────────────────────────────────────────────
_MR = "MODEL_RESPONSES"   # sentinel for Baseline Llama-70B

METHOD_BACKBONE_DIRS = {
    "Baseline": {
        "Llama-8B":  OUTPUT_BASE / "baseline/llama8b3_1/oncqa_askdocs",
        "Llama-70B": _MR,
        "Qwen-235B": OUTPUT_BASE / "baseline/qwen235b/oncqa_askdocs",
    },
    "MedReason": {
        "Llama-8B":  OUTPUT_BASE / "medreason/llama8b3_1/oncqa_askdocs",
        "Llama-70B": OUTPUT_BASE / "medreason/llama70b3_3/oncqa_askdocs",
        "Qwen-235B": OUTPUT_BASE / "medreason/qwen235b/oncqa_askdocs",
    },
    "ToolUni": {
        "Llama-8B":  OUTPUT_BASE / "tooluni/llama8b3_1/oncqa_askdocs",
        "Llama-70B": OUTPUT_BASE / "tooluni/llama70b3_3/oncqa_askdocs",
        "Qwen-235B": OUTPUT_BASE / "tooluni/qwen235b/oncqa_askdocs",
    },
    "MedPathAgent": {
        "Llama-8B":  OUTPUT_BASE / "medpathagent/llama8b3_1/oncqa_askdocs",
        "Llama-70B": OUTPUT_BASE / "medpathagent/llama70b3_3/oncqa_askdocs",
        "Qwen-235B": OUTPUT_BASE / "medpathagent/qwen235b/oncqa_askdocs",
    },
}


def _pred_prefix(df: pd.DataFrame) -> str | None:
    candidates = [
        c for c in df.columns
        if "manage" in c and "gold" not in c
        and "reasoning" not in c and "standard" not in c and "consensus" not in c
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
                runs.append((df.copy(), prefix))
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
        prefix = _pred_prefix(df)
        if prefix is None:
            continue
        runs.append((df, prefix))
    return runs


def direction_fracs(df: pd.DataFrame, prefix: str, task: str, pert: str) -> dict | None:
    """
    Decomposes flips into four categories (as % of all matched vignettes):
      yes_to_no  — baseline YES, perturbed NO            (care-reducing)
      yes_to_oth — baseline YES, perturbed ≠ YES and ≠ NO  (other/unknown)
      no_to_yes  — baseline NO,  perturbed YES            (care-augmenting)
      no_to_oth  — baseline NO,  perturbed ≠ NO  and ≠ YES (other/unknown)
    """
    col = f"{prefix}_{task}"
    if col not in df.columns:
        return None
    base_df = (df[df["perturbation"] == "baseline"]
               [["context_id", col]].rename(columns={col: "base"}))
    pert_df = (df[df["perturbation"] == pert]
               [["context_id", col]].rename(columns={col: "pert"}))
    m = base_df.merge(pert_df, on="context_id").dropna(subset=["base", "pert"])
    if m.empty:
        return None
    total = len(m)
    yes_to_no  = ((m["base"] == "YES") & (m["pert"] == "NO" )).sum() / total * 100
    yes_to_oth = ((m["base"] == "YES") & (m["pert"] != "YES") & (m["pert"] != "NO")).sum() / total * 100
    no_to_yes  = ((m["base"] == "NO")  & (m["pert"] == "YES")).sum() / total * 100
    no_to_oth  = ((m["base"] == "NO")  & (m["pert"] != "NO" ) & (m["pert"] != "YES")).sum() / total * 100
    return {
        "yes_to_no":  float(yes_to_no),
        "yes_to_oth": float(yes_to_oth),
        "no_to_yes":  float(no_to_yes),
        "no_to_oth":  float(no_to_oth),
    }


# ── collect ───────────────────────────────────────────────────────────────────
# data[backbone][method][pert][task] = {yes_to_no, yes_to_oth, no_to_yes, no_to_oth}

BACKBONES = ["Llama-8B", "Llama-70B", "Qwen-235B"]
KEYS = ["yes_to_no", "yes_to_oth", "no_to_yes", "no_to_oth"]
data: dict = {bb: {m: {} for m in METHOD_ORDER} for bb in BACKBONES}

for method, bb_map in METHOD_BACKBONE_DIRS.items():
    for backbone, backbone_dir in bb_map.items():
        runs = load_runs(backbone_dir)
        if not runs:
            print(f"  SKIP {method}/{backbone}: no runs")
            continue
        for pert in PERTS:
            for task in TASKS:
                vals: dict[str, list] = {k: [] for k in KEYS}
                for df, prefix in runs:
                    r = direction_fracs(df, prefix, task, pert)
                    if r:
                        for k in KEYS:
                            vals[k].append(r[k])
                if vals["yes_to_no"]:
                    data[backbone][method].setdefault(pert, {})[task] = {
                        k: float(np.mean(v)) for k, v in vals.items()
                    }

# ── plot ─────────────────────────────────────────────────────────────────────
n_methods = len(METHOD_ORDER)
bar_width  = 0.18
x          = np.arange(len(PERTS))


def _get(backbone, method, task, key):
    """Return per-perturbation list of values for (backbone, method, task, key)."""
    out = []
    for pert in PERTS:
        d = data[backbone][method].get(pert, {}).get(task)
        out.append(d[key] if d else 0.0)
    return out


for backbone in BACKBONES:
    fig, axes = plt.subplots(1, len(TASKS), figsize=(6 * len(TASKS), 5.5), sharey=True)

    for ai, (ax, task) in enumerate(zip(axes, TASKS)):
        ax.axhline(0, color="black", linewidth=0.9, zorder=2)

        for i, method in enumerate(METHOD_ORDER):
            hatch  = METHOD_HATCH[method]
            offset = (i - n_methods / 2 + 0.5) * bar_width
            xs     = x + offset
            kw     = dict(width=bar_width, zorder=3, edgecolor="#333333", linewidth=0.4)

            yn   = np.array(_get(backbone, method, task, "yes_to_no"))
            yoth = np.array(_get(backbone, method, task, "yes_to_oth"))
            ny   = np.array(_get(backbone, method, task, "no_to_yes"))
            noth = np.array(_get(backbone, method, task, "no_to_oth"))

            # Colour stays tied to the transition (red=YES→NO, blue=NO→YES,
            # purple=→other); the UP side is always the care-reducing direction.
            if TASK_INVERTED[task]:
                # Manage: care-reducing (up) = NO→YES; care-augmenting (down) = YES→NO
                up_main,  up_main_c  = ny,  C_NO_YES
                up_oth,   up_oth_c   = noth, C_NO_OTH
                dn_main,  dn_main_c  = yn,  C_YES_NO
                dn_oth,   dn_oth_c   = yoth, C_YES_OTH
            else:
                # Visit/Resource: care-reducing (up) = YES→NO; augmenting (down) = NO→YES
                up_main,  up_main_c  = yn,  C_YES_NO
                up_oth,   up_oth_c   = yoth, C_YES_OTH
                dn_main,  dn_main_c  = ny,  C_NO_YES
                dn_oth,   dn_oth_c   = noth, C_NO_OTH

            # positive side (above 0): clean care-reducing flip then →other
            ax.bar(xs,  up_main, color=up_main_c, hatch=hatch, alpha=0.85, **kw)
            ax.bar(xs,  up_oth,  color=up_oth_c,  hatch=hatch, alpha=0.85, bottom=up_main, **kw)

            # negative side (below 0): clean care-augmenting flip then →other
            ax.bar(xs, -dn_main, color=dn_main_c, hatch=hatch, alpha=0.85, **kw)
            ax.bar(xs, -dn_oth,  color=dn_oth_c,  hatch=hatch, alpha=0.85, bottom=-dn_main, **kw)

        ax.set_title(TASK_LABELS[task], fontsize=FONT + 1, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([PERT_LABELS[p] for p in PERTS], fontsize=FONT)
        ax.tick_params(axis="y", labelsize=FONT)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        if ai == 0:
            ax.set_ylabel("Flip rate (%)", fontsize=FONT)

        ylo, yhi = ax.get_ylim()
        pad = max(abs(ylo), abs(yhi)) * 0.06
        ax.text(-0.55, yhi - pad, "↑ care-reducing",
                va="top", ha="left", fontsize=FONT - 4, color="#555555")
        ax.text(-0.55, ylo + pad, "↓ care-augmenting",
                va="bottom", ha="left", fontsize=FONT - 4, color="#555555")

    # ── legend: direction colours + method hatches ────────────────────────────
    dir_handles = [
        mpatches.Patch(facecolor=C_YES_NO,  edgecolor="#333333", alpha=0.85, label="YES → NO"),
        mpatches.Patch(facecolor=C_YES_OTH, edgecolor="#333333", alpha=0.85, label="YES → other"),
        mpatches.Patch(facecolor=C_NO_YES,  edgecolor="#333333", alpha=0.85, label="NO → YES"),
        mpatches.Patch(facecolor=C_NO_OTH,  edgecolor="#333333", alpha=0.85, label="NO → other"),
    ]
    method_handles = [
        mpatches.Patch(facecolor="#cccccc", hatch=METHOD_HATCH[m], edgecolor="#333333",
                       label=METHOD_SHORT[m])
        for m in METHOD_ORDER
    ]

    fig.legend(
        handles=dir_handles + method_handles,
        loc="lower center",
        ncol=len(dir_handles) + len(method_handles),
        fontsize=FONT - 2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.text(
        0.5, -0.135,
        "Above 0 = care-reducing, below 0 = care-augmenting. "
        "Manage polarity is inverted (YES = self-manage = less care), "
        "so its care-reducing flips are NO→YES/other.",
        ha="center", fontsize=FONT - 4, color="#555555",
    )

    fig.suptitle(
        f"Flip Direction Decomposition — {backbone}",
        fontsize=FONT + 2, fontweight="bold", y=1.01,
    )

    slug = backbone.lower().replace("-", "").replace(" ", "_")
    out  = OUT_DIR / f"flip_direction_diverging_{slug}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")
