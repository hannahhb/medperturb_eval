"""
MedPerturb Robustness Analysis
================================
Consolidated plots:
  Fig 1 — Flip Rate          (% predictions changed baseline → perturbed)
  Fig 2 — RCER               (Reduced Care Error Rate)
  Fig 3 — Mutual Information (prediction stability)
  Fig 4 — Accuracy Drop      (Δ accuracy perturbed − baseline)

All plots: colour = backbone, hatch = method, 1 subplot per task (3 rows × 1 col).
No significance tests.
"""
from __future__ import annotations

import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from pathlib import Path
from sklearn.metrics import mutual_info_score

# ── paths ─────────────────────────────────────────────────────────────────────

OUTPUT_BASE    = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/output")
MEDPERTURB_CSV = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/medperturb_data.csv")
OUT_DIR        = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/plots/robustness")

DATASETS   = {"oncqa", "askdocs"}
TASKS      = ["manage", "visit", "resource"]
PERTS      = ["summary", "gender_swap", "uncertain_tone"]
PERT_LABELS = {"summary": "Sum.", "gender_swap": "GS", "uncertain_tone": "UT"}
TASK_LABELS = {"manage": "Manage", "visit": "Visit", "resource": "Resource"}
MIN_ROWS    = 600

# Baseline llama70b/qwen235b columns live in medperturb_data.csv
MEDPERTURB_BASELINE_COLS = {"llama70b3_3": "llama", "qwen235b": "qwen"}

METHOD_CSV_PATHS = {
    "Baseline": {
        "llama70b3_3": None,   # from medperturb_data.csv
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

NAME_MAP = {"llama70b3_3": "Llama 70B", "llama8b3_1": "Llama 8B", "qwen235b": "Qwen 235B"}

# ── visual style ──────────────────────────────────────────────────────────────

FONT = 25

BACKBONE_COLORS = {
    "Llama 70B": "#4C72B0",
    "Qwen 235B": "#DD8452",
    "Llama 8B":  "#55A868",
}
BACKBONE_MARKERS = {
    "Llama 70B": "o",
    "Qwen 235B": "^",
    "Llama 8B":  "*",
}
METHOD_HATCH = {
    "Baseline":     "",
    "MedReason":    "",
    "ToolUni":      "",
    "MedPathAgent": "",
}
METHOD_ORDER   = ["Baseline", "MedReason", "ToolUni", "MedPathAgent"]
BACKBONE_ORDER = ["Llama 70B", "Qwen 235B", "Llama 8B"]

plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    FONT + 1,
    "axes.labelsize":    FONT,
    "xtick.labelsize":   FONT,
    "ytick.labelsize":   FONT,
    "legend.fontsize":   FONT - 1,
    "legend.title_fontsize": FONT,
})

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── data loading ──────────────────────────────────────────────────────────────

def _pred_col(df, task):
    candidates = [c for c in df.columns
                  if task in c and "gold" not in c and "reasoning" not in c
                  and "standard" not in c and "consensus" not in c]
    return candidates[0] if candidates else None


def load_all() -> dict[str, tuple[pd.DataFrame, str]]:
    """
    Returns {label: (df, pred_prefix)} for every method × backbone.
    label = "Method\nBackbone display name"
    """
    mp = pd.read_csv(MEDPERTURB_CSV)
    mp = mp[mp["dataset"].isin(DATASETS)]

    result = {}
    for method, backbone_paths in METHOD_CSV_PATHS.items():
        for backbone, path in backbone_paths.items():
            display = NAME_MAP.get(backbone, backbone)
            label   = f"{method}\n{display}"

            # Baseline llama70b/qwen from medperturb_data.csv
            if path is None and method == "Baseline" and backbone in MEDPERTURB_BASELINE_COLS:
                prefix = MEDPERTURB_BASELINE_COLS[backbone]
                if f"{prefix}_manage" not in mp.columns:
                    print(f"  MISSING col {prefix}_manage in medperturb_data.csv")
                    continue
                result[label] = (mp.copy(), prefix)
                print(f"  {label}: medperturb_data.csv  prefix={prefix}  rows={len(mp)}")
                continue

            if path is None or not Path(path).exists():
                print(f"  MISSING: {label}: {path}")
                continue

            df = pd.read_csv(path)
            if "dataset" in df.columns:
                df = df[df["dataset"].isin(DATASETS)]
            if len(df) < MIN_ROWS:
                print(f"  skip {label} ({len(df)} rows)")
                continue
            col = _pred_col(df, "manage")
            if col is None:
                print(f"  WARNING: no pred col in {path}")
                continue
            prefix = col.replace("_manage", "")
            print(f"  {label}: prefix={prefix}  rows={len(df)}")
            result[label] = (df, prefix)

    return result


def _binarise(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return s.str.strip().str.upper().map({"YES": 1, "NO": 0}).astype("float")
    return pd.to_numeric(s, errors="coerce")


# ── metric helpers ────────────────────────────────────────────────────────────

def flip_rate(df, prefix, task, pert) -> float | None:
    col  = f"{prefix}_{task}"
    base = df[df["perturbation"] == "baseline"][["context_id", col]].rename(columns={col: "b"})
    p    = df[df["perturbation"] == pert][["context_id", col]].rename(columns={col: "q"})
    m    = base.merge(p, on="context_id")
    if m.empty:
        return None
    return (m["b"] != m["q"]).mean() * 100


def _load_clinician_lookup() -> dict[str, dict[str, str]]:
    """
    Returns {context_id: {task: "YES"|"NO"}} from medperturb_data.csv baseline rows.
    Uses clinician_consensus_{task} columns (0.0=NO, 1.0=YES).
    Only rows where at least one task has a non-null clinician label are included.
    """
    mp = pd.read_csv(MEDPERTURB_CSV)
    mp = mp[(mp["perturbation"] == "baseline") & mp["dataset"].isin(DATASETS)]
    clin_cols = {t: f"clinician_consensus_{t}" for t in TASKS}
    available = {t: c for t, c in clin_cols.items() if c in mp.columns}
    if not available:
        return {}
    lookup = {}
    for _, row in mp.iterrows():
        cid = str(row["context_id"])
        entry = {}
        for t, col in available.items():
            v = row.get(col)
            if pd.notna(v):
                entry[t] = "YES" if float(v) == 1.0 else "NO"
        if entry:
            lookup[cid] = entry
    return lookup


def _care_pair(df, prefix, task, pert):
    """Merge baseline and perturbed rows on context_id, including gold label."""
    col      = f"{prefix}_{task}"
    gold_col = f"gold_standard_{task}"
    has_gold = gold_col in df.columns
    base_cols = ["context_id", col] + ([gold_col] if has_gold else [])
    base = df[df["perturbation"] == "baseline"][base_cols].rename(
        columns={col: "base_ans", gold_col: "gold"} if has_gold else {col: "base_ans"})
    p = df[df["perturbation"] == pert][["context_id", col]].rename(columns={col: "pert_ans"})
    return base.merge(p, on="context_id"), has_gold


def _is_care_aug(series: pd.Series, task: str) -> pd.Series:
    """Binary care-augmenting indicator: VISIT/RESOURCE YES=1, MANAGE NO=1."""
    if task == "manage":
        return (series == "NO").astype(float)
    else:
        return (series == "YES").astype(float)


def tsr(df, prefix, task, pert) -> float | None:
    """
    Treatment Shift Rate (Eq 1):
    TSR = mean(T_b - T_p)  where T=1 means care-augmenting.
    Positive = net shift toward care-reducing after perturbation.
    """
    m, _ = _care_pair(df, prefix, task, pert)
    if m.empty:
        return None
    T_b = _is_care_aug(m["base_ans"], task)
    T_p = _is_care_aug(m["pert_ans"], task)
    valid = ~(T_b.isna() | T_p.isna())
    if valid.sum() == 0:
        return None
    return float((T_b[valid] - T_p[valid]).mean() * 100)


def rcr(df, prefix, task, pert) -> float | None:
    """
    Reduced Care Rate (Eq 2):
    RCR = sum(1[T_b=c+] * (T_b - T_p)) / sum(1[T_b=c+])
        = (base=c+ AND pert=c-) / (base=c+)
    Denominator: cases where baseline was care-augmenting.
    """
    m, _ = _care_pair(df, prefix, task, pert)
    if m.empty:
        return None
    T_b = _is_care_aug(m["base_ans"], task)
    T_p = _is_care_aug(m["pert_ans"], task)
    care_aug_base = T_b == 1
    if care_aug_base.sum() == 0:
        return None
    reduced = ((T_b == 1) & (T_p == 0)).sum()
    return float(reduced / care_aug_base.sum() * 100)


def rcer(df, prefix, task, pert,
         clinician_lookup: dict | None = None) -> float | None:
    """
    Reduced Care Error Rate (Eq 3):
    RCER = (gold=c+ AND pert=c-) / (gold=c+)
    Uses gold_standard label only.
    clinician_lookup parameter retained for API compatibility but ignored.
    """
    m, has_gold = _care_pair(df, prefix, task, pert)
    if m.empty or not has_gold or "gold" not in m.columns:
        return None

    m_use = m[m["gold"].notna()].copy()
    if m_use.empty:
        return None
    V   = _is_care_aug(m_use["gold"], task)
    T_p = _is_care_aug(m_use["pert_ans"], task)
    care_aug_v = V == 1
    if care_aug_v.sum() == 0:
        return None
    errors = ((V == 1) & (T_p == 0)).sum()
    return float(errors / care_aug_v.sum() * 100)


def mutual_info(df, prefix, task, pert) -> float | None:
    col  = f"{prefix}_{task}"
    base = df[df["perturbation"] == "baseline"][["context_id", col]].rename(columns={col: "b"})
    p    = df[df["perturbation"] == pert][["context_id", col]].rename(columns={col: "q"})
    m    = base.merge(p, on="context_id")
    b    = _binarise(m["b"]).values
    q    = _binarise(m["q"]).values
    mask = ~(np.isnan(b) | np.isnan(q))
    if mask.sum() < 5:
        return None
    return float(mutual_info_score(b[mask].astype(int), q[mask].astype(int)))


def delta_accuracy(df, prefix, task, pert) -> float | None:
    col      = f"{prefix}_{task}"
    gold_col = f"gold_standard_{task}"
    if gold_col not in df.columns:
        return None
    base = df[df["perturbation"] == "baseline"][["context_id", col, gold_col]]
    base = base.rename(columns={col: "pred_b", gold_col: "gold_b"})
    p    = df[df["perturbation"] == pert][["context_id", col, gold_col]]
    p    = p.rename(columns={col: "pred_p", gold_col: "gold_p"})
    m    = base.merge(p, on="context_id")
    if m.empty:
        return None
    def acc(pred, gold):
        pred = _binarise(pred); gold = _binarise(gold)
        mask = ~(np.isnan(pred) | np.isnan(gold))
        return float((pred[mask] == gold[mask]).mean()) if mask.sum() else np.nan
    return acc(m["pred_p"], m["gold_p"]) - acc(m["pred_b"], m["gold_b"])


# ── plot style ────────────────────────────────────────────────────────────────

# Per-method solid colour used when avg_backbones=True (backbone info shown via scatter)
METHOD_COLORS = {
    "Baseline":     "#bbbbbb",
    "MedReason":    "#4C72B0",
    "ToolUni":      "#DD8452",
    "MedPathAgent": "#55A868",
}
METHOD_SHORT = {
    "Baseline":     "Baseline",
    "MedReason":    "MRA",
    "ToolUni":      "TUA",
    "MedPathAgent": "MPA",
}


# ── generic plot builder ──────────────────────────────────────────────────────

def make_figure(
    data: dict,
    title: str,
    ylabel: str,
    fname: str,
    pct_fmt: bool = False,
    zero_line: bool = False,
    avg_backbones: bool = False,
    avg_perts: bool = False,
    show_shapes: bool = False,
) -> None:
    """
    data[label] = {(pert, task): value},  label = "Method\\nBackbone"

    avg_backbones=False, avg_perts=False  (default)
        x-axis = perturbations, bars = method × backbone (colour=backbone, hatch=method)

    avg_backbones=True, avg_perts=False
        x-axis = perturbations, bars = method (colour=method),
        scatter points on each bar = individual backbone values (colour=backbone)

    avg_backbones=False, avg_perts=True
        x-axis = single "Average" group, bars = method × backbone

    avg_backbones=True, avg_perts=True
        x-axis = single "Average" group, bars = method (colour=method),
        scatter points = individual backbone values
    """
    existing   = set(data.keys())
    x_groups   = ["Average"] if avg_perts else PERTS
    x_labels   = ["Average"] if avg_perts else [PERT_LABELS[p] for p in PERTS]
    x          = np.arange(len(x_groups))

    def _val(label, task, groups):
        """Get value(s) for a label/task averaged over groups if needed."""
        vals = [data.get(label, {}).get((g, task)) for g in groups]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    AGG_SUFFIX = "\nAll Backbones"

    if avg_backbones:
        bar_items = [m for m in METHOD_ORDER if f"{m}{AGG_SUFFIX}" in existing]
        n_bars    = len(bar_items)
        width     = 0.7 / n_bars

        fig, axes = plt.subplots(1, len(TASKS),
                                 figsize=(5 * len(TASKS) + 2 * len(x_groups), 5),
                                 sharey=True)
        if len(TASKS) == 1:
            axes = [axes]

        for ai, (ax, task) in enumerate(zip(axes, TASKS)):
            scatter_queue = []
            for i, method in enumerate(bar_items):
                color     = METHOD_COLORS.get(method, "#888888")
                hatch     = METHOD_HATCH.get(method, "")
                alpha     = 0.35 if method == "Baseline" else 0.85
                offset    = (i - n_bars / 2 + 0.5) * width
                agg_label = f"{method}{AGG_SUFFIX}"

                bar_vals = []
                for xi, grp in enumerate(x_groups):
                    pert_list = PERTS if avg_perts else [grp]
                    bar_vals.append(_val(agg_label, task, pert_list) or 0.0)
                    if show_shapes:
                        for b in BACKBONE_ORDER:
                            bb_label = f"{method}\n{b}"
                            if bb_label not in existing:
                                continue
                            bv = _val(bb_label, task, pert_list)
                            if bv is not None:
                                scatter_queue.append((xi + offset, bv,
                                                      BACKBONE_MARKERS.get(b, "o")))

                ax.bar(x + offset, bar_vals, width,
                       facecolor=color,
                       edgecolor="grey" if method == "Baseline" else "white",
                       alpha=alpha, hatch=hatch, linewidth=1.2, zorder=3)

            if show_shapes:
                for sx, sy, marker in scatter_queue:
                    ax.scatter(sx, sy, color="none", marker=marker,
                               s=70, zorder=6, linewidths=1.8, edgecolors="black")

            if zero_line:
                ax.axhline(0, color="black", linewidth=0.8, zorder=0)
            ax.set_title(TASK_LABELS[task], fontsize=FONT + 1, fontweight="bold", pad=8)
            if ai == 0:
                ax.set_ylabel(ylabel, fontsize=FONT + 1)
            if pct_fmt:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.tick_params(axis="y", labelsize=FONT)
            ax.grid(axis="y", alpha=0.3)

            if avg_perts:
                # avg_bb_pert: method names directly on x-axis, no method legend
                bar_positions   = [(i - n_bars / 2 + 0.5) * width for i in range(n_bars)]
                bar_tick_labels = [METHOD_SHORT.get(m, m) for m in bar_items]
                ax.set_xticks(bar_positions)
                ax.set_xticklabels(bar_tick_labels, fontsize=FONT, rotation=0, ha="center")
            else:
                # avg_bb: perturbation group labels on x-axis, method legend below
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, fontsize=FONT)

        backbone_handles = [
            plt.Line2D([0], [0],
                       marker=BACKBONE_MARKERS.get(b, "o"), color="w",
                       markerfacecolor="black", markeredgecolor="black",
                       markersize=10, label=b)
            for b in BACKBONE_ORDER if any(b in lbl for lbl in existing)
        ]
        # Set shared y-axis: bottom=0 (or auto for zero_line), top=auto from data
        axes[0].set_ylim(bottom=None if zero_line else 0)
        if zero_line:
            lo, hi = axes[0].get_ylim()
            axes[0].set_ylim(top=max(hi, abs(lo) * 0.28))

        method_handles = [
            mpatches.Patch(facecolor=METHOD_COLORS.get(m, "#888"),
                           alpha=0.35 if m == "Baseline" else 0.85,
                           edgecolor="grey" if m == "Baseline" else "white",
                           label=METHOD_SHORT.get(m, m))
            for m in bar_items
        ]
        leg_loc    = "lower right" if zero_line else "upper right"
        leg_anchor = (0.99, 0.08)  if zero_line else (0.99, 0.92)
        fig.legend(handles=method_handles, fontsize=FONT - 1,
                   loc=leg_loc, bbox_to_anchor=leg_anchor,
                   ncol=2, framealpha=0.9,
                   handlelength=1.0, handleheight=0.8, handletextpad=0.4,
                   columnspacing=0.8)

    else:
        all_models = [f"{m}\n{b}" for b in BACKBONE_ORDER for m in METHOD_ORDER
                      if f"{m}\n{b}" in existing]
        n_bars = len(all_models)
        width  = 0.8 / n_bars

        fig, axes = plt.subplots(1, len(TASKS),
                                 figsize=(5 * len(TASKS) + 2 * len(x_groups), 5),
                                 sharey=True)
        if len(TASKS) == 1:
            axes = [axes]

        for ai, (ax, task) in enumerate(zip(axes, TASKS)):
            for i, model in enumerate(all_models):
                method, backbone = model.split("\n")
                color    = BACKBONE_COLORS.get(backbone, "#888888")
                hatch    = METHOD_HATCH.get(method, "")
                if method == "Baseline":
                    facecolor, edgecol, alpha = color, color, 0.3
                else:
                    facecolor, edgecol, alpha = color, "white", 0.85
                offset = (i - n_bars / 2 + 0.5) * width

                bar_vals = [_val(model, task, PERTS if avg_perts else [grp]) or 0.0
                            for grp in x_groups]
                ax.bar(x + offset, bar_vals, width,
                       facecolor=facecolor, edgecolor=edgecol, alpha=alpha,
                       hatch=hatch, linewidth=1.2)

            if zero_line:
                ax.axhline(0, color="black", linewidth=0.8, zorder=0)
            ax.set_title(TASK_LABELS[task], fontsize=FONT + 1, fontweight="bold", pad=8)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=FONT)
            if ai == 0:
                ax.set_ylabel(ylabel, fontsize=FONT + 1)
            if pct_fmt:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.tick_params(axis="both", labelsize=FONT)
            ax.grid(axis="y", alpha=0.3)

        axes[0].set_ylim(bottom=None if zero_line else 0)
        if zero_line:
            lo, hi = axes[0].get_ylim()
            axes[0].set_ylim(top=max(hi, abs(lo) * 0.28))

        leg_loc    = "lower right" if zero_line else "upper right"
        leg_anchor = (0.99, 0.08)  if zero_line else (0.99, 0.92)

        backbone_handles = [
            mpatches.Patch(facecolor=c, edgecolor=c, label=b)
            for b, c in BACKBONE_COLORS.items() if any(b in m for m in existing)
        ]
        method_handles = [
            mpatches.Patch(facecolor="grey", edgecolor="grey", alpha=0.3,
                           label=METHOD_SHORT.get(m, m))
            if m == "Baseline" else
            mpatches.Patch(facecolor="grey", hatch=METHOD_HATCH[m], edgecolor="white",
                           alpha=0.85, label=METHOD_SHORT.get(m, m))
            for m in METHOD_ORDER if any(m in lbl for lbl in existing)
        ]
        fig.legend(handles=method_handles, fontsize=FONT - 1,
                   loc=leg_loc, bbox_to_anchor=leg_anchor,
                   ncol=2, framealpha=0.9,
                   handlelength=1.0, handleheight=0.8, handletextpad=0.4,
                   columnspacing=0.8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    path = OUT_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── aggregate backbone helper ─────────────────────────────────────────────────

AGG_PREFIX = "agg"   # shared column prefix after renaming

def _build_agg_df(method: str, models: dict) -> pd.DataFrame | None:
    """
    Concatenate all backbone dfs for a given method, renaming each backbone's
    pred columns {prefix}_{task} → agg_{task} so metrics can run on the pool.
    context_ids are kept as-is; rows from different backbones stack freely.
    """
    parts = []
    for backbone in BACKBONE_ORDER:
        label = f"{method}\n{backbone}"
        if label not in models:
            continue
        df, prefix = models[label]
        rename = {f"{prefix}_{t}": f"{AGG_PREFIX}_{t}" for t in TASKS}
        # keep shared columns + renamed pred cols
        keep = ["context_id", "perturbation"] + \
               [c for c in df.columns if c in
                [f"gold_standard_{t}" for t in TASKS] +
                list(rename.keys())]
        sub = df[keep].rename(columns=rename).copy()
        # Make context_id unique per backbone so merges don't cross backbones
        sub["context_id"] = sub["context_id"].astype(str) + f"__{backbone}"
        parts.append(sub)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    models = load_all()
    print(f"  {len(models)} model entries loaded")

    print("Loading clinician consensus labels...")
    clin_lookup = _load_clinician_lookup()
    print(f"  {len(clin_lookup)} context_ids with clinician labels\n")

    # Build data dicts for each metric
    flip_data, tsr_data, rcr_data, rcer_data, mi_data, acc_data = {}, {}, {}, {}, {}, {}

    # Per backbone×method
    for label, (df, prefix) in models.items():
        flip_data[label] = {}
        tsr_data[label]  = {}
        rcr_data[label]  = {}
        rcer_data[label] = {}
        mi_data[label]   = {}
        acc_data[label]  = {}
        for p in PERTS:
            for t in TASKS:
                flip_data[label][(p, t)] = flip_rate(df, prefix, t, p)
                tsr_data[label][(p, t)]  = tsr(df, prefix, t, p)
                rcr_data[label][(p, t)]  = rcr(df, prefix, t, p)
                rcer_data[label][(p, t)] = rcer(df, prefix, t, p, clin_lookup)
                mi_data[label][(p, t)]   = mutual_info(df, prefix, t, p)
                acc_data[label][(p, t)]  = delta_accuracy(df, prefix, t, p)

    # Aggregated across backbones (pool rows, then compute metric)
    AGG_LABEL_SUFFIX = "\nAll Backbones"
    for method in METHOD_ORDER:
        agg_df = _build_agg_df(method, models)
        if agg_df is None:
            continue
        label = f"{method}{AGG_LABEL_SUFFIX}"
        flip_data[label] = {}
        tsr_data[label]  = {}
        rcr_data[label]  = {}
        rcer_data[label] = {}
        mi_data[label]   = {}
        acc_data[label]  = {}
        for p in PERTS:
            for t in TASKS:
                flip_data[label][(p, t)] = flip_rate(agg_df, AGG_PREFIX, t, p)
                tsr_data[label][(p, t)]  = tsr(agg_df,  AGG_PREFIX, t, p)
                rcr_data[label][(p, t)]  = rcr(agg_df,  AGG_PREFIX, t, p)
                rcer_data[label][(p, t)] = rcer(agg_df, AGG_PREFIX, t, p, clin_lookup)
                mi_data[label][(p, t)]   = mutual_info(agg_df, AGG_PREFIX, t, p)
                acc_data[label][(p, t)]  = delta_accuracy(agg_df, AGG_PREFIX, t, p)

    METRICS = [
        dict(data=flip_data, title="Flip Rate",                               ylabel="Flip Rate (%)", slug="flip",     pct_fmt=True,  zero_line=False),
        dict(data=tsr_data,  title="Treatment Shift Rate (TSR)",              ylabel="TSR (%)",       slug="tsr",      pct_fmt=True,  zero_line=True),
        dict(data=rcr_data,  title="Reduced Care Rate (RCR)",                 ylabel="RCR (%)",       slug="rcr",      pct_fmt=True,  zero_line=False),
        dict(data=rcer_data, title="Reduced Care Error Rate (RCER)",          ylabel="RCER (%)",      slug="rcer",     pct_fmt=True,  zero_line=False),
        dict(data=mi_data,   title="Mutual Information (Prediction Stability)",ylabel="MI (bits)",    slug="mi",       pct_fmt=False, zero_line=False),
        dict(data=acc_data,  title="Accuracy Drop (Δ Accuracy)",              ylabel="Δ Accuracy",   slug="acc_drop", pct_fmt=False, zero_line=True),
    ]

    # Four variants per metric:
    #   base          — all backbones shown separately, x = perturbations
    #   avg_bb        — averaged across backbones (backbone values as scatter points), x = perturbations
    #   avg_pert      — all backbones, x = single average across perturbations
    #   avg_bb_pert   — averaged across both backbones and perturbations
    VARIANTS = [
        dict(suffix="",            avg_backbones=False, avg_perts=False),
        dict(suffix="_avg_bb",     avg_backbones=True,  avg_perts=False),
        dict(suffix="_avg_pert",   avg_backbones=False, avg_perts=True),
        dict(suffix="_avg_bb_pert",avg_backbones=True,  avg_perts=True),
    ]

    for m in METRICS:
        for v in VARIANTS:
            # Clean version (no shapes)
            make_figure(
                data          = m["data"],
                title         = m["title"],
                ylabel        = m["ylabel"],
                fname         = f"{m['slug']}{v['suffix']}.png",
                pct_fmt       = m["pct_fmt"],
                zero_line     = m["zero_line"],
                avg_backbones = v["avg_backbones"],
                avg_perts     = v["avg_perts"],
                show_shapes   = False,
            )
            # Shapes version (appendix) — only for avg_backbones variants
            if v["avg_backbones"]:
                make_figure(
                    data          = m["data"],
                    title         = m["title"],
                    ylabel        = m["ylabel"],
                    fname         = f"{m['slug']}{v['suffix']}_shapes.png",
                    pct_fmt       = m["pct_fmt"],
                    zero_line     = m["zero_line"],
                    avg_backbones = v["avg_backbones"],
                    avg_perts     = v["avg_perts"],
                    show_shapes   = True,
                )

    make_kg_split_figure(models, fname="medreason_flip_kg_split.png")

    # ── save all metric results to CSV ────────────────────────────────────────
    csv_rows = []
    all_data = {
        "flip":     flip_data,
        "tsr":      tsr_data,
        "rcr":      rcr_data,
        "rcer":     rcer_data,
        "mi":       mi_data,
        "acc_drop": acc_data,
    }
    for metric, data in all_data.items():
        for label, pt_vals in data.items():
            method_backbone = label.split("\n")
            method  = method_backbone[0]
            backbone = method_backbone[1] if len(method_backbone) > 1 else "All Backbones"
            for (pert, task), val in pt_vals.items():
                csv_rows.append({
                    "metric":   metric,
                    "method":   METHOD_SHORT.get(method, method),
                    "backbone": backbone,
                    "perturbation": pert,
                    "task":     task,
                    "value":    round(val, 4) if val is not None else None,
                })

    csv_df = pd.DataFrame(csv_rows)
    csv_path = OUT_DIR / "robustness_results.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}  ({len(csv_df)} rows)")

    print(f"\nAll figures saved to {OUT_DIR}/")


# ── KG-path split flip rate ───────────────────────────────────────────────────

def make_kg_split_figure(models: dict, fname: str) -> None:
    """
    Flip rate split by KG-path availability, aggregated (pooled) across all
    MedReason backbone models.

    Two bars per perturbation group:
      Blue  = vignettes where MedReason found KG paths
      Orange = vignettes where no KG paths were found
    """
    COLOR_BASE  = "#aaaaaa"
    COLOR_KG    = "#4C72B0"
    COLOR_NO_KG = "#DD8452"

    # Pool all MedReason backbone dfs
    parts = []
    for backbone in BACKBONE_ORDER:
        label = f"MedReason\n{backbone}"
        if label not in models:
            continue
        df, prefix = models[label]
        rename = {f"{prefix}_{t}": f"{AGG_PREFIX}_{t}" for t in TASKS}
        keep   = ["context_id", "perturbation"] + \
                 [c for c in df.columns if c in list(rename) or c == "kg_paths"]
        sub    = df[[c for c in keep if c in df.columns]].rename(columns=rename).copy()
        parts.append(sub)

    if not parts:
        print("  KG split: no MedReason data found, skipping")
        return

    agg = pd.concat(parts, ignore_index=True)

    # Pool all Baseline backbone dfs
    base_parts = []
    for backbone in BACKBONE_ORDER:
        label = f"Baseline\n{backbone}"
        if label not in models:
            continue
        df, prefix = models[label]
        rename = {f"{prefix}_{t}": f"{AGG_PREFIX}_{t}" for t in TASKS}
        keep   = ["context_id", "perturbation"] + \
                 [c for c in df.columns if c in list(rename)]
        sub    = df[[c for c in keep if c in df.columns]].rename(columns=rename).copy()
        base_parts.append(sub)
    base_agg = pd.concat(base_parts, ignore_index=True) if base_parts else None

    # KG availability per context_id from MedReason baseline rows
    bl     = agg[agg["perturbation"] == "baseline"][["context_id", "kg_paths"]].copy()
    has_kg = (
        bl["kg_paths"].notna() &
        (bl["kg_paths"].astype(str).str.strip() != "") &
        (bl["kg_paths"].astype(str).str.strip() != "nan")
    )
    kg_map = bl.assign(has_kg=has_kg).groupby("context_id")["has_kg"].any()
    agg["has_kg"] = agg["context_id"].map(kg_map)
    if base_agg is not None:
        base_agg["has_kg"] = base_agg["context_id"].map(kg_map)

    def _flip(subset, task, pert):
        col = f"{AGG_PREFIX}_{task}"
        if col not in subset.columns:
            return None
        b = subset[subset["perturbation"] == "baseline"][["context_id", col]].rename(columns={col: "b"})
        p = subset[subset["perturbation"] == pert][["context_id", col]].rename(columns={col: "q"})
        m = b.merge(p, on="context_id")
        return (m["b"] != m["q"]).mean() * 100 if not m.empty else None

    # 4 bars per group, paired by KG availability:
    #   [Base KG | MedR KG]  gap  [Base no-KG | MedR no-KG]
    # Use a small gap between the two pairs by offsetting manually.
    pair_gap = 0.08
    width    = 0.18
    # offsets relative to the perturbation centre:
    #   pair 1 (KG):    -pair_gap/2 - width,  -pair_gap/2
    #   pair 2 (no KG): +pair_gap/2,           +pair_gap/2 + width
    off_base_kg    = -pair_gap / 2 - width * 1.5
    off_mr_kg      = -pair_gap / 2 - width * 0.5
    off_base_no_kg = +pair_gap / 2 + width * 0.5
    off_mr_no_kg   = +pair_gap / 2 + width * 1.5
    x = np.arange(len(PERTS))

    fig, axes = plt.subplots(1, len(TASKS),
                             figsize=(6 * len(TASKS), 5),
                             sharey=True)

    for ai, (ax, task) in enumerate(zip(axes, TASKS)):
        mr_kg    = agg[agg["has_kg"] == True]
        mr_no_kg = agg[agg["has_kg"] == False]
        bl_kg    = base_agg[base_agg["has_kg"] == True]  if base_agg is not None else None
        bl_no_kg = base_agg[base_agg["has_kg"] == False] if base_agg is not None else None

        for pi, pert in enumerate(PERTS):
            def _bar(subset, off, color, label):
                if subset is None:
                    return
                r = _flip(subset, task, pert)
                if r is not None:
                    ax.bar(pi + off, r, width, color=color, alpha=0.85,
                           zorder=3, label=label if pi == 0 else "")

            _bar(bl_kg,    off_base_kg,    COLOR_BASE,  "Baseline (KG found rows)")
            _bar(mr_kg,    off_mr_kg,      COLOR_KG,    "MedReason (KG found)")
            _bar(bl_no_kg, off_base_no_kg, "#cccccc",   "Baseline (no KG rows)")
            _bar(mr_no_kg, off_mr_no_kg,   COLOR_NO_KG, "MedReason (no KG)")

        ax.set_title(TASK_LABELS[task], fontsize=FONT + 1, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([PERT_LABELS[p] for p in PERTS],
                           fontsize=FONT, rotation=20, ha="right")
        if ai == 0:
            ax.set_ylabel("Flip Rate (%)", fontsize=FONT + 1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.tick_params(axis="y", labelsize=FONT)
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_BASE,  alpha=0.85, label="Baseline (KG found rows)"),
        mpatches.Patch(facecolor=COLOR_KG,    alpha=0.85, label="MedReason (KG found)"),
        mpatches.Patch(facecolor="#cccccc",   alpha=0.85, label="Baseline (no KG rows)"),
        mpatches.Patch(facecolor=COLOR_NO_KG, alpha=0.85, label="MedReason (no KG)"),
    ]
    fig.legend(handles=legend_handles, fontsize=FONT - 1,
               loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncol=4, framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    path = OUT_DIR / fname
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
