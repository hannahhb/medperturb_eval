"""
MedXpertQA robustness and accuracy plots.

Produces 8 PNGs in plots/medxpertqa/:
  robustness_{aggregate, body_system, question_type, medical_task}.png
  accuracy_{aggregate, body_system, question_type, medical_task}.png

Robustness = accuracy(baseline) - accuracy(gender_swap)  [percentage points]
Significance: McNemar's on (baseline_correct, swap_correct) per model.
              Bars are hatched when p < 0.05.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT.parent / "data"
OUT_DIR   = ROOT / "output" / "medxpertqa"
PLOTS_DIR = Path(__file__).resolve().parent / "medxpertqa"
PLOTS_DIR.mkdir(exist_ok=True)

JSONL_PATH = DATA_DIR / "medxpertqa_text.jsonl"

MODELS = {
    "Llama-70B":  OUT_DIR / "aws_llama70b3_3"       / "results.csv",
    "MedReason":  OUT_DIR / "medreason_llama70b3_3"  / "results.csv",
    "AgentMD":    OUT_DIR / "agentmd_llama70b3_3"    / "results.csv",
    "ToolUni":    OUT_DIR / "tooluni_llama70b3_3"    / "results.csv",
}

COLORS = {
    "Llama-70B": "#4C72B0",
    "MedReason": "#DD8452",
    "AgentMD":   "#55A868",
    "ToolUni":   "#C44E52",
}
SIG_HATCH  = "///"
SIG_ALPHA  = 0.05

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ── data loading ──────────────────────────────────────────────────────────────

def load_meta() -> pd.DataFrame:
    rows = [json.loads(l) for l in JSONL_PATH.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)[["id", "medical_task", "body_system", "question_type"]].rename(
        columns={"question_type": "q_type"}   # avoid collision with results column
    )


def load_model(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"  WARNING: {name} results not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str)
    # Keep last occurrence of any duplicate (id, question_type) rows so
    # the baseline×swap merge in build_wide doesn't create a cross-product.
    n_before = len(df)
    df = df.drop_duplicates(subset=["id", "question_type"], keep="last")
    if len(df) < n_before:
        print(f"  {name}: dropped {n_before - len(df)} duplicate rows")
    df["correct"] = (df["model_answer_letter"].str.strip() == df["correct_label"].str.strip())
    df["model"] = name
    return df


def build_wide(dfs: dict[str, pd.DataFrame], meta: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one row per (id, model) with baseline_correct and swap_correct."""
    frames = []
    for name, df in dfs.items():
        if df.empty:
            continue
        base = df[df["question_type"] == "baseline"][["id", "correct"]].rename(columns={"correct": "base_correct"})
        swap = df[df["question_type"] == "gender_swap"][["id", "correct"]].rename(columns={"correct": "swap_correct"})
        wide = base.merge(swap, on="id", how="inner")
        wide["model"] = name
        frames.append(wide)
    wide_all = pd.concat(frames, ignore_index=True)
    wide_all = wide_all.merge(meta, on="id", how="left")
    return wide_all


# ── mutual information ────────────────────────────────────────────────────────

def _mi_bits(base_correct: pd.Series, swap_correct: pd.Series) -> float:
    """
    Mutual information (in bits) between baseline correctness and swap correctness,
    computed over matched question pairs.

    Variables:
      X = base_correct  (0 = wrong, 1 = right on baseline variant)
      Y = swap_correct  (0 = wrong, 1 = right on gender-swap variant)

    Each question contributes one (X, Y) pair → 2×2 joint distribution.

    MI = sum_{x,y} P(x,y) * log2( P(x,y) / (P(x) * P(y)) )

    Interpretation:
      MI = 0  → knowing baseline result tells you nothing about swap result
               (perturbation completely disrupts the model)
      MI high → baseline and swap correctness are strongly coupled
               (model behaves consistently across the perturbation)
    """
    n = len(base_correct)
    if n == 0:
        return float("nan")

    # 2×2 joint frequency table: rows = base_correct, cols = swap_correct
    joint = np.zeros((2, 2), dtype=float)
    b = base_correct.astype(int).values
    s = swap_correct.astype(int).values
    for bv, sv in zip(b, s):
        joint[bv, sv] += 1
    joint /= joint.sum()

    px = joint.sum(axis=1)   # marginal P(base_correct)
    py = joint.sum(axis=0)   # marginal P(swap_correct)

    mi = 0.0
    for x in range(2):
        for y in range(2):
            if joint[x, y] > 0:
                mi += joint[x, y] * np.log2(joint[x, y] / (px[x] * py[y]))
    return mi


def mi_stats(wide: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    """
    Compute MI (bits) per model × group.
    Returns DataFrame with columns: group, model, mi_bits, n.
    """
    records = []
    groups = [None] if group_col is None else wide[group_col].dropna().unique()
    for g in groups:
        subset = wide if g is None else wide[wide[group_col] == g]
        for model in wide["model"].unique():
            m = subset[subset["model"] == model]
            if len(m) == 0:
                continue
            mi = _mi_bits(m["base_correct"], m["swap_correct"])
            records.append({
                "group": str(g) if g is not None else "Aggregate",
                "model": model,
                "mi_bits": mi,
                "n": len(m),
            })
    return pd.DataFrame(records)


def plot_mi(stats: pd.DataFrame, groups: list[str], title: str, fname: str,
            horizontal: bool = False, figsize=(7, 5)):
    """
    Bar chart of mutual information (bits) between perturbation type and
    correctness, one bar cluster per group, one bar per model.
    """
    models = [m for m in COLORS if m in stats["model"].values]
    n_groups  = len(groups)
    n_models  = len(models)
    width     = min(0.18, 0.72 / max(n_models, 1))
    gap       = width * 0.15
    offsets   = np.linspace(-(n_models - 1) * (width + gap) / 2,
                             (n_models - 1) * (width + gap) / 2, n_models)
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=figsize)

    for i, model in enumerate(models):
        mdata = stats[stats["model"] == model].set_index("group")
        vals  = [mdata.loc[g, "mi_bits"] if g in mdata.index else np.nan for g in groups]
        color = COLORS[model]

        if horizontal:
            ax.barh(x + offsets[i], vals, height=width,
                    label=model, color=color, alpha=0.85, edgecolor="white")
        else:
            ax.bar(x + offsets[i], vals, width=width,
                   label=model, color=color, alpha=0.85, edgecolor="white")

    if horizontal:
        ax.set_yticks(x)
        ax.set_yticklabels(groups, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mutual Information (bits)", fontsize=9)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9, rotation=15, ha="right")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Mutual Information (bits)", fontsize=9)
        ax.set_ylim(bottom=0)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01,
             "MI(baseline correct; swap correct) — higher = more consistent across perturbation",
             fontsize=7, color="#666", va="bottom")

    fig.tight_layout()
    path = PLOTS_DIR / fname
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def flip_stats(wide: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    """
    Flip rate = % of questions where correctness changes between baseline and swap.
    A flip occurs when base_correct ≠ swap_correct (either direction).
    Returns DataFrame with columns: group, model, flip_rate, n.
    """
    records = []
    groups = [None] if group_col is None else wide[group_col].dropna().unique()
    for g in groups:
        subset = wide if g is None else wide[wide[group_col] == g]
        for model in wide["model"].unique():
            m = subset[subset["model"] == model]
            if len(m) == 0:
                continue
            flipped   = (m["base_correct"] != m["swap_correct"]).sum()
            flip_rate = flipped / len(m) * 100
            records.append({
                "group":     str(g) if g is not None else "Aggregate",
                "model":     model,
                "flip_rate": flip_rate,
                "n":         len(m),
            })
    return pd.DataFrame(records)


def plot_flip(stats: pd.DataFrame, groups: list[str], title: str, fname: str,
              horizontal: bool = False, figsize=(7, 5), show_n: bool = False):
    """
    Bar chart of flip rate (%) — proportion of questions where correctness
    changes between baseline and gender-swap, regardless of direction.
    """
    models = [m for m in COLORS if m in stats["model"].values]
    n_groups = len(groups)
    n_models = len(models)
    width    = min(0.18, 0.72 / max(n_models, 1))
    gap      = width * 0.15
    offsets  = np.linspace(-(n_models - 1) * (width + gap) / 2,
                            (n_models - 1) * (width + gap) / 2, n_models)
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=figsize)

    for i, model in enumerate(models):
        mdata = stats[stats["model"] == model].set_index("group")
        vals  = [mdata.loc[g, "flip_rate"] if g in mdata.index else np.nan for g in groups]
        ns    = [int(mdata.loc[g, "n"])    if g in mdata.index else 0      for g in groups]
        color = COLORS[model]

        if horizontal:
            bars = ax.barh(x + offsets[i], vals, height=width,
                           label=model, color=color, alpha=0.85, edgecolor="white")
            if show_n:
                for bar, val, n in zip(bars, vals, ns):
                    if not np.isnan(val):
                        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                                f"n={n}", va="center", ha="left", fontsize=6, color="#555")
        else:
            bars = ax.bar(x + offsets[i], vals, width=width,
                          label=model, color=color, alpha=0.85, edgecolor="white")
            if show_n:
                for bar, val, n in zip(bars, vals, ns):
                    if not np.isnan(val):
                        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                                f"n={n}", ha="center", va="bottom", fontsize=6, color="#555")

    if horizontal:
        ax.set_yticks(x)
        ax.set_yticklabels(groups, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Flip rate (%)", fontsize=9)
        ax.set_xlim(left=0)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Flip rate (%)", fontsize=9)
        ax.set_ylim(0, 100)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01,
             "Flip rate = % questions where correctness changes between baseline and gender-swap",
             fontsize=7, color="#666", va="bottom")

    fig.tight_layout()
    path = PLOTS_DIR / fname
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── statistics ────────────────────────────────────────────────────────────────

def mcnemar_p(base_correct: pd.Series, swap_correct: pd.Series) -> float:
    b = int(( base_correct & ~swap_correct).sum())
    c = int((~base_correct &  swap_correct).sum())
    if b + c == 0:
        return 1.0
    return binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue


def group_stats(wide: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      group, model, delta_pp, base_acc_pp, swap_acc_pp, p_value, n, sig
    """
    records = []
    groups = [None] if group_col is None else wide[group_col].dropna().unique()
    for g in groups:
        subset = wide if g is None else wide[wide[group_col] == g]
        for model in wide["model"].unique():
            m = subset[subset["model"] == model]
            if len(m) == 0:
                continue
            base_acc = m["base_correct"].mean() * 100
            swap_acc = m["swap_correct"].mean() * 100
            delta    = base_acc - swap_acc
            p        = mcnemar_p(m["base_correct"], m["swap_correct"])
            records.append({
                "group":       str(g) if g is not None else "Aggregate",
                "model":       model,
                "delta_pp":    delta,
                "base_acc_pp": base_acc,
                "swap_acc_pp": swap_acc,
                "p_value":     p,
                "n":           len(m),
                "sig":         p < SIG_ALPHA,
            })
    return pd.DataFrame(records)


# ── completeness summary ──────────────────────────────────────────────────────

def completeness_text(dfs: dict[str, pd.DataFrame]) -> str:
    parts = []
    for name, df in dfs.items():
        if df.empty:
            parts.append(f"{name}: NO DATA")
        else:
            b = (df["question_type"] == "baseline").sum()
            s = (df["question_type"] == "gender_swap").sum()
            parts.append(f"{name}: {b} baseline / {s} gender-swap")
    return "  |  ".join(parts)


# ── plotting helpers ──────────────────────────────────────────────────────────

def _bar_group(ax, stats: pd.DataFrame, models: list[str], x_labels: list[str],
               y_col: str, ylabel: str, title: str, horizontal: bool = False,
               show_n: bool = False):
    """Draw grouped bars for multiple models across groups."""
    # Only plot models that actually have data in this stats slice
    models = [m for m in models if m in stats["model"].values]
    n_groups = len(x_labels)
    n_models = len(models)
    # Keep bars narrow enough that 4 models never overlap
    width   = min(0.18, 0.72 / max(n_models, 1))
    gap     = width * 0.15           # small breathing room between bars
    offsets = np.linspace(-(n_models - 1) * (width + gap) / 2,
                           (n_models - 1) * (width + gap) / 2, n_models)
    x = np.arange(n_groups)

    for i, model in enumerate(models):
        mdata = stats[stats["model"] == model].set_index("group")
        vals  = [mdata.loc[g, y_col] if g in mdata.index else np.nan for g in x_labels]
        sigs  = [mdata.loc[g, "sig"]  if g in mdata.index else False  for g in x_labels]
        ns    = [int(mdata.loc[g, "n"]) if g in mdata.index else 0    for g in x_labels]
        color = COLORS[model]

        if horizontal:
            bars = ax.barh(x + offsets[i], vals, height=width,
                           label=model, color=color, alpha=0.85, edgecolor="white")
            for bar, sig, val, n in zip(bars, sigs, vals, ns):
                if sig:
                    bar.set_hatch(SIG_HATCH)
                if show_n and not np.isnan(val):
                    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                            f"n={n}", va="center", ha="left", fontsize=6, color="#555")
        else:
            bars = ax.bar(x + offsets[i], vals, width=width,
                          label=model, color=color, alpha=0.85, edgecolor="white")
            for bar, sig, val, n in zip(bars, sigs, vals, ns):
                if sig:
                    bar.set_hatch(SIG_HATCH)
                if show_n and not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            val + 0.3 if val >= 0 else val - 1.5,
                            f"n={n}", ha="center", va="bottom", fontsize=6, color="#555")

    if horizontal:
        ax.set_yticks(x)
        ax.set_yticklabels(x_labels, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(ylabel, fontsize=9)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9, rotation=15, ha="right")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=9)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)


def _add_sig_note(fig):
    fig.text(0.01, 0.01, f"Hatching = McNemar's p < {SIG_ALPHA}",
             fontsize=7, color="#666", va="bottom")


# ── per-graph save functions ──────────────────────────────────────────────────

def plot_robustness(stats: pd.DataFrame, groups: list[str], title: str,
                    fname: str, horizontal: bool = False, figsize=(7, 5),
                    show_n: bool = False):
    models = list(COLORS.keys())
    fig, ax = plt.subplots(figsize=figsize)
    _bar_group(ax, stats, models, groups,
               y_col="delta_pp",
               ylabel="Accuracy difference (pp): baseline − gender-swap",
               title=title, horizontal=horizontal, show_n=show_n)
    _add_sig_note(fig)
    fig.tight_layout()
    path = PLOTS_DIR / fname
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_accuracy(stats: pd.DataFrame, groups: list[str], title: str,
                  fname: str, horizontal: bool = False, figsize=(7, 5),
                  show_n: bool = False):
    """Two sub-panels: baseline accuracy and gender-swap accuracy."""
    models = list(COLORS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 1.6, figsize[1]),
                             sharey=not horizontal, sharex=horizontal)

    for ax, (qtype_col, label) in zip(axes, [
        ("base_acc_pp", "Baseline accuracy (%)"),
        ("swap_acc_pp", "Gender-swap accuracy (%)"),
    ]):
        _bar_group(ax, stats, models, groups,
                   y_col=qtype_col, ylabel=label,
                   title=f"{title}\n{label}", horizontal=horizontal, show_n=show_n)
        if not horizontal:
            ax.set_ylim(0, 105)

    _add_sig_note(fig)
    fig.tight_layout()
    path = PLOTS_DIR / fname
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    _DEFAULT_BODY_SYSTEMS = [
        "Integumetary", "Reproductive", "Urinary", "Muscular",  "Cardiovascular",  "Muscular", "Skeletal"
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--medical-task", nargs="+",
                        default=["Basic Science", "Treatment"],
                        help="Only include rows from these medical task categories "
                             "(e.g. --medical-task Diagnosis Treatment). "
                             "Pass --medical-task all to include all. "
                             "Default: Basic Science Treatment")
    parser.add_argument("--body-system", nargs="+", default=_DEFAULT_BODY_SYSTEMS,
                        metavar="SYSTEM",
                        help="Body systems to include in the AGGREGATE plots only. "
                             "Pass --body-system all to include all systems, or list "
                             "specific ones (e.g. --body-system Cardiovascular Endocrine). "
                             f"Default: {_DEFAULT_BODY_SYSTEMS}")
    args = parser.parse_args()

    print("Loading data…")
    meta = load_meta()
    dfs  = {name: load_model(name, path) for name, path in MODELS.items()}

    print("Data completeness:")
    print(" ", completeness_text(dfs))

    wide = build_wide(dfs, meta)
    # Deduplicate after merging to avoid cross-product inflation from duplicate
    # (id, question_type) rows in any model's results CSV.
    wide = wide.drop_duplicates(subset=["id", "model"])
    print(f"  Matched pairs (after dedup): {len(wide) // max(len(dfs), 1)} per model")

    if args.medical_task and args.medical_task != ["all"]:
        wide = wide[wide["medical_task"].isin(args.medical_task)]
        print(f"  After --medical-task filter {args.medical_task}: {len(wide)} rows")
        if wide.empty:
            print("  ERROR: no rows remain after filter — check task names.")
            return

    # Subset used only for aggregate plots (body-system filtered)
    if args.body_system and args.body_system != ["all"]:
        wide_agg = wide[wide["body_system"].isin(args.body_system)]
        agg_subtitle = "Body systems: " + ", ".join(args.body_system)
        print(f"  Aggregate filter to body systems {args.body_system}: "
              f"{len(wide_agg) // max(len(dfs), 1)} pairs per model")
        if wide_agg.empty:
            print("  WARNING: no rows match --body-system filter — falling back to all systems.")
            wide_agg = wide
            agg_subtitle = "All body systems"
    else:
        wide_agg = wide
        agg_subtitle = "All body systems"

    # ── 1. Aggregate ──────────────────────────────────────────────────────────
    print("\nPlotting aggregate…")
    agg = group_stats(wide_agg)
    plot_robustness(agg, ["Aggregate"],
                    f"Robustness — Aggregate\n(baseline − gender-swap accuracy)\n{agg_subtitle}",
                    "robustness_aggregate.png", figsize=(4, 4), show_n=True)
    plot_accuracy(agg, ["Aggregate"],
                  f"Accuracy — Aggregate\n{agg_subtitle}",
                  "accuracy_aggregate.png", figsize=(4, 4), show_n=True)
    agg_mi = mi_stats(wide_agg)
    plot_mi(agg_mi, ["Aggregate"],
            f"Gender Sensitivity (MI) — Aggregate\n{agg_subtitle}",
            "mi_aggregate.png", figsize=(4, 4))
    agg_flip = flip_stats(wide_agg)
    plot_flip(agg_flip, ["Aggregate"],
              f"Flip Rate — Aggregate\n{agg_subtitle}",
              "flip_aggregate.png", figsize=(4, 4), show_n=True)

    # ── 2. By body system ─────────────────────────────────────────────────────
    print("Plotting by body system…")
    body_stats = group_stats(wide, "body_system")
    body_order = (body_stats.groupby("group")["n"].sum()
                  .sort_values(ascending=True).index.tolist())
    plot_robustness(body_stats, body_order,
                    "Robustness by Body System",
                    "robustness_body_system.png", horizontal=True, figsize=(8, 6), show_n=True)
    plot_accuracy(body_stats, body_order,
                  "Accuracy by Body System",
                  "accuracy_body_system.png", horizontal=True, figsize=(10, 6), show_n=True)
    body_mi = mi_stats(wide, "body_system")
    plot_mi(body_mi, body_order,
            "Gender Sensitivity (MI) by Body System",
            "mi_body_system.png", horizontal=True, figsize=(8, 6))
    body_flip = flip_stats(wide, "body_system")
    plot_flip(body_flip, body_order,
              "Flip Rate by Body System",
              "flip_body_system.png", horizontal=True, figsize=(8, 6), show_n=True)

    # ── 3. By question type (Reasoning / Understanding) ───────────────────────
    print("Plotting by question type…")
    qt_stats = group_stats(wide, "q_type")
    qt_order  = qt_stats.groupby("group")["n"].sum().sort_values(ascending=False).index.tolist()
    plot_robustness(qt_stats, qt_order,
                    "Robustness by Question Type",
                    "robustness_question_type.png", figsize=(5, 4))
    plot_accuracy(qt_stats, qt_order,
                  "Accuracy by Question Type",
                  "accuracy_question_type.png", figsize=(6, 4))
    qt_mi = mi_stats(wide, "q_type")
    plot_mi(qt_mi, qt_order,
            "Gender Sensitivity (MI) by Question Type",
            "mi_question_type.png", figsize=(5, 4))
    qt_flip = flip_stats(wide, "q_type")
    plot_flip(qt_flip, qt_order,
              "Flip Rate by Question Type",
              "flip_question_type.png", figsize=(5, 4))

    # ── 4. By medical task ────────────────────────────────────────────────────
    print("Plotting by medical task…")
    task_stats = group_stats(wide, "medical_task")
    task_order  = task_stats.groupby("group")["n"].sum().sort_values(ascending=False).index.tolist()
    plot_robustness(task_stats, task_order,
                    "Robustness by Medical Task",
                    "robustness_medical_task.png", figsize=(5, 4))
    plot_accuracy(task_stats, task_order,
                  "Accuracy by Medical Task",
                  "accuracy_medical_task.png", figsize=(6, 4))
    task_mi = mi_stats(wide, "medical_task")
    plot_mi(task_mi, task_order,
            "Gender Sensitivity (MI) by Medical Task",
            "mi_medical_task.png", figsize=(5, 4))
    task_flip = flip_stats(wide, "medical_task")
    plot_flip(task_flip, task_order,
              "Flip Rate by Medical Task",
              "flip_medical_task.png", figsize=(5, 4))

    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
