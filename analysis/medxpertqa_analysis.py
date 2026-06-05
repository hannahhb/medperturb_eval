"""
MedXpertQA results analysis.

Compares accuracy across models and methods (e.g. baseline vs ToolUni),
broken down by body system and question type.

Designed to scale: add new runs to MODEL_FILES, re-run.

Usage:
    python analysis/medxpertqa_analysis.py
    python analysis/medxpertqa_analysis.py --task "Basic Science"
    python analysis/medxpertqa_analysis.py --no-plots      # tables only
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2

# ── config ────────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent.parent
PLOTS_DIR = BASE / "plots" / "medxpertqa"
OUTPUT_BASE = BASE / "output" / "medxpertqa"

# Map display label → results CSV path (or list of CSVs to concatenate).
# Add new runs here as they complete.
MODEL_FILES: Dict[str, str | list] = {
    # qwen235b, Basic Science subset
    "Baseline (Qwen)":  OUTPUT_BASE / "aws_qwen235b_mt-Basic_Science/results.csv",
    "ToolUni (Qwen)":   OUTPUT_BASE / "tooluni_qwen235b_mt-Basic_Science/results.csv",
    # add more as available, e.g.:
    # "MedReason (Qwen)": OUTPUT_BASE / "medreason_qwen235b_mt-Basic_Science/results.csv",
}

BODY_SYSTEM_ORDER = [
    "Cardiovascular", "Digestive", "Endocrine", "Integumentary",
    "Lymphatic", "Muscular", "Nervous", "Reproductive",
    "Respiratory", "Skeletal", "Urinary", "Other / NA",
]

# Colour palette (one per model label, extended as needed)
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

# ── data loading ──────────────────────────────────────────────────────────────

def _load_one(path: str | Path) -> pd.DataFrame:
    """Load a single results CSV and ensure required columns exist."""
    df = pd.read_csv(path)
    # Derive correct flag
    df["correct"] = (df["model_answer_letter"].astype(str).str.strip()
                     == df["correct_label"].astype(str).str.strip()).astype(int)
    # Older runs may lack body_system / medical_task
    for col in ("body_system", "medical_task", "question_type"):
        if col not in df.columns:
            df[col] = None
    return df


def load_all(
    model_files: Dict[str, str | list] = MODEL_FILES,
    task_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load all model CSVs, concatenate into a long-form DataFrame with a
    'model' column (display label from MODEL_FILES keys).
    """
    frames = []
    for label, path_or_list in model_files.items():
        paths = path_or_list if isinstance(path_or_list, list) else [path_or_list]
        dfs = []
        for p in paths:
            p = Path(p)
            if not p.exists():
                print(f"  [SKIP] {label}: file not found: {p}")
                continue
            dfs.append(_load_one(p))
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True)
        df["model"] = label
        if task_filter and "medical_task" in df.columns:
            df = df[df["medical_task"].fillna("").str.strip() == task_filter]
        frames.append(df)
        print(f"  Loaded {label}: {len(df)} rows, acc={df['correct'].mean():.3f}")

    if not frames:
        raise RuntimeError("No data loaded — check MODEL_FILES paths.")
    return pd.concat(frames, ignore_index=True)


# ── statistics ────────────────────────────────────────────────────────────────

def bootstrap_ci(
    correct: pd.Series, n_boot: int = 2000, ci: float = 0.95
) -> tuple[float, float]:
    """Return (lo, hi) bootstrap confidence interval for accuracy."""
    vals = correct.values.astype(float)
    rng = np.random.default_rng(42)
    boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boots, 100 * (1 + ci) / 2))
    return lo, hi


def mcnemar(correct_a: pd.Series, correct_b: pd.Series) -> tuple[float, float]:
    """
    Two-sided McNemar test on paired binary outcomes (no continuity correction).
    Returns (statistic, p-value).
    """
    b01 = int(((1 - correct_a) * correct_b).sum())   # A wrong, B right
    b10 = int((correct_a * (1 - correct_b)).sum())    # A right, B wrong
    if b01 + b10 == 0:
        return 0.0, 1.0
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p = 1 - chi2.cdf(stat, df=1)
    return float(stat), float(p)


# ── accuracy tables ───────────────────────────────────────────────────────────

def overall_table(df: pd.DataFrame) -> pd.DataFrame:
    """Overall accuracy (± 95% CI) per model."""
    rows = []
    for model, g in df.groupby("model"):
        acc = g["correct"].mean()
        lo, hi = bootstrap_ci(g["correct"])
        rows.append({
            "Model": model,
            "N": len(g),
            "Accuracy": acc,
            "CI_lo": lo,
            "CI_hi": hi,
            "Acc (95% CI)": f"{acc:.3f} [{lo:.3f}–{hi:.3f}]",
        })
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False)


def body_system_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per body system × model."""
    out = {}
    systems = [s for s in BODY_SYSTEM_ORDER
               if s in df["body_system"].unique()]
    systems += [s for s in df["body_system"].dropna().unique()
                if s not in systems]

    for model, g in df.groupby("model"):
        accs = {}
        for sys in systems:
            sub = g[g["body_system"] == sys]
            accs[sys] = sub["correct"].mean() if len(sub) else float("nan")
        out[model] = accs
    return pd.DataFrame(out).loc[systems]


def question_type_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per question type (Reasoning / Understanding) × model."""
    rows = []
    for (model, qtype), g in df.groupby(["model", "question_type"]):
        rows.append({
            "Model": model,
            "Question Type": qtype,
            "N": len(g),
            "Accuracy": round(g["correct"].mean(), 4),
        })
    return pd.DataFrame(rows).pivot(
        index="Question Type", columns="Model", values="Accuracy"
    )


def mcnemar_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise McNemar tests on shared question IDs."""
    models = df["model"].unique().tolist()
    rows = []
    for m_a, m_b in itertools.combinations(models, 2):
        ga = df[df["model"] == m_a].set_index("id")["correct"]
        gb = df[df["model"] == m_b].set_index("id")["correct"]
        shared = ga.index.intersection(gb.index)
        if len(shared) < 10:
            continue
        stat, p = mcnemar(ga.loc[shared], gb.loc[shared])
        rows.append({
            "Model A": m_a,
            "Model B": m_b,
            "N": len(shared),
            "McNemar stat": round(stat, 3),
            "p-value": round(p, 4),
            "sig": "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "",
        })
    return pd.DataFrame(rows)


# ── plots ─────────────────────────────────────────────────────────────────────

def _colour_map(models: List[str]) -> Dict[str, str]:
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(sorted(models))}


def plot_overall(df: pd.DataFrame, save_dir: Path) -> None:
    import matplotlib.pyplot as plt

    tbl = overall_table(df).sort_values("Accuracy", ascending=True)
    colours = _colour_map(df["model"].unique().tolist())
    cols = [colours[m] for m in tbl["Model"]]

    fig, ax = plt.subplots(figsize=(7, max(2.5, len(tbl) * 0.7)))
    bars = ax.barh(tbl["Model"], tbl["Accuracy"], color=cols, height=0.55, zorder=2)
    # Error bars
    xerr_lo = tbl["Accuracy"] - tbl["CI_lo"]
    xerr_hi = tbl["CI_hi"] - tbl["Accuracy"]
    ax.errorbar(tbl["Accuracy"], tbl["Model"],
                xerr=[xerr_lo, xerr_hi], fmt="none",
                color="black", capsize=4, linewidth=1.2, zorder=3)
    for bar, acc in zip(bars, tbl["Accuracy"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Accuracy (95% CI)")
    ax.set_xlim(0, min(1.0, tbl["Accuracy"].max() + 0.12))
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_dir / "overall_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: overall_accuracy.png")


def plot_body_system(df: pd.DataFrame, save_dir: Path) -> None:
    import matplotlib.pyplot as plt

    tbl = body_system_table(df)
    models = tbl.columns.tolist()
    n_models = len(models)
    systems = tbl.index.tolist()
    colours = _colour_map(models)

    x = np.arange(len(systems))
    width = 0.8 / n_models
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(max(12, len(systems) * 1.1), 5))
    for i, model in enumerate(models):
        vals = tbl[model].values.astype(float)
        ax.bar(x + offsets[i], vals, width * 0.92,
               label=model, color=colours[model], zorder=2, alpha=0.88)
        # Annotate bars
        for xi, v in zip(x + offsets[i], vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.008, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_dir / "accuracy_by_body_system.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: accuracy_by_body_system.png")


def plot_body_system_delta(df: pd.DataFrame, save_dir: Path,
                           reference: Optional[str] = None) -> None:
    """
    For each model vs a reference model (default: first alphabetically),
    show accuracy delta per body system as a heatmap.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    tbl = body_system_table(df)
    models = tbl.columns.tolist()
    if reference is None:
        reference = models[0]
    if reference not in models or len(models) < 2:
        return

    comparators = [m for m in models if m != reference]
    delta = tbl[comparators].subtract(tbl[reference], axis=0)

    vmax = max(0.15, float(delta.abs().max().max()))
    cmap = "RdYlGn"
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(6, len(comparators) * 1.8), max(4, len(delta) * 0.55)))
    im = ax.imshow(delta.values.T, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(delta.index)))
    ax.set_xticklabels(delta.index, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(comparators)))
    ax.set_yticklabels(comparators, fontsize=9)
    # Annotate cells
    for yi, _ in enumerate(comparators):
        for xi, _ in enumerate(delta.index):
            v = delta.values[xi, yi]
            if not np.isnan(v):
                ax.text(xi, yi, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    ax.set_title(f"Accuracy Δ vs {reference}", fontsize=10)
    plt.colorbar(im, ax=ax, label="Δ Accuracy", shrink=0.7)
    fig.tight_layout()
    fig.savefig(save_dir / "delta_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: delta_heatmap.png")


def plot_question_type(df: pd.DataFrame, save_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if df["question_type"].isna().all():
        return
    tbl = question_type_table(df)
    models = tbl.columns.tolist()
    n_models = len(models)
    colours = _colour_map(models)

    qtypes = tbl.index.tolist()
    x = np.arange(len(qtypes))
    width = 0.7 / n_models
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, model in enumerate(models):
        vals = tbl[model].values.astype(float)
        ax.bar(x + offsets[i], vals, width * 0.92,
               label=model, color=colours[model], zorder=2, alpha=0.88)
        for xi, v in zip(x + offsets[i], vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(qtypes, fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_dir / "accuracy_by_question_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: accuracy_by_question_type.png")


def plot_sample_size(df: pd.DataFrame, save_dir: Path) -> None:
    """Bar chart of N per body system (same for all models — just shown once)."""
    import matplotlib.pyplot as plt

    sub = df[df["body_system"].notna()]
    if sub.empty:
        return
    counts = (sub.groupby(["body_system", "model"])
              .size()
              .unstack(fill_value=0))
    # all models should have same N; just show first model
    col = counts.columns[0]
    ns = counts[col].reindex(BODY_SYSTEM_ORDER).dropna()

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(ns)), ns.values, color="#4C72B0", alpha=0.75, zorder=2)
    for xi, (sys, n) in enumerate(zip(ns.index, ns.values)):
        ax.text(xi, n + 1, str(int(n)), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("N questions")
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(ns.index, rotation=35, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_dir / "sample_size_by_system.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: sample_size_by_system.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MedXpertQA results analysis")
    parser.add_argument("--task", default=None,
                        help="Filter by medical_task value, e.g. 'Basic Science'")
    parser.add_argument("--reference", default=None,
                        help="Reference model for delta heatmap (default: first model)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation (tables only)")
    parser.add_argument("--out-dir", default=None,
                        help="Override plot output directory")
    args = parser.parse_args()

    save_dir = Path(args.out_dir) if args.out_dir else PLOTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Loading data ──────────────────────────────────────────────────")
    df = load_all(task_filter=args.task)

    # ── Overall accuracy ─────────────────────────────────────────────────────
    print("\n── Overall accuracy ──────────────────────────────────────────────")
    ov = overall_table(df)
    print(ov[["Model", "N", "Acc (95% CI)"]].to_string(index=False))

    # ── By body system ────────────────────────────────────────────────────────
    if df["body_system"].notna().any():
        print("\n── Accuracy by body system ───────────────────────────────────────")
        bs = body_system_table(df)
        # Append row counts
        ns = df[df["body_system"].notna()].groupby("body_system").size().rename("N")
        bs.insert(0, "N", ns)
        print(bs.round(3).to_string())

    # ── By question type ──────────────────────────────────────────────────────
    if df["question_type"].notna().any():
        print("\n── Accuracy by question type ─────────────────────────────────────")
        qt = question_type_table(df)
        print(qt.round(3).to_string())

    # ── McNemar tests ─────────────────────────────────────────────────────────
    print("\n── Pairwise McNemar tests ────────────────────────────────────────")
    mc = mcnemar_table(df)
    if mc.empty:
        print("  (need ≥2 models with shared question IDs)")
    else:
        print(mc.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print(f"\n── Saving plots → {save_dir} ─────────────────────────────────")
        try:
            import matplotlib
            matplotlib.use("Agg")
            plot_overall(df, save_dir)
            if df["body_system"].notna().any():
                plot_body_system(df, save_dir)
                plot_body_system_delta(df, save_dir, reference=args.reference)
                plot_sample_size(df, save_dir)
            if df["question_type"].notna().any():
                plot_question_type(df, save_dir)
        except ImportError:
            print("  matplotlib not available — skipping plots")

    print("\nDone.")


if __name__ == "__main__":
    main()
