"""
Distribution of KG path counts per question from MedReason results,
faceted by body system (change=0 / baseline questions only).

Outputs saved to plots/medxpertqa/kg_path_dist/

Usage:
    python scripts/kg_path_dist.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT.parent / "data"
OUT_DIR   = ROOT / "plots" / "medxpertqa" / "kg_path_dist"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = ROOT / "output" / "medxpertqa" / "medreason_llama70b3_3" / "results.csv"
JSONL_PATH  = DATA_DIR / "medxpertqa_text.jsonl"

BODY_SYSTEM_ORDER = [
    "Cardiovascular", "Nervous", "Endocrine", "Digestive",
    "Respiratory", "Reproductive", "Urinary", "Skeletal",
    "Muscular", "Lymphatic", "Integumentary", "Other / NA",
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

BIN_EDGES  = [0, 1, 2, 3, 4, 5, 6, 8, 11, 999]
BIN_LABELS = ["0", "1", "2", "3", "4", "5", "6–7", "8–10", "11+"]
ZERO_COLOR = "#d9534f"
BAR_COLOR  = "#55A868"

# ── helpers ────────────────────────────────────────────────────────────────────

def count_paths(x) -> int:
    if pd.isna(x) or str(x).strip() == "":
        return 0
    return len([l for l in str(x).strip().splitlines() if l.strip()])


def make_bins(series: pd.Series) -> list[int]:
    counts = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        counts.append(int(((series >= lo) & (series < hi)).sum()))
    return counts


# ── load data ──────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    meta = {
        json.loads(l)["id"]: json.loads(l)
        for l in JSONL_PATH.read_text().splitlines() if l.strip()
    }
    df = pd.read_csv(RESULTS_CSV)
    df["n_paths"] = df["kg_paths"].apply(count_paths)

    # Average path count across both question types per question id.
    # Gender swapping occasionally changes which entities are extracted
    # (161/1850 questions differ), so averaging gives a fairer picture
    # than using baseline alone.
    df = (
        df.groupby("id", as_index=False)["n_paths"]
          .mean()
          .rename(columns={"n_paths": "n_paths_mean"})
    )
    # Round to nearest int for binning
    df["n_paths"] = df["n_paths_mean"].round().astype(int)
    df["body_system"] = df["id"].map(
        lambda qid: meta.get(qid, {}).get("body_system", "Other / NA")
    )
    return df


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_overall(df: pd.DataFrame) -> None:
    counts = make_bins(df["n_paths"])
    colors = [ZERO_COLOR if i == 0 else BAR_COLOR for i in range(len(BIN_LABELS))]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(BIN_LABELS, counts, color=colors, edgecolor="white", width=0.7)
    for xi, c in enumerate(counts):
        ax.text(xi, c + 5, str(c), ha="center", va="bottom", fontsize=8)

    pct_zero = 100 * counts[0] / len(df)
    ax.set_xlabel("KG paths generated per question", fontsize=9)
    ax.set_ylabel("Number of questions", fontsize=9)
    ax.set_title(
        f"KG path count distribution — MedReason (all body systems)\n"
        f"{counts[0]} questions ({pct_zero:.0f}%) produced 0 paths  |  "
        f"n={len(df)}",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    p = OUT_DIR / "kg_path_dist_overall.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def plot_by_system(df: pd.DataFrame) -> None:
    n_cols = 3
    n_rows = int(np.ceil(len(BODY_SYSTEM_ORDER) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 4.5, n_rows * 3.2),
                              sharey=False)
    axes_flat = axes.flatten()

    for ax, system in zip(axes_flat, BODY_SYSTEM_ORDER):
        subset = df[df["body_system"] == system]["n_paths"]
        counts = make_bins(subset)
        colors = [ZERO_COLOR if i == 0 else BAR_COLOR
                  for i in range(len(BIN_LABELS))]
        ax.bar(BIN_LABELS, counts, color=colors, edgecolor="white", width=0.7)

        for xi, c in enumerate(counts):
            if c > 0:
                ax.text(xi, c + 0.3, str(c),
                        ha="center", va="bottom", fontsize=6, color="#333")

        n_total   = len(subset)
        pct_zero  = 100 * counts[0] / n_total if n_total else 0
        mean_path = subset.mean()
        ax.set_title(
            f"{system}  (n={n_total})\n"
            f"{pct_zero:.0f}% zero paths  |  mean={mean_path:.1f}",
            fontsize=8, fontweight="bold",
        )
        ax.set_xlabel("KG paths", fontsize=7)
        ax.set_ylabel("Questions", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))

    for ax in axes_flat[len(BODY_SYSTEM_ORDER):]:
        ax.set_visible(False)

    fig.suptitle(
        "KG path count distribution by body system — MedReason\n"
        "(red bars = 0 paths generated, no KG signal for that question)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    p = OUT_DIR / "kg_path_dist_by_system.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


def plot_zero_pct_bar(df: pd.DataFrame) -> None:
    """Horizontal bar: % of questions with 0 paths, one bar per body system."""
    rows = []
    for system in BODY_SYSTEM_ORDER:
        subset = df[df["body_system"] == system]["n_paths"]
        if len(subset) == 0:
            continue
        rows.append({
            "body_system": system,
            "pct_zero":    100 * (subset == 0).mean(),
            "mean_paths":  subset.mean(),
            "n":           len(subset),
        })
    summary = pd.DataFrame(rows).sort_values("pct_zero", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: % zero paths
    ax = axes[0]
    colors = plt.cm.RdYlGn(1 - summary["pct_zero"].values / 100)
    bars = ax.barh(summary["body_system"], summary["pct_zero"],
                   color=colors, edgecolor="white", height=0.65)
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{row['pct_zero']:.0f}%  (n={row['n']})",
                va="center", ha="left", fontsize=8)
    ax.set_xlabel("Questions with 0 KG paths (%)", fontsize=9)
    ax.set_title("% questions with NO KG paths\n(MedReason signal gap)",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(0, 105)

    # Right: mean paths
    summary2 = summary.sort_values("mean_paths", ascending=True)
    colors2 = plt.cm.Blues(
        0.3 + 0.7 * summary2["mean_paths"] / summary2["mean_paths"].max()
    )
    bars2 = axes[1].barh(summary2["body_system"], summary2["mean_paths"],
                         color=colors2, edgecolor="white", height=0.65)
    for bar, (_, row) in zip(bars2, summary2.iterrows()):
        axes[1].text(bar.get_width() + 0.05,
                     bar.get_y() + bar.get_height() / 2,
                     f"{row['mean_paths']:.1f}",
                     va="center", ha="left", fontsize=8)
    axes[1].set_xlabel("Mean KG paths per question", fontsize=9)
    axes[1].set_title("Mean KG paths generated\n(MedReason signal strength)",
                      fontsize=10, fontweight="bold")
    axes[1].set_xlim(0, summary2["mean_paths"].max() * 1.3)

    fig.tight_layout()
    p = OUT_DIR / "kg_path_coverage_by_system.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data…")
    df = load()
    print(f"  {len(df):,} questions (path count averaged over baseline + gender_swap)")
    print(f"  0-path questions: {(df['n_paths']==0).sum()} "
          f"({100*(df['n_paths']==0).mean():.1f}%)")
    print(f"  mean paths: {df['n_paths'].mean():.2f}  "
          f"max: {df['n_paths'].max()}")
    print()

    print("Plotting…")
    plot_overall(df)
    plot_by_system(df)
    plot_zero_pct_bar(df)

    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
