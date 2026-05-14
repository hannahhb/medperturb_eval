"""
Distribution of drug-mention counts per question (change=0 only),
faceted by body system.

One figure: 12 small subplots (one per body system), each showing
how many questions have 0, 1, 2, 3, 4, 5+ drug mentions.

Usage:
    python scripts/drug_mention_dist.py
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.parent / "data"
OUT_DIR  = ROOT / "plots" / "medxpertqa" / "drug_freq"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATH   = DATA_DIR / "medxpertqa_text.jsonl"
PERTURB_PATH = DATA_DIR / "medxpertqa_gender_perturb.csv"
KG_PATH      = DATA_DIR / "kg.csv"

BODY_SYSTEM_ORDER = [
    "Cardiovascular", "Nervous", "Endocrine", "Digestive",
    "Respiratory", "Reproductive", "Urinary", "Skeletal",
    "Muscular", "Lymphatic", "Integumentary", "Other / NA",
]

# ── helpers (same as drug_freq_by_body.py) ────────────────────────────────────

def load_drug_vocab(min_length: int = 4) -> tuple[set[str], set[str]]:
    kg = pd.read_csv(KG_PATH, low_memory=False,
                     usecols=["x_type", "x_name", "y_type", "y_name"])
    names: set[str] = set()
    for col_type, col_name in [("x_type", "x_name"), ("y_type", "y_name")]:
        names.update(kg.loc[kg[col_type] == "drug", col_name]
                       .dropna().str.lower().unique())
    names = {n for n in names if len(n) >= min_length}
    return {n for n in names if " " not in n}, {n for n in names if " " in n}


def find_drugs(text: str, unigrams: set[str], bigrams: set[str]) -> list[str]:
    tokens = re.findall(r"[a-z][a-z\-]*[a-z]|[a-z]", text.lower())
    found, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            bg = tokens[i] + " " + tokens[i + 1]
            if bg in bigrams:
                found.append(bg)
                i += 2
                continue
        if tokens[i] in unigrams:
            found.append(tokens[i])
        i += 1
    return found


def load_data(unigrams, bigrams) -> pd.DataFrame:
    meta = {
        json.loads(l)["id"]: json.loads(l)
        for l in JSONL_PATH.read_text().splitlines() if l.strip()
    }
    perturb  = pd.read_csv(PERTURB_PATH, dtype=str)
    change0  = perturb[perturb["change"].str.strip() == "0"]
    rows = []
    for _, row in change0.iterrows():
        m = meta.get(row["id"], {})
        drugs = find_drugs(row["original_question"], unigrams, bigrams)
        rows.append({
            "id":          row["id"],
            "body_system": m.get("body_system", "Other / NA"),
            "n_mentions":  len(drugs),
        })
    return pd.DataFrame(rows)


# ── plot ───────────────────────────────────────────────────────────────────────

BIN_EDGES   = [0, 1, 2, 3, 4, 5, 6, 8, 999]   # last bin = "8+"
BIN_LABELS  = ["0", "1", "2", "3", "4", "5", "6–7", "8+"]
BAR_COLOR   = "#4C72B0"
ZERO_COLOR  = "#d9534f"   # red for zero-drug bars to highlight "no signal" cases


def make_bins(series: pd.Series) -> list[int]:
    """Count questions per bin for a single body system."""
    counts = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        counts.append(int(((series >= lo) & (series < hi)).sum()))
    return counts


def main():
    print("Loading vocab…")
    unigrams, bigrams = load_drug_vocab()

    print("Loading data…")
    df = load_data(unigrams, bigrams)
    print(f"  {len(df):,} change=0 questions")

    # ── overall distribution ──────────────────────────────────────────────────
    fig_all, ax_all = plt.subplots(figsize=(8, 4))
    overall_counts = make_bins(df["n_mentions"])
    colors = [ZERO_COLOR if i == 0 else BAR_COLOR for i in range(len(BIN_LABELS))]
    ax_all.bar(BIN_LABELS, overall_counts, color=colors, edgecolor="white", width=0.7)
    for xi, c in enumerate(overall_counts):
        ax_all.text(xi, c + 5, str(c), ha="center", va="bottom", fontsize=8)
    ax_all.set_xlabel("Drug mentions per question", fontsize=9)
    ax_all.set_ylabel("Number of questions", fontsize=9)
    ax_all.set_title(
        "Distribution of drug mentions per question (all body systems)\n"
        "change=0 gender-perturbation questions only",
        fontsize=10, fontweight="bold",
    )
    fig_all.tight_layout()
    p = OUT_DIR / "drug_mention_dist_overall.png"
    fig_all.savefig(p, bbox_inches="tight")
    plt.close(fig_all)
    print(f"  Saved: {p}")

    # ── faceted by body system ────────────────────────────────────────────────
    n_cols = 3
    n_rows = int(np.ceil(len(BODY_SYSTEM_ORDER) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.5, n_rows * 3.2),
                             sharey=False)
    axes_flat = axes.flatten()

    for ax, system in zip(axes_flat, BODY_SYSTEM_ORDER):
        subset = df[df["body_system"] == system]["n_mentions"]
        counts = make_bins(subset)
        colors = [ZERO_COLOR if i == 0 else BAR_COLOR for i in range(len(BIN_LABELS))]
        ax.bar(BIN_LABELS, counts, color=colors, edgecolor="white", width=0.7)

        # Annotate non-zero bars
        for xi, c in enumerate(counts):
            if c > 0:
                ax.text(xi, c + 0.3, str(c),
                        ha="center", va="bottom", fontsize=6.5, color="#333")

        pct_zero = 100 * counts[0] / len(subset) if len(subset) else 0
        ax.set_title(
            f"{system}  (n={len(subset)})\n{pct_zero:.0f}% with 0 drug mentions",
            fontsize=8, fontweight="bold",
        )
        ax.set_xlabel("Drug mentions", fontsize=7)
        ax.set_ylabel("Questions", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))

    # Hide unused subplots
    for ax in axes_flat[len(BODY_SYSTEM_ORDER):]:
        ax.set_visible(False)

    fig.suptitle(
        "Drug mention count distribution by body system\n"
        "(change=0 questions — red bars = no drug signal for ToolUni)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    p = OUT_DIR / "drug_mention_dist_by_system.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")


if __name__ == "__main__":
    main()
