"""
Drug mention frequency in MedXpertQA change=0 questions, by body system.

For each body system: bar chart of the top-N most-mentioned drug names,
where bar height = total number of mentions across all change=0 questions
in that system.

Also produces one summary chart: mean drug mentions per question by body system.

Outputs saved to plots/medxpertqa/drug_freq/

Usage:
    python scripts/drug_freq_by_body.py
    python scripts/drug_freq_by_body.py --top-n 15 --min-length 4
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.parent / "data"
OUT_DIR  = ROOT / "plots" / "medxpertqa" / "drug_freq"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATH   = DATA_DIR / "medxpertqa_text.jsonl"
PERTURB_PATH = DATA_DIR / "medxpertqa_gender_perturb.csv"
KG_PATH      = DATA_DIR / "kg.csv"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

BODY_SYSTEM_ORDER = [
    "Cardiovascular", "Nervous", "Endocrine", "Digestive",
    "Respiratory", "Reproductive", "Urinary", "Skeletal",
    "Muscular", "Lymphatic", "Integumentary", "Other / NA",
]

# ── drug vocabulary ────────────────────────────────────────────────────────────

def load_drug_vocab(min_length: int = 4) -> tuple[set[str], set[str]]:
    kg = pd.read_csv(KG_PATH, low_memory=False,
                     usecols=["x_type", "x_name", "y_type", "y_name"])
    names: set[str] = set()
    for col_type, col_name in [("x_type", "x_name"), ("y_type", "y_name")]:
        names.update(kg.loc[kg[col_type] == "drug", col_name]
                       .dropna().str.lower().unique())
    names = {n for n in names if len(n) >= min_length}
    unigrams = {n for n in names if " " not in n}
    bigrams  = {n for n in names if " " in n}
    print(f"  Drug vocab: {len(unigrams):,} unigrams  {len(bigrams):,} bigrams")
    return unigrams, bigrams


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


# ── data loading ───────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Return one row per change=0 question with body_system and drug list."""
    meta = {
        r["id"]: r
        for l in JSONL_PATH.read_text().splitlines()
        if l.strip()
        for r in [json.loads(l)]
    }
    perturb = pd.read_csv(PERTURB_PATH, dtype=str)
    change0 = perturb[perturb["change"].str.strip() == "0"].copy()
    print(f"  change=0 questions: {len(change0):,} / {len(perturb):,} total")

    rows = []
    for _, row in change0.iterrows():
        qid = row["id"]
        m   = meta.get(qid, {})
        rows.append({
            "id":          qid,
            "body_system": m.get("body_system", "Unknown"),
            "text":        row["original_question"],
        })
    return pd.DataFrame(rows)


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_per_system(system: str, drug_counts: Counter, top_n: int) -> None:
    """Bar chart of top-N drugs for one body system."""
    if not drug_counts:
        return

    top = drug_counts.most_common(top_n)
    drugs, counts = zip(*top)

    fig, ax = plt.subplots(figsize=(max(7, top_n * 0.55), 4))
    x = range(len(drugs))
    ax.bar(x, counts, color="#4C72B0", edgecolor="white", width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(drugs, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Total mentions (change=0 questions)", fontsize=9)
    ax.set_title(f"Drug mentions — {system}", fontsize=10, fontweight="bold")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Annotate bars
    for xi, c in zip(x, counts):
        ax.text(xi, c + 0.1, str(c), ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    safe = re.sub(r"[^\w]+", "_", system).strip("_").lower()
    path = OUT_DIR / f"drug_freq_{safe}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_mean_per_question(summary: pd.DataFrame) -> None:
    """Horizontal bar: mean drug mentions per question, sorted, by body system."""
    df = summary[summary["body_system"].isin(BODY_SYSTEM_ORDER)].copy()
    # Preserve logical order
    df["_order"] = df["body_system"].map(
        {s: i for i, s in enumerate(BODY_SYSTEM_ORDER)}
    )
    df = df.sort_values("mean_mentions", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Blues(
        0.35 + 0.65 * df["mean_mentions"] / df["mean_mentions"].max()
    )
    bars = ax.barh(df["body_system"], df["mean_mentions"],
                   color=colors, edgecolor="white", height=0.65)

    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{row['mean_mentions']:.2f}  (n={row['n_questions']})",
                va="center", ha="left", fontsize=8, color="#333")

    ax.set_xlabel("Mean drug mentions per question", fontsize=9)
    ax.set_title(
        "Mean drug mentions per question by body system\n"
        "(change=0 gender-perturbation questions only)",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlim(0, df["mean_mentions"].max() * 1.45)
    fig.tight_layout()
    path = OUT_DIR / "mean_mentions_by_system.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_clustered_top_drugs(
    drug_by_system: dict[str, Counter], top_n: int
) -> None:
    """
    Grouped bar chart: top-N drugs overall, bar heights = mentions per system.
    Each drug is a group; each bar within the group is a body system.
    Only systems that mention the drug at all are shown.
    """
    # Compute global top drugs
    global_counts: Counter = Counter()
    for c in drug_by_system.values():
        global_counts.update(c)
    top_drugs = [d for d, _ in global_counts.most_common(top_n)]

    systems = [s for s in BODY_SYSTEM_ORDER if s in drug_by_system]
    n_drugs   = len(top_drugs)
    n_systems = len(systems)

    cmap   = plt.cm.get_cmap("tab20", n_systems)
    width  = 0.8 / n_systems
    x      = range(n_drugs)

    fig, ax = plt.subplots(figsize=(max(12, n_drugs * 0.9), 5))
    for i, system in enumerate(systems):
        offsets = [xi - 0.4 + (i + 0.5) * width for xi in x]
        vals    = [drug_by_system[system].get(d, 0) for d in top_drugs]
        ax.bar(offsets, vals, width=width * 0.9,
               label=system, color=cmap(i), edgecolor="white")

    ax.set_xticks(list(x))
    ax.set_xticklabels(top_drugs, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Total mentions", fontsize=9)
    ax.set_title(
        f"Top-{top_n} drugs by mention count — breakdown by body system\n"
        "(change=0 questions only)",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    path = OUT_DIR / "clustered_top_drugs.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n",      type=int, default=12,
                        help="Top-N drugs to show per system (default 12)")
    parser.add_argument("--min-length", type=int, default=4,
                        help="Minimum drug name length (default 4)")
    args = parser.parse_args()

    print("Loading drug vocabulary…")
    unigrams, bigrams = load_drug_vocab(args.min_length)

    print("Loading questions…")
    df = load_data()

    print("Counting drug mentions…")
    df["drugs"]     = df["text"].apply(lambda t: find_drugs(t, unigrams, bigrams))
    df["n_mentions"] = df["drugs"].apply(len)

    # ── per-system counters ───────────────────────────────────────────────────
    drug_by_system: dict[str, Counter] = defaultdict(Counter)
    summary_rows = []
    for system, grp in df.groupby("body_system"):
        c: Counter = Counter()
        for drugs in grp["drugs"]:
            c.update(drugs)
        drug_by_system[system] = c
        summary_rows.append({
            "body_system":    system,
            "n_questions":    len(grp),
            "total_mentions": grp["n_mentions"].sum(),
            "mean_mentions":  round(grp["n_mentions"].mean(), 3),
            "pct_with_drug":  round(100 * (grp["n_mentions"] > 0).mean(), 1),
        })

    summary = pd.DataFrame(summary_rows).sort_values("mean_mentions", ascending=False)

    print("\n── Summary ────────────────────────────────────────────────")
    print(summary.to_string(index=False))

    csv_path = OUT_DIR / "drug_freq_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    print("\nPlotting…")
    plot_mean_per_question(summary)
    plot_clustered_top_drugs(drug_by_system, top_n=args.top_n)
    for system, counts in drug_by_system.items():
        plot_per_system(system, counts, top_n=args.top_n)

    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
