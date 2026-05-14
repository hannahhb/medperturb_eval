"""
Drug-name analysis of MedXpertQA — ToolUni relevance by body system.

For each body_system:
  - Detect questions that contain at least one named drug (matched against
    the PrimeKG drug vocabulary).
  - Compute the percentage of questions with ≥1 drug mention.
  - Collect all drug tokens found and build a word-cloud sized by frequency.

Outputs (saved to plots/medxpertqa/drug_wordclouds/):
  summary_bar.png           — % drug-mention questions per body system
  wordcloud_{system}.png    — drug frequency word-cloud per body system
  drug_mention_summary.csv  — raw counts (body_system, n_total, n_drug, pct)

Usage:
  python scripts/drug_wordcloud.py
  python scripts/drug_wordcloud.py --min-length 4   # ignore short tokens like "iron"
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT.parent / "data"
OUT_DIR   = ROOT / "plots" / "medxpertqa" / "drug_wordclouds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATH = DATA_DIR / "medxpertqa_text.jsonl"
KG_PATH    = DATA_DIR / "kg.csv"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── drug vocabulary from PrimeKG ──────────────────────────────────────────────

def load_drug_vocab(kg_path: Path, min_length: int = 3) -> set[str]:
    """Return lower-cased drug names from PrimeKG, length-filtered."""
    kg = pd.read_csv(kg_path, low_memory=False,
                     usecols=["x_type", "x_name", "y_type", "y_name"])
    names = set()
    for col_type, col_name in [("x_type", "x_name"), ("y_type", "y_name")]:
        mask = kg[col_type] == "drug"
        names.update(kg.loc[mask, col_name].dropna().str.lower().unique())
    # Remove very short / ambiguous tokens
    names = {n for n in names if len(n) >= min_length}
    print(f"  Drug vocabulary: {len(names):,} terms from PrimeKG")
    return names


# ── drug detection ────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """Lower-case, split on non-alpha (keep hyphens inside words)."""
    return re.findall(r"[a-z][a-z\-]*[a-z]|[a-z]", text.lower())


def find_drugs_in_text(text: str, vocab: set[str],
                       bigrams: set[str]) -> list[str]:
    """
    Return every drug name found in text.
    Checks single tokens and consecutive bigrams against the vocabulary.
    """
    tokens = _tokenise(text)
    found = []
    i = 0
    while i < len(tokens):
        # Try bigram first (e.g. "warfarin sodium")
        if i + 1 < len(tokens):
            bg = tokens[i] + " " + tokens[i + 1]
            if bg in bigrams:
                found.append(bg)
                i += 2
                continue
        if tokens[i] in vocab:
            found.append(tokens[i])
        i += 1
    return found


# ── data loading ──────────────────────────────────────────────────────────────

def load_questions(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ── analysis ──────────────────────────────────────────────────────────────────

def analyse(questions: list[dict], vocab: set[str],
            bigrams: set[str]) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per question:
      id, body_system, medical_task, has_drug, drugs_found
    """
    rows = []
    for q in questions:
        # Search question text + all option texts
        full_text = q["question"]
        if isinstance(q.get("options"), dict):
            full_text += " " + " ".join(q["options"].values())
        elif isinstance(q.get("options"), list):
            full_text += " " + " ".join(str(o) for o in q["options"])

        drugs = find_drugs_in_text(full_text, vocab, bigrams)
        rows.append({
            "id":           q["id"],
            "body_system":  q["body_system"],
            "medical_task": q["medical_task"],
            "has_drug":     len(drugs) > 0,
            "drugs_found":  drugs,
        })
    return pd.DataFrame(rows)


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_summary_bar(summary: pd.DataFrame) -> None:
    """Horizontal bar chart: % questions with drug mentions per body system."""
    summary_sorted = summary.sort_values("pct_drug", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn(summary_sorted["pct_drug"].values / 100)
    bars = ax.barh(summary_sorted["body_system"], summary_sorted["pct_drug"],
                   color=colors, edgecolor="white", height=0.65)

    for bar, (_, row) in zip(bars, summary_sorted.iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{row['pct_drug']:.1f}%  (n={row['n_drug']}/{row['n_total']})",
                va="center", ha="left", fontsize=8, color="#333")

    ax.set_xlabel("Questions with ≥1 drug mention (%)", fontsize=9)
    ax.set_title("ToolUni relevance by body system\n"
                 "(% of MedXpertQA questions containing a named drug)",
                 fontsize=10, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlim(0, max(summary_sorted["pct_drug"]) * 1.35)
    fig.tight_layout()
    path = OUT_DIR / "summary_bar.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_wordcloud(drug_counts: Counter, body_system: str) -> None:
    """Word-cloud for a single body system."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("  wordcloud not installed — skipping word-cloud plots")
        return

    if not drug_counts:
        return

    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap="Blues",
        max_words=60,
        prefer_horizontal=0.9,
        min_font_size=10,
    ).generate_from_frequencies(drug_counts)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Drug mentions — {body_system}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    safe_name = re.sub(r"[^\w]+", "_", body_system).strip("_").lower()
    path = OUT_DIR / f"wordcloud_{safe_name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-length", type=int, default=4,
                        help="Minimum drug name length to include (default 4)")
    args = parser.parse_args()

    print("Loading drug vocabulary from PrimeKG…")
    vocab   = load_drug_vocab(KG_PATH, min_length=args.min_length)
    bigrams = {n for n in vocab if " " in n}          # multi-word drug names
    vocab   = {n for n in vocab if " " not in n}      # single-token drugs

    print("Loading MedXpertQA questions…")
    questions = load_questions(JSONL_PATH)
    print(f"  {len(questions):,} questions")

    print("Detecting drug mentions…")
    df = analyse(questions, vocab, bigrams)

    # ── per-system summary ────────────────────────────────────────────────────
    summary_rows = []
    drug_by_system: dict[str, Counter] = defaultdict(Counter)

    for system, grp in df.groupby("body_system"):
        n_total = len(grp)
        n_drug  = grp["has_drug"].sum()
        pct     = 100 * n_drug / n_total
        summary_rows.append({
            "body_system": system,
            "n_total":     n_total,
            "n_drug":      n_drug,
            "pct_drug":    round(pct, 1),
        })
        # Accumulate drug frequencies for this system
        for drugs in grp["drugs_found"]:
            drug_by_system[system].update(drugs)

    summary = pd.DataFrame(summary_rows).sort_values("pct_drug", ascending=False)

    # ── overall stats ─────────────────────────────────────────────────────────
    total   = len(df)
    n_drug  = df["has_drug"].sum()
    print(f"\nOverall: {n_drug}/{total} questions ({100*n_drug/total:.1f}%) "
          f"contain ≥1 drug name")
    print()
    print(summary.to_string(index=False))

    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "drug_mention_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nPlotting…")
    plot_summary_bar(summary)

    for system in df["body_system"].unique():
        plot_wordcloud(drug_by_system[system], system)

    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
