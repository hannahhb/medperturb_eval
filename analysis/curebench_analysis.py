"""
CureBench Analysis
==================
Compares final model results across AWS, MedReason, ToolUni, and MedPathAgent.
Saves plots to plots/curebench/.

Usage:
    python analysis/curebench_analysis.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2

def mcnemar(b01: int, b10: int):
    """McNemar test with continuity correction. Returns p-value."""
    if b01 + b10 == 0:
        return 1.0
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    return float(1 - _chi2.cdf(stat, df=1))

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data")
OUT_DIR    = ROOT / "output" / "curebench"
PLOT_DIR   = ROOT / "plots" / "curebench"
VAL_JSONL  = DATA_DIR / "curebench_valset_phase1.jsonl"

MODEL_FILES: Dict[str, Path] = {
    "Llama70B":     OUT_DIR / "aws_llama70b3_3"          / "results.csv",
    "MedReason":    OUT_DIR / "medreason_llama70b3_3"    / "results_0.csv",
    "ToolUni":      OUT_DIR / "tooluni_llama70b3_3"      / "results_final.csv",
    "MedPathAgent": OUT_DIR / "medpathagent_llama70b3_3" / "results_final.csv",
}

MODEL_COLORS = {
    "Llama70B":     "#6c757d",
    "MedReason":    "#4e9af1",
    "ToolUni":      "#f4a020",
    "MedPathAgent": "#2ca02c",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def savefig(name: str):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → saved {path}")


def load_questions(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "id":             d["id"],
                "question_type":  d.get("question_type", "unknown"),
                "correct_answer": d.get("correct_answer", ""),
                "n_options":      len(d.get("options") or {}),
            })
    return pd.DataFrame(rows).set_index("id")


def load_results(files: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    dfs = {}
    for name, path in files.items():
        if not path.exists():
            print(f"  [WARN] {name}: {path} not found — skipping")
            continue
        df = pd.read_csv(path)
        df["id"] = df["id"].astype(str)
        dfs[name] = df.set_index("id")
        print(f"  {name:<20} {len(df):>4} rows  ({path.name})")
    return dfs


def bootstrap_ci(correct: pd.Series, n: int = 2000, alpha: float = 0.05):
    vals = correct.values.astype(float)
    means = [np.mean(np.random.choice(vals, size=len(vals), replace=True)) for _ in range(n)]
    lo, hi = np.percentile(means, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return lo * 100, hi * 100


def sep(title: str = "", width: int = 70):
    if title:
        print(f"\n{'─'*3} {title} {'─'*(max(0,width-5-len(title)))}")
    else:
        print("─" * width)

# ── plot functions ────────────────────────────────────────────────────────────

def plot_overall(dfs: Dict[str, pd.DataFrame]):
    """Bar chart of overall accuracy with 95% CI error bars."""
    names, accs, los, his = [], [], [], []
    for name, df in dfs.items():
        acc = df["correct"].mean() * 100
        lo, hi = bootstrap_ci(df["correct"])
        names.append(name); accs.append(acc)
        los.append(acc - lo); his.append(hi - acc)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, accs,
                  color=[MODEL_COLORS[n] for n in names],
                  yerr=[los, his], capsize=5, error_kw={"elinewidth": 1.5},
                  edgecolor="white", linewidth=0.8)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CureBench Overall Accuracy (with 95% CI)")
    ax.set_ylim(70, 100)
    ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout()
    savefig("01_overall_accuracy.png")


def plot_by_question_type(dfs: Dict[str, pd.DataFrame], questions: pd.DataFrame):
    """Grouped bar chart: accuracy by question type per model."""
    qtypes = sorted(questions["question_type"].unique())
    x = np.arange(len(qtypes))
    width = 0.2
    names = list(dfs.keys())

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(names):
        df = dfs[name]
        merged = df.join(questions[["question_type"]], how="left")
        accs = [merged[merged["question_type"]==qt]["correct"].mean()*100 for qt in qtypes]
        ax.bar(x + i*width, accs, width, label=name,
               color=MODEL_COLORS[name], edgecolor="white")

    ax.set_xticks(x + width * (len(names)-1)/2)
    ax.set_xticklabels([qt.replace("_", "\n") for qt in qtypes], fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Question Type")
    ax.set_ylim(70, 100)
    ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout()
    savefig("02_accuracy_by_qtype.png")


def plot_kg_path_split(dfs: Dict[str, pd.DataFrame]):
    """Grouped bars: accuracy with vs without KG paths.
    Uses results_3.csv (relaxed threshold) as path source so all MPA-augmented
    rows are counted in the 'with paths' bucket."""
    # Use the relaxed-threshold MedReason run as the path definition
    mr3_path = OUT_DIR / "medreason_llama70b3_3" / "results_final.csv"
    if mr3_path.exists():
        mr = pd.read_csv(mr3_path).set_index("id")
    elif "MedReason" in dfs:
        mr = dfs["MedReason"]
    else:
        return
    has = mr["kg_paths"].notna() & (mr["kg_paths"].str.strip() != "") & \
          (mr["kg_paths"].str.strip().str.lower() != "nopaths")
    path_ids    = set(mr[has].index)
    no_path_ids = set(mr[~has].index)

    names = list(dfs.keys())
    x = np.arange(len(names))
    w = 0.35

    with_accs = [dfs[n][dfs[n].index.isin(path_ids)]["correct"].mean()*100 for n in names]
    wo_accs   = [dfs[n][dfs[n].index.isin(no_path_ids)]["correct"].mean()*100 for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars_with = ax.bar(x - w/2, with_accs, w, label=f"With KG paths (n={len(path_ids)})",
                       color=[MODEL_COLORS[n] for n in names], edgecolor="white")
    bars_wo   = ax.bar(x + w/2, wo_accs,   w, label=f"Without KG paths (n={len(no_path_ids)})",
                       color=[MODEL_COLORS[n] for n in names], edgecolor="white", alpha=0.45, hatch="//")

    # Accuracy labels on each bar
    for bar, acc in zip(bars_with, with_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=7.5)
    for bar, acc in zip(bars_wo, wo_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Accuracy: With vs Without KG Paths")
    ax.set_ylim(70, 102)

    solid = mpatches.Patch(color="grey", label=f"With KG paths (n={len(path_ids)})")
    hatch = mpatches.Patch(facecolor="grey", hatch="//", alpha=0.5,
                           label=f"Without KG paths (n={len(no_path_ids)})")
    ax.legend(handles=[solid, hatch], fontsize=8)
    ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout()
    savefig("03_kg_path_split.png")


def plot_pairwise_heatmap(dfs: Dict[str, pd.DataFrame]):
    """Heatmap of pairwise wins (row beats column)."""
    names = list(dfs.keys())
    common = list(set.intersection(*[set(df.index) for df in dfs.values()]))
    n = len(names)
    mat = np.zeros((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                da = dfs[a].loc[common, "correct"]
                db = dfs[b].loc[common, "correct"]
                mat[i, j] = ((da == 1) & (db == 0)).sum()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="YlOrRd")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title("Pairwise Wins (row correct, col wrong)")
    for i in range(n):
        for j in range(n):
            val = int(mat[i, j])
            ax.text(j, i, str(val) if i != j else "—",
                    ha="center", va="center",
                    color="white" if mat[i,j] > mat.max()*0.6 else "black", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    savefig("04_pairwise_wins.png")


def plot_agreement_pie(dfs: Dict[str, pd.DataFrame]):
    """Pie chart of all-correct / all-wrong / split questions."""
    common = sorted(set.intersection(*[set(df.index) for df in dfs.values()]))
    mat = pd.DataFrame({n: dfs[n].loc[common, "correct"] for n in dfs}).fillna(0)
    row_sums = mat.sum(axis=1)
    all_c = (row_sums == len(dfs)).sum()
    all_w = (row_sums == 0).sum()
    split = len(common) - all_c - all_w

    fig, ax = plt.subplots(figsize=(5, 5))
    sizes  = [all_c, split, all_w]
    labels = [f"All correct\n{all_c} ({all_c/len(common)*100:.1f}%)",
              f"Split\n{split} ({split/len(common)*100:.1f}%)",
              f"All wrong\n{all_w} ({all_w/len(common)*100:.1f}%)"]
    colors = ["#2ca02c", "#f4a020", "#d62728"]
    ax.pie(sizes, labels=labels, colors=colors, startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title("Model Agreement Across All Questions")
    plt.tight_layout()
    savefig("05_agreement_pie.png")


def plot_option_bias(dfs: Dict[str, pd.DataFrame], questions: pd.DataFrame):
    """Grouped bar: % answers per letter A-D vs gold distribution."""
    letters = list("ABCD")
    gold_ids = list(set.union(*[set(df.index) for df in dfs.values()]))
    gold = questions.loc[[i for i in gold_ids if i in questions.index], "correct_answer"]
    gold_dist = [(gold == l).sum() / len(gold) * 100 for l in letters]

    names = list(dfs.keys())
    x = np.arange(len(letters))
    width = 0.15

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(names):
        choices = dfs[name]["choice"].astype(str).str.strip().str.upper()
        dist = [(choices == l).sum() / len(choices) * 100 for l in letters]
        ax.bar(x + i*width, dist, width, label=name,
               color=MODEL_COLORS[name], edgecolor="white")

    # Gold as step line
    gold_x = x + width * (len(names)-1) / 2
    ax.step(np.append(gold_x - width/2, gold_x[-1]+width/2),
            np.append(gold_dist, gold_dist[-1]),
            where="post", color="black", linewidth=2, linestyle="--", label="Gold")

    ax.set_xticks(x + width*(len(names)-1)/2)
    ax.set_xticklabels(letters)
    ax.set_ylabel("% of answers")
    ax.set_title("Answer Letter Distribution vs Gold")
    ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout()
    savefig("06_option_bias.png")


def plot_difficulty(dfs: Dict[str, pd.DataFrame]):
    """Bar chart: # of questions by how many models answered correctly."""
    common = sorted(set.intersection(*[set(df.index) for df in dfs.values()]))
    mat = pd.DataFrame({n: dfs[n].loc[common, "correct"] for n in dfs}).fillna(0)
    counts = mat.sum(axis=1).astype(int).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color="#4e9af1", edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("# Models Correct")
    ax.set_ylabel("# Questions")
    ax.set_title("Question Difficulty Distribution")
    ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout()
    savefig("07_difficulty_distribution.png")


def plot_mcnemar(dfs: Dict[str, pd.DataFrame]):
    """Heatmap of McNemar p-values for pairwise significance testing."""
    names = list(dfs.keys())
    common = list(set.intersection(*[set(df.index) for df in dfs.values()]))
    n = len(names)
    pmat = np.ones((n, n))

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            da = dfs[a].loc[common, "correct"].values
            db = dfs[b].loc[common, "correct"].values
            # Contingency: b=1 a=0, b=0 a=1
            b01 = ((da == 0) & (db == 1)).sum()
            b10 = ((da == 1) & (db == 0)).sum()
            pmat[i, j] = mcnemar(b01, b10)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(np.where(np.eye(n, dtype=bool), np.nan, pmat),
                   cmap="RdYlGn_r", vmin=0, vmax=0.1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title("McNemar p-values (row vs col)\ngreen=significant (p<0.05)")
    for i in range(n):
        for j in range(n):
            txt = "—" if i == j else f"{pmat[i,j]:.3f}"
            color = "white" if i != j and pmat[i,j] < 0.01 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label="p-value")
    plt.tight_layout()
    savefig("08_mcnemar_pvalues.png")


def plot_incremental_gain(dfs: Dict[str, pd.DataFrame]):
    """Horizontal bar chart of Δacc vs AWS baseline."""
    if "Llama70B" not in dfs:
        return
    aws = dfs["Llama70B"]
    names, deltas = [], []
    for name, df in dfs.items():
        if name == "Llama70B":
            continue
        common = list(set(aws.index) & set(df.index))
        delta = (df.loc[common, "correct"].mean() - aws.loc[common, "correct"].mean()) * 100
        names.append(name); deltas.append(delta)

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = [MODEL_COLORS[n] for n in names]
    bars = ax.barh(names, deltas, color=colors, edgecolor="white")
    for bar, d in zip(bars, deltas):
        ax.text(d + 0.1 if d >= 0 else d - 0.1, bar.get_y() + bar.get_height()/2,
                f"{d:+.2f}pp", va="center", ha="left" if d >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ Accuracy (pp) vs AWS baseline")
    ax.set_title("Incremental Gain over AWS Baseline")
    ax.xaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout()
    savefig("09_incremental_gain.png")


# ── text sections ─────────────────────────────────────────────────────────────

def section_overall(dfs):
    sep("1. Overall Accuracy")
    rows = []
    for name, df in dfs.items():
        lo, hi = bootstrap_ci(df["correct"])
        rows.append({"Model": name, "N": len(df), "Correct": int(df["correct"].sum()),
                     "Accuracy": f"{df['correct'].mean()*100:.2f}%",
                     "95% CI": f"[{lo:.1f}, {hi:.1f}]"})
    print(pd.DataFrame(rows).to_string(index=False))


def section_pairwise(dfs):
    sep("4. Pairwise Win / Loss  (row beats column)")
    names = list(dfs.keys())
    common = list(set.intersection(*[set(df.index) for df in dfs.values()]))
    print(f"{'':>14}" + "".join(f"{n:>14}" for n in names))
    for a in names:
        row = f"{a:>14}"
        for b in names:
            if a == b: row += f"{'—':>14}"
            else:
                da = dfs[a].loc[common, "correct"]
                db = dfs[b].loc[common, "correct"]
                w = ((da==1)&(db==0)).sum(); l = ((da==0)&(db==1)).sum()
                row += f"{w}W/{l}L".rjust(14)
        print(row)


def section_mcnemar(dfs):
    sep("10. McNemar Significance Tests")
    names = list(dfs.keys())
    common = list(set.intersection(*[set(df.index) for df in dfs.values()]))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i: continue
            da = dfs[a].loc[common, "correct"].values
            db = dfs[b].loc[common, "correct"].values
            b01 = ((da==0)&(db==1)).sum()
            b10 = ((da==1)&(db==0)).sum()
            if b01+b10 == 0:
                print(f"  {a} vs {b}: no discordant pairs")
                continue
            p = mcnemar(b01, b10)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {a:<14} vs {b:<14}  p={p:.4f}  {sig}"
                  f"  (discordant: {b10} favour {a}, {b01} favour {b})")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading questions...")
    questions = load_questions(VAL_JSONL)
    print(f"  {len(questions)} questions\n")

    print("Loading model results...")
    dfs = load_results(MODEL_FILES)
    if not dfs:
        print("No results found."); return

    # Text summaries
    section_overall(dfs)
    section_pairwise(dfs)
    section_mcnemar(dfs)

    # Plots
    sep("Generating plots")
    plot_overall(dfs)
    plot_by_question_type(dfs, questions)
    plot_kg_path_split(dfs)
    plot_pairwise_heatmap(dfs)
    plot_agreement_pie(dfs)
    plot_option_bias(dfs, questions)
    plot_difficulty(dfs)
    plot_mcnemar(dfs)
    plot_incremental_gain(dfs)

    print(f"\nAll plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
