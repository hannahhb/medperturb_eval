"""
Robustness Evaluation Pipeline

Compares MedPathAgent-LLaMA against LLaMA and clinician consensus on the
same computed subset of MedPerturb rows.

Metrics:
  - ATR: Average Treatment Ratio
  - MI: Mutual Information between baseline and perturbed predictions
  - PC: Percent Change / flip rate from baseline to perturbed predictions
  - Cohen's Kappa: human-vs-model agreement for LLaMA and MedPathAgent

Outputs:
  output_dir/atr_all.csv
  output_dir/mi_all.csv
  output_dir/pc_all.csv
  output_dir/pairwise_kappa.csv
  output_dir/kappa_vs_human_consensus.csv
  output_dir/summary_comparison.csv
  output_dir/plots/atr_comparison_by_perturbation.png
  output_dir/plots/mi_comparison_by_perturbation.png
  output_dir/plots/pc_comparison_by_perturbation.png
  output_dir/plots/kappa_vs_human_by_perturbation.png

Usage:
    python robustness_eval_all_in_one.py \
        --kg_results results.csv \
        --orig_dataset medperturb_data.csv \
        --output_dir eval_output \
        --baseline_label baseline
"""

import argparse
import os
import warnings
from itertools import combinations
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

QUESTIONS = ["manage", "visit", "resource"]

# Models from the original MedPerturb dataset.
ORIG_MODELS = {
    "llama": {
        "manage": "llama_manage",
        "visit": "llama_visit",
        "resource": "llama_resource",
    },
}

# KG-augmented model columns from your computed results file.
KG_MODEL = {
    "medpathagent": {
        "manage": "medpathagent_llama70b3_3_manage",
        "visit": "medpathagent_llama70b3_3_visit",
        "resource": "medpathagent_llama70b3_3_resource",
    }
}

# Clinician consensus columns from the original MedPerturb dataset.
CLINICIAN = {
    "clinician": {
        "manage": "clinician_consensus_manage",
        "visit": "clinician_consensus_visit",
        "resource": "clinician_consensus_resource",
    }
}

GOLD = {
    "manage": "gold_standard_manage",
    "visit": "gold_standard_visit",
    "resource": "gold_standard_resource",
}

FOCUS_MODELS = ["llama", "medpathagent", "clinician"]
KAPPA_MODELS = ["llama", "medpathagent"]
ALPHA = 0.01
BONFERRONI_FACTOR = 5


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

def binarise(series: pd.Series) -> pd.Series:
    """Convert yes/no/1/0/true/false values to nullable integer binary labels."""
    s = series.copy()
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower()
        s = s.replace({
            "yes": 1,
            "y": 1,
            "true": 1,
            "t": 1,
            "1": 1,
            "no": 0,
            "n": 0,
            "false": 0,
            "f": 0,
            "0": 0,
            "nan": np.nan,
            "none": np.nan,
            "": np.nan,
        })
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def load_and_merge(kg_path: str, orig_path: str) -> pd.DataFrame:
    """
    Load KG results and original MedPerturb data, then inner-merge them.

    Important: this uses ONLY the rows present in your computed KG results,
    because the merge is inner on context_id, perturbation, and dataset.
    """
    kg = pd.read_csv(kg_path)
    orig = pd.read_csv(orig_path)

    required_keys = ["context_id", "perturbation", "dataset"]
    for name, frame in [("kg_results", kg), ("orig_dataset", orig)]:
        missing = [c for c in required_keys if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    for frame in [kg, orig]:
        frame["context_id"] = frame["context_id"].astype(str).str.strip()
        frame["perturbation"] = frame["perturbation"].astype(str).str.strip().str.lower()
        frame["dataset"] = frame["dataset"].astype(str).str.strip()

    merged = pd.merge(
        orig,
        kg,
        on=["context_id", "perturbation", "dataset"],
        how="inner",
        suffixes=("", "_kg"),
    )

    all_models = {}
    all_models.update(ORIG_MODELS)
    all_models.update(KG_MODEL)
    all_models.update(CLINICIAN)

    answer_cols: List[str] = []
    for model_cols in all_models.values():
        answer_cols.extend(model_cols.values())
    answer_cols.extend(GOLD.values())

    for col in answer_cols:
        if col in merged.columns:
            merged[col] = binarise(merged[col])

    print(f"Original dataset rows: {len(orig)}")
    print(f"KG results rows:       {len(kg)}")
    print(f"Merged/evaluated rows: {len(merged)}")
    print(f"Perturbations used:    {sorted(merged['perturbation'].unique())}")

    if len(merged) == 0:
        raise ValueError(
            "The merge produced 0 rows. Check that context_id, perturbation, "
            "and dataset match between the two files."
        )

    return merged


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, all_models: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """ATR = mean of binary treatment decisions per model, perturbation, and question."""
    rows = []
    for pert in sorted(df["perturbation"].unique()):
        subset = df[df["perturbation"] == pert]
        for model_name, cols in all_models.items():
            for q in QUESTIONS:
                col = cols[q]
                if col not in subset.columns:
                    continue
                vals = subset[col].dropna().astype(float)
                if vals.empty:
                    continue
                rows.append({
                    "perturbation": pert,
                    "model": model_name,
                    "question": q,
                    "ATR": vals.mean(),
                    "n": len(vals),
                })
    return pd.DataFrame(rows)


def _mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Compute binary mutual information in bits."""
    x = x.astype(int)
    y = y.astype(int)
    joint = np.zeros((2, 2), dtype=float)

    for a, b in zip(x, y):
        if a in (0, 1) and b in (0, 1):
            joint[a, b] += 1

    total = joint.sum()
    if total == 0:
        return np.nan

    joint /= total
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0
    for i in range(2):
        for j in range(2):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    return mi


def _get_baseline(df: pd.DataFrame, baseline_label: str) -> tuple[pd.DataFrame, str]:
    baseline_label = baseline_label.lower().strip()
    baseline_df = df[df["perturbation"] == baseline_label].copy()

    if baseline_df.empty and baseline_label != "vignette":
        baseline_label = "vignette"
        baseline_df = df[df["perturbation"] == baseline_label].copy()

    if baseline_df.empty:
        raise ValueError(
            "No baseline rows found. Use --baseline_label to set the correct baseline, "
            "for example 'baseline' or 'vignette'."
        )

    return baseline_df, baseline_label


def compute_mi(
    df: pd.DataFrame,
    all_models: Dict[str, Dict[str, str]],
    baseline_label: str = "baseline",
) -> pd.DataFrame:
    """
    MI between baseline and perturbed decisions.
    Higher MI = more stable/robust decisions under perturbation.
    """
    baseline_df, baseline_label = _get_baseline(df, baseline_label)
    perturbations = [p for p in sorted(df["perturbation"].unique()) if p != baseline_label]

    rows = []
    for pert in perturbations:
        pert_df = df[df["perturbation"] == pert].copy()
        merged = pd.merge(
            baseline_df,
            pert_df,
            on="context_id",
            suffixes=("_base", "_pert"),
            how="inner",
        )

        for model_name, cols in all_models.items():
            for q in QUESTIONS:
                col = cols[q]
                base_col = f"{col}_base"
                pert_col = f"{col}_pert"
                if base_col not in merged.columns or pert_col not in merged.columns:
                    continue
                valid = merged[[base_col, pert_col]].dropna()
                if len(valid) < 5:
                    continue
                mi = _mutual_information(valid[base_col].to_numpy(), valid[pert_col].to_numpy())
                rows.append({
                    "perturbation": pert,
                    "model": model_name,
                    "question": q,
                    "MI": mi,
                    "n": len(valid),
                })

    return pd.DataFrame(rows)


def compute_pc(
    df: pd.DataFrame,
    all_models: Dict[str, Dict[str, str]],
    baseline_label: str = "baseline",
) -> pd.DataFrame:
    """
    Percent change from baseline to perturbation.

    PC is reported as signed mean difference: perturbed - baseline.
    flip_rate is reported as the proportion of changed recommendations.
    """
    baseline_df, baseline_label = _get_baseline(df, baseline_label)
    perturbations = [p for p in sorted(df["perturbation"].unique()) if p != baseline_label]

    rows = []
    for pert in perturbations:
        pert_df = df[df["perturbation"] == pert].copy()
        merged = pd.merge(
            baseline_df,
            pert_df,
            on="context_id",
            suffixes=("_base", "_pert"),
            how="inner",
        )

        for model_name, cols in all_models.items():
            for q in QUESTIONS:
                col = cols[q]
                base_col = f"{col}_base"
                pert_col = f"{col}_pert"
                if base_col not in merged.columns or pert_col not in merged.columns:
                    continue
                valid = merged[[base_col, pert_col]].dropna().astype(float)
                if len(valid) < 5:
                    continue
                diff = valid[pert_col].to_numpy() - valid[base_col].to_numpy()
                pc = diff.mean()
                flip_rate = (diff != 0).mean()

                if np.std(diff) > 0:
                    _, p_val = stats.ttest_1samp(diff, 0)
                    p_val_corrected = min(float(p_val) * BONFERRONI_FACTOR, 1.0)
                else:
                    p_val_corrected = 1.0

                rows.append({
                    "perturbation": pert,
                    "model": model_name,
                    "question": q,
                    "PC": pc,
                    "flip_rate": flip_rate,
                    "p_value_corrected": p_val_corrected,
                    "significant": p_val_corrected < ALPHA,
                    "n": len(valid),
                })

    return pd.DataFrame(rows)


def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's Kappa for two binary raters."""
    a = a.astype(int)
    b = b.astype(int)
    if len(a) == 0:
        return np.nan
    p_o = (a == b).mean()
    p_a = a.mean()
    p_b = b.mean()
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def compute_pairwise_kappa(df: pd.DataFrame, all_models: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Cohen's Kappa between model pairs, per perturbation and question."""
    rows = []
    model_names = list(all_models.keys())

    for pert in sorted(df["perturbation"].unique()):
        subset = df[df["perturbation"] == pert]
        for q in QUESTIONS:
            for m1, m2 in combinations(model_names, 2):
                col1 = all_models[m1][q]
                col2 = all_models[m2][q]
                if col1 not in subset.columns or col2 not in subset.columns:
                    continue
                valid = subset[[col1, col2]].dropna()
                if len(valid) < 5:
                    continue
                a = valid[col1].to_numpy()
                b = valid[col2].to_numpy()
                rows.append({
                    "perturbation": pert,
                    "question": q,
                    "model_1": m1,
                    "model_2": m2,
                    "cohens_kappa": _cohens_kappa(a, b),
                    "agreement_pct": (a == b).mean() * 100,
                    "n": len(valid),
                })

    return pd.DataFrame(rows)


def extract_vs_clinician(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Keep Cohen's Kappa comparisons where one rater is clinician consensus."""
    if pairwise.empty:
        return pd.DataFrame()

    df = pairwise[
        (pairwise["model_1"] == "clinician") | (pairwise["model_2"] == "clinician")
    ].copy()

    if df.empty:
        return df

    df["model"] = df.apply(
        lambda r: r["model_1"] if r["model_2"] == "clinician" else r["model_2"],
        axis=1,
    )
    return df[df["model"].isin(KAPPA_MODELS)]


# ---------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------

def _ensure_plot_dir(output_dir: str) -> str:
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def _plot_grouped_bars(ax, pivot: pd.DataFrame, ylabel: str, title: str):
    labels = list(pivot.index)
    models = list(pivot.columns)
    x = np.arange(len(labels))
    width = 0.8 / max(len(models), 1)

    for i, model in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * width, pivot[model].values, width, label=model)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, linewidth=0.8)
    ax.legend(loc="best")


def plot_metric_by_perturbation(
    metric_df: pd.DataFrame,
    metric_col: str,
    output_dir: str,
    filename: str,
    title: str,
    ylabel: str,
    models: List[str] | None = None,
):
    """
    Create one figure with one row per perturbation.
    Within each row: x-axis is manage/visit/resource, bars are models.
    """
    if metric_df.empty:
        print(f"Skipping {filename}: metric dataframe is empty.")
        return

    models = models or FOCUS_MODELS
    df = metric_df[metric_df["model"].isin(models)].copy()
    if df.empty:
        print(f"Skipping {filename}: no requested models found.")
        return

    perturbations = sorted(df["perturbation"].unique())
    nrows = len(perturbations)
    fig_height = max(3.0 * nrows, 4.0)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(9, fig_height), squeeze=False)

    for idx, pert in enumerate(perturbations):
        ax = axes[idx][0]
        sub = df[df["perturbation"] == pert]
        pivot = sub.pivot_table(index="question", columns="model", values=metric_col, aggfunc="mean")
        pivot = pivot.reindex(index=QUESTIONS)
        pivot = pivot[[m for m in models if m in pivot.columns]]
        _plot_grouped_bars(ax, pivot, ylabel, f"{pert}")

    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    path = os.path.join(_ensure_plot_dir(output_dir), filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def plot_metric_averaged_rows(
    metric_df: pd.DataFrame,
    metric_col: str,
    output_dir: str,
    filename: str,
    title: str,
    ylabel: str,
    models: List[str] | None = None,
):
    """
    Create a compact grouped bar chart with perturbations as rows/x labels,
    averaged across manage, visit, and resource.
    """
    if metric_df.empty:
        return
    models = models or FOCUS_MODELS
    df = metric_df[metric_df["model"].isin(models)].copy()
    avg = df.groupby(["perturbation", "model"], as_index=False)[metric_col].mean()
    pivot = avg.pivot(index="perturbation", columns="model", values=metric_col)
    pivot = pivot[[m for m in models if m in pivot.columns]]

    fig, ax = plt.subplots(figsize=(9, 5))
    _plot_grouped_bars(ax, pivot, ylabel, title)
    ax.set_xlabel("Perturbation")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    path = os.path.join(_ensure_plot_dir(output_dir), filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")


def plot_kappa_vs_human(vs_clinician: pd.DataFrame, output_dir: str):
    """
    One figure with one row per perturbation.
    Within each row: x-axis is manage/visit/resource, bars are LLaMA vs MedPathAgent.
    """
    if vs_clinician.empty:
        print("Skipping kappa plot: no human-vs-model kappa rows found.")
        return

    plot_metric_by_perturbation(
        vs_clinician.rename(columns={"cohens_kappa": "kappa"}),
        metric_col="kappa",
        output_dir=output_dir,
        filename="kappa_vs_human_by_perturbation.png",
        title="Cohen's Kappa vs Clinician Consensus by Perturbation",
        ylabel="Cohen's Kappa",
        models=KAPPA_MODELS,
    )

    plot_metric_averaged_rows(
        vs_clinician.rename(columns={"cohens_kappa": "kappa"}),
        metric_col="kappa",
        output_dir=output_dir,
        filename="kappa_vs_human_average_by_perturbation.png",
        title="Average Cohen's Kappa vs Clinician Consensus",
        ylabel="Mean Cohen's Kappa",
        models=KAPPA_MODELS,
    )


# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

def build_summary(atr_df: pd.DataFrame, mi_df: pd.DataFrame, pc_df: pd.DataFrame) -> pd.DataFrame:
    """Build a wide summary table for the focus models."""
    parts = []

    if not atr_df.empty:
        tmp = atr_df[atr_df["model"].isin(FOCUS_MODELS)]
        pivot = tmp.pivot_table(index=["perturbation", "question"], columns="model", values="ATR")
        pivot.columns = [f"ATR_{c}" for c in pivot.columns]
        parts.append(pivot)

    if not mi_df.empty:
        tmp = mi_df[mi_df["model"].isin(FOCUS_MODELS)]
        pivot = tmp.pivot_table(index=["perturbation", "question"], columns="model", values="MI")
        pivot.columns = [f"MI_{c}" for c in pivot.columns]
        parts.append(pivot)

    if not pc_df.empty:
        tmp = pc_df[pc_df["model"].isin(FOCUS_MODELS)]
        pivot = tmp.pivot_table(index=["perturbation", "question"], columns="model", values="PC")
        pivot.columns = [f"PC_{c}" for c in pivot.columns]
        parts.append(pivot)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=1).round(4)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation pipeline")
    parser.add_argument("--kg_results", required=True, help="Path to MedPathAgent/KG results CSV")
    parser.add_argument("--orig_dataset", required=True, help="Path to original MedPerturb dataset CSV")
    parser.add_argument("--output_dir", default="eval_output", help="Output directory")
    parser.add_argument("--baseline_label", default="baseline", help="Baseline perturbation label")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    _ensure_plot_dir(args.output_dir)

    all_models = {}
    all_models.update(ORIG_MODELS)
    all_models.update(KG_MODEL)
    all_models.update(CLINICIAN)

    print("=" * 70)
    print("LOADING AND MATCHING DATA")
    print("=" * 70)
    df = load_and_merge(args.kg_results, args.orig_dataset)

    print("\n" + "=" * 70)
    print("COMPUTING ATR")
    print("=" * 70)
    atr_df = compute_atr(df, all_models)
    atr_path = os.path.join(args.output_dir, "atr_all.csv")
    atr_df.to_csv(atr_path, index=False)
    print(f"Saved: {atr_path}")

    print("\n" + "=" * 70)
    print("COMPUTING MI")
    print("=" * 70)
    mi_df = compute_mi(df, all_models, baseline_label=args.baseline_label)
    mi_path = os.path.join(args.output_dir, "mi_all.csv")
    mi_df.to_csv(mi_path, index=False)
    print(f"Saved: {mi_path}")

    print("\n" + "=" * 70)
    print("COMPUTING PC")
    print("=" * 70)
    pc_df = compute_pc(df, all_models, baseline_label=args.baseline_label)
    pc_path = os.path.join(args.output_dir, "pc_all.csv")
    pc_df.to_csv(pc_path, index=False)
    print(f"Saved: {pc_path}")

    print("\n" + "=" * 70)
    print("COMPUTING COHEN'S KAPPA VS CLINICIAN")
    print("=" * 70)
    pairwise = compute_pairwise_kappa(df, all_models)
    pairwise_path = os.path.join(args.output_dir, "pairwise_kappa.csv")
    pairwise.to_csv(pairwise_path, index=False)
    print(f"Saved: {pairwise_path}")

    vs_clinician = extract_vs_clinician(pairwise)
    kappa_path = os.path.join(args.output_dir, "kappa_vs_human_consensus.csv")
    vs_clinician.to_csv(kappa_path, index=False)
    print(f"Saved: {kappa_path}")

    summary = build_summary(atr_df, mi_df, pc_df)
    summary_path = os.path.join(args.output_dir, "summary_comparison.csv")
    summary.to_csv(summary_path)
    print(f"Saved: {summary_path}")

    print("\n" + "=" * 70)
    print("CREATING PLOTS")
    print("=" * 70)

    # Full figures: one row per perturbation and bars for each model/question.
    plot_metric_by_perturbation(
        atr_df,
        metric_col="ATR",
        output_dir=args.output_dir,
        filename="atr_comparison_by_perturbation.png",
        title="ATR Comparison: LLaMA vs MedPathAgent vs Clinician",
        ylabel="Average Treatment Ratio",
        models=FOCUS_MODELS,
    )

    plot_metric_by_perturbation(
        mi_df,
        metric_col="MI",
        output_dir=args.output_dir,
        filename="mi_comparison_by_perturbation.png",
        title="MI Robustness Comparison: LLaMA vs MedPathAgent vs Clinician",
        ylabel="Mutual Information",
        models=FOCUS_MODELS,
    )

    plot_metric_by_perturbation(
        pc_df,
        metric_col="flip_rate",
        output_dir=args.output_dir,
        filename="pc_comparison_by_perturbation.png",
        title="Percent Changed Recommendations: LLaMA vs MedPathAgent vs Clinician",
        ylabel="Flip Rate / Percent Changed",
        models=FOCUS_MODELS,
    )

    plot_kappa_vs_human(vs_clinician, args.output_dir)

    # Compact averaged figures across manage/visit/resource.
    plot_metric_averaged_rows(
        atr_df,
        metric_col="ATR",
        output_dir=args.output_dir,
        filename="atr_average_by_perturbation.png",
        title="Average ATR Across Tasks",
        ylabel="Mean ATR",
        models=FOCUS_MODELS,
    )

    plot_metric_averaged_rows(
        mi_df,
        metric_col="MI",
        output_dir=args.output_dir,
        filename="mi_average_by_perturbation.png",
        title="Average MI Across Tasks",
        ylabel="Mean MI",
        models=FOCUS_MODELS,
    )

    plot_metric_averaged_rows(
        pc_df,
        metric_col="flip_rate",
        output_dir=args.output_dir,
        filename="pc_average_by_perturbation.png",
        title="Average Percent Changed Recommendations Across Tasks",
        ylabel="Mean Flip Rate",
        models=FOCUS_MODELS,
    )

    print("\nDone.")
    print(f"CSV files saved in:   {args.output_dir}")
    print(f"Plot images saved in: {os.path.join(args.output_dir, 'plots')}")
    print("\nInterpretation:")
    print("  - ATR: treatment recommendation tendency; direction depends on task.")
    print("  - MI: higher means more robust/stable under perturbation.")
    print("  - PC/flip_rate: lower means fewer changed recommendations.")
    print("  - Cohen's Kappa: higher means stronger agreement with clinician consensus.")


if __name__ == "__main__":
    main()
