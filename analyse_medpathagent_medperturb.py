"""
Robustness Evaluation Pipeline
Compares KG-augmented method (MedPathAgent) vs baseline LLMs
using MedPerturb metrics: ATR, MI, PC, and Fleiss' Kappa

Usage:
    python robustness_eval.py \
        --kg_results results.csv \
        --orig_dataset orig_dataset.csv \
        --output_dir eval_output
"""

import argparse
import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

QUESTIONS = ["manage", "visit", "resource"]

# Models in the orig dataset (adjust if your column names differ)
ORIG_MODELS = {
    "gpt-4o":    {"manage": "gpt-4o_manage",    "visit": "gpt-4o_visit",    "resource": "gpt-4o_resource"},
    "llama":     {"manage": "llama_manage",      "visit": "llama_visit",     "resource": "llama_resource"},
    "deepseek":  {"manage": "deepseek_manage",   "visit": "deepseek_visit",  "resource": "deepseek_resource"},
    "qwen":      {"manage": "qwen_manage",       "visit": "qwen_visit",      "resource": "qwen_resource"},
    "medgemma":  {"manage": "medgemma_manage",    "visit": "medgemma_visit",  "resource": "medgemma_resource"},
}

# KG model columns from results.csv
KG_MODEL = {
    "medpathagent": {
        "manage": "medpathagent_llama70b3_3_manage",
        "visit":  "medpathagent_llama70b3_3_visit",
        "resource": "medpathagent_llama70b3_3_resource",
    }
}

# Clinician columns (from orig dataset)
CLINICIAN = {
    "clinician": {
        "manage": "clinician_consensus_manage",
        "visit":  "clinician_consensus_visit",
        "resource": "clinician_consensus_resource",
    }
}

# Gold standard columns
GOLD = {
    "manage": "gold_standard_manage",
    "visit":  "gold_standard_visit",
    "resource": "gold_standard_resource",
}

ALPHA = 0.01  # significance level
BONFERRONI_FACTOR = 5  # following MedPerturb


# ─────────────────────────────────────────────────────────────────────
# DATA LOADING & MERGING
# ─────────────────────────────────────────────────────────────────────

def load_and_merge(kg_path: str, orig_path: str) -> pd.DataFrame:
    """Load both CSVs and merge on (context_id, perturbation, dataset)."""
    kg = pd.read_csv(kg_path)
    orig = pd.read_csv(orig_path)

    # Standardise perturbation labels
    for df in [kg, orig]:
        df["perturbation"] = df["perturbation"].str.strip().str.lower()
        df["context_id"] = df["context_id"].astype(str).str.strip()

    # Merge
    merged = pd.merge(
        orig, kg,
        on=["context_id", "perturbation", "dataset"],
        how="inner",
        suffixes=("", "_kg"),
    )

    print(f"Orig rows: {len(orig)}, KG rows: {len(kg)}, Merged: {len(merged)}")
    print(f"Perturbation types: {sorted(merged['perturbation'].unique())}")
    print(f"Datasets: {sorted(merged['dataset'].unique())}")

    # Binarise all answer columns (handle yes/no strings and numeric)
    answer_cols = []
    for model_cols in list(ORIG_MODELS.values()) + list(KG_MODEL.values()) + list(CLINICIAN.values()):
        answer_cols.extend(model_cols.values())
    answer_cols.extend(GOLD.values())

    for col in answer_cols:
        if col in merged.columns:
            merged[col] = binarise(merged[col])

    return merged


def binarise(series: pd.Series) -> pd.Series:
    """Convert yes/no/1/0/True/False to binary int."""
    s = series.copy()
    if s.dtype == object:
        s = s.str.strip().str.lower()
        s = s.map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    return pd.to_numeric(s, errors="coerce").astype("Int64")


# ─────────────────────────────────────────────────────────────────────
# METRIC 1: AVERAGE TREATMENT RATIO (ATR)
# ─────────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, all_models: dict) -> pd.DataFrame:
    """
    ATR = mean of binary treatment decisions per model, per perturbation, per question.
    """
    rows = []
    for pert in sorted(df["perturbation"].unique()):
        subset = df[df["perturbation"] == pert]
        for model_name, cols in all_models.items():
            for q in QUESTIONS:
                col = cols[q]
                if col not in subset.columns:
                    continue
                vals = subset[col].dropna()
                if len(vals) == 0:
                    continue
                atr = vals.mean()
                rows.append({
                    "perturbation": pert,
                    "model": model_name,
                    "question": q,
                    "ATR": round(atr, 4),
                    "n": len(vals),
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# METRIC 2: MUTUAL INFORMATION (MI)
# ─────────────────────────────────────────────────────────────────────

def compute_mi(df: pd.DataFrame, all_models: dict, baseline_label: str = "baseline") -> pd.DataFrame:
    """
    MI between baseline and perturbed decisions for each model and question.
    Higher MI = more stable decisions under perturbation.
    """
    baseline_df = df[df["perturbation"] == baseline_label].copy()

    if len(baseline_df) == 0:
        # Try "vignette" as baseline (format perturbation experiments)
        baseline_label = "vignette"
        baseline_df = df[df["perturbation"] == baseline_label].copy()

    if len(baseline_df) == 0:
        print("WARNING: No baseline rows found. Check perturbation labels.")
        return pd.DataFrame()

    perturbations = [p for p in df["perturbation"].unique() if p != baseline_label]
    rows = []

    for pert in sorted(perturbations):
        pert_df = df[df["perturbation"] == pert].copy()

        # Align on context_id
        merged = pd.merge(
            baseline_df, pert_df,
            on="context_id",
            suffixes=("_base", "_pert"),
            how="inner",
        )

        for model_name, cols in all_models.items():
            for q in QUESTIONS:
                col = cols[q]
                base_col = f"{col}_base" if f"{col}_base" in merged.columns else col
                pert_col = f"{col}_pert" if f"{col}_pert" in merged.columns else col

                if base_col not in merged.columns or pert_col not in merged.columns:
                    continue

                base_vals = merged[base_col].dropna()
                pert_vals = merged[pert_col].dropna()
                valid = merged[[base_col, pert_col]].dropna()

                if len(valid) < 10:
                    continue

                mi = _mutual_information(valid[base_col].values, valid[pert_col].values)

                rows.append({
                    "perturbation": pert,
                    "model": model_name,
                    "question": q,
                    "MI": round(mi, 6),
                    "n": len(valid),
                })

    return pd.DataFrame(rows)


def _mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Compute MI for two binary arrays."""
    # Build joint distribution
    joint = np.zeros((2, 2))
    for a, b in zip(x.astype(int), y.astype(int)):
        if 0 <= a <= 1 and 0 <= b <= 1:
            joint[a, b] += 1

    joint = joint / joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0
    for i in range(2):
        for j in range(2):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    return mi


# ─────────────────────────────────────────────────────────────────────
# METRIC 3: PERCENT CHANGE (PC)
# ─────────────────────────────────────────────────────────────────────

def compute_pc(df: pd.DataFrame, all_models: dict, baseline_label: str = "baseline") -> pd.DataFrame:
    """
    PC = mean(perturbed - baseline) across all vignettes.
    Positive = perturbation increases yes-rate. Closer to 0 = more robust.
    """
    baseline_df = df[df["perturbation"] == baseline_label].copy()

    if len(baseline_df) == 0:
        baseline_label = "vignette"
        baseline_df = df[df["perturbation"] == baseline_label].copy()

    if len(baseline_df) == 0:
        print("WARNING: No baseline rows found.")
        return pd.DataFrame()

    perturbations = [p for p in df["perturbation"].unique() if p != baseline_label]
    rows = []

    for pert in sorted(perturbations):
        pert_df = df[df["perturbation"] == pert].copy()

        merged = pd.merge(
            baseline_df, pert_df,
            on="context_id",
            suffixes=("_base", "_pert"),
            how="inner",
        )

        for model_name, cols in all_models.items():
            for q in QUESTIONS:
                col = cols[q]
                base_col = f"{col}_base" if f"{col}_base" in merged.columns else col
                pert_col = f"{col}_pert" if f"{col}_pert" in merged.columns else col

                if base_col not in merged.columns or pert_col not in merged.columns:
                    continue

                valid = merged[[base_col, pert_col]].dropna()
                if len(valid) < 5:
                    continue

                diff = valid[pert_col].values - valid[base_col].values
                pc = diff.mean()
                flip_rate = (diff != 0).mean()

                # Paired t-test for significance
                if np.std(diff) > 0:
                    t_stat, p_val = stats.ttest_1samp(diff, 0)
                    p_val_corrected = min(p_val * BONFERRONI_FACTOR, 1.0)
                else:
                    t_stat, p_val_corrected = 0.0, 1.0

                rows.append({
                    "perturbation": pert,
                    "model": model_name,
                    "question": q,
                    "PC": round(pc, 4),
                    "flip_rate": round(flip_rate, 4),
                    "p_value_corrected": round(p_val_corrected, 6),
                    "significant": p_val_corrected < ALPHA,
                    "n": len(valid),
                })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# METRIC 4: FLEISS' KAPPA
# ─────────────────────────────────────────────────────────────────────

def compute_fleiss_kappa(df: pd.DataFrame, all_models: dict, include_clinician: bool = True) -> pd.DataFrame:
    """
    Fleiss' Kappa for inter-rater agreement across models (and optionally clinicians).
    Computed per perturbation and per question.
    """
    rows = []

    for pert in sorted(df["perturbation"].unique()):
        subset = df[df["perturbation"] == pert]

        for q in QUESTIONS:
            # Gather ratings from all models
            raters = {}
            for model_name, cols in all_models.items():
                col = cols[q]
                if col in subset.columns:
                    raters[model_name] = subset[col].values

            if len(raters) < 2:
                continue

            # Build ratings matrix: rows=subjects, cols=raters
            rater_names = list(raters.keys())
            ratings_matrix = np.column_stack([raters[r] for r in rater_names])

            # Drop rows with NaN
            mask = ~np.isnan(ratings_matrix.astype(float)).any(axis=1)
            ratings_matrix = ratings_matrix[mask].astype(int)

            if len(ratings_matrix) < 5:
                continue

            kappa = _fleiss_kappa(ratings_matrix, n_categories=2)

            rows.append({
                "perturbation": pert,
                "question": q,
                "fleiss_kappa": round(kappa, 4),
                "n_raters": len(rater_names),
                "raters": ", ".join(rater_names),
                "n_subjects": len(ratings_matrix),
            })

    return pd.DataFrame(rows)


def _fleiss_kappa(ratings: np.ndarray, n_categories: int = 2) -> float:
    """
    Compute Fleiss' Kappa from a ratings matrix (n_subjects x n_raters).
    Each cell is a category label (0 or 1 for binary).
    """
    n_subjects, n_raters = ratings.shape

    # Build category count matrix
    counts = np.zeros((n_subjects, n_categories))
    for cat in range(n_categories):
        counts[:, cat] = (ratings == cat).sum(axis=1)

    # Proportion of agreements per subject
    p_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(p_i)

    # Expected agreement by chance
    p_j = counts.sum(axis=0) / (n_subjects * n_raters)
    P_e = np.sum(p_j ** 2)

    if P_e == 1.0:
        return 1.0

    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


# ─────────────────────────────────────────────────────────────────────
# PAIRWISE MODEL COMPARISON (Cohen's Kappa)
# ─────────────────────────────────────────────────────────────────────

def compute_pairwise_kappa(df: pd.DataFrame, all_models: dict) -> pd.DataFrame:
    """Cohen's Kappa between each pair of models, per perturbation and question."""
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

                a = valid[col1].astype(int).values
                b = valid[col2].astype(int).values

                kappa = _cohens_kappa(a, b)

                rows.append({
                    "perturbation": pert,
                    "question": q,
                    "model_1": m1,
                    "model_2": m2,
                    "cohens_kappa": round(kappa, 4),
                    "agreement_pct": round((a == b).mean() * 100, 1),
                    "n": len(valid),
                })

    return pd.DataFrame(rows)


def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's Kappa for two raters on binary data."""
    n = len(a)
    if n == 0:
        return 0.0
    p_o = (a == b).mean()  # observed agreement
    p_a = a.mean()
    p_b = b.mean()
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)  # expected agreement

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


# ─────────────────────────────────────────────────────────────────────
# MI SIGNIFICANCE TESTING (Mann-Whitney U)
# ─────────────────────────────────────────────────────────────────────

def compare_mi_significance(mi_df: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    """
    For each perturbation and question, test whether model_a MI differs
    from model_b MI using Mann-Whitney U (following MedPerturb).
    
    NOTE: This requires per-vignette MI contributions. For a single MI value
    per condition, we instead report the difference directly.
    """
    rows = []
    if mi_df.empty:
        return pd.DataFrame()

    for pert in mi_df["perturbation"].unique():
        for q in QUESTIONS:
            mi_a = mi_df[(mi_df["model"] == model_a) &
                         (mi_df["perturbation"] == pert) &
                         (mi_df["question"] == q)]["MI"]
            mi_b = mi_df[(mi_df["model"] == model_b) &
                         (mi_df["perturbation"] == pert) &
                         (mi_df["question"] == q)]["MI"]

            if len(mi_a) == 0 or len(mi_b) == 0:
                continue

            rows.append({
                "perturbation": pert,
                "question": q,
                f"MI_{model_a}": mi_a.values[0],
                f"MI_{model_b}": mi_b.values[0],
                "MI_diff": round(mi_a.values[0] - mi_b.values[0], 6),
                "more_robust": model_a if mi_a.values[0] > mi_b.values[0] else model_b,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────

def build_summary(atr_df, mi_df, pc_df, focus_models=None):
    """Build a single comparison table."""
    if focus_models:
        atr_df = atr_df[atr_df["model"].isin(focus_models)]
        mi_df = mi_df[mi_df["model"].isin(focus_models)] if not mi_df.empty else mi_df
        pc_df = pc_df[pc_df["model"].isin(focus_models)] if not pc_df.empty else pc_df

    summary_parts = []

    if not atr_df.empty:
        atr_pivot = atr_df.pivot_table(
            index=["perturbation", "question"],
            columns="model",
            values="ATR",
        )
        atr_pivot.columns = [f"ATR_{c}" for c in atr_pivot.columns]
        summary_parts.append(atr_pivot)

    if not mi_df.empty:
        mi_pivot = mi_df.pivot_table(
            index=["perturbation", "question"],
            columns="model",
            values="MI",
        )
        mi_pivot.columns = [f"MI_{c}" for c in mi_pivot.columns]
        summary_parts.append(mi_pivot)

    if not pc_df.empty:
        pc_pivot = pc_df.pivot_table(
            index=["perturbation", "question"],
            columns="model",
            values="PC",
        )
        pc_pivot.columns = [f"PC_{c}" for c in pc_pivot.columns]
        summary_parts.append(pc_pivot)

    if summary_parts:
        summary = pd.concat(summary_parts, axis=1)
        return summary.round(4)
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def plot_fleiss_vs_clinician(fleiss_df: pd.DataFrame, output_dir: str):
    """
    Grouped bar chart: Fleiss' Kappa vs clinician for each model,
    grouped by (perturbation, question).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # Filter to just the models we want to compare
    plot_models = ["medpathagent", "llama"]
    model_labels = {"medpathagent": "MedPathAgent (KG)", "llama": "Llama-70B"}
    model_colors = {"medpathagent": "#2D6A4F", "llama": "#E76F51"}

    plot_df = fleiss_df[fleiss_df["model"].isin(plot_models)].copy()
    if plot_df.empty:
        print("WARNING: No data for plotting. Check model names.")
        return

    # Create group labels
    plot_df["group"] = plot_df["perturbation"] + "\n" + plot_df["question"].str.upper()

    groups = sorted(plot_df["group"].unique())
    n_groups = len(groups)
    n_models = len(plot_models)

    bar_width = 0.35
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.2), 6))

    for i, model in enumerate(plot_models):
        model_data = plot_df[plot_df["model"] == model]
        vals = []
        for g in groups:
            row = model_data[model_data["group"] == g]
            vals.append(row["fleiss_kappa_vs_clinician"].values[0] if len(row) > 0 else 0)

        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, vals,
            width=bar_width,
            label=model_labels.get(model, model),
            color=model_colors.get(model, f"C{i}"),
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
        )

        # Value labels on bars
        for bar, v in zip(bars, vals):
            y_pos = bar.get_height()
            va = "bottom" if y_pos >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos + (0.01 if y_pos >= 0 else -0.01),
                f"{v:.2f}",
                ha="center", va=va,
                fontsize=8, fontweight="600",
                color="#333",
            )

    # Reference lines
    ax.axhline(y=0, color="#999", linewidth=0.8, linestyle="-")
    ax.axhline(y=0.2, color="#ccc", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.axhline(y=0.4, color="#ccc", linewidth=0.6, linestyle="--", alpha=0.7)
    ax.axhline(y=0.6, color="#ccc", linewidth=0.6, linestyle="--", alpha=0.7)

    # Kappa interpretation bands (subtle background)
    ax.axhspan(-1, 0, alpha=0.03, color="red")
    ax.axhspan(0, 0.2, alpha=0.03, color="orange")
    ax.axhspan(0.2, 0.4, alpha=0.03, color="yellow")
    ax.axhspan(0.4, 0.6, alpha=0.03, color="lightgreen")
    ax.axhspan(0.6, 1.0, alpha=0.03, color="green")

    # Interpretation labels on right side
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    interp_positions = {0.1: "Slight", 0.3: "Fair", 0.5: "Moderate", 0.7: "Substantial"}
    for pos, label in interp_positions.items():
        if ax.get_ylim()[0] <= pos <= ax.get_ylim()[1]:
            ax2.annotate(
                label, xy=(1.02, pos), xycoords=("axes fraction", "data"),
                fontsize=7, color="#888", va="center",
            )
    ax2.set_yticks([])

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("Fleiss' κ (vs Clinician)", fontsize=11, fontweight="600")
    ax.set_title(
        "Agreement with Clinician: MedPathAgent (KG) vs Llama-70B",
        fontsize=13, fontweight="700", pad=15,
    )
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, "fleiss_vs_clinician_plot.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nPlot saved: {plot_path}")


def compute_model_vs_clinician_fleiss(df: pd.DataFrame, model_dict: dict, clinician_dict: dict) -> pd.DataFrame:
    """
    Compute Fleiss' Kappa between a single model and clinician,
    per perturbation and question. This gives a 2-rater Fleiss (equivalent to Cohen's).
    """
    rows = []
    for model_name, cols in model_dict.items():
        paired = {model_name: cols}
        paired.update(clinician_dict)

        for pert in sorted(df["perturbation"].unique()):
            subset = df[df["perturbation"] == pert]
            for q in QUESTIONS:
                model_col = cols[q]
                clin_col = list(clinician_dict.values())[0][q]

                if model_col not in subset.columns or clin_col not in subset.columns:
                    continue

                valid = subset[[model_col, clin_col]].dropna()
                if len(valid) < 5:
                    continue

                ratings = np.column_stack([
                    valid[model_col].astype(int).values,
                    valid[clin_col].astype(int).values,
                ])
                kappa = _fleiss_kappa(ratings, n_categories=2)

                rows.append({
                    "perturbation": pert,
                    "model": model_name,
                    "question": q,
                    "fleiss_kappa_vs_clinician": round(kappa, 4),
                    "n": len(valid),
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation pipeline")
    parser.add_argument("--kg_results", required=True, help="Path to KG model results.csv")
    parser.add_argument("--orig_dataset", required=True, help="Path to original dataset CSV")
    parser.add_argument("--output_dir", default="eval_output", help="Output directory")
    parser.add_argument("--baseline_label", default="baseline",
                        help="Perturbation label for baseline (e.g. 'baseline' or 'vignette')")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ──
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = load_and_merge(args.kg_results, args.orig_dataset)

    # ── Combine all model dicts ──
    all_models = {}
    all_models.update(ORIG_MODELS)
    all_models.update(KG_MODEL)

    # All models + clinician (for Fleiss comparison)
    all_with_clinician = {}
    all_with_clinician.update(all_models)
    all_with_clinician.update(CLINICIAN)

    # ── Focus comparison models ──
    focus = ["medpathagent", "llama", "gpt-4o"]

    # ── ATR ──
    print("\n" + "=" * 60)
    print("COMPUTING ATR")
    print("=" * 60)
    atr_df = compute_atr(df, all_with_clinician)
    atr_df.to_csv(os.path.join(args.output_dir, "atr_all.csv"), index=False)

    focus_atr = atr_df[atr_df["model"].isin(focus + ["clinician"])]
    print(focus_atr.to_string(index=False))

    # ── MI ──
    print("\n" + "=" * 60)
    print("COMPUTING MUTUAL INFORMATION")
    print("=" * 60)
    mi_df = compute_mi(df, all_with_clinician, baseline_label=args.baseline_label)
    if not mi_df.empty:
        mi_df.to_csv(os.path.join(args.output_dir, "mi_all.csv"), index=False)
        focus_mi = mi_df[mi_df["model"].isin(focus + ["clinician"])]
        print(focus_mi.to_string(index=False))

        # MI comparison: KG vs Llama, KG vs GPT
        for comparison_model in ["llama", "gpt-4o"]:
            mi_comp = compare_mi_significance(mi_df, "medpathagent", comparison_model)
            if not mi_comp.empty:
                print(f"\nMI comparison: medpathagent vs {comparison_model}")
                print(mi_comp.to_string(index=False))
                mi_comp.to_csv(
                    os.path.join(args.output_dir, f"mi_comparison_kg_vs_{comparison_model}.csv"),
                    index=False,
                )

    # ── PC ──
    print("\n" + "=" * 60)
    print("COMPUTING PERCENT CHANGE")
    print("=" * 60)
    pc_df = compute_pc(df, all_with_clinician, baseline_label=args.baseline_label)
    if not pc_df.empty:
        pc_df.to_csv(os.path.join(args.output_dir, "pc_all.csv"), index=False)
        focus_pc = pc_df[pc_df["model"].isin(focus + ["clinician"])]
        print(focus_pc.to_string(index=False))

    # ── Fleiss' Kappa: all models vs clinicians ──
    print("\n" + "=" * 60)
    print("COMPUTING FLEISS' KAPPA (all models + clinician)")
    print("=" * 60)
    fleiss_all = compute_fleiss_kappa(df, all_with_clinician)
    if not fleiss_all.empty:
        fleiss_all.to_csv(os.path.join(args.output_dir, "fleiss_kappa_all.csv"), index=False)
        print(fleiss_all.to_string(index=False))

    # Fleiss' Kappa: just focus models + clinician
    focus_models_dict = {k: v for k, v in all_with_clinician.items() if k in focus + ["clinician"]}
    fleiss_focus = compute_fleiss_kappa(df, focus_models_dict)
    if not fleiss_focus.empty:
        fleiss_focus.to_csv(os.path.join(args.output_dir, "fleiss_kappa_focus.csv"), index=False)
        print("\nFocus models + clinician:")
        print(fleiss_focus.to_string(index=False))

    # ── Pairwise Cohen's Kappa ──
    print("\n" + "=" * 60)
    print("COMPUTING PAIRWISE COHEN'S KAPPA")
    print("=" * 60)
    pairwise = compute_pairwise_kappa(df, all_with_clinician)
    if not pairwise.empty:
        pairwise.to_csv(os.path.join(args.output_dir, "pairwise_kappa.csv"), index=False)
        focus_pairs = pairwise[
            (pairwise["model_1"].isin(focus + ["clinician"])) &
            (pairwise["model_2"].isin(focus + ["clinician"]))
        ]
        print(focus_pairs.to_string(index=False))

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("SUMMARY: medpathagent vs llama vs gpt-4o")
    print("=" * 60)
    summary = build_summary(atr_df, mi_df, pc_df, focus_models=focus)
    if not summary.empty:
        summary.to_csv(os.path.join(args.output_dir, "summary_comparison.csv"))
        print(summary.to_string())

    # ── Model vs Clinician Fleiss' Kappa + Plot ──
    print("\n" + "=" * 60)
    print("COMPUTING MODEL vs CLINICIAN FLEISS' KAPPA")
    print("=" * 60)

    # Compute for focus models only
    focus_model_dict = {k: v for k, v in all_models.items() if k in focus}
    fleiss_vs_clin = compute_model_vs_clinician_fleiss(df, focus_model_dict, CLINICIAN)

    if not fleiss_vs_clin.empty:
        fleiss_vs_clin.to_csv(
            os.path.join(args.output_dir, "fleiss_vs_clinician.csv"), index=False
        )
        print(fleiss_vs_clin.to_string(index=False))

        # ── Generate comparison plot ──
        plot_fleiss_vs_clinician(fleiss_vs_clin, args.output_dir)

    print(f"\nAll results saved to: {args.output_dir}/")
    print("Files: atr_all.csv, mi_all.csv, pc_all.csv, fleiss_kappa_all.csv,")
    print("       fleiss_kappa_focus.csv, pairwise_kappa.csv, summary_comparison.csv,")
    print("       fleiss_vs_clinician.csv, fleiss_vs_clinician_plot.png")


if __name__ == "__main__":
    main()