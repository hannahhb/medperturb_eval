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
    "llama":     {"manage": "llama_manage",      "visit": "llama_visit",     "resource": "llama_resource"},
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

def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation pipeline")
    parser.add_argument("--kg_results", default= "/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/output/medpathagent_llama70b3_3/oncqa_askdocs/run_0/results.csv", help="Path to KG model results.csv")
    parser.add_argument("--orig_dataset", default="/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/medperturb_data.csv", help="Path to original dataset CSV")
    parser.add_argument("--output_dir", default="output", help="Output directory")
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

    # ── Focus comparison: llama vs medpathagent vs human consensus ──
    focus = ["medpathagent", "llama"]

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

        # MI comparison: medpathagent vs llama, and each vs human consensus
        mi_comp = compare_mi_significance(mi_df, "medpathagent", "llama")
        if not mi_comp.empty:
            print("\nMI comparison: medpathagent vs llama")
            print(mi_comp.to_string(index=False))
            mi_comp.to_csv(
                os.path.join(args.output_dir, "mi_comparison_medpathagent_vs_llama.csv"),
                index=False,
            )

        mi_comp_clinician_mp = compare_mi_significance(mi_df, "medpathagent", "clinician")
        if not mi_comp_clinician_mp.empty:
            print("\nMI comparison: medpathagent vs human consensus")
            print(mi_comp_clinician_mp.to_string(index=False))
            mi_comp_clinician_mp.to_csv(
                os.path.join(args.output_dir, "mi_comparison_medpathagent_vs_clinician.csv"),
                index=False,
            )

        mi_comp_clinician_ll = compare_mi_significance(mi_df, "llama", "clinician")
        if not mi_comp_clinician_ll.empty:
            print("\nMI comparison: llama vs human consensus")
            print(mi_comp_clinician_ll.to_string(index=False))
            mi_comp_clinician_ll.to_csv(
                os.path.join(args.output_dir, "mi_comparison_llama_vs_clinician.csv"),
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

    focus_models_dict = {k: v for k, v in all_with_clinician.items() if k in focus + ["clinician"]}

    # ── Alignment with human consensus (pairwise kappa: each model vs clinician) ──
    print("\n" + "=" * 60)
    print("ALIGNMENT WITH HUMAN CONSENSUS (Cohen's Kappa vs clinician)")
    print("=" * 60)
    pairwise = compute_pairwise_kappa(df, focus_models_dict)
    if not pairwise.empty:
        pairwise.to_csv(os.path.join(args.output_dir, "pairwise_kappa.csv"), index=False)

        vs_clinician = pairwise[
            (pairwise["model_1"] == "clinician") | (pairwise["model_2"] == "clinician")
        ].copy()
        vs_clinician["model"] = vs_clinician.apply(
            lambda r: r["model_1"] if r["model_2"] == "clinician" else r["model_2"], axis=1
        )

        if not vs_clinician.empty:
            comparison = vs_clinician.pivot_table(
                index=["perturbation", "question"],
                columns="model",
                values="cohens_kappa",
            )
            comparison.columns = [f"kappa_vs_human_{c}" for c in comparison.columns]
            comparison.to_csv(os.path.join(args.output_dir, "kappa_vs_human_consensus.csv"))
            print(comparison.to_string())

        # Also print the raw pairwise for reference
        print("\nAll pairwise kappas:")
        print(pairwise.to_string(index=False))

    # ── Summary table ──
    print("\n" + "=" * 60)
    print("SUMMARY: medpathagent vs llama vs human consensus")
    print("=" * 60)
    summary = build_summary(atr_df, mi_df, pc_df, focus_models=focus + ["clinician"])
    if not summary.empty:
        summary.to_csv(os.path.join(args.output_dir, "summary_comparison.csv"))
        print(summary.to_string())

    print(f"\nAll results saved to: {args.output_dir}/")
    print("Files: atr_all.csv, mi_all.csv, pc_all.csv,")
    print("       mi_comparison_medpathagent_vs_llama.csv,")
    print("       mi_comparison_medpathagent_vs_clinician.csv,")
    print("       mi_comparison_llama_vs_clinician.csv,")
    print("       kappa_vs_human_consensus.csv, pairwise_kappa.csv,")
    print("       pairwise_kappa.csv, summary_comparison.csv")


if __name__ == "__main__":
    main()