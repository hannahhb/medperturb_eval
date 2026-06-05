#!/usr/bin/env python3
"""
Bonferroni-corrected McNemar's test: MedReason vs ToolUni vs Llama 70B (baseline)

For each perturbation type × task, we test whether two models differ
significantly in their flip rates (prediction changed baseline → perturbed).

Since the same vignettes are evaluated by all models, predictions are PAIRED,
so McNemar's test (not chi-squared) is the correct test.

Bonferroni correction: total comparisons = 2 pairs × 3 perturbations × 3 tasks = 18
→ corrected α = 0.05 / 18 ≈ 0.0028

Usage:
    python medperturb_eval/analysis/bonferroni_mcnemar.py
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2
from itertools import product

# ── File paths ────────────────────────────────────────────────────────────────
MEDREASON_CSV = (
    "medperturb_eval/output/medreason_llama70b3_3/oncqa_askdocs/run_0/results.csv"
)
TOOLUNI_CSV = (
    "medperturb_eval/output/tooluni_agent/llama70b3_3/askdocs_oncqa/results_1.csv"
)
MEDPERTURB_CSV = "data/medperturb_data.csv"

TASKS = ["manage", "visit", "resource"]
PERTURBATIONS = ["gender_swap", "summary", "uncertain_tone"]

# ── Load data ─────────────────────────────────────────────────────────────────

def load_and_normalise():
    mr = pd.read_csv(MEDREASON_CSV)
    mr = mr.rename(columns={
        "medpathagent_llama70b3_3_manage":   "medreason_manage",
        "medpathagent_llama70b3_3_visit":    "medreason_visit",
        "medpathagent_llama70b3_3_resource": "medreason_resource",
    })
    mr = mr[["context_id", "perturbation", "dataset",
             "medreason_manage", "medreason_visit", "medreason_resource"]]

    tu = pd.read_csv(TOOLUNI_CSV)
    tu = tu.rename(columns={
        "tooluni_agent_manage":   "tooluni_manage",
        "tooluni_agent_visit":    "tooluni_visit",
        "tooluni_agent_resource": "tooluni_resource",
    })
    tu = tu[["context_id", "perturbation",
             "tooluni_manage", "tooluni_visit", "tooluni_resource"]]

    # Llama 70B from medperturb_data — filter to oncqa + askdocs only
    md = pd.read_csv(MEDPERTURB_CSV, low_memory=False)
    md = md[md["dataset"].isin(["oncqa", "askdocs"])]
    md = md.rename(columns={
        "llama_manage":   "llama_manage",
        "llama_visit":    "llama_visit",
        "llama_resource": "llama_resource",
    })
    md = md[["context_id", "perturbation",
             "llama_manage", "llama_visit", "llama_resource"]]

    # Merge all three on context_id × perturbation
    df = mr.merge(tu, on=["context_id", "perturbation"], how="inner")
    df = df.merge(md, on=["context_id", "perturbation"], how="inner")

    # Upper-case all prediction values for consistency
    for col in [c for c in df.columns if any(t in c for t in TASKS)]:
        df[col] = df[col].astype(str).str.upper().str.strip()

    return df


# ── Flip calculation ──────────────────────────────────────────────────────────

def compute_flips(df, model_col):
    """
    For each context_id × perturbation, flip=1 if prediction differs from baseline.
    Returns dataframe with added column f"{model_col}_flip".
    """
    baseline = df[df["perturbation"] == "baseline"][["context_id", model_col]].copy()
    baseline = baseline.rename(columns={model_col: f"{model_col}_base"})

    out = df[df["perturbation"] != "baseline"].copy()
    out = out.merge(baseline, on="context_id", how="inner")
    out[f"{model_col}_flip"] = (out[model_col] != out[f"{model_col}_base"]).astype(int)
    # Only count flips when both are valid YES/NO predictions
    valid = out[model_col].isin(["YES", "NO"]) & out[f"{model_col}_base"].isin(["YES", "NO"])
    out.loc[~valid, f"{model_col}_flip"] = np.nan
    return out


# ── McNemar's test ────────────────────────────────────────────────────────────

def mcnemar_test(flip_a: pd.Series, flip_b: pd.Series):
    """
    McNemar's test on paired binary flip indicators.
    Returns (statistic, p_value, b, c) where b = A flips, B doesn't; c = B flips, A doesn't.
    Uses continuity correction when b+c < 25.
    """
    mask = flip_a.notna() & flip_b.notna()
    a_arr = flip_a[mask].astype(int).values
    b_arr = flip_b[mask].astype(int).values

    # 2×2 table: rows = model A, cols = model B
    n11 = ((a_arr == 1) & (b_arr == 1)).sum()  # both flip
    n10 = ((a_arr == 1) & (b_arr == 0)).sum()  # A flips, B doesn't  (b)
    n01 = ((a_arr == 0) & (b_arr == 1)).sum()  # B flips, A doesn't  (c)
    n00 = ((a_arr == 0) & (b_arr == 0)).sum()  # neither flips

    b, c = n10, n01
    denom = b + c
    if denom == 0:
        return np.nan, np.nan, b, c, mask.sum()

    # Continuity-corrected McNemar
    stat = (abs(b - c) - 1) ** 2 / denom
    p = 1 - chi2.cdf(stat, df=1)
    return stat, p, b, c, mask.sum()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df_raw = load_and_normalise()

    model_pairs = [
        ("medreason", "tooluni",   "MedReason vs ToolUni"),
        ("medreason", "llama",     "MedReason vs Llama 70B"),
    ]

    # Compute flip columns for all models × tasks
    flip_dfs = {}
    for model in ["medreason", "tooluni", "llama"]:
        frames = []
        for task in TASKS:
            col = f"{model}_{task}"
            tmp = compute_flips(df_raw, col)
            tmp = tmp[["context_id", "perturbation", f"{col}_flip"]].copy()
            frames.append(tmp)
        # Merge per-task flips into one frame keyed by context_id × perturbation
        merged = frames[0]
        for f in frames[1:]:
            merged = merged.merge(f, on=["context_id", "perturbation"], how="outer")
        flip_dfs[model] = merged

    # Merge all model flip frames together
    flip_all = flip_dfs["medreason"]
    for model in ["tooluni", "llama"]:
        flip_all = flip_all.merge(
            flip_dfs[model], on=["context_id", "perturbation"], how="outer"
        )

    # ── Run tests ────────────────────────────────────────────────────────────
    n_comparisons = len(model_pairs) * len(PERTURBATIONS) * len(TASKS)
    alpha = 0.05
    alpha_bonf = alpha / n_comparisons

    print(f"\nBonferroni-corrected McNemar's Test")
    print(f"Total comparisons : {n_comparisons}")
    print(f"Uncorrected α     : {alpha}")
    print(f"Corrected α       : {alpha_bonf:.4f}  (= {alpha}/{n_comparisons})")
    print("=" * 90)

    results = []
    for (m_a, m_b, label), task, pert in product(model_pairs, TASKS, PERTURBATIONS):
        subset = flip_all[flip_all["perturbation"] == pert]
        col_a = f"{m_a}_{task}_flip"
        col_b = f"{m_b}_{task}_flip"

        if col_a not in subset.columns or col_b not in subset.columns:
            continue

        stat, p, b, c, n = mcnemar_test(subset[col_a], subset[col_b])

        flip_rate_a = subset[col_a].mean() * 100 if not subset[col_a].isna().all() else np.nan
        flip_rate_b = subset[col_b].mean() * 100 if not subset[col_b].isna().all() else np.nan

        sig_raw  = "*" if (p is not None and not np.isnan(p) and p < alpha)      else ""
        sig_bonf = "*" if (p is not None and not np.isnan(p) and p < alpha_bonf) else ""

        results.append({
            "comparison":     label,
            "task":           task,
            "perturbation":   pert,
            "n_pairs":        n,
            f"flip%_{m_a}":   round(flip_rate_a, 1),
            f"flip%_{m_b}":   round(flip_rate_b, 1),
            "b (A not B)":    b,
            "c (B not A)":    c,
            "McNemar χ²":     round(stat, 3) if not np.isnan(stat) else "—",
            "p-value":        round(p, 4)    if not np.isnan(p)    else "—",
            "p < 0.05":       sig_raw,
            f"p < {alpha_bonf:.4f} (Bonf)": sig_bonf,
        })

    results_df = pd.DataFrame(results)

    # Pretty print
    for label in [p[2] for p in model_pairs]:
        print(f"\n{'─'*90}")
        print(f"  {label}")
        print(f"{'─'*90}")
        sub = results_df[results_df["comparison"] == label]

        # Determine column names dynamically
        m_a = label.split(" vs ")[0].lower().replace(" ", "")
        m_b = label.split(" vs ")[1].lower().replace(" ", "").replace("70b", "70b")
        # Map display names back to model keys
        model_key_map = {"medreason": "medreason", "tooluni": "tooluni",
                         "llama70b": "llama", "llama": "llama"}
        mk_a = "medreason"
        mk_b = "tooluni" if "ToolUni" in label else "llama"

        col_fa = f"flip%_{mk_a}"
        col_fb = f"flip%_{mk_b}"
        bonf_col = f"p < {alpha_bonf:.4f} (Bonf)"

        header = f"{'Task':<10} {'Perturbation':<17} {'n':>5}  {'Flip%_A':>8} {'Flip%_B':>8}  {'b':>5} {'c':>5}  {'χ²':>8}  {'p':>8}  {'<0.05':>6}  {'<Bonf':>6}"
        print(header)
        print("-" * len(header))
        for _, row in sub.iterrows():
            print(
                f"{row['task']:<10} {row['perturbation']:<17} {row['n_pairs']:>5}  "
                f"{row[col_fa]:>7.1f}% {row[col_fb]:>7.1f}%  "
                f"{row['b (A not B)']:>5} {row['c (B not A)']:>5}  "
                f"{str(row['McNemar χ²']):>8}  {str(row['p-value']):>8}  "
                f"{'yes' if row['p < 0.05'] else 'no':>6}  "
                f"{'yes' if row[bonf_col] else 'no':>6}"
            )

    # Save CSV
    out_path = "medperturb_eval/output/bonferroni_mcnemar_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n✅ Results saved to {out_path}")
    print(f"\nInterpretation guide:")
    print(f"  b = cases where Model A flipped but Model B did NOT (A more sensitive)")
    print(f"  c = cases where Model B flipped but Model A did NOT (B more sensitive)")
    print(f"  A significant result (Bonf) means the two models have DIFFERENT flip rates")
    print(f"  on that perturbation × task, beyond chance.")


if __name__ == "__main__":
    main()
