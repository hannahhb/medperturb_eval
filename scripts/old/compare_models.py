"""
Four-way MedPerturb comparison:
  tooluni_agent  |  medreasonagent (medpathagent_llama70b3_3)  |  llama  |  clinician

Metrics (following MedPerturb paper appendices K.2–K.4):

  1.  Accuracy vs gold standard  (all 4 models)
  2.  Average Treatment Rate (ATR)  per model × perturbation × question
  3.  Paired t-Test: ATR shift baseline→perturbed  (K.3)
  4.  Mutual Information: decision stability under perturbation  (K.4)
  5.  Mann–Whitney U: compare MI distributions across models  (K.4)
  6.  Fleiss' κ: inter-rater agreement (all raters as a group)  (K.2)
  7.  Wilcoxon Signed-Rank: κ shift baseline→perturbed  (K.2)

Usage (defaults work out of the box):
    python compare_models.py
    python compare_models.py --output_dir my_eval
    python compare_models.py --tooluni <path> --medpathagent <path> --orig <path>
"""
from __future__ import annotations

import argparse
import os
import warnings
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

QUESTIONS   = ["manage", "visit", "resource"]
PERTS_ALL   = ["baseline", "summary", "gender_swap", "uncertain_tone"]
ALPHA       = 0.05

# Model display names used throughout
MODEL_TOOLUNI      = "tooluni_agent"
MODEL_MEDREASONAGENT = "medreasonagent"
MODEL_LLAMA        = "llama"
MODEL_CLINICIAN    = "clinician"
ALL_MODELS         = [MODEL_TOOLUNI, MODEL_MEDREASONAGENT, MODEL_LLAMA, MODEL_CLINICIAN]


# ── Data loading ──────────────────────────────────────────────────────────────

def _binarise(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.str.strip().str.lower().map(
            {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0, "1.0": 1, "0.0": 0}
        )
    else:
        s = s.replace({1.0: 1, 0.0: 0})
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def load_data(
    tooluni_path: str,
    medpathagent_path: str,
    orig_path: str,
    datasets: list[str] = ("askdocs", "oncqa"),
    perturbations: list[str] = ("baseline", "summary", "gender_swap", "uncertain_tone"),
) -> pd.DataFrame:
    """
    Merge all three sources on (context_id, perturbation).

    Returns a flat DataFrame with unified column names:
      tooluni_agent_{q}, medreasonagent_{q}, llama_{q}, clinician_{q}
      gold_{q}, context_id, perturbation, dataset
    """
    tu   = pd.read_csv(tooluni_path,    low_memory=False)
    mp   = pd.read_csv(medpathagent_path, low_memory=False)
    orig = pd.read_csv(orig_path,       low_memory=False)

    # Normalise keys
    for df in (tu, mp, orig):
        df["context_id"]   = df["context_id"].astype(str).str.strip()
        df["perturbation"] = df["perturbation"].str.strip().str.lower()

    # Restrict orig to the target datasets and perturbations
    orig = orig[
        orig["dataset"].str.lower().isin([d.lower() for d in datasets]) &
        orig["perturbation"].isin([p.lower() for p in perturbations])
    ].copy()

    # ── Select and rename columns ──────────────────────────────────────────
    # tooluni
    tu_cols = {"context_id": "context_id", "perturbation": "perturbation"}
    for q in QUESTIONS:
        tu_cols[f"tooluni_agent_{q}"] = f"{MODEL_TOOLUNI}_{q}"
    tu_slim = tu[[c for c in tu_cols if c in tu.columns]].rename(columns=tu_cols)

    # medpathagent → medreasonagent
    mp_cols = {"context_id": "context_id", "perturbation": "perturbation"}
    for q in QUESTIONS:
        mp_cols[f"medpathagent_llama70b3_3_{q}"] = f"{MODEL_MEDREASONAGENT}_{q}"
    mp_slim = mp[[c for c in mp_cols if c in mp.columns]].rename(columns=mp_cols)

    # orig: llama + clinician + gold
    orig_cols = {
        "context_id": "context_id",
        "perturbation": "perturbation",
        "dataset": "dataset",
    }
    for q in QUESTIONS:
        orig_cols[f"llama_{q}"]                    = f"{MODEL_LLAMA}_{q}"
        orig_cols[f"clinician_consensus_{q}"]      = f"{MODEL_CLINICIAN}_{q}"
        orig_cols[f"gold_standard_{q}"]            = f"gold_{q}"
    orig_slim = orig[[c for c in orig_cols if c in orig.columns]].rename(columns=orig_cols)

    # ── Merge ────────────────────────────────────────────────────────────
    merged = orig_slim.merge(tu_slim,  on=["context_id", "perturbation"], how="inner")
    merged = merged.merge(mp_slim, on=["context_id", "perturbation"], how="inner")

    # Binarise all decision columns
    for m in ALL_MODELS:
        for q in QUESTIONS:
            col = f"{m}_{q}"
            if col in merged.columns:
                merged[col] = _binarise(merged[col])
    for q in QUESTIONS:
        col = f"gold_{q}"
        if col in merged.columns:
            merged[col] = _binarise(merged[col])

    print(f"Rows after merge:  {len(merged)}")
    print(f"Perturbations:     {sorted(merged['perturbation'].unique())}")
    print(f"Datasets:          {sorted(merged['dataset'].unique())}")
    for p in sorted(merged["perturbation"].unique()):
        print(f"  {p}: {(merged['perturbation']==p).sum()} rows")

    # Clinician coverage
    clin_rows = merged[f"{MODEL_CLINICIAN}_manage"].notna().sum()
    print(f"Clinician consensus available: {clin_rows}/{len(merged)} rows")

    return merged


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col(model: str, q: str) -> str:
    return f"{model}_{q}"


def _section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _print(df: pd.DataFrame):
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", "{:.4f}".format,
    ):
        print(df.to_string(index=False))


# ── 1. Accuracy ───────────────────────────────────────────────────────────────

def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy vs gold standard, per model × perturbation × question."""
    rows = []
    for pert in sorted(df["perturbation"].unique()):
        sub = df[df["perturbation"] == pert]
        for q in QUESTIONS:
            gold_col = f"gold_{q}"
            if gold_col not in sub.columns:
                continue
            for model in ALL_MODELS:
                pred_col = _col(model, q)
                if pred_col not in sub.columns:
                    continue
                valid = sub[[pred_col, gold_col]].dropna()
                if len(valid) < 3:
                    continue
                acc = (valid[pred_col] == valid[gold_col]).mean()
                rows.append({
                    "perturbation": pert, "question": q, "model": model,
                    "accuracy": round(float(acc), 4), "n": len(valid),
                })
    return pd.DataFrame(rows)


def summarise_accuracy(acc_df: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy across perturbations, per model × question."""
    return (
        acc_df.groupby(["model", "question"])["accuracy"]
        .mean().round(4).reset_index()
        .rename(columns={"accuracy": "mean_accuracy"})
    )


# ── 2 & 3. ATR + Paired t-Test (K.3) ─────────────────────────────────────────

def compute_atr(df: pd.DataFrame) -> pd.DataFrame:
    """ATR = mean binary decision per model × perturbation × question."""
    rows = []
    for pert in sorted(df["perturbation"].unique()):
        sub = df[df["perturbation"] == pert]
        for q in QUESTIONS:
            for model in ALL_MODELS:
                col = _col(model, q)
                if col not in sub.columns:
                    continue
                vals = sub[col].dropna()
                if len(vals) < 3:
                    continue
                rows.append({
                    "perturbation": pert, "question": q, "model": model,
                    "ATR": round(float(vals.mean()), 4), "n": len(vals),
                })
    return pd.DataFrame(rows)


def atr_paired_ttest(df: pd.DataFrame) -> pd.DataFrame:
    """
    K.3 Paired t-Test: for each (model, question, perturbation),
    align baseline and perturbed rows on context_id,
    compute D_i = decision_pert_i - decision_base_i, test H0: mean(D) = 0.
    """
    base_df = df[df["perturbation"] == "baseline"].copy()
    rows = []
    for pert in sorted(df["perturbation"].unique()):
        if pert == "baseline":
            continue
        pert_df = df[df["perturbation"] == pert].copy()
        paired = pd.merge(
            base_df[["context_id"] + [_col(m, q) for m in ALL_MODELS for q in QUESTIONS
                                       if _col(m, q) in base_df.columns]],
            pert_df[["context_id"] + [_col(m, q) for m in ALL_MODELS for q in QUESTIONS
                                       if _col(m, q) in pert_df.columns]],
            on="context_id", suffixes=("_base", "_pert"),
        )
        for model in ALL_MODELS:
            for q in QUESTIONS:
                bc = f"{_col(model, q)}_base"
                pc = f"{_col(model, q)}_pert"
                if bc not in paired.columns or pc not in paired.columns:
                    continue
                valid = paired[[bc, pc]].dropna()
                if len(valid) < 5:
                    continue
                D = (valid[pc] - valid[bc]).astype(float).values
                n = len(D)
                D_bar = D.mean()
                sd = D.std(ddof=1)
                if sd == 0:
                    t_stat, p_val = 0.0, 1.0
                else:
                    t_stat, p_val = stats.ttest_1samp(D, 0)
                rows.append({
                    "perturbation": pert, "question": q, "model": model,
                    "ATR_base": round(float(valid[bc].mean()), 4),
                    "ATR_pert": round(float(valid[pc].mean()), 4),
                    "mean_D": round(float(D_bar), 4),
                    "t_stat": round(float(t_stat), 4),
                    "p_value": round(float(p_val), 6),
                    "significant": float(p_val) < ALPHA,
                    "N": n,
                })
    return pd.DataFrame(rows)


# ── 4 & 5. MI + Mann-Whitney U (K.4) ─────────────────────────────────────────

def _mi_binary(x: np.ndarray, y: np.ndarray) -> float:
    """MI between two binary arrays (nats → bits via log2)."""
    joint = np.zeros((2, 2))
    for a, b in zip(x.astype(int), y.astype(int)):
        if 0 <= a <= 1 and 0 <= b <= 1:
            joint[a, b] += 1
    total = joint.sum()
    if total == 0:
        return float("nan")
    joint /= total
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    mi = 0.0
    for i in range(2):
        for j in range(2):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    return float(mi)


def compute_mi(df: pd.DataFrame) -> pd.DataFrame:
    """
    MI between baseline and perturbed decisions for each model × question.
    One MI value per (model, question, perturbation).
    """
    base = df[df["perturbation"] == "baseline"].copy()
    rows = []
    for pert in sorted(df["perturbation"].unique()):
        if pert == "baseline":
            continue
        pert_df = df[df["perturbation"] == pert].copy()
        for model in ALL_MODELS:
            for q in QUESTIONS:
                col = _col(model, q)
                if col not in base.columns or col not in pert_df.columns:
                    continue
                paired = pd.merge(
                    base[["context_id", col]],
                    pert_df[["context_id", col]],
                    on="context_id",
                    suffixes=("_base", "_pert"),
                )
                valid = paired[[f"{col}_base", f"{col}_pert"]].dropna()
                if len(valid) < 10:
                    continue
                mi = _mi_binary(
                    valid[f"{col}_base"].values,
                    valid[f"{col}_pert"].values,
                )
                rows.append({
                    "perturbation": pert, "question": q, "model": model,
                    "MI": round(mi, 6), "n": len(valid),
                })
    return pd.DataFrame(rows)


def mi_mannwhitney(mi_df: pd.DataFrame) -> pd.DataFrame:
    """
    K.4 Mann–Whitney U: for each question, compare the distribution of
    MI values (across perturbations) between every pair of models.
    """
    rows = []
    for q in QUESTIONS:
        sub = mi_df[mi_df["question"] == q]
        model_vals: Dict[str, np.ndarray] = {}
        for model in ALL_MODELS:
            vals = sub[sub["model"] == model]["MI"].dropna().values
            if len(vals) > 0:
                model_vals[model] = vals

        for m_a, m_b in combinations(model_vals.keys(), 2):
            a, b = model_vals[m_a], model_vals[m_b]
            if len(a) < 2 or len(b) < 2:
                continue
            # Compute U statistic manually following K.4
            combined = np.concatenate([a, b])
            ranks = stats.rankdata(combined)
            n_a, n_b = len(a), len(b)
            R_a = ranks[:n_a].sum()
            U_a = R_a - n_a * (n_a + 1) / 2
            U_b = n_a * n_b - U_a
            U = min(U_a, U_b)
            # p-value from scipy (two-sided)
            _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append({
                "question": q,
                "model_A": m_a, "model_B": m_b,
                f"mean_MI_{m_a}": round(float(a.mean()), 6),
                f"mean_MI_{m_b}": round(float(b.mean()), 6),
                "higher_MI": m_a if a.mean() >= b.mean() else m_b,
                "U_stat": round(float(U), 3),
                "p_value": round(float(p), 6),
                "significant": float(p) < ALPHA,
                "n_A": len(a), "n_B": len(b),
            })
    return pd.DataFrame(rows)


# ── 6 & 7. Fleiss' κ + Wilcoxon (K.2) ───────────────────────────────────────

def _fleiss_kappa(ratings: np.ndarray) -> float:
    """
    Fleiss' κ from an (n_subjects × n_raters) binary matrix.
    """
    n_subj, n_rat = ratings.shape
    if n_rat < 2 or n_subj < 2:
        return float("nan")
    counts = np.column_stack([(ratings == c).sum(axis=1) for c in range(2)])
    p_i = (np.sum(counts ** 2, axis=1) - n_rat) / (n_rat * (n_rat - 1))
    P_bar = p_i.mean()
    p_j = counts.sum(axis=0) / (n_subj * n_rat)
    P_e = np.sum(p_j ** 2)
    if P_e >= 1.0:
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))


def compute_fleiss_kappa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fleiss' κ per (perturbation × question) with all available raters.
    Reports both the full 4-rater κ (clinician-subset rows) and
    the 3-rater κ excluding clinician (all rows).
    """
    rows = []
    for pert in sorted(df["perturbation"].unique()):
        sub = df[df["perturbation"] == pert]
        for q in QUESTIONS:
            cols_3 = [_col(m, q) for m in [MODEL_TOOLUNI, MODEL_MEDREASONAGENT, MODEL_LLAMA]
                      if _col(m, q) in sub.columns]
            cols_4 = cols_3 + [_col(MODEL_CLINICIAN, q)] \
                if _col(MODEL_CLINICIAN, q) in sub.columns else cols_3

            # 3-rater (all rows)
            valid3 = sub[cols_3].dropna()
            kappa3 = _fleiss_kappa(valid3.values.astype(int)) if len(valid3) >= 5 else float("nan")

            # 4-rater (clinician-available rows only)
            valid4 = sub[cols_4].dropna()
            kappa4 = _fleiss_kappa(valid4.values.astype(int)) if len(valid4) >= 5 else float("nan")

            rows.append({
                "perturbation": pert, "question": q,
                "fleiss_kappa_3rater": round(kappa3, 4),
                "n_3rater": len(valid3),
                "fleiss_kappa_4rater": round(kappa4, 4),
                "n_4rater": len(valid4),
                "raters_3": ", ".join([MODEL_TOOLUNI, MODEL_MEDREASONAGENT, MODEL_LLAMA]),
                "raters_4": ", ".join([MODEL_TOOLUNI, MODEL_MEDREASONAGENT, MODEL_LLAMA, MODEL_CLINICIAN]),
            })
    return pd.DataFrame(rows)


def kappa_wilcoxon(kappa_df: pd.DataFrame) -> pd.DataFrame:
    """
    K.2 Wilcoxon Signed-Rank: tests whether perturbation shifts Fleiss' κ.

    For each question q:
      κ_base_q  = κ at baseline
      κ_pert_{p,q} = κ at perturbation p
      d_{p,q} = κ_pert - κ_base  →  N = n_perturbations × n_questions pairs
    Tests H0: Median(d) = 0.

    Also reports per-question results using the 3-perturbation × 1-question
    paired differences (n=3 per question).
    """
    kappa_col = "fleiss_kappa_3rater"  # use all-row version

    # Per question: 3 differences (one per non-baseline perturbation)
    question_rows = []
    all_diffs = []

    for q in QUESTIONS:
        base_row = kappa_df[(kappa_df["perturbation"] == "baseline") &
                            (kappa_df["question"] == q)]
        if len(base_row) == 0:
            continue
        k_base = float(base_row.iloc[0][kappa_col])

        diffs = []
        for pert in sorted(kappa_df["perturbation"].unique()):
            if pert == "baseline":
                continue
            pert_row = kappa_df[(kappa_df["perturbation"] == pert) &
                                (kappa_df["question"] == q)]
            if len(pert_row) == 0:
                continue
            k_pert = float(pert_row.iloc[0][kappa_col])
            d = k_pert - k_base
            diffs.append(d)
            all_diffs.append({"question": q, "perturbation": pert,
                               "kappa_base": round(k_base, 4),
                               "kappa_pert": round(k_pert, 4),
                               "d": round(d, 4)})

        if len(diffs) < 3:
            note = f"n={len(diffs)} (need ≥3 for reliable test)"
        else:
            note = ""

        if len(diffs) >= 2:
            nonzero = [d for d in diffs if d != 0]
            if len(nonzero) < 2:
                stat, p = 0.0, 1.0
            else:
                try:
                    stat, p = stats.wilcoxon(
                        [k_base] * len(diffs),
                        [k_base + d for d in diffs],
                        alternative="two-sided",
                    )
                except Exception:
                    stat, p = 0.0, 1.0
        else:
            stat, p = float("nan"), float("nan")

        question_rows.append({
            "question": q,
            "kappa_baseline": round(k_base, 4),
            "mean_kappa_perturbed": round(float(np.mean([k_base + d for d in diffs])), 4) if diffs else float("nan"),
            "mean_d": round(float(np.mean(diffs)), 4) if diffs else float("nan"),
            "Wilcoxon_W": round(float(stat), 3) if not np.isnan(stat) else float("nan"),
            "p_value": round(float(p), 6) if not np.isnan(p) else float("nan"),
            "significant": float(p) < ALPHA if not np.isnan(p) else False,
            "n_perturbations": len(diffs),
            "note": note,
        })

    # Combined test across all questions (9 paired differences)
    all_d_df = pd.DataFrame(all_diffs)
    combined = {}
    if len(all_d_df) >= 3:
        d_vals = all_d_df["d"].values
        nonzero = d_vals[d_vals != 0]
        if len(nonzero) >= 2:
            try:
                stat, p = stats.wilcoxon(d_vals, alternative="two-sided")
            except Exception:
                stat, p = 0.0, 1.0
        else:
            stat, p = 0.0, 1.0
        combined = {
            "question": "ALL",
            "kappa_baseline": float("nan"),
            "mean_kappa_perturbed": float("nan"),
            "mean_d": round(float(d_vals.mean()), 4),
            "Wilcoxon_W": round(float(stat), 3),
            "p_value": round(float(p), 6),
            "significant": float(p) < ALPHA,
            "n_perturbations": len(all_d_df),
            "note": f"Combined over all {len(all_d_df)} (pert×question) pairs",
        }

    result_df = pd.DataFrame(question_rows)
    if combined:
        result_df = pd.concat([result_df, pd.DataFrame([combined])], ignore_index=True)
    return result_df, pd.DataFrame(all_diffs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tooluni",
        default="output/tooluni_agent/llama70b3_3/askdocs_oncqa/results.csv",
    )
    parser.add_argument(
        "--medpathagent",
        default="output/medpathagent_llama70b3_3/oncqa_askdocs/run_0/results.csv",
    )
    parser.add_argument(
        "--orig",
        default="/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/medperturb_data.csv",
    )
    parser.add_argument("--output_dir", default="eval_output/four_way_comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    _section("LOADING & MERGING DATA")
    df = load_data(args.tooluni, args.medpathagent, args.orig)

    # ── 1. Accuracy ───────────────────────────────────────────────────────
    _section("1. ACCURACY vs GOLD STANDARD")
    acc_df = compute_accuracy(df)
    acc_df.to_csv(os.path.join(args.output_dir, "accuracy.csv"), index=False)
    _print(acc_df)

    acc_summary = summarise_accuracy(acc_df)
    acc_summary.to_csv(os.path.join(args.output_dir, "accuracy_summary.csv"), index=False)
    print("\nMean accuracy across perturbations:")
    _print(acc_summary.pivot(index="model", columns="question", values="mean_accuracy").reset_index())

    # ── 2. ATR ────────────────────────────────────────────────────────────
    _section("2. AVERAGE TREATMENT RATE (ATR)")
    atr_df = compute_atr(df)
    atr_df.to_csv(os.path.join(args.output_dir, "atr.csv"), index=False)
    # Pivot for readability
    atr_pivot = atr_df.pivot_table(
        index=["perturbation", "question"], columns="model", values="ATR"
    ).reset_index()
    _print(atr_pivot)

    # ── 3. Paired t-Test for ATR ──────────────────────────────────────────
    _section("3. PAIRED t-TEST: ATR shift baseline→perturbed  (K.3)")
    ttest_df = atr_paired_ttest(df)
    ttest_df.to_csv(os.path.join(args.output_dir, "atr_ttest.csv"), index=False)
    _print(ttest_df)

    # ── 4. MI ─────────────────────────────────────────────────────────────
    _section("4. MUTUAL INFORMATION: decision stability under perturbation  (K.4)")
    mi_df = compute_mi(df)
    mi_df.to_csv(os.path.join(args.output_dir, "mi.csv"), index=False)
    mi_pivot = mi_df.pivot_table(
        index=["perturbation", "question"], columns="model", values="MI"
    ).reset_index()
    _print(mi_pivot)

    # ── 5. Mann-Whitney U ─────────────────────────────────────────────────
    _section("5. MANN–WHITNEY U: comparing MI distributions across models  (K.4)")
    mw_df = mi_mannwhitney(mi_df)
    mw_df.to_csv(os.path.join(args.output_dir, "mi_mannwhitney.csv"), index=False)
    _print(mw_df)

    # ── 6. Fleiss' κ ──────────────────────────────────────────────────────
    _section("6. FLEISS' κ: inter-rater agreement across models  (K.2)")
    kappa_df = compute_fleiss_kappa(df)
    kappa_df.to_csv(os.path.join(args.output_dir, "fleiss_kappa.csv"), index=False)
    print("3-rater (tooluni + medreasonagent + llama, all rows):")
    k3 = kappa_df[["perturbation","question","fleiss_kappa_3rater","n_3rater"]]
    _print(k3.pivot_table(index="perturbation", columns="question", values="fleiss_kappa_3rater").reset_index())
    print("\n4-rater (+ clinician, consensus-subset rows only):")
    k4 = kappa_df[["perturbation","question","fleiss_kappa_4rater","n_4rater"]]
    _print(k4.pivot_table(index="perturbation", columns="question", values="fleiss_kappa_4rater").reset_index())

    # ── 7. Wilcoxon on κ ──────────────────────────────────────────────────
    _section("7. WILCOXON SIGNED-RANK: κ shift baseline→perturbed  (K.2)")
    wilcox_df, kappa_diffs_df = kappa_wilcoxon(kappa_df)
    wilcox_df.to_csv(os.path.join(args.output_dir, "kappa_wilcoxon.csv"), index=False)
    kappa_diffs_df.to_csv(os.path.join(args.output_dir, "kappa_diffs.csv"), index=False)
    print("\nPaired differences (κ_pert - κ_base) per condition:")
    _print(kappa_diffs_df)
    print("\nWilcoxon test results:")
    _print(wilcox_df)

    # ── Summary table ─────────────────────────────────────────────────────
    _section("SUMMARY: mean across perturbations")
    summary_rows = []
    for model in ALL_MODELS:
        for q in QUESTIONS:
            mi_vals = mi_df[(mi_df["model"] == model) & (mi_df["question"] == q)]["MI"].dropna()
            atr_vals = atr_df[(atr_df["model"] == model) & (atr_df["question"] == q)]["ATR"].dropna()
            acc_vals = acc_df[(acc_df["model"] == model) & (acc_df["question"] == q)]["accuracy"].dropna()
            summary_rows.append({
                "model": model, "question": q,
                "mean_MI":       round(float(mi_vals.mean()),  4) if len(mi_vals)  else float("nan"),
                "mean_ATR":      round(float(atr_vals.mean()), 4) if len(atr_vals) else float("nan"),
                "mean_accuracy": round(float(acc_vals.mean()), 4) if len(acc_vals) else float("nan"),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    for q in QUESTIONS:
        print(f"\n  question = {q}")
        _print(summary_df[summary_df["question"] == q].drop(columns="question"))

    print(f"\nAll outputs → {args.output_dir}/")
    for f in ["accuracy.csv","accuracy_summary.csv","atr.csv","atr_ttest.csv",
              "mi.csv","mi_mannwhitney.csv","fleiss_kappa.csv",
              "kappa_wilcoxon.csv","kappa_diffs.csv","summary.csv"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
