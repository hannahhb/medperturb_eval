"""
MedPerturb analysis — basic + advanced.

Usage:
    python analysis.py                                  # all sections, fetch from HF
    python analysis.py --local data/medperturb_data.csv # use cached local copy
    python analysis.py --local data/medperturb_data.csv --sections overview gender kappa
    python analysis.py --list                           # list available sections

Available sections:
    overview       Dataset size, sources, perturbation counts, gold label rates, age/gender
    accuracy       Per-task accuracy + missing predictions
    f1             Per-task macro-F1 and per-class F1
    kappa          Cohen's kappa — model vs gold standard
    ima            Inter-model pairwise kappa
    consistency    Cross-perturbation answer consistency per model
    gender         Gender-swap flip rates + McNemar's test
    uncertainty    YES-rate shift under uncertain_tone phrasing
    source         Accuracy broken down by source dataset
    age            Accuracy broken down by age group
    clinician      Model vs clinician consensus agreement
    summary        Accuracy drop on summary perturbation vs baseline
"""
import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, f1_score

warnings.filterwarnings("ignore")

HF_URL = "https://huggingface.co/datasets/abinitha/MedPerturb/resolve/main/data.csv"
TASKS   = ["manage", "visit", "resource"]
MODELS  = ["gpt-4o", "medgemma", "deepseek", "qwen", "llama"]


# ── helpers ───────────────────────────────────────────────────────────────────

def load(local_path=None) -> pd.DataFrame:
    src = local_path or HF_URL
    print(f"Loading: {src}")
    df = pd.read_csv(src)
    for m in MODELS:
        for t in TASKS:
            df[f"{m}_{t}"] = df[f"{m}_{t}"].map({"YES": 1, "NO": 0})
    for t in TASKS:
        df[f"gold_standard_{t}"] = df[f"gold_standard_{t}"].map({"YES": 1, "NO": 0})
    print(f"Loaded {len(df):,} rows, {df['context_id'].nunique():,} unique vignettes.\n")
    return df


def _valid_pair(df, col_a, col_b):
    mask = df[col_a].isin([0, 1]) & df[col_b].isin([0, 1])
    return df.loc[mask, col_a].astype(int), df.loc[mask, col_b].astype(int)


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ── basic sections ────────────────────────────────────────────────────────────

def sec_overview(df):
    section("DATASET OVERVIEW")
    print(f"  Total rows       : {len(df):,}")
    print(f"  Unique vignettes : {df['context_id'].nunique():,}")
    print(f"  Perturbations    : {sorted(df['perturbation'].unique())}")
    print(f"  Source datasets  : {sorted(df['dataset'].unique())}")

    print("\n--- Source breakdown ---")
    print(df["dataset"].value_counts().to_string())

    print("\n--- Perturbation counts ---")
    print(df["perturbation"].value_counts().to_string())

    print("\n--- Gold standard YES rates ---")
    for t in TASKS:
        yes_pct = df[f"gold_standard_{t}"].mean() * 100
        print(f"  {t:10s}: YES={yes_pct:.1f}%  NO={100-yes_pct:.1f}%")

    print("\n--- Gender distribution ---")
    print(df["original_gender"].value_counts().to_string())

    print("\n--- Age ---")
    age = pd.to_numeric(df["age"], errors="coerce")
    unknown = df["age"].astype(str).str.upper().eq("X").sum()
    print(f"  min={age.min():.0f}  max={age.max():.0f}  median={age.median():.0f}  "
          f"mean={age.mean():.1f}  unknown={unknown}")

    print("\n--- Clinician consensus coverage ---")
    for t in TASKS:
        n = df[f"clinician_consensus_{t}"].notna().sum()
        print(f"  {t:10s}: {n} rows ({n/len(df)*100:.1f}%)")


def sec_accuracy(df):
    section("MODEL ACCURACY vs GOLD STANDARD")
    print(f"  {'model':12s}  {'manage':>8}  {'visit':>8}  {'resource':>8}  {'avg':>8}  {'n':>6}")
    print("  " + "-" * 58)
    for m in MODELS:
        accs, n_total = [], None
        for t in TASKS:
            y_true, y_pred = _valid_pair(df, f"gold_standard_{t}", f"{m}_{t}")
            accs.append((y_true == y_pred).mean() * 100 if len(y_true) > 0 else float("nan"))
            n_total = len(y_true)
        avg = np.nanmean(accs)
        vals = "  ".join(f"{v:7.1f}%" if not np.isnan(v) else "     N/A" for v in accs)
        print(f"  {m:12s}  {vals}  {avg:7.1f}%  {n_total:>6,}")

    print("\n--- Missing / invalid predictions ---")
    any_missing = False
    for m in MODELS:
        for t in TASKS:
            col = f"{m}_{t}"
            n = df[col].isna().sum()
            if n > 0:
                print(f"  {m}_{t}: {n:,} missing ({n/len(df)*100:.1f}%)")
                any_missing = True
    if not any_missing:
        print("  None.")


# ── advanced sections ─────────────────────────────────────────────────────────

def sec_f1(df):
    section("PER-TASK F1 (macro) vs GOLD STANDARD")
    for t in TASKS:
        gold_col = f"gold_standard_{t}"
        print(f"\n  Task: {t.upper()}")
        for m in MODELS:
            y_true, y_pred = _valid_pair(df, gold_col, f"{m}_{t}")
            if len(y_true) < 10:
                print(f"    {m:12s}: insufficient data")
                continue
            f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
            f1_yes = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            f1_no  = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
            print(f"    {m:12s}: macro-F1={f1:.3f}  F1(YES)={f1_yes:.3f}  F1(NO)={f1_no:.3f}  n={len(y_true):,}")


def sec_kappa(df):
    section("COHEN'S KAPPA — model vs gold standard")
    print(f"  {'model':12s}  {'manage':>8}  {'visit':>8}  {'resource':>8}  {'avg':>8}")
    print("  " + "-" * 52)
    for m in MODELS:
        kappas = []
        for t in TASKS:
            y_true, y_pred = _valid_pair(df, f"gold_standard_{t}", f"{m}_{t}")
            kappas.append(cohen_kappa_score(y_true, y_pred) if len(y_true) >= 10 else float("nan"))
        avg = np.nanmean(kappas)
        vals = "  ".join(f"{k:8.3f}" if not np.isnan(k) else "     N/A" for k in kappas)
        print(f"  {m:12s}  {vals}  {avg:8.3f}")


def sec_ima(df):
    section("INTER-MODEL PAIRWISE KAPPA (averaged across tasks)")
    print(f"  {'pair':28s}  {'manage':>8}  {'visit':>8}  {'resource':>8}  {'avg':>8}")
    print("  " + "-" * 66)
    for m1, m2 in itertools.combinations(MODELS, 2):
        kappas = []
        for t in TASKS:
            a, b = _valid_pair(df, f"{m1}_{t}", f"{m2}_{t}")
            kappas.append(cohen_kappa_score(a, b) if len(a) >= 10 else float("nan"))
        avg = np.nanmean(kappas)
        vals = "  ".join(f"{k:8.3f}" if not np.isnan(k) else "     N/A" for k in kappas)
        print(f"  {m1+' vs '+m2:28s}  {vals}  {avg:8.3f}")


def sec_consistency(df):
    section("CROSS-PERTURBATION CONSISTENCY (% vignettes with unanimous answer across perturbations)")
    print(f"  {'model':14s}  {'manage':>8}  {'visit':>8}  {'resource':>8}  {'avg':>8}")
    print("  " + "-" * 54)
    all_models = MODELS + ["gold_standard"]
    for m in all_models:
        consistencies = []
        for t in TASKS:
            col = f"{m}_{t}" if m != "gold_standard" else f"gold_standard_{t}"
            grp = df.groupby("context_id")[col]
            unanimous = grp.apply(
                lambda s: float(s.dropna().nunique() == 1) if s.dropna().shape[0] >= 2 else float("nan")
            ).dropna()
            consistencies.append(unanimous.mean() * 100 if len(unanimous) > 0 else float("nan"))
        avg = np.nanmean(consistencies)
        vals = "  ".join(f"{v:7.1f}%" if not np.isnan(v) else "     N/A" for v in consistencies)
        print(f"  {m:14s}  {vals}  {avg:7.1f}%")


def sec_gender(df):
    section("GENDER BIAS — flip rate baseline → gender_swap")
    print("  (% vignettes where prediction changes when patient gender is swapped)\n")
    base = df[df["perturbation"] == "baseline"].set_index("context_id")
    swap = df[df["perturbation"] == "gender_swap"].set_index("context_id")
    both = base.index.intersection(swap.index)
    print(f"  Paired vignettes: {len(both):,}\n")
    print(f"  {'model':14s}  {'manage':>8}  {'visit':>8}  {'resource':>8}  {'avg':>8}")
    print("  " + "-" * 54)
    all_models = MODELS + ["gold_standard"]
    for m in all_models:
        flips = []
        for t in TASKS:
            col = f"{m}_{t}" if m != "gold_standard" else f"gold_standard_{t}"
            b = base.loc[both, col]
            s = swap.loc[both, col]
            valid = b.isin([0, 1]) & s.isin([0, 1])
            flips.append((b[valid] != s[valid]).mean() * 100)
        avg = np.mean(flips)
        vals = "  ".join(f"{v:7.1f}%" for v in flips)
        print(f"  {m:14s}  {vals}  {avg:7.1f}%")

    # McNemar's test per model
    print(f"\n--- McNemar's test (is the flip rate statistically significant?) ---")
    print(f"  {'model':14s}  {'task':10s}  {'NO→YES':>8}  {'YES→NO':>8}  {'p-value':>10}")
    print("  " + "-" * 56)
    for m in MODELS:
        for t in TASKS:
            col = f"{m}_{t}"
            b = base.loc[both, col].dropna()
            s = swap.loc[b.index, col].dropna()
            idx = b.index.intersection(s.index)
            b2, s2 = b.loc[idx], s.loc[idx]
            n01 = int(((b2 == 0) & (s2 == 1)).sum())
            n10 = int(((b2 == 1) & (s2 == 0)).sum())
            if n01 + n10 > 0:
                chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
                p = 1 - stats.chi2.cdf(chi2, df=1)
                sig = " *" if p < 0.05 else ""
                print(f"  {m:14s}  {t:10s}  {n01:>8}  {n10:>8}  {p:>10.4f}{sig}")


def sec_uncertainty(df):
    section("UNCERTAINTY PERTURBATION — YES-rate shift vs baseline")
    print("  (negative Δ = model says NO more often under uncertain phrasing)\n")
    base = df[df["perturbation"] == "baseline"].set_index("context_id")
    unct = df[df["perturbation"] == "uncertain_tone"].set_index("context_id")
    both = base.index.intersection(unct.index)
    print(f"  Paired vignettes: {len(both):,}\n")
    print(f"  {'model':14s}  {'manage Δ':>10}  {'visit Δ':>10}  {'resource Δ':>12}")
    print("  " + "-" * 52)
    for m in MODELS:
        deltas = []
        for t in TASKS:
            col = f"{m}_{t}"
            b_yes = base.loc[both, col].mean()
            u_yes = unct.loc[both, col].mean()
            deltas.append(u_yes - b_yes)
        print(f"  {m:14s}  {deltas[0]:+9.3f}   {deltas[1]:+9.3f}   {deltas[2]:+11.3f}")


def sec_source(df):
    section("ACCURACY BY SOURCE DATASET (avg across tasks)")
    sources = sorted(df["dataset"].unique())
    w = max(len(s) for s in sources)
    print(f"  {'dataset':{w}}  " + "  ".join(f"{m:>10}" for m in MODELS))
    print("  " + "-" * (w + 2 + 12 * len(MODELS)))
    for src in sources:
        sub = df[df["dataset"] == src]
        accs = []
        for m in MODELS:
            task_accs = []
            for t in TASKS:
                y_true, y_pred = _valid_pair(sub, f"gold_standard_{t}", f"{m}_{t}")
                if len(y_true) >= 5:
                    task_accs.append((y_true == y_pred).mean() * 100)
            accs.append(np.mean(task_accs) if task_accs else float("nan"))
        vals = "  ".join(f"{v:9.1f}%" if not np.isnan(v) else "       N/A" for v in accs)
        print(f"  {src:{w}}  {vals}")


def sec_age(df):
    section("ACCURACY BY AGE GROUP (avg across tasks, all models)")
    df2 = df.copy()
    df2["age_num"] = pd.to_numeric(df2["age"], errors="coerce")
    bins   = [0, 17, 39, 64, 200]
    labels = ["0-17 (peds)", "18-39 (young adult)", "40-64 (middle-aged)", "65+ (elderly)"]
    df2["age_group"] = pd.cut(df2["age_num"], bins=bins, labels=labels, right=True)
    print(f"  {'age group':22s}  {'n':>6}  " + "  ".join(f"{m:>9}" for m in MODELS))
    print("  " + "-" * (30 + 11 * len(MODELS)))
    for grp in labels:
        sub = df2[df2["age_group"] == grp]
        accs = []
        for m in MODELS:
            task_accs = []
            for t in TASKS:
                y_true, y_pred = _valid_pair(sub, f"gold_standard_{t}", f"{m}_{t}")
                if len(y_true) >= 5:
                    task_accs.append((y_true == y_pred).mean() * 100)
            accs.append(np.mean(task_accs) if task_accs else float("nan"))
        vals = "  ".join(f"{v:8.1f}%" if not np.isnan(v) else "      N/A" for v in accs)
        print(f"  {grp:22s}  {len(sub):>6}  {vals}")


def sec_clinician(df):
    section("CLINICIAN CONSENSUS vs MODEL PREDICTIONS")
    clin = df[df["clinician_consensus_manage"].notna()].copy()
    print(f"  Rows with clinician consensus: {len(clin):,}\n")
    for t in TASKS:
        clin[f"clin_bin_{t}"] = (clin[f"clinician_consensus_{t}"] >= 0.5).astype(int)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    print(f"  --- Accuracy ---")
    print(f"  {'model':12s}  {'manage':>8}  {'visit':>8}  {'resource':>10}  {'avg':>8}  {'n':>6}")
    print("  " + "-" * 62)
    for m in MODELS:
        accs, n = [], 0
        for t in TASKS:
            y_clin, y_pred = _valid_pair(clin, f"clin_bin_{t}", f"{m}_{t}")
            accs.append((y_clin == y_pred).mean() * 100 if len(y_clin) else float("nan"))
            n = max(n, len(y_clin))
        avg = np.nanmean(accs)
        vals = "  ".join(f"{v:7.1f}%" if not np.isnan(v) else "     N/A" for v in accs)
        print(f"  {m:12s}  {vals}  {avg:7.1f}%  {n:>6,}")

    # ── Cohen's kappa ─────────────────────────────────────────────────────────
    print(f"\n  --- Cohen's Kappa ---")
    print(f"  {'model':12s}  {'manage κ':>10}  {'visit κ':>10}  {'resource κ':>12}  {'avg κ':>8}  {'n':>6}")
    print("  " + "-" * 68)
    for m in MODELS:
        kappas, n = [], 0
        for t in TASKS:
            y_clin, y_pred = _valid_pair(clin, f"clin_bin_{t}", f"{m}_{t}")
            kappas.append(cohen_kappa_score(y_clin, y_pred) if len(y_clin) >= 5 else float("nan"))
            n = max(n, len(y_clin))
        avg = np.nanmean(kappas)
        vals = "  ".join(f"{k:9.3f}" if not np.isnan(k) else "       N/A" for k in kappas)
        print(f"  {m:12s}  {vals}  {avg:8.3f}  {n:>6,}")

    # ── Macro-F1 ──────────────────────────────────────────────────────────────
    print(f"\n  --- Macro-F1 ---")
    print(f"  {'model':12s}  {'manage':>8}  {'visit':>8}  {'resource':>10}  {'avg':>8}")
    print("  " + "-" * 54)
    for m in MODELS:
        f1s = []
        for t in TASKS:
            y_clin, y_pred = _valid_pair(clin, f"clin_bin_{t}", f"{m}_{t}")
            f1s.append(f1_score(y_clin, y_pred, average="macro", zero_division=0) if len(y_clin) >= 5 else float("nan"))
        avg = np.nanmean(f1s)
        vals = "  ".join(f"{v:8.3f}" if not np.isnan(v) else "     N/A" for v in f1s)
        print(f"  {m:12s}  {vals}  {avg:8.3f}")


def sec_summary(df):
    section("SUMMARY PERTURBATION — accuracy vs baseline")
    print(f"  {'model + task':26s}  {'baseline':>10}  {'summary':>10}  {'Δ':>8}")
    print("  " + "-" * 60)
    for m in MODELS:
        for t in TASKS:
            base_sub = df[df["perturbation"] == "baseline"]
            summ_sub = df[df["perturbation"] == "summary"]
            yb, pb = _valid_pair(base_sub, f"gold_standard_{t}", f"{m}_{t}")
            ys, ps = _valid_pair(summ_sub, f"gold_standard_{t}", f"{m}_{t}")
            if len(yb) < 5 or len(ys) < 5:
                continue
            acc_b = (yb == pb).mean() * 100
            acc_s = (ys == ps).mean() * 100
            delta = acc_s - acc_b
            print(f"  {m+'_'+t:26s}  {acc_b:9.1f}%  {acc_s:9.1f}%  {delta:+7.1f}%")


# ── main ──────────────────────────────────────────────────────────────────────

SECTION_MAP = {
    "overview":    sec_overview,
    "accuracy":    sec_accuracy,
    "f1":          sec_f1,
    "kappa":       sec_kappa,
    "ima":         sec_ima,
    "consistency": sec_consistency,
    "gender":      sec_gender,
    "uncertainty": sec_uncertainty,
    "source":      sec_source,
    "age":         sec_age,
    "clinician":   sec_clinician,
    "summary":     sec_summary,
}


def main():
    parser = argparse.ArgumentParser(description="MedPerturb analysis")
    parser.add_argument("--local", default=None, help="Path to local data.csv")
    parser.add_argument(
        "--sections", nargs="+", default=None,
        help=f"Sections to run (default: all). Choices: {', '.join(SECTION_MAP)}"
    )
    parser.add_argument("--list", action="store_true", help="List available sections and exit")
    args = parser.parse_args()

    if args.list:
        print("Available sections:")
        for k in SECTION_MAP:
            print(f"  {k}")
        return

    df = load(args.local)
    to_run = args.sections or list(SECTION_MAP.keys())
    for key in to_run:
        if key in SECTION_MAP:
            SECTION_MAP[key](df)
        else:
            print(f"Unknown section: {key}  (run --list to see options)")

    print("\nDone.")


if __name__ == "__main__":
    main()
