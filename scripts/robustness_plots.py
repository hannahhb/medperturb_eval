"""
MedPerturb Robustness Analysis Plots
=====================================
Fig 1 (3 panels)  — Delta accuracy vs baseline, aggregated (OncQA + AskDocs only)
Fig 2 (6 panels)  — Delta accuracy vs baseline, split by oncqa / askdocs
Fig 3 (3 panels)  — Mutual Information (prediction stability), aggregated (OncQA + AskDocs only)
Fig 4 (6 panels)  — Mutual Information, split by oncqa / askdocs
Fig 5 (9 panels)  — Pairwise one-sided McNemar's: is model A better than model B?
                     3 tasks × 3 perturbations, each panel a 5×5 matrix heatmap
Fig 6 (3 panels)  — Flip rate (% predictions changed), aggregated (OncQA + AskDocs only)
Fig 7 (6 panels)  — Flip rate, split by oncqa / askdocs

Models   : ToolUni | AgentMD | MedReason | Llama 70B | Clinician
Tasks    : manage  | visit   | resource
Perturbs : gender_swap | summary | uncertain_tone   (all vs baseline)

Visual encoding:
  Bar colour  → perturbation type
  Bar hatch   → significance level  (none = n.s., / = p<0.05, X = p<0.01)

Significance tests:
  Delta accuracy — McNemar's test  (H0: no accuracy change baseline→perturbed)
  MI             — Mann-Whitney U on bootstrap MI distributions
                   (H0: MI(base,pert) == MI(base,base), i.e. no stability loss)

Usage:
    cd medperturb_eval
    python robustness_plots.py
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu, binomtest, chi2
from sklearn.metrics import mutual_info_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

TOOLUNI_PATH   = "output/tooluni_agent/llama70b3_3/askdocs_oncqa/results.csv"
AGENTMD_PATH   = "output/agentmd_llama70b3_3/oncqa_askdocs/run_0/results.csv"
MEDREASON_PATH = "output/medreason_llama70b3_3/oncqa_askdocs/run_0/results.csv"

ORIG_PATH      = "/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/medperturb_data.csv"

OUT_DIR        = "plots/medperturb_robustness"

# ── Constants ─────────────────────────────────────────────────────────────────

TASKS     = ["manage", "visit", "resource"]
PERTURBS  = ["gender_swap", "summary", "uncertain_tone"]
DATASETS  = ["askdocs", "oncqa"]
ALPHA     = 0.05
N_BOOT    = 1000

MODEL_ORDER  = ["tooluni", "agentmd", "medreason", "llama", "clinician"]
MODEL_LABELS = {
    "tooluni":   "ToolUni",
    "agentmd":   "AgentMD",
    "medreason": "MedReasonAgent",
    "llama":     "Llama 70B",
    "clinician": "Clinician",
}

# Perturbation colours
PERT_COLORS = {
    "gender_swap":    "#4C72B0",   # blue
    "summary":        "#DD8452",   # orange
    "uncertain_tone": "#55A868",   # green
}
PERT_LABELS = {
    "gender_swap":    "Gender Swap",
    "summary":        "Summary",
    "uncertain_tone": "Uncertain Tone",
}
TASK_LABELS = {"manage": "Manage", "visit": "Visit", "resource": "Resource"}
DS_LABELS   = {"askdocs": "AskDocs", "oncqa": "OncQA"}

# Significance hatching
SIG_HATCH = {
    "ns":   "",      # not significant
    "p05":  "////",  # p < 0.05
    "p01":  "xxxx",  # p < 0.01
}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
})

os.makedirs(OUT_DIR, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def _binarise(s: pd.Series) -> pd.Series:
    """Convert YES/NO/1/0/1.0/0.0/UNKNOWN → 1/0/NaN."""
    if s.dtype == object:
        return s.str.strip().str.upper().map({"YES": 1, "NO": 0}).astype("float")
    return pd.to_numeric(s, errors="coerce").map({1.0: 1.0, 0.0: 0.0, 1: 1.0, 0: 0.0})


def load_data() -> pd.DataFrame:
    """
    Returns one row per (context_id, perturbation, dataset) with columns:
      gold_{t}, tooluni_{t}, agentmd_{t}, medreason_{t}, llama_{t}, clinician_{t}
    Restricted to DATASETS and [baseline] + PERTURBS.
    """
    tooluni   = pd.read_csv(TOOLUNI_PATH,   low_memory=False)
    agentmd   = pd.read_csv(AGENTMD_PATH,   low_memory=False)
    medreason = pd.read_csv(MEDREASON_PATH, low_memory=False)
    orig      = pd.read_csv(ORIG_PATH,      low_memory=False)

    orig = orig[
        orig["dataset"].isin(DATASETS) &
        orig["perturbation"].isin(["baseline"] + PERTURBS)
    ].copy()

    key_cols = ["context_id", "perturbation", "dataset"]
    for df in (tooluni, agentmd, medreason, orig):
        df["context_id"]   = df["context_id"].astype(str).str.strip()
        df["perturbation"] = df["perturbation"].str.strip().str.lower()

    # ── orig: gold + llama + clinician ────────────────────────────────────────
    keep = key_cols + [f"gold_standard_{t}" for t in TASKS] + \
                      [f"llama_{t}"          for t in TASKS] + \
                      [f"clinician_consensus_{t}" for t in TASKS]
    merged = orig[keep].copy()
    for t in TASKS:
        merged[f"gold_{t}"]      = _binarise(merged.pop(f"gold_standard_{t}"))
        merged[f"llama_{t}"]     = _binarise(merged[f"llama_{t}"])
        merged[f"clinician_{t}"] = _binarise(merged.pop(f"clinician_consensus_{t}"))

    # ── ToolUni ───────────────────────────────────────────────────────────────
    # tooluni file uses gold_manage (not gold_standard_manage)
    tu = tooluni[key_cols + [f"tooluni_agent_{t}" for t in TASKS]].copy()
    tu = tu.rename(columns={f"tooluni_agent_{t}": f"tooluni_{t}" for t in TASKS})
    for t in TASKS:
        tu[f"tooluni_{t}"] = _binarise(tu[f"tooluni_{t}"])
    merged = merged.merge(tu, on=key_cols, how="left")

    # ── AgentMD ───────────────────────────────────────────────────────────────
    ag = agentmd[key_cols + [f"agentmd_llama70b3_3_{t}" for t in TASKS]].copy()
    ag = ag.rename(columns={f"agentmd_llama70b3_3_{t}": f"agentmd_{t}" for t in TASKS})
    for t in TASKS:
        ag[f"agentmd_{t}"] = _binarise(ag[f"agentmd_{t}"])
    merged = merged.merge(ag, on=key_cols, how="left")

    # ── MedReason ────────────────────────────────────────────────────────────
    mr = medreason[key_cols + [f"medpathagent_llama70b3_3_{t}" for t in TASKS]].copy()
    mr = mr.rename(columns={f"medpathagent_llama70b3_3_{t}": f"medreason_{t}" for t in TASKS})
    for t in TASKS:
        mr[f"medreason_{t}"] = _binarise(mr[f"medreason_{t}"])
    merged = merged.merge(mr, on=key_cols, how="left")

    return merged.reset_index(drop=True)


# ── Accuracy helpers ──────────────────────────────────────────────────────────

def _pair(df_base, df_pert, model, task):
    """Inner join baseline and perturbed on context_id, return paired arrays."""
    col  = f"{model}_{task}"
    gold = f"gold_{task}"
    m = df_base[["context_id", col, gold]].merge(
        df_pert[["context_id", col, gold]].rename(
            columns={col: f"{col}_p", gold: f"{gold}_p"}),
        on="context_id", how="inner"
    )
    return m


def delta_accuracy(df_base, df_pert, model, task):
    m = _pair(df_base, df_pert, model, task)
    col, gold = f"{model}_{task}", f"gold_{task}"
    if len(m) == 0:
        return np.nan, 0
    def acc(pred, g):
        mask = ~(np.isnan(pred) | np.isnan(g))
        return float((pred[mask] == g[mask]).mean()) if mask.sum() else np.nan
    return acc(m[f"{col}_p"].values, m[f"{gold}_p"].values) - \
           acc(m[col].values,        m[gold].values), len(m)


# ── McNemar's test ────────────────────────────────────────────────────────────

def mcnemars(df_base, df_pert, model, task):
    """H0: accuracy unchanged baseline → perturbed. Returns (p, sig_level)."""
    m = _pair(df_base, df_pert, model, task)
    if len(m) == 0:
        return np.nan, "ns"
    col, gold = f"{model}_{task}", f"gold_{task}"

    pred_b = m[col].values
    pred_p = m[f"{col}_p"].values
    gold_b = m[gold].values
    gold_p = m[f"{gold}_p"].values

    valid  = ~(np.isnan(pred_b) | np.isnan(pred_p) |
               np.isnan(gold_b) | np.isnan(gold_p))
    bc = (pred_b[valid] == gold_b[valid]).astype(float)
    pc = (pred_p[valid] == gold_p[valid]).astype(float)

    b = int(((bc == 1) & (pc == 0)).sum())
    c = int(((bc == 0) & (pc == 1)).sum())

    if b + c == 0:
        return 1.0, "ns"

    if b + c < 25:
        p = float(binomtest(min(b, c), b + c, 0.5).pvalue)
    else:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p = float(chi2.sf(stat, df=1))

    if p < 0.01:
        return p, "p01"
    if p < 0.05:
        return p, "p05"
    return p, "ns"


# ── Flip rate helpers ─────────────────────────────────────────────────────────

def compute_flip_rate(df_base, df_pert, model, task):
    """
    Flip rate = % of vignettes where model prediction changes between
    baseline and perturbed variant (regardless of direction or correctness).

    Returns (flip_rate_pct, n_pairs).
    """
    col = f"{model}_{task}"
    m = df_base[["context_id", col]].merge(
        df_pert[["context_id", col]].rename(columns={col: f"{col}_p"}),
        on="context_id", how="inner"
    )
    bp = m[col].values
    pp = m[f"{col}_p"].values
    mask = ~(np.isnan(bp) | np.isnan(pp))
    if mask.sum() == 0:
        return np.nan, 0
    flipped = (bp[mask] != pp[mask]).sum()
    return float(flipped / mask.sum() * 100), int(mask.sum())


def compute_flip_stats(df, subset=None):
    """
    Returns flip_stats[model][task][pert] = (flip_rate_pct, "ns")
    (no significance test — flip rate is descriptive).
    """
    if subset:
        df = df[df["dataset"].isin(subset)]

    stats = {m: {t: {} for t in TASKS} for m in MODEL_ORDER}
    df_base = df[df["perturbation"] == "baseline"].copy()

    for pert in PERTURBS:
        df_pert = df[df["perturbation"] == pert].copy()
        if df_pert.empty:
            continue
        for model in MODEL_ORDER:
            for task in TASKS:
                rate, _ = compute_flip_rate(df_base, df_pert, model, task)
                stats[model][task][pert] = (rate, "ns")

    return stats


# ── MI helpers ────────────────────────────────────────────────────────────────

def compute_mi(df_base, df_pert, model, task):
    col = f"{model}_{task}"
    m = df_base[["context_id", col]].merge(
        df_pert[["context_id", col]].rename(columns={col: f"{col}_p"}),
        on="context_id", how="inner"
    )
    bp = m[col].values
    pp = m[f"{col}_p"].values
    mask = ~(np.isnan(bp) | np.isnan(pp))
    if mask.sum() < 2:
        return np.nan, 0
    return float(mutual_info_score(bp[mask].astype(int), pp[mask].astype(int))), int(mask.sum())


def mi_mannwhitney(df_base, df_pert, model, task, n_boot=N_BOOT, seed=42):
    """Bootstrap MI(base,pert) vs MI(base,base). Returns (p, sig_level)."""
    col = f"{model}_{task}"
    m = df_base[["context_id", col]].merge(
        df_pert[["context_id", col]].rename(columns={col: f"{col}_p"}),
        on="context_id", how="inner"
    )
    bp = m[col].values
    pp = m[f"{col}_p"].values
    mask = ~(np.isnan(bp) | np.isnan(pp))
    bp, pp = bp[mask].astype(int), pp[mask].astype(int)
    if len(bp) < 5:
        return np.nan, "ns"

    rng = np.random.default_rng(seed)

    def boot(a, b):
        out = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(a), size=len(a))
            out.append(mutual_info_score(a[idx], b[idx]))
        return np.array(out)

    mi_pert = boot(bp, pp)
    mi_base = boot(bp, bp)
    _, p = mannwhitneyu(mi_pert, mi_base, alternative="two-sided")
    p = float(p)

    if p < 0.01:
        return p, "p01"
    if p < 0.05:
        return p, "p05"
    return p, "ns"


# ── Pairwise one-sided McNemar's ─────────────────────────────────────────────

def mcnemars_onesided(df_base, df_pert, model_a, model_b, task):
    """
    One-sided McNemar's: H1 = model_a more accurate than model_b on perturbed vignettes.

    Pairs vignettes via context_id. Discordant cells:
      b = A correct, B wrong   (favours A)
      c = A wrong,   B correct (favours B)
    p = Binomial(b | n=b+c, p=0.5, alternative='greater')

    Returns p-value (NaN if insufficient paired data).
    """
    col_a = f"{model_a}_{task}"
    col_b = f"{model_b}_{task}"
    gold  = f"gold_{task}"

    # Merge perturbed predictions for both models with gold on same vignettes
    ma = df_pert[["context_id", col_a, gold]].dropna(subset=[col_a, gold])
    mb = df_pert[["context_id", col_b]].dropna(subset=[col_b])
    merged = ma.merge(mb, on="context_id", how="inner")
    if len(merged) == 0:
        return np.nan

    a_correct = (merged[col_a].values == merged[gold].values).astype(float)
    b_correct = (merged[col_b].values == merged[gold].values).astype(float)

    b = int(((a_correct == 1) & (b_correct == 0)).sum())   # A wins
    c = int(((a_correct == 0) & (b_correct == 1)).sum())   # B wins

    if b + c == 0:
        return 1.0

    return float(binomtest(b, b + c, 0.5, alternative="greater").pvalue)


def compute_pairwise(df, subset=None):
    """
    Returns pairwise[pert][task] = DataFrame (models × models) of p-values.
    matrix[i][j] = p-value for H1: model_i better than model_j.
    """
    if subset:
        df = df[df["dataset"].isin(subset)]

    df_base = df[df["perturbation"] == "baseline"].copy()
    result  = {}

    for pert in PERTURBS:
        df_pert = df[df["perturbation"] == pert].copy()
        result[pert] = {}
        for task in TASKS:
            mat = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
            for ma in MODEL_ORDER:
                for mb in MODEL_ORDER:
                    if ma == mb:
                        mat.loc[ma, mb] = np.nan
                    else:
                        mat.loc[ma, mb] = mcnemars_onesided(df_base, df_pert, ma, mb, task)
            result[pert][task] = mat

    return result


# ── Compute all stats ─────────────────────────────────────────────────────────

def compute_stats(df, subset=None):
    """
    Returns:
      acc_stats[model][task][pert] = (delta, sig_level)
      mi_stats [model][task][pert] = (mi,    sig_level)
    """
    if subset:
        df = df[df["dataset"].isin(subset)]

    acc_stats = {m: {t: {} for t in TASKS} for m in MODEL_ORDER}
    mi_stats  = {m: {t: {} for t in TASKS} for m in MODEL_ORDER}

    df_base = df[df["perturbation"] == "baseline"].copy()

    for pert in PERTURBS:
        df_pert = df[df["perturbation"] == pert].copy()
        if df_pert.empty:
            continue
        for model in MODEL_ORDER:
            for task in TASKS:
                delta, _   = delta_accuracy(df_base, df_pert, model, task)
                _, sig_acc = mcnemars(df_base, df_pert, model, task)
                acc_stats[model][task][pert] = (delta, sig_acc)

                mi, _     = compute_mi(df_base, df_pert, model, task)
                _, sig_mi = mi_mannwhitney(df_base, df_pert, model, task)
                mi_stats[model][task][pert] = (mi, sig_mi)

    return acc_stats, mi_stats


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _draw_bars(ax, data, ylabel, title, zero_line=True):
    """
    data[model][pert] = (value, sig_level)
    Colors by perturbation; hatch by significance.
    """
    n_models = len(MODEL_ORDER)
    n_perts  = len(PERTURBS)
    width    = 0.18
    offsets  = np.linspace(-(n_perts - 1) / 2, (n_perts - 1) / 2, n_perts) * width

    for pi, pert in enumerate(PERTURBS):
        for mi, model in enumerate(MODEL_ORDER):
            val, sig = data.get(model, {}).get(pert, (np.nan, "ns"))
            if np.isnan(val):
                continue
            x     = mi + offsets[pi]
            color = PERT_COLORS[pert]
            hatch = SIG_HATCH[sig]
            ax.bar(x, val, width,
                   color=color, hatch=hatch,
                   edgecolor="white" if not hatch else "black",
                   linewidth=0.5 if not hatch else 0.8,
                   alpha=0.85)

    if zero_line:
        ax.axhline(0, color="black", linewidth=0.9, zorder=0)

    ax.set_xticks(range(n_models))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)


def make_legend(n_per_pert: dict | None = None):
    """
    Perturbation colours + significance hatches.
    n_per_pert: {pert: (n_total, n_clinician)} — adds sample sizes to labels.
    """
    pert_patches = []
    for p in PERTURBS:
        if n_per_pert and p in n_per_pert:
            n, _ = n_per_pert[p]
            label = f"{PERT_LABELS[p]}  (n={n})"
        else:
            label = PERT_LABELS[p]
        pert_patches.append(mpatches.Patch(facecolor=PERT_COLORS[p], label=label))

    sig_patches = [
        mpatches.Patch(facecolor="lightgrey", hatch=SIG_HATCH["p05"],
                       edgecolor="black", label="p < 0.05"),
        mpatches.Patch(facecolor="lightgrey", hatch=SIG_HATCH["p01"],
                       edgecolor="black", label="p < 0.01"),
    ]
    return pert_patches + sig_patches


def _n_counts(df, subset=None):
    """
    Returns {pert: (n_total, n_clinician)} for the given dataset subset.
    """
    if subset:
        df = df[df["dataset"].isin(subset)]
    counts = {}
    for p in PERTURBS:
        sub = df[df["perturbation"] == p]
        n_total = len(sub)
        n_clin  = int(sub["clinician_manage"].notna().sum()) if "clinician_manage" in sub.columns else 0
        counts[p] = (n_total, n_clin)
    return counts


def _task_title(task, n_clinician):
    """Task label with clinician n appended."""
    return f"{TASK_LABELS[task]}  (clinician n={n_clinician})"


# ── Figure builders ───────────────────────────────────────────────────────────

def _fig_1row(stats, ylabel, suptitle, fname, zero_line=True, df=None):
    """3 panels (one per task), single dataset group."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    n_per_pert = _n_counts(df) if df is not None else {}

    for ax, task in zip(axes, TASKS):
        data = {m: {p: stats[m][task].get(p, (np.nan, "ns")) for p in PERTURBS}
                for m in MODEL_ORDER}
        # Clinician n is the min across perturbations (conservative label)
        n_clin = min((v[1] for v in n_per_pert.values()), default=0) if n_per_pert else ""
        title  = _task_title(task, n_clin) if n_clin else TASK_LABELS[task]
        _draw_bars(ax, data, ylabel=ylabel if task == "manage" else "",
                   title=title, zero_line=zero_line)
        if zero_line:
            lim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]), 0.12)
            ax.set_ylim(-lim, lim + 0.05)
        else:
            ax.set_ylim(0, ax.get_ylim()[1] + 0.02)

    legend = make_legend(n_per_pert)
    fig.legend(handles=legend, loc="lower center", ncol=len(legend),
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def _fig_2row(stats_ds, ylabel, suptitle, fname, zero_line=True, df=None):
    """6 panels (2 datasets × 3 tasks)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey="row")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    for ri, ds in enumerate(DATASETS):
        stats = stats_ds[ds]
        n_per_pert = _n_counts(df, subset=[ds]) if df is not None else {}
        for ci, task in enumerate(TASKS):
            ax = axes[ri, ci]
            data = {m: {p: stats[m][task].get(p, (np.nan, "ns")) for p in PERTURBS}
                    for m in MODEL_ORDER}
            n_clin = min((v[1] for v in n_per_pert.values()), default=0) if n_per_pert else ""
            title  = f"{DS_LABELS[ds]} — {_task_title(task, n_clin)}" if n_clin else f"{DS_LABELS[ds]} — {TASK_LABELS[task]}"
            _draw_bars(ax, data,
                       ylabel=ylabel if ci == 0 else "",
                       title=title,
                       zero_line=zero_line)
            if zero_line:
                lim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]), 0.12)
                ax.set_ylim(-lim, lim + 0.05)
            else:
                ax.set_ylim(0, ax.get_ylim()[1] + 0.02)

        legend = make_legend(n_per_pert)
        if ri == len(DATASETS) - 1:
            fig.legend(handles=legend, loc="lower center", ncol=len(legend),
                       frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── Fig 5: Pairwise delta accuracy heatmap ────────────────────────────────────

def _pairwise_robustness(df_base, df_pert, task):
    """
    For every pair (A, B) compute:
      cell[A,B] = delta_acc(A) − delta_acc(B)
      where delta_acc(X) = acc(X, perturbed) − acc(X, baseline)

    Positive → A drops less than B (A more robust).

    Per-vignette robustness score for model X:
      r_X_i = correct(X, pert_i) − correct(X, base_i)  ∈ {-1, 0, 1}

    Significance: Wilcoxon signed-rank on (r_A_i − r_B_i), Bonferroni corrected.
    """
    gold = f"gold_{task}"
    n    = len(MODEL_ORDER)
    delta_mat = pd.DataFrame(np.nan, index=MODEL_ORDER, columns=MODEL_ORDER)
    sig_mat   = pd.DataFrame("ns",   index=MODEL_ORDER, columns=MODEL_ORDER, dtype=object)

    for ma in MODEL_ORDER:
        for mb in MODEL_ORDER:
            if ma == mb:
                continue
            col_a, col_b = f"{ma}_{task}", f"{mb}_{task}"

            # Need baseline AND perturbed predictions for both models + gold
            # Merge baseline rows
            base_cols = ["context_id", col_a, col_b, gold]
            pert_cols = ["context_id", col_a, col_b, gold]
            avail_b = [c for c in base_cols if c in df_base.columns]
            avail_p = [c for c in pert_cols if c in df_pert.columns]
            if len(avail_b) < 4 or len(avail_p) < 4:
                continue

            base = df_base[avail_b].dropna().rename(
                columns={col_a: f"{col_a}_b", col_b: f"{col_b}_b", gold: f"{gold}_b"})
            pert = df_pert[avail_p].dropna().rename(
                columns={col_a: f"{col_a}_p", col_b: f"{col_b}_p", gold: f"{gold}_p"})

            m = base.merge(pert, on="context_id", how="inner")
            if len(m) == 0:
                continue

            # Per-vignette robustness: correct(pert) - correct(base)
            r_a = ((m[f"{col_a}_p"] == m[f"{gold}_p"]).astype(float) -
                   (m[f"{col_a}_b"] == m[f"{gold}_b"]).astype(float))
            r_b = ((m[f"{col_b}_p"] == m[f"{gold}_p"]).astype(float) -
                   (m[f"{col_b}_b"] == m[f"{gold}_b"]).astype(float))

            delta_mat.loc[ma, mb] = float(r_a.mean() - r_b.mean())

            # Wilcoxon signed-rank on per-vignette difference
            diff = (r_a - r_b).values
            if len(diff) < 5 or np.all(diff == 0):
                p = 1.0
            else:
                from scipy.stats import wilcoxon
                try:
                    _, p = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
                    p = float(p)
                except Exception:
                    p = 1.0

            # Bonferroni
            p_corr = min(p * n * (n - 1), 1.0)
            if p_corr < 0.01:
                sig_mat.loc[ma, mb] = "p01"
            elif p_corr < 0.05:
                sig_mat.loc[ma, mb] = "p05"
            else:
                sig_mat.loc[ma, mb] = "ns"

    return delta_mat, sig_mat


def build_fig_pairwise(df, suptitle, fname):
    """
    3 rows (tasks) × 3 cols (perturbations), each panel a 5×5 delta-accuracy matrix.

    Colour  : diverging RdBu — blue = row model better, red = col model better.
    Text    : Δ value with significance star(s) (* p<0.05, ** p<0.01).
    Diagonal: grey.
    """
    n_models = len(MODEL_ORDER)
    short    = [MODEL_LABELS[m] for m in MODEL_ORDER]
    cmap     = plt.cm.get_cmap("RdBu")          # red=negative, blue=positive

    fig, axes = plt.subplots(len(TASKS), len(PERTURBS),
                             figsize=(4.5 * len(PERTURBS), 4.2 * len(TASKS)))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)

    df_base = df[df["perturbation"] == "baseline"].copy()

    # Global colour scale across all panels
    all_deltas = []
    precomp = {}
    for pert in PERTURBS:
        df_pert = df[df["perturbation"] == pert].copy()
        precomp[pert] = {}
        for task in TASKS:
            dm, sm = _pairwise_robustness(df_base, df_pert, task)
            precomp[pert][task] = (dm, sm)
            all_deltas += dm.values.flatten().tolist()

    all_deltas = [v for v in all_deltas if not np.isnan(v)]
    vlim = max(abs(np.nanmin(all_deltas)), abs(np.nanmax(all_deltas)), 0.05)

    for ri, task in enumerate(TASKS):
        for ci, pert in enumerate(PERTURBS):
            ax = axes[ri, ci]
            dm, sm = precomp[pert][task]

            vals = dm.values.astype(float)
            img  = ax.imshow(vals, vmin=-vlim, vmax=vlim, cmap=cmap, aspect="auto")

            # Blank lower triangle and diagonal
            for i in range(n_models):
                for j in range(i + 1):   # j <= i → lower triangle + diagonal
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 color="white", zorder=2))
                    if i == j:
                        ax.text(j, i, "—", ha="center", va="center",
                                fontsize=9, color="#aaaaaa", zorder=3)

            # Annotate upper triangle only
            for i in range(n_models):
                for j in range(i + 1, n_models):   # j > i → upper triangle
                    v   = vals[i, j]
                    sig = sm.loc[MODEL_ORDER[i], MODEL_ORDER[j]]
                    if np.isnan(v):
                        ax.text(j, i, "n/a", ha="center", va="center",
                                fontsize=7, color="grey")
                        continue
                    star = {"p01": "**", "p05": "*", "ns": ""}[sig]
                    txt_col = "white" if abs(v) > vlim * 0.6 else "black"
                    ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center",
                            fontsize=7.5, color=txt_col, fontweight="bold" if star else "normal")

            ax.set_xticks(range(n_models))
            ax.set_yticks(range(n_models))
            ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
            ax.set_yticklabels(short, fontsize=8)
            ax.set_xlabel("Model B", fontsize=8)
            ax.set_ylabel("Model A" if ci == 0 else "", fontsize=8)
            ax.set_title(f"{TASK_LABELS[task]} — {PERT_LABELS[pert]}",
                         fontweight="bold", fontsize=9)

    # Shared colorbar
    cbar_ax = fig.add_axes([1.01, 0.15, 0.015, 0.7])
    sm_cb   = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-vlim, vmax=vlim))
    sm_cb.set_array([])
    cb = fig.colorbar(sm_cb, cax=cbar_ax)
    cb.set_label("Δ Accuracy  [A − B]", fontsize=9)

    fig.text(0.5, -0.02,
             "Cell [A, B] = Δacc(A) − Δacc(B)  where Δacc = acc(perturbed) − acc(baseline).  "
             "Blue = A drops less (more robust).  Red = B drops less.\n"
             "* p<0.05   ** p<0.01   (Wilcoxon signed-rank on per-vignette robustness, Bonferroni corrected)",
             ha="center", fontsize=8.5, style="italic")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _bonferroni_correct(pairwise):
    """
    Apply Bonferroni correction within each (pert, task) matrix.
    n_comparisons = n_models * (n_models - 1)  (all off-diagonal cells).
    Modifies p-values in-place.
    """
    n_comp = len(MODEL_ORDER) * (len(MODEL_ORDER) - 1)
    for pert in PERTURBS:
        for task in TASKS:
            mat = pairwise[pert][task]
            for ma in MODEL_ORDER:
                for mb in MODEL_ORDER:
                    if ma != mb and not np.isnan(mat.loc[ma, mb]):
                        mat.loc[ma, mb] = min(mat.loc[ma, mb] * n_comp, 1.0)


def main():
    print("Loading data...")
    df = load_data()
    print(f"  Loaded {len(df)} rows  |  models: {MODEL_ORDER}")

    print("Computing aggregated stats (OncQA + AskDocs)...")
    acc_agg, mi_agg = compute_stats(df)
    flip_agg = compute_flip_stats(df)

    print("Computing per-dataset stats...")
    acc_ds, mi_ds, flip_ds = {}, {}, {}
    for ds in DATASETS:
        acc_ds[ds],  mi_ds[ds]  = compute_stats(df, subset=[ds])
        flip_ds[ds]             = compute_flip_stats(df, subset=[ds])

    print("Generating figures...")

    _fig_1row(acc_agg,
              ylabel="Δ Accuracy (perturbed − baseline)",
              suptitle="Fig 1 — Δ Accuracy vs Baseline  (OncQA + AskDocs only)",
              fname="fig1_delta_accuracy_aggregate.png",
              zero_line=True, df=df)

    _fig_2row(acc_ds,
              ylabel="Δ Accuracy (perturbed − baseline)",
              suptitle="Fig 2 — Δ Accuracy vs Baseline  by Dataset",
              fname="fig2_delta_accuracy_by_dataset.png",
              zero_line=True, df=df)

    _fig_1row(mi_agg,
              ylabel="Mutual Information (bits)",
              suptitle="Fig 3 — MI: Prediction Stability vs Baseline  (OncQA + AskDocs only)\n"
                       "Higher MI = predictions more stable under perturbation",
              fname="fig3_mi_aggregate.png",
              zero_line=False, df=df)

    _fig_2row(mi_ds,
              ylabel="Mutual Information (bits)",
              suptitle="Fig 4 — MI: Prediction Stability vs Baseline  by Dataset",
              fname="fig4_mi_by_dataset.png",
              zero_line=False, df=df)

    build_fig_pairwise(
        df,
        suptitle="Fig 5 — Pairwise Robustness Comparison  [Δacc(A) − Δacc(B)]\n"
                 "Blue = A more robust (drops less).  Red = B more robust.  * p<0.05  ** p<0.01",
        fname="fig5_pairwise_robustness.png",
    )

    _fig_1row(flip_agg,
              ylabel="Flip rate (%)",
              suptitle="Fig 6 — Flip Rate: % Predictions Changed  (OncQA + AskDocs only)\n"
                       "Flip = prediction differs between baseline and perturbed (any direction)",
              fname="fig6_flip_rate_aggregate.png",
              zero_line=False, df=df)

    _fig_2row(flip_ds,
              ylabel="Flip rate (%)",
              suptitle="Fig 7 — Flip Rate: % Predictions Changed  by Dataset",
              fname="fig7_flip_rate_by_dataset.png",
              zero_line=False, df=df)

    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Hatch key:  //// = p<0.05   xxxx = p<0.01  (McNemar's for acc, Mann-Whitney U for MI)")
    print("Fig 5:  one-sided McNemar's, Bonferroni corrected within each (pert × task)")
    print("Figs 6–7: flip rate is descriptive (no significance test)")


if __name__ == "__main__":
    main()
