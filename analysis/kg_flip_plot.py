"""
KG-split Flip Rate plot — built iteratively.

Current state:
  MedReason (Llama 8B) vs Baseline (Llama 8B) on KG-found rows.
  1 row × 3 cols (tasks), 2 bars per perturbation, shared y-axis.
  Bar height = mean across 3 runs; error bar = std across 3 runs.

Output:
  plots/rcer/kg_flip_llama8b_mra_vs_base.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick

REPO_ROOT    = Path(__file__).resolve().parent.parent
OUTPUT_BASE  = REPO_ROOT / "output"
MODEL_RESP   = REPO_ROOT.parent / "data" / "model_responses.csv"
PLOT_DIR     = REPO_ROOT / "plots" / "rcer"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS    = {"oncqa", "askdocs"}
TASKS       = ["manage", "visit", "resource"]
PERTS       = ["summary", "gender_swap", "uncertain_tone"]
PERT_LABELS = {"summary": "Sum.", "gender_swap": "GS", "uncertain_tone": "UT"}
TASK_LABELS = {"manage": "Manage", "visit": "Visit", "resource": "Resource"}

C_MRA  = "#5c85d6"   # blue  — MedReason
C_BASE = "#9e9e9e"   # grey  — Baseline
FONT = 25


# ── helpers ────────────────────────────────────────────────────────────────────

def get_kg_cids(backbone: str) -> set[str]:
    """Context IDs where MedReason found KG paths (from run_0)."""
    df = pd.read_csv(OUTPUT_BASE / f"medreason/{backbone}/oncqa_askdocs/run_0/results.csv")
    has_kg = (df["kg_paths"].notna()
              & (df["kg_paths"].astype(str).str.strip() != "")
              & (df["kg_paths"].astype(str) != "nan"))
    return set(df[has_kg]["context_id"].astype(str).unique())


def flip_rate(pert_df: pd.DataFrame, base_df: pd.DataFrame,
              pred_col: str, cid_filter: set) -> float:
    """Flip rate on a filtered subset of context_ids."""
    sub = pert_df[pert_df["context_id"].astype(str).isin(cid_filter)]
    if pred_col not in sub.columns:
        return np.nan
    m = (sub[["context_id", pred_col]].rename(columns={pred_col: "pp"})
         .merge(base_df[["context_id", pred_col]].rename(columns={pred_col: "pb"}),
                on="context_id"))
    pp    = m["pp"].fillna("").astype(str).str.upper()
    pb    = m["pb"].fillna("").astype(str).str.upper()
    valid = (pp != "") & (pb != "")
    return 100.0 * (pp[valid] != pb[valid]).mean() if valid.sum() > 0 else np.nan


def get_pred_prefix(df: pd.DataFrame) -> str | None:
    col = next((c for c in df.columns
                if "manage" in c and "gold" not in c
                and "reasoning" not in c and "standard" not in c), None)
    return col.replace("_manage", "") if col else None


# ── data collection ────────────────────────────────────────────────────────────

def collect_records(backbone: str, kg_cids: set) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      model, run, pert, task, flip
    for both MedReason and Baseline on KG-found rows.
    """
    mra_prefix = f"medreason_{backbone}"
    records = []

    # MedReason runs
    for run_n in range(3):
        p = OUTPUT_BASE / f"medreason/{backbone}/oncqa_askdocs/run_{run_n}/results.csv"
        if not p.exists():
            continue
        df      = pd.read_csv(p)
        base_df = df[df["perturbation"] == "baseline"]
        for pert in PERTS:
            pert_df = df[df["perturbation"] == pert]
            for task in TASKS:
                v = flip_rate(pert_df, base_df, f"{mra_prefix}_{task}", kg_cids)
                records.append({"model": "MedReason", "run": run_n,
                                 "pert": pert, "task": task, "flip": v})

    # Baseline runs
    for run_n in range(3):
        p = OUTPUT_BASE / f"baseline/{backbone}/oncqa_askdocs/run_{run_n}/results.csv"
        if not p.exists():
            continue
        df      = pd.read_csv(p)
        pfx     = get_pred_prefix(df)
        if pfx is None:
            continue
        base_df = df[df["perturbation"] == "baseline"]
        for pert in PERTS:
            pert_df = df[df["perturbation"] == pert]
            for task in TASKS:
                v = flip_rate(pert_df, base_df, f"{pfx}_{task}", kg_cids)
                records.append({"model": "Baseline", "run": run_n,
                                 "pert": pert, "task": task, "flip": v})

    return pd.DataFrame(records)


# ── plot ───────────────────────────────────────────────────────────────────────

def plot_mra_vs_base(backbone: str = "llama8b3_1",
                     outname: str = "kg_flip_llama8b_mra_vs_base.png"):
    kg_cids = get_kg_cids(backbone)
    print(f"KG-found cids: {len(kg_cids)}")

    df_rec = collect_records(backbone, kg_cids)
    stats  = (df_rec.groupby(["model", "pert", "task"])["flip"]
              .agg(mean="mean", std=lambda x: x.std(ddof=0))
              .reset_index())

    FONT  = 25
    bar_w = 0.32
    x     = np.arange(len(PERTS))
    models = [("MedReason", C_MRA,  "MedReason (KG-found rows)"),
              ("Baseline",  C_BASE, "Baseline (KG-found rows)")]

    # KG-found vs No KG counts per perturbation (including baseline)
    df0 = pd.read_csv(OUTPUT_BASE / f"medreason/{backbone}/oncqa_askdocs/run_0/results.csv")
    df0["has_kg"] = (df0["kg_paths"].notna()
                     & (df0["kg_paths"].astype(str).str.strip() != "")
                     & (df0["kg_paths"].astype(str) != "nan"))
    all_perts = ["baseline"] + PERTS
    all_pert_labels = {"baseline": "Base.", **PERT_LABELS}
    kg_counts  = [int(df0[df0["perturbation"] == p]["has_kg"].sum())    for p in all_perts]
    nokg_counts= [int((~df0[df0["perturbation"] == p]["has_kg"]).sum()) for p in all_perts]
    for p, k, n in zip(all_perts, kg_counts, nokg_counts):
        print(f"  {p:<20} KG-found={k}  No KG={n}  Total={k+n}")

    # ── Figure: 1×3 flip rate panels ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.patch.set_facecolor("white")

    for ci, task in enumerate(TASKS):
        ax = axes[ci]
        ax.set_facecolor("white")

        for mi, (model, color, _) in enumerate(models):
            sub   = stats[(stats["model"] == model) & (stats["task"] == task)].set_index("pert")
            means = [sub.loc[p, "mean"] if p in sub.index else 0.0 for p in PERTS]
            stds  = [sub.loc[p, "std"]  if p in sub.index else 0.0 for p in PERTS]
            offset = (mi - 0.5) * bar_w
            ax.bar(x + offset, means, bar_w * 0.92, color=color,
                   edgecolor="white", linewidth=0.5,
                   yerr=stds, error_kw=dict(elinewidth=1.4, capsize=4,
                                             ecolor="black", capthick=1.2))

        ax.set_title(TASK_LABELS[task], fontsize=FONT + 1, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([PERT_LABELS[p] for p in PERTS], fontsize=FONT)

        ax.set_ylim(0, None)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.tick_params(axis="y", labelsize=FONT - 2)
        if ci == 0:
            ax.set_ylabel("Flip Rate (%)", fontsize=FONT)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(facecolor=c, label=lbl) for _, c, lbl in models]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=FONT - 1,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out = PLOT_DIR / outname
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {out}")


# ── path-length helpers ────────────────────────────────────────────────────────

import ast as _ast


def _parse_paths(v) -> list[str]:
    """Return list of individual path strings from kg_paths cell value."""
    if pd.isna(v) or str(v).strip() in ("", "nan", "[]"):
        return []
    try:
        parsed = _ast.literal_eval(str(v))
        raw = "\n".join(str(p) for p in parsed) if isinstance(parsed, list) else str(parsed)
    except Exception:
        raw = str(v)
    return [p.strip() for p in raw.split("\n") if p.strip()]


def _mean_path_len(v) -> float:
    """Mean number of nodes (edges+1) across paths for one context."""
    paths = _parse_paths(v)
    if not paths:
        return np.nan
    return float(np.mean([len(p.split(" -> ")) for p in paths]))


# ── rank-based equal-count bins ───────────────────────────────────────────────

def _get_bins_for_pert(pert: str, df0: pd.DataFrame, base0: pd.DataFrame,
                       n_bins: int = 5):
    """
    Compute n_bins equal-count bins for the contexts that belong to `pert`
    and have KG paths.

    Returns
    -------
    ctx_df     : DataFrame with columns [context_id, mean_path_len, quintile]
    bin_labels : dict  quintile → x-axis label (mean path len, 2 dp)
    bin_n      : Series quintile → count
    pert_kg_cids : set of context_id strings that are in this perturbation's
                   KG-found set
    """
    pert_cids = set(df0[df0["perturbation"] == pert]["context_id"].astype(str))
    sub = base0[base0["context_id"].astype(str).isin(pert_cids)
                & base0["mean_path_len"].notna()].copy()
    sub["context_id"] = sub["context_id"].astype(str)

    len_lookup = sub.set_index("context_id")["mean_path_len"].to_dict()

    ctx_df = (pd.DataFrame({"context_id": list(len_lookup),
                             "mean_path_len": list(len_lookup.values())})
              .sort_values("mean_path_len")
              .reset_index(drop=True))
    n = len(ctx_df)
    ctx_df["quintile"] = ctx_df.index * n_bins // n   # rank-based equal-count

    bin_avg    = ctx_df.groupby("quintile")["mean_path_len"].mean().round(2)
    bin_n      = ctx_df.groupby("quintile").size()
    bin_labels = {q: f"{avg:.2f}" for q, avg in bin_avg.items()}

    return ctx_df, bin_labels, bin_n, set(len_lookup.keys())


def _collect_bin_rates(run_dfs_prefixes: list, pert: str,
                       ctx_df: pd.DataFrame, pert_kg_cids: set,
                       n_bins: int = 5) -> dict[int, list[float]]:
    """
    For each run, compute the overall flip rate (across all tasks) for each bin.

    Parameters
    ----------
    run_dfs_prefixes : list of (df, prefix) tuples — one entry per run
    pert             : perturbation name
    ctx_df           : output of _get_bins_for_pert — has context_id + quintile
    pert_kg_cids     : KG-found context_ids for this perturbation
    """
    run_bin_rates: dict[int, list[float]] = {q: [] for q in range(n_bins)}

    for df, prefix in run_dfs_prefixes:
        base_df = df[df["perturbation"] == "baseline"]
        sub     = df[df["perturbation"] == pert]
        sub     = sub[sub["context_id"].astype(str).isin(pert_kg_cids)]

        for q in range(n_bins):
            bin_cids = set(ctx_df[ctx_df["quintile"] == q]["context_id"].astype(str))
            flips, total = 0, 0
            for task in TASKS:
                col = f"{prefix}_{task}"
                if col not in df.columns:
                    continue
                m = (sub[sub["context_id"].astype(str).isin(bin_cids)]
                     [["context_id", col]]
                     .rename(columns={col: "pp"})
                     .merge(base_df[["context_id", col]]
                            .rename(columns={col: "pb"}),
                            on="context_id"))
                pp    = m["pp"].fillna("").astype(str).str.upper()
                pb    = m["pb"].fillna("").astype(str).str.upper()
                valid = (pp != "") & (pb != "")
                flips += int((pp[valid] != pb[valid]).sum())
                total += int(valid.sum())
            if total > 0:
                run_bin_rates[q].append(100.0 * flips / total)

    return run_bin_rates


# ── main path-length plot ──────────────────────────────────────────────────────

def plot_flip_by_path_length_all_perts(
        backbone:      str  = "llama8b3_1",
        outname:       str  = "kg_flip_path_length_all_perts_llama8b.png",
        bins_per_pert: dict | None = None):
    """
    1×3 panel plot (one panel per perturbation) of flip rate vs mean KG path
    length, using equal-count (rank-based) bins per perturbation.

    bins_per_pert : dict mapping perturbation name → number of bins.
                    Defaults to 5 for summary, 10 for gender_swap / uncertain_tone.

    Two lines per panel:
      - MedReason (KG-found rows)  [blue]
      - Baseline  (same rows)      [grey]

    Error bars = std across 3 runs.
    Bin sizes are printed to terminal; no annotations in the figure.
    """
    if bins_per_pert is None:
        bins_per_pert = {"summary": 8, "gender_swap": 10, "uncertain_tone": 10}

    PERT_FULL = {"summary": "Summary",
                 "gender_swap": "Gender Swap",
                 "uncertain_tone": "Uncertain Tone"}

    # ── load run_0 once for path-length lookup ────────────────────────────────
    df0   = pd.read_csv(OUTPUT_BASE / f"medreason/{backbone}/oncqa_askdocs/run_0/results.csv")
    base0 = df0[df0["perturbation"] == "baseline"].copy()
    base0["mean_path_len"] = base0["kg_paths"].apply(_mean_path_len)
    base0["context_id"]    = base0["context_id"].astype(str)

    mra_prefix = f"medreason_{backbone}"

    # ── load all runs (MedReason + Baseline) ─────────────────────────────────
    mra_run_dfs: list = []
    for run_n in range(3):
        p = OUTPUT_BASE / f"medreason/{backbone}/oncqa_askdocs/run_{run_n}/results.csv"
        if p.exists():
            mra_run_dfs.append((pd.read_csv(p), mra_prefix))

    base_run_dfs: list = []
    for run_n in range(3):
        p = OUTPUT_BASE / f"baseline/{backbone}/oncqa_askdocs/run_{run_n}/results.csv"
        if not p.exists():
            continue
        df  = pd.read_csv(p)
        pfx = get_pred_prefix(df)
        if pfx:
            base_run_dfs.append((df, pfx))

    # ── one panel per perturbation ────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(PERTS), figsize=(22, 5), sharey=True)
    fig.patch.set_facecolor("white")

    for ci, pert in enumerate(PERTS):
        ax     = axes[ci]
        n_bins = bins_per_pert.get(pert, 5)
        ax.set_facecolor("white")

        ctx_df, bin_labels, bin_n, pert_kg_cids = _get_bins_for_pert(
            pert, df0, base0, n_bins)

        # terminal output
        sizes_str = "  ".join(f"Q{q+1} n={int(bin_n[q])}" for q in range(n_bins))
        print(f"{PERT_FULL[pert]} ({n_bins} bins) — {sizes_str}")

        x_pos  = np.arange(n_bins)
        x_lbls = [bin_labels[q] for q in range(n_bins)]

        for model_label, run_dfs, color in [
            ("MedReason (KG-found rows)", mra_run_dfs,  C_MRA),
            ("Baseline (KG-found rows)",  base_run_dfs, C_BASE),
        ]:
            rates = _collect_bin_rates(run_dfs, pert, ctx_df, pert_kg_cids, n_bins)
            means = [np.mean(rates[q]) if rates[q] else np.nan for q in range(n_bins)]
            stds  = [np.std(rates[q], ddof=0) if len(rates[q]) > 1 else 0.0
                     for q in range(n_bins)]
            ax.errorbar(x_pos, means, yerr=stds,
                        marker="o", linewidth=2, markersize=7,
                        capsize=5, capthick=1.5, elinewidth=1.5,
                        color=color, label=model_label)

        ax.set_title(PERT_FULL[pert], fontsize=FONT, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_lbls, fontsize=FONT - 5, rotation=45, ha="right")
        ax.set_xlabel("Mean KG path length (nodes)", fontsize=FONT - 2, labelpad=8)
        if ci == 0:
            ax.set_ylabel("Flip Rate (%)", fontsize=FONT)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(0, None)
        ax.tick_params(axis="y", labelsize=FONT - 4)
        ax.grid(alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    # shared legend below figure
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2,
               fontsize=FONT - 3, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = PLOT_DIR / outname
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved → {out}")


def print_upside_semidev(backbone: str = "llama8b3_1",
                         bins_per_pert: dict | None = None):
    """
    For each perturbation × model, compute upside semi-deviation from Q1:

        USD = sqrt( mean( max(0, rate_i - rate_Q1)^2 )  for i = Q2..QN )

    where rate_i is the mean flip rate (across runs) for bin i.
    Higher USD = flip rate degraded more at longer path lengths vs the reference.
    Printed to terminal only.
    """
    if bins_per_pert is None:
        bins_per_pert = {"summary": 8, "gender_swap": 10, "uncertain_tone": 10}

    PERT_FULL  = {"summary": "Summary",
                  "gender_swap": "Gender Swap",
                  "uncertain_tone": "Uncertain Tone"}

    df0   = pd.read_csv(OUTPUT_BASE / f"medreason/{backbone}/oncqa_askdocs/run_0/results.csv")
    base0 = df0[df0["perturbation"] == "baseline"].copy()
    base0["mean_path_len"] = base0["kg_paths"].apply(_mean_path_len)
    base0["context_id"]    = base0["context_id"].astype(str)

    mra_prefix = f"medreason_{backbone}"

    mra_run_dfs: list = []
    for run_n in range(3):
        p = OUTPUT_BASE / f"medreason/{backbone}/oncqa_askdocs/run_{run_n}/results.csv"
        if p.exists():
            mra_run_dfs.append((pd.read_csv(p), mra_prefix))

    base_run_dfs: list = []
    for run_n in range(3):
        p = OUTPUT_BASE / f"baseline/{backbone}/oncqa_askdocs/run_{run_n}/results.csv"
        if not p.exists():
            continue
        df  = pd.read_csv(p)
        pfx = get_pred_prefix(df)
        if pfx:
            base_run_dfs.append((df, pfx))

    print("\n" + "=" * 100)
    print("Directional stability metrics (reference = Q1, shortest-path bin)")
    print("  Volatility = std of all bin means  |  USD = sqrt(mean(max(0, r_i - r_Q1)^2))")
    print("=" * 100)

    results = []
    for pert in PERTS:
        n_bins = bins_per_pert.get(pert, 5)
        ctx_df, bin_labels, bin_n, pert_kg_cids = _get_bins_for_pert(
            pert, df0, base0, n_bins)

        for model_label, run_dfs in [("MedReason", mra_run_dfs),
                                      ("Baseline",  base_run_dfs)]:
            rates     = _collect_bin_rates(run_dfs, pert, ctx_df, pert_kg_cids, n_bins)
            bin_means = np.array([np.mean(rates[q]) if rates[q] else np.nan
                                  for q in range(n_bins)])

            volatility = float(np.nanstd(bin_means, ddof=0))

            ref    = bin_means[0]
            deltas = bin_means[1:] - ref
            usd    = float(np.sqrt(np.nanmean(np.maximum(0, deltas) ** 2)))

            bm_str = "  ".join(f"{v:5.1f}" for v in bin_means)
            print(f"{PERT_FULL[pert]:<16}  {model_label:<12}  "
                  f"[{bm_str}]  vol={volatility:5.2f}%  USD={usd:5.2f}%")
            results.append({"pert": PERT_FULL[pert], "model": model_label,
                            "Volatility": round(volatility, 2), "USD": round(usd, 2)})

        print()

    print("Summary:")
    df_res = pd.DataFrame(results)
    for metric in ["Volatility", "USD"]:
        piv = df_res.pivot(index="pert", columns="model", values=metric)
        piv["Ratio (Base/MRA)"] = (piv["Baseline"] / piv["MedReason"]).round(1)
        print(f"\n{metric} (%):")
        print(piv.to_string())


if __name__ == "__main__":
    plot_mra_vs_base()
    plot_flip_by_path_length_all_perts()
    print_upside_semidev()
