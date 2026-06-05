"""
Pairwise significance heatmap for Cochran's Q post-hoc McNemar tests.

Two panels side by side:
  Left:  RCER  — binary correct/error on care-augmenting cases (gold standard)
  Right: Flip  — binary stable/flipped from baseline

Rows    = backbone × task (9 combinations)
Columns = 6 method pairs
Cells   = Bonferroni-corrected McNemar p-value, coloured by significance tier

Usage:
    cd medperturb_eval
    conda run -n curebench python analysis/plot_cochrans_heatmap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parent.parent
OUTPUT_BASE    = REPO_ROOT / "output"
MEDPERTURB_CSV = REPO_ROOT.parent / "data" / "medperturb_data.csv"
PLOT_DIR       = REPO_ROOT / "plots" / "cochrans"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent / "tooluni_agent"))

DATASETS = {"oncqa", "askdocs"}
TASKS    = ["manage", "visit", "resource"]
PERTS    = ["summary", "gender_swap", "uncertain_tone"]

BACKBONE_ORDER  = ["llama70b3_3", "qwen235b", "llama8b3_1"]
BACKBONE_LABELS = {
    "llama70b3_3": "Llama 70B",
    "qwen235b":    "Qwen 235B",
    "llama8b3_1":  "Llama 8B",
}
TASK_LABELS = {"manage": "Manage", "visit": "Visit", "resource": "Resource"}

METHOD_ORDER  = ["Baseline", "MRA", "TUA", "MPA"]
METHOD_DIRS   = {"Baseline": "baseline", "MRA": "medreason",
                 "TUA": "tooluni",      "MPA": "medpathagent"}
BASELINE_PREFIX = {"llama70b3_3": "llama"}

PAIRS = [(a, b) for i, a in enumerate(METHOD_ORDER)
                for b in METHOD_ORDER[i+1:]]
PAIR_LABELS = [f"{a}\nvs {b}" for a, b in PAIRS]

# Direction-aware colours
# RIGHT method wins (e.g. TUA beats Baseline in "Baseline vs TUA") → blue
# LEFT  method wins (unexpected direction)                          → orange
COLORS_RIGHT = {
    "ns":  "#f0f0f0",
    "*":   "#a8d8ea",   # light blue
    "**":  "#3a86ff",   # blue
    "***": "#023e8a",   # dark blue
}
COLORS_LEFT = {
    "ns":  "#f0f0f0",
    "*":   "#ffd6a5",   # light orange
    "**":  "#f4845f",   # orange
    "***": "#9b2226",   # dark red-orange
}
TEXT_COLOR = {
    "ns": "#aaaaaa", "*": "#1a1a2e", "**": "white", "***": "white"
}

# ── helpers ────────────────────────────────────────────────────────────────────

def care_augmenting(task):  return "NO"  if task == "manage" else "YES"
def reduced_care(task):     return "YES" if task == "manage" else "NO"


def build_gold_lookup():
    mp = pd.read_csv(MEDPERTURB_CSV)
    mp = mp[(mp["perturbation"] == "baseline") & mp["dataset"].isin(DATASETS)]
    lookup = {}
    for _, row in mp.iterrows():
        cid = str(row["context_id"])
        for task in TASKS:
            gold = row.get(f"gold_standard_{task}", "")
            if pd.notna(gold) and str(gold).strip():
                lookup[(cid, task)] = str(gold).strip().upper()
    return lookup


def load_pred_lookup(method, backbone):
    if method == "Baseline" and backbone in BASELINE_PREFIX:
        prefix = BASELINE_PREFIX[backbone]
        mp = pd.read_csv(MEDPERTURB_CSV)
        mp = mp[mp["dataset"].isin(DATASETS)]
        lookup = {}
        for _, row in mp.iterrows():
            cid = str(row["context_id"]); pert = str(row["perturbation"])
            for task in TASKS:
                col = f"{prefix}_{task}"
                if col in mp.columns and pd.notna(row[col]):
                    lookup[(cid, pert, task)] = str(row[col]).strip().upper()
        return lookup

    path = OUTPUT_BASE / f"{METHOD_DIRS[method]}/{backbone}/oncqa_askdocs/run_0/results.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    prefix = None
    for col in df.columns:
        if col.endswith("_manage") and backbone in col:
            prefix = col.replace("_manage", ""); break
    if prefix is None:
        return {}
    lookup = {}
    for _, row in df.iterrows():
        cid = str(row["context_id"]); pert = str(row["perturbation"])
        for task in TASKS:
            col = f"{prefix}_{task}"
            if col in df.columns and pd.notna(row[col]):
                lookup[(cid, pert, task)] = str(row[col]).strip().upper()
    return lookup


def mcnemar_p(b, c):
    if b + c == 0:
        return None
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(1 - stats.chi2.cdf(stat, df=1))


def sig_tier(p_bonf):
    if p_bonf is None:    return "ns"
    if p_bonf < 0.001:    return "***"
    if p_bonf < 0.01:     return "**"
    if p_bonf < 0.05:     return "*"
    return "ns"


# ── build binary matrix per (backbone, task, metric) ──────────────────────────

def build_matrix(backbone, task, metric, gold_lookup, pred_lookups,
                 single_pert=None):
    """
    Returns dict: method → np.array of 0/1 for each aligned case.
    If single_pert is given, only that perturbation is used (independent rows).
    """
    c_plus  = care_augmenting(task)
    c_minus = reduced_care(task)
    cases = {}   # cid → {method: 0|1}

    for pert in ([single_pert] if single_pert else PERTS):
        if metric == "rcer":
            for (cid, t), label in gold_lookup.items():
                if t != task or label != c_plus:
                    continue
                entry = {}
                for m in METHOD_ORDER:
                    pred = pred_lookups[m].get((cid, pert, task), "")
                    if not pred:
                        break
                    entry[m] = 0 if pred == c_minus else 1
                else:
                    cases[f"{cid}__{pert}"] = entry
        else:  # flip
            base_preds = {m: {} for m in METHOD_ORDER}
            for m in METHOD_ORDER:
                for (cid, p, t), pred in pred_lookups[m].items():
                    if p == "baseline" and t == task:
                        base_preds[m][cid] = pred
            all_cids = set.intersection(*[set(base_preds[m].keys())
                                          for m in METHOD_ORDER])
            for cid in all_cids:
                entry = {}
                for m in METHOD_ORDER:
                    pred_p = pred_lookups[m].get((cid, pert, task), "")
                    pred_b = base_preds[m].get(cid, "")
                    if not pred_p or not pred_b:
                        break
                    entry[m] = 1 if pred_p == pred_b else 0
                else:
                    cases[f"{cid}__{pert}"] = entry

    if not cases:
        return None
    arrays = {m: np.array([v[m] for v in cases.values()]) for m in METHOD_ORDER}
    return arrays


def compute_pairwise(arrays, n_pairs=6):
    """
    Returns dict: (a,b) → (p_bonf, tier, b_count, c_count, winner)
    For a single binary matrix (one perturbation or concatenated).
    """
    results = {}
    for a, b in PAIRS:
        col_a = arrays[a]; col_b = arrays[b]
        b_count = int(((col_a == 1) & (col_b == 0)).sum())
        c_count = int(((col_a == 0) & (col_b == 1)).sum())
        p_raw   = mcnemar_p(b_count, c_count)
        p_bonf  = min(p_raw * n_pairs, 1.0) if p_raw is not None else None
        tier    = sig_tier(p_bonf)
        winner  = "ns" if tier == "ns" else ("right" if c_count >= b_count else "left")
        results[(a, b)] = (p_bonf, tier, b_count, c_count, winner)
    return results


def compute_pairwise_fisher(arrays_per_pert: list[dict], n_pairs=6):
    """
    Fisher's combined McNemar across perturbations for one (backbone, task).

    arrays_per_pert: list of per-perturbation {method: np.array} dicts
                     — each list entry is one independent perturbation.

    Returns dict: (a,b) → (p_fisher, tier, total_b, total_c, winner)
    Direction (b vs c) is determined from the aggregated discordant counts.
    """
    results = {}
    for a, b in PAIRS:
        p_list   = []
        total_bc = 0  # sum of b_count across perts (left wins)
        total_cb = 0  # sum of c_count across perts (right wins)
        for arrays in arrays_per_pert:
            col_a   = arrays[a]; col_b = arrays[b]
            b_count = int(((col_a == 1) & (col_b == 0)).sum())
            c_count = int(((col_a == 0) & (col_b == 1)).sum())
            total_bc += b_count; total_cb += c_count
            p_raw = mcnemar_p(b_count, c_count)
            if p_raw is not None:
                p_list.append(p_raw)

        if len(p_list) >= 2:
            _, p_fisher = stats.combine_pvalues(p_list, method="fisher")
        elif len(p_list) == 1:
            p_fisher = p_list[0]
        else:
            p_fisher = 1.0

        p_bonf  = min(float(p_fisher) * n_pairs, 1.0)
        tier    = sig_tier(p_bonf)
        winner  = "ns" if tier == "ns" else ("right" if total_cb >= total_bc else "left")
        results[(a, b)] = (p_bonf, tier, total_bc, total_cb, winner)
    return results


# ── plot ───────────────────────────────────────────────────────────────────────

def build_heatmap_data(metric, gold_lookup):
    """
    Returns:
        sig_matrix  : (n_rows × n_pairs) array of tier strings
        p_matrix    : (n_rows × n_pairs) array of p_bonf floats
        row_labels  : list of strings
        n_list      : list of N (number of cases per row)
        dividers    : row indices where a thick divider should be drawn
        is_pooled   : list of bools — True for pooled/backbone-level rows
    """
    row_labels = []
    sig_rows   = []
    p_rows     = []
    win_rows   = []
    n_rows     = []
    dividers   = []
    is_pooled  = []

    # ── build per-perturbation arrays for every (backbone, task) ─────────────
    # Structure: per_pert_arrays[backbone][task][pert] = {method: np.array}
    per_pert = {}
    for backbone in BACKBONE_ORDER:
        pred_lookups = {m: load_pred_lookup(m, backbone) for m in METHOD_ORDER}
        per_pert[backbone] = {}
        for task in TASKS:
            per_pert[backbone][task] = {}
            for pert in PERTS:
                arr = build_matrix(backbone, task, metric, gold_lookup,
                                   pred_lookups, single_pert=pert)
                if arr is not None:
                    per_pert[backbone][task][pert] = arr

    # ── pooled row: Fisher's combined across all (backbone, task, pert) ───────
    # Collect per-perturbation arrays pooled across backbone × task
    pooled_per_pert: dict[str, list[dict]] = {p: [] for p in PERTS}
    for backbone in BACKBONE_ORDER:
        for task in TASKS:
            for pert in PERTS:
                if pert in per_pert[backbone][task]:
                    pooled_per_pert[pert].append(per_pert[backbone][task][pert])

    # Merge backbone×task arrays within each perturbation, then Fisher-combine
    pert_arrays_for_pooled = []
    total_n_pooled = 0
    for pert in PERTS:
        if pooled_per_pert[pert]:
            merged = {m: np.concatenate([a[m] for a in pooled_per_pert[pert]])
                      for m in METHOD_ORDER}
            pert_arrays_for_pooled.append(merged)
            total_n_pooled += len(merged[METHOD_ORDER[0]])

    if pert_arrays_for_pooled:
        pw = compute_pairwise_fisher(pert_arrays_for_pooled)
        row_labels.append("Fisher\nPooled")
        n_rows.append(total_n_pooled)
        sig_rows.append([pw[(a, b)][1] for a, b in PAIRS])
        p_rows.append([pw[(a, b)][0] for a, b in PAIRS])
        win_rows.append([pw[(a, b)][4] for a, b in PAIRS])
        is_pooled.append(True)

    dividers.append(len(sig_rows))   # divider after pooled row

    # ── per backbone × task: Fisher's combined across 3 perturbations ─────────
    for bi, backbone in enumerate(BACKBONE_ORDER):
        for task in TASKS:
            row_labels.append(f"{BACKBONE_LABELS[backbone]}\n{TASK_LABELS[task]}")
            is_pooled.append(False)
            pert_arrays = [per_pert[backbone][task][p]
                           for p in PERTS if p in per_pert[backbone][task]]
            if not pert_arrays:
                sig_rows.append(["ns"] * len(PAIRS))
                p_rows.append([None] * len(PAIRS))
                win_rows.append(["ns"] * len(PAIRS))
                n_rows.append(0)
                continue
            pw = compute_pairwise_fisher(pert_arrays)
            n_total = sum(len(a[METHOD_ORDER[0]]) for a in pert_arrays)
            n_rows.append(n_total)
            sig_rows.append([pw[(a, b)][1] for a, b in PAIRS])
            p_rows.append([pw[(a, b)][0] for a, b in PAIRS])
            win_rows.append([pw[(a, b)][4] for a, b in PAIRS])

        if bi < len(BACKBONE_ORDER) - 1:
            dividers.append(len(sig_rows))

    return sig_rows, p_rows, win_rows, row_labels, n_rows, dividers, is_pooled


def cell_color(tier, winner):
    """Return (facecolor) based on significance tier and direction."""
    if tier == "ns":
        return COLORS_RIGHT["ns"]
    return COLORS_RIGHT[tier] if winner == "right" else COLORS_LEFT[tier]


def plot_heatmap(ax, sig_matrix, p_matrix, win_matrix, row_labels, n_list,
                 dividers, is_pooled, title):
    n_rows = len(sig_matrix)
    n_cols = len(PAIRS)

    for ri in range(n_rows):
        for ci in range(n_cols):
            tier   = sig_matrix[ri][ci]
            winner = win_matrix[ri][ci]
            color  = cell_color(tier, winner)
            rect = mpatches.FancyBboxPatch(
                (ci + 0.05, ri + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor="white", linewidth=1.5,
                transform=ax.transData, clip_on=False,
            )
            ax.add_patch(rect)
            label = tier if tier != "ns" else "ns"
            ax.text(ci + 0.5, ri + 0.5, label,
                    ha="center", va="center", fontsize=10,
                    fontweight="bold" if tier != "ns" else "normal",
                    color=TEXT_COLOR[tier])

    # axes
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(PAIR_LABELS, fontsize=9)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(
        [f"{rl}  (N={n})" for rl, n in zip(row_labels, n_list)],
        fontsize=9,
        fontweight="bold" if False else "normal",  # placeholder
    )
    # bold pooled tick labels
    for i, (lbl, tick) in enumerate(zip(row_labels, ax.get_yticklabels())):
        if is_pooled[i]:
            tick.set_fontweight("bold")

    ax.tick_params(length=0)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # grid
    for x in range(n_cols + 1):
        ax.axvline(x, color="white", lw=2)
    for y in range(n_rows + 1):
        ax.axhline(y, color="white", lw=1)

    # dividers
    for d in dividers:
        lw   = 2.5 if d == 1 else 1.5   # thicker after pooled row
        ls   = "-" if d == 1 else "--"
        col  = "#222222" if d == 1 else "#555555"
        ax.axhline(d, color=col, lw=lw, linestyle=ls)

    ax.invert_yaxis()
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")


def main():
    print("Loading gold labels...")
    gold_lookup = build_gold_lookup()
    print(f"  {len(gold_lookup)} (context_id, task) pairs")

    print("Computing RCER pairwise tests...")
    rcer_sig, rcer_p, rcer_win, row_labels, rcer_n, dividers, is_pooled = \
        build_heatmap_data("rcer", gold_lookup)

    print("Computing Flip pairwise tests...")
    flip_sig, flip_p, flip_win, _, flip_n, _, _ = \
        build_heatmap_data("flip", gold_lookup)

    # shared legend handles
    legend_patches = [
        mpatches.Patch(facecolor=COLORS_RIGHT["ns"],  label="ns  (p ≥ 0.05)"),
        mpatches.Patch(facecolor=COLORS_RIGHT["*"],   label="*   right method wins"),
        mpatches.Patch(facecolor=COLORS_RIGHT["**"],  label="**  right method wins"),
        mpatches.Patch(facecolor=COLORS_RIGHT["***"], label="*** right method wins"),
        mpatches.Patch(facecolor=COLORS_LEFT["*"],    label="*   left method wins"),
        mpatches.Patch(facecolor=COLORS_LEFT["**"],   label="**  left method wins"),
        mpatches.Patch(facecolor=COLORS_LEFT["***"],  label="*** left method wins"),
    ]

    # ── FULL heatmap (9 backbone×task rows + pooled) ──────────────────────────
    n_rows = len(rcer_sig)
    fig, axes = plt.subplots(1, 2, figsize=(16, 1.0 + n_rows * 0.85))
    fig.patch.set_facecolor("white")

    plot_heatmap(axes[0], rcer_sig, rcer_p, rcer_win, row_labels, rcer_n,
                 dividers, is_pooled,
                 "RCER — Pairwise McNemar\n(Bonferroni corrected)")
    plot_heatmap(axes[1], flip_sig, flip_p, flip_win, row_labels, flip_n,
                 dividers, is_pooled,
                 "Flip Rate — Pairwise McNemar\n(Bonferroni corrected)")

    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Pairwise Method Comparisons — Cochran's Q Post-hoc McNemar Tests\n"
        "Blue = right method wins   |   Orange = left method wins   |   "
        "Dashed lines separate backbones",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out = PLOT_DIR / "cochrans_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {out}")

    # ── POOLED-ONLY plot (2 rows: RCER + Flip) ────────────────────────────────
    # Extract just the pooled row (index 0)
    pooled_rcer_sig = [rcer_sig[0]]
    pooled_rcer_p   = [rcer_p[0]]
    pooled_rcer_win = [rcer_win[0]]
    pooled_flip_sig = [flip_sig[0]]
    pooled_flip_p   = [flip_p[0]]
    pooled_flip_win = [flip_win[0]]

    fig2, ax2 = plt.subplots(figsize=(13, 3.2))
    fig2.patch.set_facecolor("white")
    ax2.set_facecolor("white")
    ax2.set_xlim(0, len(PAIRS))
    ax2.set_ylim(0, 2)
    ax2.invert_yaxis()
    ax2.xaxis.set_ticks_position("top")
    ax2.xaxis.set_label_position("top")

    row_data = [
        ("RCER", pooled_rcer_sig[0], pooled_rcer_win[0], rcer_n[0]),
        ("Flip Rate", pooled_flip_sig[0], pooled_flip_win[0], flip_n[0]),
    ]
    for ri, (label, sigs, wins, n) in enumerate(row_data):
        for ci, ((a, b), tier, winner) in enumerate(zip(PAIRS, sigs, wins)):
            color = cell_color(tier, winner)
            rect = mpatches.FancyBboxPatch(
                (ci + 0.05, ri + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor="white", linewidth=2,
                transform=ax2.transData, clip_on=False,
            )
            ax2.add_patch(rect)
            lbl = tier if tier != "ns" else "ns"
            ax2.text(ci + 0.5, ri + 0.5, lbl,
                     ha="center", va="center", fontsize=13,
                     fontweight="bold" if tier != "ns" else "normal",
                     color=TEXT_COLOR[tier])

    ax2.set_xticks(np.arange(len(PAIRS)) + 0.5)
    ax2.set_xticklabels(PAIR_LABELS, fontsize=11)
    ax2.set_yticks([0.5, 1.5])
    ax2.set_yticklabels(
        [f"RCER  (N={rcer_n[0]})", f"Flip Rate  (N={flip_n[0]})"],
        fontsize=12, fontweight="bold"
    )
    ax2.tick_params(length=0)
    ax2.axhline(1, color="#333333", lw=1.5, linestyle="--")
    for x in range(len(PAIRS) + 1):
        ax2.axvline(x, color="white", lw=2)

    fig2.legend(handles=legend_patches, loc="lower center", ncol=4,
                fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.18))
    fig2.suptitle(
        "Pooled Pairwise McNemar Tests (all backbones × tasks × perturbations)\n"
        "Blue = right method wins   |   Orange = left method wins",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out2 = PLOT_DIR / "cochrans_pooled.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {out2}")

    # also print summary table
    print("\n=== RCER significance summary ===")
    print(f"{'Condition':<22}", "  ".join(f"{p[0]}v{p[1]}" for p in PAIRS))
    for rl, row in zip(row_labels, rcer_sig):
        label = rl.replace("\n", " ")
        print(f"{label:<22}", "  ".join(f"{t:>5}" for t in row))


if __name__ == "__main__":
    main()
