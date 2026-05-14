"""
Generate publication-quality figures from eval_output/four_way_comparison/.

Outputs saved to eval_output/figures/
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── config ────────────────────────────────────────────────────────────────────

OUT_DIR = "eval_output/figures"
DATA_DIR = "eval_output/four_way_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_LABELS = {
    "tooluni_agent":  "ToolUni",
    "medreasonagent": "MedReason",
    "llama":          "Llama 70B",
    "clinician":      "Clinician",
}
MODEL_ORDER = ["tooluni_agent", "medreasonagent", "llama", "clinician"]
COLORS      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]   # muted blue/orange/green/red
PERT_ORDER  = ["baseline", "gender_swap", "summary", "uncertain_tone"]
PERT_LABELS = {
    "baseline":       "Baseline",
    "gender_swap":    "Gender Swap",
    "summary":        "Summary",
    "uncertain_tone": "Uncertain Tone",
}
QUESTION_LABELS = {"manage": "Manage", "visit": "Visit", "resource": "Resource"}

MARKER_MAP = {"tooluni_agent": "o", "medreasonagent": "s", "llama": "^", "clinician": "D"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
})

# ── helpers ───────────────────────────────────────────────────────────────────

def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))

def model_color(m):
    return COLORS[MODEL_ORDER.index(m)] if m in MODEL_ORDER else "#888"


# ════════════════════════════════════════════════════════════════════════════
# Fig 1 — Accuracy by perturbation × question  (3×4 heatmaps per model)
# ════════════════════════════════════════════════════════════════════════════

def fig_accuracy_heatmaps():
    acc = load("accuracy.csv")
    acc = acc[acc["model"].isin(MODEL_ORDER)]

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(14, 4), sharey=True)
    fig.suptitle("Accuracy by Perturbation & Triage Question", fontsize=13, fontweight="bold", y=1.02)

    for ax, model in zip(axes, MODEL_ORDER):
        sub = acc[acc["model"] == model].copy()
        pivot = sub.pivot(index="perturbation", columns="question", values="accuracy")
        pivot = pivot.reindex(index=PERT_ORDER, columns=list(QUESTION_LABELS.keys()))

        im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8.5, color="black" if 0.3 < val < 0.8 else "white")

        ax.set_xticks(range(3))
        ax.set_xticklabels([QUESTION_LABELS[q] for q in QUESTION_LABELS], rotation=30, ha="right")
        ax.set_yticks(range(len(PERT_ORDER)))
        ax.set_yticklabels([PERT_LABELS[p] for p in PERT_ORDER])
        ax.set_title(MODEL_LABELS[model], fontweight="bold", pad=8)
        ax.grid(False)

    plt.colorbar(im, ax=axes[-1], shrink=0.8, label="Accuracy")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_accuracy_heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ════════════════════════════════════════════════════════════════════════════
# Fig 2 — Grouped bar: accuracy across perturbations for each question
# ════════════════════════════════════════════════════════════════════════════

def fig_accuracy_grouped_bars():
    acc = load("accuracy.csv")
    acc = acc[acc["model"].isin(MODEL_ORDER)]

    questions = list(QUESTION_LABELS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Model Accuracy per Perturbation", fontsize=13, fontweight="bold")

    x = np.arange(len(PERT_ORDER))
    width = 0.18
    offsets = np.linspace(-(len(MODEL_ORDER)-1)/2, (len(MODEL_ORDER)-1)/2, len(MODEL_ORDER)) * width

    for ax, q in zip(axes, questions):
        sub = acc[acc["question"] == q]
        for i, model in enumerate(MODEL_ORDER):
            msub = sub[sub["model"] == model]
            vals = [
                msub.loc[msub["perturbation"] == p, "accuracy"].values[0]
                if p in msub["perturbation"].values else np.nan
                for p in PERT_ORDER
            ]
            ax.bar(x + offsets[i], vals, width, label=MODEL_LABELS[model],
                   color=model_color(model), alpha=0.85, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([PERT_LABELS[p] for p in PERT_ORDER], rotation=20, ha="right")
        ax.set_title(QUESTION_LABELS[q], fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Accuracy" if q == "manage" else "")
        ax.axhline(0.5, color="gray", lw=0.8, linestyle="--", alpha=0.5)

    handles = [mpatches.Patch(color=model_color(m), label=MODEL_LABELS[m]) for m in MODEL_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_accuracy_grouped_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ════════════════════════════════════════════════════════════════════════════
# Fig 3 — MI across perturbations (line plot, one panel per question)
# ════════════════════════════════════════════════════════════════════════════

def fig_mi_line():
    mi = load("mi.csv")
    mi = mi[mi["model"].isin(MODEL_ORDER)]

    questions = list(QUESTION_LABELS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle("Mutual Information (MI) Across Perturbations", fontsize=13, fontweight="bold")

    for ax, q in zip(axes, questions):
        sub = mi[mi["question"] == q]
        for model in MODEL_ORDER:
            msub = sub[sub["model"] == model].set_index("perturbation")
            vals = [msub.loc[p, "MI"] if p in msub.index else np.nan for p in PERT_ORDER]
            ax.plot(
                range(len(PERT_ORDER)), vals,
                marker=MARKER_MAP[model], color=model_color(model),
                linewidth=1.8, markersize=7, label=MODEL_LABELS[model],
                linestyle="-" if model != "clinician" else "--",
            )
        ax.set_xticks(range(len(PERT_ORDER)))
        ax.set_xticklabels([PERT_LABELS[p] for p in PERT_ORDER], rotation=20, ha="right")
        ax.set_title(QUESTION_LABELS[q], fontweight="bold")
        ax.set_ylabel("MI (bits)" if q == "manage" else "")

    handles = [
        plt.Line2D([0], [0], color=model_color(m), marker=MARKER_MAP[m],
                   linestyle="-" if m != "clinician" else "--",
                   label=MODEL_LABELS[m], markersize=7)
        for m in MODEL_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_mi_line.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ════════════════════════════════════════════════════════════════════════════
# Fig 4 — ATR grouped bar (same layout as accuracy)
# ════════════════════════════════════════════════════════════════════════════

def fig_atr_grouped_bars():
    atr = load("atr.csv")
    atr = atr[atr["model"].isin(MODEL_ORDER)]

    questions = list(QUESTION_LABELS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Agreement-to-Reference (ATR) per Perturbation", fontsize=13, fontweight="bold")

    x = np.arange(len(PERT_ORDER))
    width = 0.18
    offsets = np.linspace(-(len(MODEL_ORDER)-1)/2, (len(MODEL_ORDER)-1)/2, len(MODEL_ORDER)) * width

    for ax, q in zip(axes, questions):
        sub = atr[atr["question"] == q]
        for i, model in enumerate(MODEL_ORDER):
            msub = sub[sub["model"] == model]
            vals = [
                msub.loc[msub["perturbation"] == p, "ATR"].values[0]
                if p in msub["perturbation"].values else np.nan
                for p in PERT_ORDER
            ]
            ax.bar(x + offsets[i], vals, width, label=MODEL_LABELS[model],
                   color=model_color(model), alpha=0.85, edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([PERT_LABELS[p] for p in PERT_ORDER], rotation=20, ha="right")
        ax.set_title(QUESTION_LABELS[q], fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("ATR" if q == "manage" else "")

    handles = [mpatches.Patch(color=model_color(m), label=MODEL_LABELS[m]) for m in MODEL_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_atr_grouped_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ════════════════════════════════════════════════════════════════════════════
# Fig 5 — Summary radar / spider chart per model (mean across perturbations)
# ════════════════════════════════════════════════════════════════════════════

def fig_summary_radar():
    summary = load("summary.csv")
    summary = summary[summary["model"].isin(MODEL_ORDER)]

    # Normalise MI to [0,1] for radar (divide by max across dataset)
    max_mi = summary["mean_MI"].max()
    summary = summary.copy()
    summary["mean_MI_norm"] = summary["mean_MI"] / max_mi

    metrics = ["mean_accuracy", "mean_MI_norm", "mean_ATR"]
    metric_labels = ["Accuracy", "MI (norm)", "ATR"]
    questions = list(QUESTION_LABELS.keys())

    # One radar per model, 3 axes = 3 questions, 3 metrics as spokes
    # Actually: one radar per model, spokes = questions × metrics combo is complex;
    # Instead: one radar per model with 9 spokes (3 metrics × 3 questions)
    spokes = [f"{QUESTION_LABELS[q]}\n{m.replace('mean_','').replace('_norm','').upper()}"
              for q in questions for m in metrics]
    N = len(spokes)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close

    fig, axes = plt.subplots(2, 2, figsize=(10, 10),
                             subplot_kw=dict(polar=True))
    fig.suptitle("Per-Model Performance Radar\n(mean over perturbations)",
                 fontsize=13, fontweight="bold")

    for ax, model in zip(axes.flat, MODEL_ORDER):
        msub = summary[summary["model"] == model].set_index("question")
        vals = []
        for q in questions:
            for m in metrics:
                v = msub.loc[q, m] if q in msub.index else 0
                vals.append(float(v) if not np.isnan(v) else 0)
        vals += vals[:1]

        ax.plot(angles, vals, color=model_color(model), linewidth=2)
        ax.fill(angles, vals, color=model_color(model), alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(spokes, size=7.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], size=6.5)
        ax.set_title(MODEL_LABELS[model], size=11, fontweight="bold", pad=14,
                     color=model_color(model))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_radar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ════════════════════════════════════════════════════════════════════════════
# Fig 6 — Fleiss' κ by perturbation and question
# ════════════════════════════════════════════════════════════════════════════

def fig_kappa():
    kap = load("fleiss_kappa.csv")
    kap = kap[kap["perturbation"].isin(PERT_ORDER)]

    questions = list(QUESTION_LABELS.keys())
    x = np.arange(len(PERT_ORDER))
    width = 0.25
    offsets = [-width, 0, width]
    q_colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Fleiss' κ Inter-Rater Agreement", fontsize=13, fontweight="bold")

    for ax, col, title in zip(axes,
                               ["fleiss_kappa_3rater", "fleiss_kappa_4rater"],
                               ["3-Rater (ToolUni, MedReason, Llama)",
                                "4-Rater (+ Clinician)"]):
        for i, q in enumerate(questions):
            sub = kap[kap["question"] == q].set_index("perturbation")
            vals = [sub.loc[p, col] if p in sub.index else np.nan for p in PERT_ORDER]
            bars = ax.bar(x + offsets[i], vals, width, label=QUESTION_LABELS[q],
                          color=q_colors[i], alpha=0.85, edgecolor="white")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax.axhline(0.2, color="gray", lw=0.8, ls="--", alpha=0.6, label="Fair (0.2)")
        ax.axhline(0.4, color="gray", lw=0.8, ls=":",  alpha=0.6, label="Mod (0.4)")
        ax.set_xticks(x)
        ax.set_xticklabels([PERT_LABELS[p] for p in PERT_ORDER], rotation=20, ha="right")
        ax.set_ylabel("Fleiss' κ" if col == "fleiss_kappa_3rater" else "")
        ax.set_title(title, fontsize=10)
        ax.legend(ncol=2, frameon=False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig6_fleiss_kappa.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ════════════════════════════════════════════════════════════════════════════
# Fig 7 — Accuracy drop from baseline (robustness)
# ════════════════════════════════════════════════════════════════════════════

def fig_robustness_drop():
    acc = load("accuracy.csv")
    acc = acc[acc["model"].isin(MODEL_ORDER)]

    questions = list(QUESTION_LABELS.keys())
    perturbs = ["gender_swap", "summary", "uncertain_tone"]

    # Compute delta = pert_accuracy - baseline_accuracy
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle("Accuracy Drop vs Baseline (Robustness)", fontsize=13, fontweight="bold")

    x = np.arange(len(perturbs))
    width = 0.18
    offsets = np.linspace(-(len(MODEL_ORDER)-1)/2, (len(MODEL_ORDER)-1)/2, len(MODEL_ORDER)) * width

    for ax, q in zip(axes, questions):
        sub = acc[acc["question"] == q]
        baseline = sub[sub["perturbation"] == "baseline"].set_index("model")["accuracy"]

        for i, model in enumerate(MODEL_ORDER):
            msub = sub[sub["model"] == model]
            base_val = baseline.get(model, np.nan)
            deltas = []
            for p in perturbs:
                row = msub[msub["perturbation"] == p]
                if row.empty or np.isnan(base_val):
                    deltas.append(np.nan)
                else:
                    deltas.append(float(row["accuracy"].values[0]) - base_val)

            ax.bar(x + offsets[i], deltas, width, label=MODEL_LABELS[model],
                   color=model_color(model), alpha=0.85, edgecolor="white", linewidth=0.5)

        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([PERT_LABELS[p] for p in perturbs], rotation=20, ha="right")
        ax.set_title(QUESTION_LABELS[q], fontweight="bold")
        ax.set_ylabel("Δ Accuracy" if q == "manage" else "")

    handles = [mpatches.Patch(color=model_color(m), label=MODEL_LABELS[m]) for m in MODEL_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig7_robustness_drop.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ════════════════════════════════════════════════════════════════════════════
# Run all
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Saving figures to {OUT_DIR}/\n")
    fig_accuracy_heatmaps()
    fig_accuracy_grouped_bars()
    fig_mi_line()
    fig_atr_grouped_bars()
    fig_summary_radar()
    fig_kappa()
    fig_robustness_drop()
    print("\nDone.")
