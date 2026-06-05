"""
Convert flip rate CSV data to LaTeX table.

Highlighting: for each column (pert × task + avg), the lowest value
within each model group is highlighted by backbone colour:
  Llama 8B   → \cellcolor{rowgreen}   (light green)
  Llama 70B  → \cellcolor{rowblue}    (light blue)
  Qwen 235B  → \cellcolor{rowyellow}  (light yellow)

Add to LaTeX preamble:
  \\usepackage{colortbl}
  \\usepackage[table]{xcolor}
  \\definecolor{rowgreen}{RGB}{198,239,206}
  \\definecolor{rowblue}{RGB}{189,215,238}
  \\definecolor{rowyellow}{RGB}{255,242,204}
"""
import io
import pandas as pd
import numpy as np

DATA = """flip,Baseline,Llama 70B,summary,manage,9.6386
flip,Baseline,Llama 70B,summary,visit,9.6386
flip,Baseline,Llama 70B,summary,resource,6.0241
flip,Baseline,Llama 70B,gender_swap,manage,2.6667
flip,Baseline,Llama 70B,gender_swap,visit,4.4444
flip,Baseline,Llama 70B,gender_swap,resource,4.0
flip,Baseline,Llama 70B,uncertain_tone,manage,9.7778
flip,Baseline,Llama 70B,uncertain_tone,visit,9.7778
flip,Baseline,Llama 70B,uncertain_tone,resource,8.8889
flip,Baseline,Llama 8B,summary,manage,46.988
flip,Baseline,Llama 8B,summary,visit,44.5783
flip,Baseline,Llama 8B,summary,resource,43.9759
flip,Baseline,Llama 8B,gender_swap,manage,8.8889
flip,Baseline,Llama 8B,gender_swap,visit,8.4444
flip,Baseline,Llama 8B,gender_swap,resource,7.5556
flip,Baseline,Llama 8B,uncertain_tone,manage,20.4444
flip,Baseline,Llama 8B,uncertain_tone,visit,19.5556
flip,Baseline,Llama 8B,uncertain_tone,resource,19.1111
flip,Baseline,Qwen 235B,summary,manage,13.253
flip,Baseline,Qwen 235B,summary,visit,12.6506
flip,Baseline,Qwen 235B,summary,resource,9.6386
flip,Baseline,Qwen 235B,gender_swap,manage,2.2222
flip,Baseline,Qwen 235B,gender_swap,visit,3.5556
flip,Baseline,Qwen 235B,gender_swap,resource,2.6667
flip,Baseline,Qwen 235B,uncertain_tone,manage,4.8889
flip,Baseline,Qwen 235B,uncertain_tone,visit,7.1111
flip,Baseline,Qwen 235B,uncertain_tone,resource,6.2222
flip,MRA,Llama 70B,summary,manage,7.2289
flip,MRA,Llama 70B,summary,visit,6.6265
flip,MRA,Llama 70B,summary,resource,5.4217
flip,MRA,Llama 70B,gender_swap,manage,1.3333
flip,MRA,Llama 70B,gender_swap,visit,3.1111
flip,MRA,Llama 70B,gender_swap,resource,3.1111
flip,MRA,Llama 70B,uncertain_tone,manage,8.0
flip,MRA,Llama 70B,uncertain_tone,visit,8.4444
flip,MRA,Llama 70B,uncertain_tone,resource,7.5556
flip,MRA,Llama 8B,summary,manage,36.747
flip,MRA,Llama 8B,summary,visit,34.3373
flip,MRA,Llama 8B,summary,resource,33.7349
flip,MRA,Llama 8B,gender_swap,manage,4.0
flip,MRA,Llama 8B,gender_swap,visit,4.0
flip,MRA,Llama 8B,gender_swap,resource,3.5556
flip,MRA,Llama 8B,uncertain_tone,manage,15.5556
flip,MRA,Llama 8B,uncertain_tone,visit,16.0
flip,MRA,Llama 8B,uncertain_tone,resource,16.0
flip,MRA,Qwen 235B,summary,manage,13.8554
flip,MRA,Qwen 235B,summary,visit,11.4458
flip,MRA,Qwen 235B,summary,resource,10.241
flip,MRA,Qwen 235B,gender_swap,manage,1.7778
flip,MRA,Qwen 235B,gender_swap,visit,2.2222
flip,MRA,Qwen 235B,gender_swap,resource,3.5556
flip,MRA,Qwen 235B,uncertain_tone,manage,4.4444
flip,MRA,Qwen 235B,uncertain_tone,visit,4.4444
flip,MRA,Qwen 235B,uncertain_tone,resource,4.4444
flip,TUA,Llama 70B,summary,manage,9.0361
flip,TUA,Llama 70B,summary,visit,9.0361
flip,TUA,Llama 70B,summary,resource,9.0361
flip,TUA,Llama 70B,gender_swap,manage,5.7778
flip,TUA,Llama 70B,gender_swap,visit,5.7778
flip,TUA,Llama 70B,gender_swap,resource,4.4444
flip,TUA,Llama 70B,uncertain_tone,manage,6.2222
flip,TUA,Llama 70B,uncertain_tone,visit,6.2222
flip,TUA,Llama 70B,uncertain_tone,resource,4.8889
flip,TUA,Llama 8B,summary,manage,22.8916
flip,TUA,Llama 8B,summary,visit,21.0843
flip,TUA,Llama 8B,summary,resource,21.0843
flip,TUA,Llama 8B,gender_swap,manage,11.1111
flip,TUA,Llama 8B,gender_swap,visit,11.5556
flip,TUA,Llama 8B,gender_swap,resource,12.0
flip,TUA,Llama 8B,uncertain_tone,manage,14.6667
flip,TUA,Llama 8B,uncertain_tone,visit,14.6667
flip,TUA,Llama 8B,uncertain_tone,resource,14.6667
flip,TUA,Qwen 235B,summary,manage,10.241
flip,TUA,Qwen 235B,summary,visit,9.6386
flip,TUA,Qwen 235B,summary,resource,8.4337
flip,TUA,Qwen 235B,gender_swap,manage,3.5556
flip,TUA,Qwen 235B,gender_swap,visit,4.0
flip,TUA,Qwen 235B,gender_swap,resource,3.1111
flip,TUA,Qwen 235B,uncertain_tone,manage,8.0
flip,TUA,Qwen 235B,uncertain_tone,visit,7.5556
flip,TUA,Qwen 235B,uncertain_tone,resource,8.0
flip,MPA,Llama 70B,summary,manage,10.241
flip,MPA,Llama 70B,summary,visit,10.241
flip,MPA,Llama 70B,summary,resource,7.8313
flip,MPA,Llama 70B,gender_swap,manage,4.0
flip,MPA,Llama 70B,gender_swap,visit,4.4444
flip,MPA,Llama 70B,gender_swap,resource,2.2222
flip,MPA,Llama 70B,uncertain_tone,manage,6.6667
flip,MPA,Llama 70B,uncertain_tone,visit,6.2222
flip,MPA,Llama 70B,uncertain_tone,resource,4.4444
flip,MPA,Llama 8B,summary,manage,19.2771
flip,MPA,Llama 8B,summary,visit,18.6747
flip,MPA,Llama 8B,summary,resource,18.0723
flip,MPA,Llama 8B,gender_swap,manage,16.0
flip,MPA,Llama 8B,gender_swap,visit,16.8889
flip,MPA,Llama 8B,gender_swap,resource,16.0
flip,MPA,Llama 8B,uncertain_tone,manage,18.2222
flip,MPA,Llama 8B,uncertain_tone,visit,18.2222
flip,MPA,Llama 8B,uncertain_tone,resource,17.3333
flip,MPA,Qwen 235B,summary,manage,12.6506
flip,MPA,Qwen 235B,summary,visit,12.0482
flip,MPA,Qwen 235B,summary,resource,9.6386
flip,MPA,Qwen 235B,gender_swap,manage,3.1111
flip,MPA,Qwen 235B,gender_swap,visit,2.6667
flip,MPA,Qwen 235B,gender_swap,resource,2.6667
flip,MPA,Qwen 235B,uncertain_tone,manage,7.5556
flip,MPA,Qwen 235B,uncertain_tone,visit,7.5556
flip,MPA,Qwen 235B,uncertain_tone,resource,7.1111"""

# ── config ─────────────────────────────────────────────────────────────────────

MODEL_ORDER = ["Baseline", "MRA", "TUA", "MPA"]
MODEL_LABELS = {
    "Baseline": "Baseline",
    "MRA":      "MedReasonAgent",
    "TUA":      "ToolUniAgent",
    "MPA":      "MedPathAgent",
}
BACKBONE_ORDER = ["Llama 8B", "Llama 70B", "Qwen 235B"]
BACKBONE_LABELS = {
    "Llama 8B":  "L8B-3.1",
    "Llama 70B": "L70B-3.3",
    "Qwen 235B": "Q235B",
}
# LaTeX cellcolor name per backbone
BACKBONE_COLOR = {
    "Llama 8B":  "rowgreen",
    "Llama 70B": "rowblue",
    "Qwen 235B": "rowyellow",
}
PERT_ORDER = ["summary", "gender_swap", "uncertain_tone"]
TASK_ORDER = ["manage", "visit", "resource"]

# ── parse ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(
    io.StringIO(DATA),
    header=None,
    names=["metric", "model", "backbone", "perturbation", "task", "value"],
)
df["value"] = df["value"].astype(float)

pivot = df.set_index(["model", "backbone", "perturbation", "task"])["value"].to_dict()

def get(model, backbone, pert, task):
    return pivot.get((model, backbone, pert, task), float("nan"))

def avg(model, backbone, task):
    vals = [get(model, backbone, p, task) for p in PERT_ORDER]
    vals = [v for v in vals if not np.isnan(v)]
    return np.mean(vals) if vals else float("nan")

# ── find minimums per column ──────────────────────────────────────────────────

def col_vals(pert, task):
    """All (model, backbone) → rounded value for this column."""
    out = {}
    for m in MODEL_ORDER:
        for b in BACKBONE_ORDER:
            v = get(m, b, pert, task) if pert != "avg" else avg(m, b, task)
            if not np.isnan(v):
                out[(m, b)] = round(v, 1)
    return out

# Overall minimum per column → bold
global_min: dict[tuple, float] = {}
# Per-backbone minimum per column → cellcolor
backbone_min: dict[tuple, dict] = {}   # (pert, task) → {backbone: min_val}

for p in PERT_ORDER + ["avg"]:
    for t in TASK_ORDER:
        cv = col_vals(p, t)
        if cv:
            global_min[(p, t)] = min(cv.values())
        bm = {}
        for b in BACKBONE_ORDER:
            bb_vals = {m: v for (m, bb), v in cv.items() if bb == b}
            if bb_vals:
                bm[b] = min(bb_vals.values())
        backbone_min[(p, t)] = bm

def fmt(val, model, backbone, pert, task, decimals=1):
    if np.isnan(val):
        return "--"
    rounded = round(val, decimals)
    s = f"{rounded:.{decimals}f}"

    # italic: lowest value for this backbone in this column
    bm = backbone_min.get((pert, task), {})
    bb_min = bm.get(backbone)
    if bb_min is not None and abs(rounded - bb_min) < 1e-9:
        s = r"\textit{" + s + r"}"

    # bold: overall lowest value in this column
    gm = global_min.get((pert, task))
    if gm is not None and abs(rounded - gm) < 1e-9:
        s = r"\textbf{" + s + r"}"

    return s

# ── build LaTeX ────────────────────────────────────────────────────────────────

lines = []
lines.append(r"% Required in preamble:")
lines.append(r"% \usepackage{colortbl}")
lines.append(r"% \usepackage[table]{xcolor}")
lines.append(r"% \definecolor{rowgreen}{RGB}{198,239,206}")
lines.append(r"% \definecolor{rowblue}{RGB}{189,215,238}")
lines.append(r"% \definecolor{rowyellow}{RGB}{255,242,204}")
lines.append(r"")
lines.append(r"\begin{table}[h]")
lines.append(r"\centering")
lines.append(r"\resizebox{\textwidth}{!}{%")
lines.append(r"\begin{tabular}{ll ccc ccc ccc ccc}")
lines.append(r"\toprule")
lines.append(
    r" &  & \multicolumn{3}{c}{\textbf{Summary (Sum.)}} "
    r"& \multicolumn{3}{c}{\textbf{Gender Swap (GS)}} "
    r"& \multicolumn{3}{c}{\textbf{Uncertain Tone (UT)}} "
    r"& \multicolumn{3}{c}{\textbf{Average}} \\"
)
lines.append(
    r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}"
)
lines.append(r"Model & Backbone & M & V & R & M & V & R & M & V & R & M & V & R \\")
lines.append(r"")

for mi, model in enumerate(MODEL_ORDER):
    lines.append(r"\midrule")
    n_backbones = len(BACKBONE_ORDER)
    for bi, backbone in enumerate(BACKBONE_ORDER):
        prefix = (r"\multirow{" + str(n_backbones) + r"}{*}{"
                  + MODEL_LABELS[model] + r"}") if bi == 0 else ""
        bb_label = BACKBONE_LABELS[backbone]

        cells = []
        for pert in PERT_ORDER:
            for task in TASK_ORDER:
                v = get(model, backbone, pert, task)
                cells.append(fmt(v, model, backbone, pert, task))
        for task in TASK_ORDER:
            v = avg(model, backbone, task)
            cells.append(fmt(v, model, backbone, "avg", task))

        row = f"{prefix} & {bb_label} & " + " & ".join(cells) + r" \\"
        lines.append(row)

    lines.append(r"")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}}")
lines.append(
    r"\caption{Flip rate (\%) per model, backbone (L=Llama, Q=Qwen3), perturbation type, "
    r"and triage task (M=\textsc{Manage}, V=\textsc{Visit}, R=\textsc{Resource}). "
    r"Shading indicates the lowest flip rate within each model group per column: "
    r"\colorbox{rowgreen}{green}~=~Llama~8B, "
    r"\colorbox{rowblue}{blue}~=~Llama~70B, "
    r"\colorbox{rowyellow}{yellow}~=~Qwen~235B. "
    r"Average columns report the mean across perturbation types per task.}"
)
lines.append(r"\label{tab:flip_rate_full}")
lines.append(r"\end{table}")

print("\n".join(lines))
