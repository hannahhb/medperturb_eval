"""
Triage accuracy against labels, per backbone x system x condition.

Labels:
  consensus - clinician_consensus_{task} (100 vignettes with clinician consensus)
  gold      - gold_standard_{task} (all 225 vignettes)
Reports accuracy (%), under-triage rate (label c+, pred c-) and
over-triage rate (label c-, pred c+). c+: MANAGE=NO, VISIT/RESOURCE=YES.
Predictions that are not YES/NO (refusals) count as incorrect.
Usage: python analysis/accuracy_table.py
"""
import sys, io, contextlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import medperturb_robustness as mr  # noqa: E402

T = mr.TASKS; CONDS = ["baseline"] + mr.PERTS
METHODS = [("Baseline", "Baseline"), ("MedReason", "MRA"), ("ToolUni", "TUA"), ("MedPathAgent", "MPA")]
BACKBONES = [("Llama 8B", "Llama-8B"), ("Llama 70B", "Llama-70B"), ("Qwen 235B", "Qwen-235B")]

mp = pd.read_csv(mr.MEDPERTURB_CSV)
mp = mp[(mp.perturbation == "baseline") & mp.dataset.isin(mr.DATASETS)].set_index("context_id")
labels = {}
for t in T:
    labels[("consensus", t)] = mp[f"clinician_consensus_{t}"].dropna().map(lambda v: "YES" if float(v) == 1 else "NO")
    labels[("gold", t)] = mp[f"gold_standard_{t}"].dropna().astype(str).str.strip().str.upper()

with contextlib.redirect_stdout(io.StringIO()):
    data = mr.load_all()

cplus = lambda t: "NO" if t == "manage" else "YES"
rows = []
for src in ["consensus", "gold"]:
    for bk, bn in BACKBONES:
        for mk, mn in METHODS:
            df, pre = data[f"{mk}\n{bk}"]
            df = df.drop_duplicates(["context_id", "perturbation"], keep="last")
            for cond in CONDS:
                d = df[df.perturbation == cond].set_index("context_id")
                acc, under, over = [], [], []
                for t in T:
                    lab = labels[(src, t)]; ids = lab.index.intersection(d.index)
                    pred = d.loc[ids, f"{pre}_{t}"].astype(str).str.strip().str.upper(); y = lab.loc[ids]
                    acc.append((pred == y).mean() * 100)
                    pos = y == cplus(t)
                    under.append((pred[pos] != cplus(t)).mean() * 100)
                    over.append((pred[~pos] == cplus(t)).mean() * 100 if (~pos).any() else np.nan)
                rows.append(dict(labels=src, backbone=bn, model=mn, condition=cond, n=len(ids),
                                 acc=np.mean(acc), under=np.mean(under), over=np.nanmean(over)))
R = pd.DataFrame(rows).round(1)
R.to_csv(ROOT / "analysis" / "accuracy_table.csv", index=False)
pd.set_option("display.width", 200)
for src in ["consensus", "gold"]:
    s = R[R.labels == src]
    print(f"\n### labels = {src}  (n per condition shown)")
    print(s.pivot_table(index=["backbone", "model"], columns="condition", values="acc", sort=False)[CONDS].to_string())
    print("  under-triage (avg over conditions):"); print(s.groupby(["backbone","model"],sort=False)[["under","over"]].mean().round(1).T.to_string())
for t in T:
    for src in ["consensus","gold"]:
        y = labels[(src,t)]; print(src, t, "n=",len(y), "c+ share=", round((y==cplus(t)).mean()*100,1))
