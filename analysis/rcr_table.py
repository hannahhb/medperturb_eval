"""
Reduced Care Rate (RCR) table, Gourabathina et al. (FAccT '25) Eq. 2:

    RCR = #(baseline pred = c+ AND perturbed pred = c-) / #(baseline pred = c+)

c+ (care-augmenting): MANAGE=NO, VISIT=YES, RESOURCE=YES.
RCR uses model predictions only, no gold labels.

Reuses rcr() and the run_0 loaders from scripts/medperturb_robustness.py.
Two conventions:
  paper    — as in medperturb_robustness.py: a refusal/unparsed answer on the
             perturbed input counts as care-reducing.
  no_ref   — rows where either prediction is not YES/NO are dropped.

Outputs analysis/rcr_table.csv and prints a markdown table.
Usage: python analysis/rcr_table.py
"""
import sys, contextlib, io
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import medperturb_robustness as mr  # noqa: E402

TASKS, PERTS = mr.TASKS, mr.PERTS
PL = {"summary": "Sum", "gender_swap": "GS", "uncertain_tone": "UT"}
METHODS = [("Baseline", "Baseline"), ("MedReason", "MRA"), ("ToolUni", "TUA"), ("MedPathAgent", "MPA")]
BACKBONES = [("Llama 8B", "Llama-8B"), ("Llama 70B", "Llama-70B"), ("Qwen 235B", "Qwen-235B")]

def clean(df, prefix):
    df = df.drop_duplicates(["context_id", "perturbation"], keep="last").copy()
    for t in TASKS:
        df[f"{prefix}_{t}"] = df[f"{prefix}_{t}"].astype(str).str.strip().str.upper()
    return df

def no_refusals(df, prefix, task):
    ok = df[f"{prefix}_{task}"].isin(["YES", "NO"])
    bad = set(df.loc[~ok & (df.perturbation == "baseline"), "context_id"])
    d = df[ok]
    return d[~d.context_id.isin(bad)]

with contextlib.redirect_stdout(io.StringIO()):
    data = mr.load_all()

rows = []
for conv in ["paper", "no_ref"]:
    for bkey, bname in BACKBONES:
        for mkey, mname in METHODS:
            lab = f"{mkey}\n{bkey}"
            if lab not in data:
                continue
            df, pre = data[lab]; df = clean(df, pre)
            r = {"convention": conv, "backbone": bname, "model": mname}
            for p in PERTS:
                for t in TASKS:
                    d = no_refusals(df, pre, t) if conv == "no_ref" else df
                    r[f"{PL[p]}-{t[0].upper()}"] = mr.rcr(d, pre, t, p)
            for t in TASKS:
                r[f"Avg-{t[0].upper()}"] = np.mean([r[f"{PL[p]}-{t[0].upper()}"] for p in PERTS])
            r["Overall"] = np.mean([r[f"{PL[p]}-{t[0].upper()}"] for p in PERTS for t in TASKS])
            rows.append(r)

res = pd.DataFrame(rows).round(1)
res.to_csv(ROOT / "analysis" / "rcr_table.csv", index=False)

cols = [f"{PL[p]}-{t[0].upper()}" for p in PERTS for t in TASKS] + [f"Avg-{t[0].upper()}" for t in TASKS]
for conv in ["paper", "no_ref"]:
    sub = res[res.convention == conv]
    print(f"\n### convention = {conv}\n")
    print("| Backbone | Model | " + " | ".join(cols) + " | Overall |")
    print("|:--|:--|" + "--:|" * (len(cols) + 1))
    for b in sub.backbone.unique():
        g = sub[sub.backbone == b]
        best = {c: g[c].min() for c in cols + ["Overall"]}
        for i, (_, r) in enumerate(g.iterrows()):
            cells = [f"**{r[c]:.1f}**" if r[c] == best[c] else f"{r[c]:.1f}" for c in cols + ["Overall"]]
            print(f"| {b if i == 0 else ''} | {r.model} | " + " | ".join(cells) + " |")
    print("\nMean over backbones (overall):", sub.groupby("model", sort=False).Overall.mean().round(1).to_dict())
