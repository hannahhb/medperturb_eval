"""
Novel-tool split: does MPA's robustness gain concentrate in vignettes where
KG-conditioned planning retrieved novel tools?

A vignette is "novel" if, on its unperturbed (baseline) row, MPA's kg_novel
step (mpa_tu_rounds) returned at least one tool call with non-empty output.
This reproduces the 124 / 101 Llama-8B split.

Metric: mean flip rate (%) vs the unperturbed prediction, averaged over
3 perturbations x 3 tasks (same as flip_rate_summary.py).
Paired test: Wilcoxon signed-rank on per-vignette flip counts (0-9), MPA vs TUA.

Usage:  python analysis/novel_tool_split.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
MP_CSV = ROOT.parent / "data" / "medperturb_data.csv"
T = ["manage", "visit", "resource"]; P = ["summary", "gender_swap", "uncertain_tone"]
DS = {"oncqa", "askdocs"}

def norm(df, prefix):
    if "dataset" in df: df = df[df.dataset.isin(DS)]
    out = df[["context_id", "perturbation"] + [f"{prefix}_{t}" for t in T]].copy()
    out.columns = ["context_id", "perturbation"] + T
    for t in T: out[t] = out[t].astype(str).str.strip().str.upper()
    return out.drop_duplicates(["context_id", "perturbation"], keep="last")

def systems(bb):
    s = {}
    if bb == "llama70b3_3":
        s["Baseline"] = norm(pd.read_csv(MP_CSV), "llama")
    else:
        s["Baseline"] = norm(pd.read_csv(OUT / f"baseline/{bb}/oncqa_askdocs/run_0/results.csv"), bb)
    for name, d, pre in [("MRA", "medreason", "medreason"), ("TUA", "tooluni", "tooluni"),
                         ("MPA", "medpathagent", "medpathagent"), ("MPA-concat", "medpathagent_concat", "medpathagent")]:
        p = OUT / f"{d}/{bb}/oncqa_askdocs/run_0/results.csv"
        if p.exists(): s[name] = norm(pd.read_csv(p), f"{pre}_{bb}")
    return s

def n_novel(raw):
    try: r = json.loads(str(raw))
    except Exception: return 0
    return sum(1 for rd in r for t in rd.get("tools", []) if str(t.get("output", "") or "").strip())

def flip_matrix(df):
    """context_id x (pert, task) boolean flip matrix."""
    base = df[df.perturbation == "baseline"].set_index("context_id")[T]
    cols = {}
    for p in P:
        q = df[df.perturbation == p].set_index("context_id")[T]
        ids = base.index.intersection(q.index)
        for t in T: cols[(p, t)] = (base.loc[ids, t] != q.loc[ids, t]).reindex(base.index)
    return pd.DataFrame(cols)

def mean_flip(fm, ids, pert=None):
    sub = fm.loc[fm.index.intersection(ids)]
    cells = [(p, t) for p in (P if pert is None else [pert]) for t in T]
    return np.mean([sub[c].dropna().mean() * 100 for c in cells])

rows = []
for bb in ["llama8b3_1", "llama70b3_3", "qwen235b"]:
    mpa = pd.read_csv(OUT / f"medpathagent/{bb}/oncqa_askdocs/run_0/results.csv")
    b0 = mpa[mpa.perturbation == "baseline"]
    novel = set(b0[b0.mpa_tu_rounds.apply(n_novel) > 0].context_id)
    allc = set(b0.context_id); groups = {"novel": novel, "no_novel": allc - novel}
    fms = {k: flip_matrix(v) for k, v in systems(bb).items()}
    for g, ids in groups.items():
        r = {"backbone": bb, "group": g, "n": len(ids)}
        for k, fm in fms.items():
            r[k] = round(mean_flip(fm, ids), 1)
            for p in P: r[f"{k}_{p}"] = round(mean_flip(fm, ids, p), 1)
        for other in ["TUA", "MPA-concat"]:
            if other not in fms or "MPA" not in fms: continue
            a = fms["MPA"].loc[list(ids)].fillna(False).astype(int).sum(1)
            b = fms[other].loc[list(ids)].fillna(False).astype(int).sum(1)
            r[f"MPA_vs_{other}_p"] = round(wilcoxon(a, b).pvalue, 4) if (a != b).any() else 1.0
            r[f"MPA<{other}"] = int((a < b).sum()); r[f"MPA>{other}"] = int((a > b).sum())
        rows.append(r)

res = pd.DataFrame(rows)
res.to_csv(ROOT / "analysis" / "novel_tool_split.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print(res[["backbone", "group", "n", "Baseline", "MRA", "TUA", "MPA", "MPA-concat"]].to_string(index=False))
print()
print(res[["backbone", "group"] + [c for c in res.columns if c.startswith("MPA_vs") or c.startswith("MPA<") or c.startswith("MPA>")]].to_string(index=False))
print()
print(res[["backbone", "group"] + [c for c in res.columns if any(c.endswith(p) for p in P)]].to_string(index=False))
