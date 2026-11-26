"""
Exact McNemar tests: MPA vs MPA-concat, per backbone x perturbation.

Outcome per vignette = "flipped" (perturbed prediction != unperturbed prediction).
  b = vignettes where concat flips but MPA is stable   (MPA wins)
  c = vignettes where MPA flips but concat is stable   (concat wins)
Exact two-sided McNemar p = binomial test of b vs c (p=0.5). Holm correction
across the 9 backbone x perturbation tests.

Two units of analysis:
  vignette  - flipped if ANY of MANAGE/VISIT/RESOURCE flips (one pair per vignette,
              so pairs are independent; this is the main test)
  per task  - one test per backbone x perturbation x task (27 tests, Holm over 27)

Usage: python analysis/mpa_vs_concat_mcnemar.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
T = ["manage", "visit", "resource"]; P = ["summary", "gender_swap", "uncertain_tone"]
BB = {"llama8b3_1": "Llama-8B", "llama70b3_3": "Llama-70B", "qwen235b": "Qwen-235B"}
PL = {"summary": "Summary", "gender_swap": "Gender swap", "uncertain_tone": "Uncertain tone"}

def load(path, bb):
    d = pd.read_csv(path).drop_duplicates(["context_id", "perturbation"], keep="last")
    d = d[["context_id", "perturbation"] + [f"medpathagent_{bb}_{t}" for t in T]]
    d.columns = ["context_id", "perturbation"] + T
    for t in T: d[t] = d[t].astype(str).str.strip().str.upper()
    return d

def flips(d, pert):
    b = d[d.perturbation == "baseline"].set_index("context_id")[T]
    q = d[d.perturbation == pert].set_index("context_id")[T]
    ids = b.index.intersection(q.index)
    return b.loc[ids] != q.loc[ids]

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
    return adj

def mcn(a_flip, c_flip):
    b = int((c_flip & ~a_flip).sum()); c = int((a_flip & ~c_flip).sum())
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)

rows_v, rows_t = [], []
for bb, name in BB.items():
    mpa = load(OUT / f"medpathagent/{bb}/oncqa_askdocs/run_0/results.csv", bb)
    con = load(OUT / f"medpathagent_concat/{bb}/oncqa_askdocs/run_0/results.csv", bb)
    for p in P:
        fm, fc = flips(mpa, p), flips(con, p)
        ids = fm.index.intersection(fc.index); fm, fc = fm.loc[ids], fc.loc[ids]
        b, c, pv = mcn(fm.any(axis=1), fc.any(axis=1))
        rows_v.append(dict(backbone=name, perturbation=PL[p], n=len(ids),
                           flip_concat=round(fc.any(axis=1).mean()*100, 1),
                           flip_mpa=round(fm.any(axis=1).mean()*100, 1),
                           b_mpa_wins=b, c_concat_wins=c, p=pv))
        for t in T:
            b, c, pv = mcn(fm[t], fc[t])
            rows_t.append(dict(backbone=name, perturbation=PL[p], task=t, n=len(ids),
                               b_mpa_wins=b, c_concat_wins=c, p=pv))

V = pd.DataFrame(rows_v); V["p_holm"] = holm(V.p.values)
Tt = pd.DataFrame(rows_t); Tt["p_holm"] = holm(Tt.p.values)
V.to_csv(ROOT / "analysis" / "mpa_vs_concat_mcnemar_vignette.csv", index=False)
Tt.to_csv(ROOT / "analysis" / "mpa_vs_concat_mcnemar_task.csv", index=False)
pd.set_option("display.width", 200)
print("== vignette level (any task flips)\n", V.round(4).to_string(index=False))
print("\n== per task\n", Tt.round(4).to_string(index=False))
