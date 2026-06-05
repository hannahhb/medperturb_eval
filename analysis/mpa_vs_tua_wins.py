"""
MPA vs ToolUni — paired reduced-care-error (RCER) win analysis.
================================================================

A *win case* is a care-augmenting vignette (clinician/gold label = c+) under a
perturbation where exactly one of MPA / TUA makes a reduced-care error:

    MPA win  = MPA keeps care (correct)   AND TUA reduces care (error)
    TUA win  = TUA keeps care (correct)   AND MPA reduces care (error)

Concordant cases (both correct / both error) are ties and discarded — these are
exactly the discordant pairs of McNemar's test.

Care-augmenting (c+) per task:
    MANAGE  : NO   (reduced-care error = YES, told to self-manage)
    VISIT   : YES  (reduced-care error = NO,  told not to visit)
    RESOURCE: YES  (reduced-care error = NO,  denied resource/referral)

Label source: clinician consensus where available, else gold standard.

Outputs:
    analysis/mpa_vs_tua_wins.csv        — one row per win case (both directions)
    analysis/mpa_vs_tua_wins_summary.csv — counts by backbone × task × perturbation
    prints McNemar p-values per (backbone, task, perturbation) and pooled.

Usage:
    python analysis/mpa_vs_tua_wins.py
    python analysis/mpa_vs_tua_wins.py --backbones qwen235b llama70b3_3
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

# ── config ─────────────────────────────────────────────────────────────────────

OUTPUT_BASE    = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/output")
MEDPERTURB_CSV = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/medperturb_data.csv")
ANALYSIS_DIR   = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/analysis")

DATASETS = {"oncqa", "askdocs"}
TASKS    = ["manage", "visit", "resource"]
PERTS    = ["summary", "gender_swap", "uncertain_tone"]

ALL_BACKBONES = ["llama70b3_3", "qwen235b", "llama8b3_1"]
NAME_MAP = {"llama70b3_3": "Llama 70B", "qwen235b": "Qwen 235B", "llama8b3_1": "Llama 8B"}


def care_augmenting(task: str) -> str:
    """Label value that means 'augment care' (the clinically safe direction)."""
    return "NO" if task == "manage" else "YES"


def reduced_care_value(task: str) -> str:
    """Label value that constitutes a reduced-care error."""
    return "YES" if task == "manage" else "NO"


def has_val(s) -> bool:
    return str(s).strip() not in ("", "nan", "NaN", "None", "[]")


def mcnemar_p(b: int, c: int) -> float | None:
    """Continuity-corrected McNemar p-value. b, c = discordant counts."""
    if b + c == 0:
        return None
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(1 - stats.chi2.cdf(stat, df=1))


# ── label lookup (clinician consensus → gold fallback) ─────────────────────────

def build_label_lookup() -> dict[tuple[str, str], str]:
    """(context_id, task) → 'YES'|'NO' using clinician consensus, else gold."""
    mp = pd.read_csv(MEDPERTURB_CSV)
    mp = mp[(mp["perturbation"] == "baseline") & mp["dataset"].isin(DATASETS)]
    lookup: dict[tuple[str, str], str] = {}
    for _, row in mp.iterrows():
        cid = str(row["context_id"])
        for task in TASKS:
            clin = row.get(f"clinician_consensus_{task}")
            if pd.notna(clin):
                lookup[(cid, task)] = "YES" if float(clin) == 1.0 else "NO"
            else:
                gold = row.get(f"gold_standard_{task}")
                if pd.notna(gold) and str(gold).strip():
                    lookup[(cid, task)] = str(gold).strip().upper()
    return lookup


# ── per-backbone win extraction ────────────────────────────────────────────────

def extract_wins(backbone: str, label_lookup: dict) -> pd.DataFrame:
    mpa_path = OUTPUT_BASE / f"medpathagent/{backbone}/oncqa_askdocs/run_0/results.csv"
    tu_path  = OUTPUT_BASE / f"tooluni/{backbone}/oncqa_askdocs/run_0/results.csv"
    if not mpa_path.exists() or not tu_path.exists():
        print(f"  skip {backbone}: missing results.csv")
        return pd.DataFrame()

    mpa = pd.read_csv(mpa_path)
    tu  = pd.read_csv(tu_path)

    rows = []
    for task in TASKS:
        mpa_col = f"medpathagent_{backbone}_{task}"
        tu_col  = f"tooluni_{backbone}_{task}"
        if mpa_col not in mpa.columns or tu_col not in tu.columns:
            continue
        c_minus = reduced_care_value(task)

        for pert in PERTS:
            keep = ["context_id", mpa_col]
            for extra in ["mpa_kg_paths", "mpa_tool_context", "mpa_tu_rounds"]:
                if extra in mpa.columns:
                    keep.append(extra)
            mp_p = mpa[mpa["perturbation"] == pert][keep].rename(columns={mpa_col: "mpa_pred"})
            tu_p = tu[tu["perturbation"] == pert][["context_id", tu_col]].rename(columns={tu_col: "tu_pred"})
            m = mp_p.merge(tu_p, on="context_id", how="inner")
            if m.empty:
                continue

            # attach care-augmenting label, restrict to c+ cases
            m["label"] = m["context_id"].apply(lambda cid: label_lookup.get((str(cid), task), ""))
            care = m[m["label"] == care_augmenting(task)].copy()
            if care.empty:
                continue

            mpa_err = care["mpa_pred"].astype(str).str.upper() == c_minus
            tu_err  = care["tu_pred"].astype(str).str.upper()  == c_minus

            for _, r in care.iterrows():
                me = str(r["mpa_pred"]).upper() == c_minus
                te = str(r["tu_pred"]).upper()  == c_minus
                if me == te:
                    continue  # tie (both correct or both error) — discard
                rows.append({
                    "backbone":     NAME_MAP.get(backbone, backbone),
                    "backbone_key": backbone,
                    "task":         task,
                    "perturbation": pert,
                    "context_id":   r["context_id"],
                    "dataset":      "oncqa" if str(r["context_id"]).startswith("oncqa") else "askdocs",
                    "winner":       "MPA" if (te and not me) else "TUA",
                    "label":        r["label"],
                    "mpa_pred":     r["mpa_pred"],
                    "tu_pred":      r["tu_pred"],
                    "has_kg":       has_val(r.get("mpa_kg_paths", "")),
                    "has_novel_tools": has_val(r.get("mpa_tool_context", "")),
                    "kg_anchor":    (str(r.get("mpa_kg_paths", "")).split("->")[0].strip()
                                     if has_val(r.get("mpa_kg_paths", "")) else ""),
                    "kg_paths":     str(r.get("mpa_kg_paths", ""))[:300],
                    "tool_context": str(r.get("mpa_tool_context", ""))[:300],
                })
    return pd.DataFrame(rows)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=ALL_BACKBONES)
    args = ap.parse_args()

    label_lookup = build_label_lookup()
    print(f"Labels loaded: {len(label_lookup)} (context_id, task) pairs\n")

    all_wins = []
    for bb in args.backbones:
        print(f"Processing {bb}...")
        all_wins.append(extract_wins(bb, label_lookup))
    wins = pd.concat([w for w in all_wins if not w.empty], ignore_index=True)

    out_csv = ANALYSIS_DIR / "mpa_vs_tua_wins.csv"
    wins.to_csv(out_csv, index=False)
    print(f"\nSaved per-case wins → {out_csv}  ({len(wins)} discordant cases)\n")

    # ── headline counts ─────────────────────────────────────────────────────────
    mpa_w = (wins["winner"] == "MPA").sum()
    tua_w = (wins["winner"] == "TUA").sum()
    print("=" * 50)
    print(f"TOTAL  MPA wins: {mpa_w}   TUA wins: {tua_w}   net(MPA-TUA): {mpa_w - tua_w}")
    print("=" * 50)

    # ── breakdowns ──────────────────────────────────────────────────────────────
    for axis in ["backbone", "task", "perturbation", "dataset"]:
        tbl = (wins.groupby([axis, "winner"]).size()
                   .unstack(fill_value=0)
                   .reindex(columns=["MPA", "TUA"], fill_value=0))
        tbl["net"] = tbl["MPA"] - tbl["TUA"]
        print(f"\n--- by {axis} ---")
        print(tbl.to_string())

    # ── win mechanism (MPA wins only) ───────────────────────────────────────────
    mpa_only = wins[wins["winner"] == "MPA"]
    if not mpa_only.empty:
        print("\n--- MPA win mechanism ---")
        mech = pd.DataFrame({
            "KG + novel tools": [(mpa_only["has_kg"] & mpa_only["has_novel_tools"]).sum()],
            "KG only":          [(mpa_only["has_kg"] & ~mpa_only["has_novel_tools"]).sum()],
            "neither":          [(~mpa_only["has_kg"] & ~mpa_only["has_novel_tools"]).sum()],
        })
        print(mech.to_string(index=False))
        print("\n--- MPA win top KG anchors ---")
        print(mpa_only[mpa_only["has_kg"]]["kg_anchor"].value_counts().head(12).to_string())

    # ── McNemar tests ───────────────────────────────────────────────────────────
    print("\n--- McNemar (per backbone × task × perturbation) ---")
    print(f"{'backbone':<11}{'task':<10}{'pert':<16}{'MPA':>4}{'TUA':>5}{'p':>9}  sig")
    grp = wins.groupby(["backbone_key", "task", "perturbation", "winner"]).size().unstack(fill_value=0)
    for (bb, task, pert), r in grp.iterrows():
        b = int(r.get("MPA", 0)); c = int(r.get("TUA", 0))
        p = mcnemar_p(b, c)
        sig = "***" if p and p < .001 else "**" if p and p < .01 else "*" if p and p < .05 else "ns"
        ps = f"{p:.4f}" if p is not None else "N/A"
        print(f"{bb:<11}{task:<10}{pert:<16}{b:>4}{c:>5}{ps:>9}  {sig}")

    # pooled
    print("\n--- McNemar (pooled, all backbones+tasks+perts) ---")
    p = mcnemar_p(mpa_w, tua_w)
    sig = "***" if p and p < .001 else "**" if p and p < .01 else "*" if p and p < .05 else "ns"
    ps = f"{p:.4f}" if p is not None else "N/A"
    print(f"MPA={mpa_w} TUA={tua_w}  p={ps}  {sig}")

    # ── summary CSV ─────────────────────────────────────────────────────────────
    summary = (wins.groupby(["backbone", "task", "perturbation", "winner"]).size()
                   .unstack(fill_value=0).reset_index())
    summary_csv = ANALYSIS_DIR / "mpa_vs_tua_wins_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\nSaved summary → {summary_csv}")


if __name__ == "__main__":
    main()
