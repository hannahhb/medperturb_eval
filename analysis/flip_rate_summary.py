"""
Compute average flip rate per method, averaged across backbone models and runs.

Methods and their available backbone paths:
  - Baseline:      llama70b (medperturb_data.csv), llama8b, qwen235b
  - MedReason:     llama70b, llama8b, qwen235b
  - ToolUni:       llama70b, qwen235b  (no llama8b)
  - MedPathAgent:  llama70b only

All runs under each backbone are averaged together.
Output: one row per method in flip_rate_summary.csv
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────

OUTPUT_BASE    = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/output")
MEDPERTURB_CSV = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/medperturb_data.csv")
OUT_CSV         = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/analysis/flip_rate_summary.csv")
OUT_CSV_BACKBONE = Path("/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/analysis/flip_rate_per_backbone.csv")

DATASETS = {"oncqa", "askdocs"}
RUN_RE   = re.compile(r"^run_\d+$")
MIN_ROWS = 600

TASKS = ["manage", "visit", "resource"]
PERTS = ["summary", "gender_swap", "uncertain_tone"]

# ── explicit method → backbone → run_0 path (+ extra runs auto-discovered) ───

# For each method, list the backbone dirs to scan (all run_N under oncqa_askdocs/)
METHOD_BACKBONE_DIRS = {
    "Baseline": [
        OUTPUT_BASE / "baseline" / "llama8b3_1" / "oncqa_askdocs",
        OUTPUT_BASE / "baseline" / "qwen235b"   / "oncqa_askdocs",
        # llama70b handled separately via medperturb_data.csv below
    ],
    "MedReason": [
        OUTPUT_BASE / "medreason" / "llama70b3_3" / "oncqa_askdocs",
        OUTPUT_BASE / "medreason" / "llama8b3_1"  / "oncqa_askdocs",
        OUTPUT_BASE / "medreason" / "qwen235b"    / "oncqa_askdocs",
    ],
    "ToolUni": [
        OUTPUT_BASE / "tooluni" / "llama70b3_3" / "oncqa_askdocs",
        OUTPUT_BASE / "tooluni" / "qwen235b"    / "oncqa_askdocs",
    ],
    "MedPathAgent": [
        OUTPUT_BASE / "medpathagent" / "llama70b3_3" / "oncqa_askdocs",
    ],
}

# Baseline llama70b comes from medperturb_data.csv column "llama"
MEDPERTURB_BASELINE = {"col_prefix": "llama"}

# For per-backbone breakdown: explicit run_0 path per method × backbone
MEDPERTURB_COLS = {"llama70b3_3": "llama", "qwen235b": "qwen"}
BACKBONE_LABEL  = {"llama70b3_3": "Llama 70B", "llama8b3_1": "Llama 8B", "qwen235b": "Qwen 235B"}

METHOD_BACKBONE_RUN0 = {
    "Baseline": {
        "llama70b3_3": None,  # from medperturb_data.csv
        "llama8b3_1":  OUTPUT_BASE / "baseline/llama8b3_1/oncqa_askdocs/run_0/results.csv",
        "qwen235b":    OUTPUT_BASE / "baseline/qwen235b/oncqa_askdocs/run_0/results.csv",
    },
    "MedReason": {
        "llama70b3_3": OUTPUT_BASE / "medreason/llama70b3_3/oncqa_askdocs/run_0/results.csv",
        "llama8b3_1":  OUTPUT_BASE / "medreason/llama8b3_1/oncqa_askdocs/run_0/results.csv",
        "qwen235b":    OUTPUT_BASE / "medreason/qwen235b/oncqa_askdocs/run_0/results.csv",
    },
    "ToolUni": {
        "llama70b3_3": OUTPUT_BASE / "tooluni/llama70b3_3/oncqa_askdocs/run_0/results.csv",
        "qwen235b":    OUTPUT_BASE / "tooluni/qwen235b/oncqa_askdocs/run_0/results.csv",
        "llama8b3_1":    OUTPUT_BASE / "tooluni/llama8b3_1/oncqa_askdocs/run_0/results.csv",

    },
    "MedPathAgent": {
        "llama70b3_3": OUTPUT_BASE / "medpathagent/llama70b3_3/oncqa_askdocs/run_0/results.csv",
        "qwen235b":    OUTPUT_BASE / "medpathagent/qwen235b/oncqa_askdocs/run_0/results.csv",
        "llama8b3_1":    OUTPUT_BASE / "medpathagent/llama8b3_1/oncqa_askdocs/run_0/results.csv",

    },
}

# ── helpers ───────────────────────────────────────────────────────────────────

def get_pred_prefix(df: pd.DataFrame) -> str | None:
    candidates = [
        c for c in df.columns
        if "manage" in c
        and "gold" not in c
        and "reasoning" not in c
        and "standard" not in c
        and "consensus" not in c
    ]
    return candidates[0].replace("_manage", "") if candidates else None


def flip_rates_df(df: pd.DataFrame, col: str, task: str) -> dict[str, float]:
    base = df[df["perturbation"] == "baseline"][["context_id", col]].rename(columns={col: "b"})
    rates = {}
    for p in PERTS:
        pert = df[df["perturbation"] == p][["context_id", col]].rename(columns={col: "q"})
        m = base.merge(pert, on="context_id")
        if not m.empty:
            rates[p] = (m["b"] != m["q"]).mean() * 100
    return rates


def load_runs_from_dir(backbone_dir: Path) -> list[tuple[pd.DataFrame, str]]:
    """Load all valid canonical run_N results from a backbone/oncqa_askdocs dir."""
    pairs = []
    if not backbone_dir.exists():
        print(f"  MISSING: {backbone_dir}")
        return pairs
    for run_dir in sorted(backbone_dir.iterdir()):
        if not run_dir.is_dir() or not RUN_RE.match(run_dir.name):
            continue
        csv = run_dir / "results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if "dataset" in df.columns:
            df = df[df["dataset"].isin(DATASETS)]
        if len(df) < MIN_ROWS:
            print(f"  skip {csv.parent.name}/{run_dir.name} ({len(df)} rows)")
            continue
        prefix = get_pred_prefix(df)
        if prefix is None:
            print(f"  WARNING: no pred col in {csv}")
            continue
        print(f"  {backbone_dir.parent.name}/{backbone_dir.name}/{run_dir.name}  prefix={prefix}  rows={len(df)}")
        pairs.append((df, prefix))
    return pairs


def summarise(pairs: list[tuple[pd.DataFrame, str]], label: str) -> dict:
    # (pert, task) → list of flip rates across runs/backbones
    vals: dict[tuple, list[float]] = {(p, t): [] for p in PERTS for t in TASKS}

    for df, prefix in pairs:
        for t in TASKS:
            col = f"{prefix}_{t}"
            if col not in df.columns:
                continue
            rates = flip_rates_df(df, col, t)
            for p, r in rates.items():
                vals[(p, t)].append(r)

    row = {"method": label}

    # Per-perturbation × per-task
    for p in PERTS:
        for t in TASKS:
            v = vals[(p, t)]
            row[f"{p}_{t}"] = round(np.mean(v), 1) if v else None
        # Per-perturbation average across tasks
        pert_all = [r for t in TASKS for r in vals[(p, t)]]
        row[f"{p}_avg"] = round(np.mean(pert_all), 1) if pert_all else None

    # Per-task average across perturbations
    for t in TASKS:
        task_all = [r for p in PERTS for r in vals[(p, t)]]
        row[f"{t}_avg"] = round(np.mean(task_all), 1) if task_all else None
        row[f"{t}_std"] = round(np.std(task_all), 1)  if task_all else None

    # Overall
    all_vals = [r for v in vals.values() for r in v]
    row["overall_avg"] = round(np.mean(all_vals), 1) if all_vals else None
    row["overall_std"] = round(np.std(all_vals), 1)  if all_vals else None
    return row


def flip_single(df: pd.DataFrame, prefix: str, task: str, pert: str):
    col = f"{prefix}_{task}"
    if col not in df.columns:
        return None
    base = df[df["perturbation"] == "baseline"][["context_id", col]].rename(columns={col: "b"})
    pert_df = df[df["perturbation"] == pert][["context_id", col]].rename(columns={col: "q"})
    m = base.merge(pert_df, on="context_id")
    return round((m["b"] != m["q"]).mean() * 100, 1) if not m.empty else None


def summarise_backbone(method: str, backbone: str, path) -> dict:
    mp = pd.read_csv(MEDPERTURB_CSV)
    mp = mp[mp["dataset"].isin(DATASETS)]

    if path is None:
        prefix = MEDPERTURB_COLS[backbone]
        df = mp
    else:
        if not Path(path).exists():
            print(f"  MISSING: {path}")
            return {}
        df = pd.read_csv(path)
        if "dataset" in df.columns:
            df = df[df["dataset"].isin(DATASETS)]
        prefix = get_pred_prefix(df)
        if prefix is None:
            print(f"  WARNING: no pred col in {path}")
            return {}

    row = {"Method": method, "Backbone": BACKBONE_LABEL[backbone]}
    all_vals = []
    for p in PERTS:
        for t in TASKS:
            v = flip_single(df, prefix, t, p)
            row[f"{p}_{t}"] = v
            if v is not None:
                all_vals.append(v)
        pert_vals = [flip_single(df, prefix, t, p) for t in TASKS]
        pert_vals = [v for v in pert_vals if v is not None]
        row[f"{p}_avg"] = round(np.mean(pert_vals), 1) if pert_vals else None
    for t in TASKS:
        task_vals = [flip_single(df, prefix, t, p) for p in PERTS]
        task_vals = [v for v in task_vals if v is not None]
        row[t.capitalize()] = round(np.mean(task_vals), 1) if task_vals else None
    row["Overall"] = round(np.mean(all_vals), 1) if all_vals else None
    return row


# ── main ─────────────────────────────────────────────────────────────────────

rows = []

for method, backbone_dirs in METHOD_BACKBONE_DIRS.items():
    print(f"\n=== {method} ===")
    pairs = []

    # Special case: Baseline llama70b from medperturb_data.csv
    if method == "Baseline":
        mp = pd.read_csv(MEDPERTURB_CSV)
        mp = mp[mp["dataset"].isin(DATASETS)]
        prefix = MEDPERTURB_BASELINE["col_prefix"]
        print(f"  medperturb_data.csv  prefix={prefix}  rows={len(mp)}")
        pairs.append((mp, prefix))

    for d in backbone_dirs:
        pairs.extend(load_runs_from_dir(d))

    rows.append(summarise(pairs, method))

out_df = pd.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUT_CSV, index=False)

print(f"\nSaved to {OUT_CSV}")
print(out_df.to_string(index=False))

# ── per-backbone breakdown (run_0 only) ───────────────────────────────────────

print("\n=== Per-backbone (run_0) ===")
backbone_rows = []
for method, backbones in METHOD_BACKBONE_RUN0.items():
    for backbone, path in backbones.items():
        r = summarise_backbone(method, backbone, path)
        if r:
            backbone_rows.append(r)

backbone_df = pd.DataFrame(backbone_rows)
backbone_df.to_csv(OUT_CSV_BACKBONE, index=False)
print(f"Saved to {OUT_CSV_BACKBONE}")
print(backbone_df[["Method", "Backbone", "summary_avg", "gender_swap_avg", "uncertain_tone_avg",
                    "Manage", "Visit", "Resource", "Overall"]].to_string(index=False))
