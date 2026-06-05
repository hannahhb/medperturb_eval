"""
Analyse ToolUni tool usage from per-question detailed JSON logs.

Supports two log formats:
  medxpertqa  — files: detailed_Text-N_<qtype>.json
                fields: id, question_type, tu_rounds
                group:  body_system (from --meta-jsonl)
  medperturb  — files: detailed_<dataset>_<id>__<perturbation>.json
                fields: row_id, perturbation, tu_rounds
                group:  dataset  (parsed from row_id)

Outputs:
  1. Top-10 tools by call count  (+ error rate per tool)
  2. Top-10 tools by error count
  3. Top-10 tools by successful execution
  4. Heatmap: tool usage by group  (body_system / dataset)
  5. Heatmap: tool usage by subgroup  (question_type / perturbation)
  6. Avg successful tool calls per question by group

Usage:
    # MedXpertQA (auto-detected)
    python scripts/analyse_tooluni_logs.py
    python scripts/analyse_tooluni_logs.py --logs-dir logs/medxpertqa/tooluni_llama70b3_3

    # MedPerturb (auto-detected from row_id field)
    python scripts/analyse_tooluni_logs.py \\
        --logs-dir logs/tooluni_llama70b3_3/oncqa_askdocs/run_0

    # Force a mode
    python scripts/analyse_tooluni_logs.py --mode medperturb \\
        --logs-dir logs/tooluni_llama70b3_3/oncqa_askdocs/run_0

    # Filter
    python scripts/analyse_tooluni_logs.py --subgroup baseline
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── default paths ──────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT.parent / "data"
JSONL_PATH = DATA_DIR / "medxpertqa_text.jsonl"

DEFAULT_MEDXPERTQA_LOGS  = ROOT / "logs" / "medxpertqa" / "tooluni_llama70b3_3"
DEFAULT_MEDPERTURB_LOGS  = ROOT / "logs" / "tooluni_llama70b3_3" / "oncqa_askdocs" / "run_0"
# MedReason (MedPathAgent) logs — KG paths only, no tu_rounds (tool stats n/a):
#   logs/medreason_llama70b3_3/oncqa_askdocs/run_0/


# ── log loading ────────────────────────────────────────────────────────────────

def _detect_mode(logs_dir: Path) -> str:
    """
    Auto-detect log mode by peeking at the first JSON file.
    Returns "medperturb" if records have a 'row_id' field, else "medxpertqa".
    """
    for f in sorted(logs_dir.glob("detailed_*.json"))[:3]:
        try:
            d = json.loads(f.read_text())
            if "row_id" in d:
                return "medperturb"
            if "id" in d:
                return "medxpertqa"
        except Exception:
            continue
    return "medxpertqa"


def _parse_medperturb_row_id(row_id: str) -> tuple[str, str]:
    """
    'askdocs_52__baseline'  → dataset='askdocs',  perturbation='baseline'
    'oncqa_12__gender_swap' → dataset='oncqa',    perturbation='gender_swap'
    """
    m = re.match(r"^([a-zA-Z]+)_.*?__(.+)$", row_id)
    if m:
        return m.group(1), m.group(2)
    return "unknown", "unknown"


def load_logs(
    logs_dir: Path,
    mode: str,
    subgroup_filter: str = "all",
    meta: dict[str, dict] | None = None,
) -> list[dict]:
    """
    Load all detailed_*.json files and normalise to a common schema:
      {id, group, subgroup, tu_rounds, ...original fields}

    medxpertqa: id=id,     group=body_system (from meta), subgroup=question_type
    medperturb: id=row_id, group=dataset,                 subgroup=perturbation
    """
    records = []
    for f in sorted(logs_dir.glob("detailed_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue

        if mode == "medperturb":
            row_id = d.get("row_id", f.stem)
            dataset, perturbation = _parse_medperturb_row_id(row_id)
            d["_id"]       = row_id
            d["_group"]    = dataset
            d["_subgroup"] = d.get("perturbation", perturbation)
            # tu_rounds is stored as a JSON-encoded string by processor.py
            if isinstance(d.get("tu_rounds"), str):
                try:
                    d["tu_rounds"] = json.loads(d["tu_rounds"])
                except Exception:
                    d["tu_rounds"] = []
        else:
            qid = d.get("id", f.stem)
            d["_id"]       = qid
            d["_group"]    = (meta or {}).get(qid, {}).get("body_system", "Unknown")
            d["_subgroup"] = d.get("question_type", "unknown")

        if subgroup_filter != "all" and d["_subgroup"] != subgroup_filter:
            continue

        records.append(d)
    return records


def load_medxpertqa_meta(jsonl_path: Path) -> dict[str, dict]:
    """Return {id: {body_system, medical_task}} from MedXpertQA JSONL."""
    meta = {}
    if not jsonl_path.exists():
        return meta
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        meta[row["id"]] = {
            "body_system":  row.get("body_system", "Unknown"),
            "medical_task": row.get("medical_task", "Unknown"),
        }
    return meta


# ── tool stats ─────────────────────────────────────────────────────────────────

_ERROR_SUBSTRINGS = (
    "error", "invalid", "not found", "no result", "failed", "exception",
    "unable to", "could not", "no data", "no match", "none found",
    "returned empty", "empty response", "status code",
)


def _is_real_output(output) -> bool:
    if not output:
        return False
    text = output if isinstance(output, str) else json.dumps(output, default=str)
    text_l = text.lower().strip()
    if not text_l or text_l in ("{}", "[]", "null", "none"):
        return False
    for kw in _ERROR_SUBSTRINGS:
        if kw in text_l and len(text_l) < 300:
            return False
    return True


def aggregate_tool_stats(records: list[dict]) -> pd.DataFrame:
    tool_calls   = defaultdict(int)
    tool_errors  = defaultdict(int)
    tool_success = defaultdict(int)
    tool_qs      = defaultdict(set)

    for rec in records:
        qid = rec.get("_id", "")
        for step in rec.get("tu_rounds", []):
            for t in step.get("tools", []):
                name = t.get("tool_name", "unknown")
                tool_calls[name]  += 1
                tool_qs[name].add(qid)
                if t.get("error"):
                    tool_errors[name] += 1
                elif _is_real_output(t.get("output")):
                    tool_success[name] += 1

    rows = []
    for name in tool_calls:
        c = tool_calls[name]
        e = tool_errors[name]
        s = tool_success[name]
        rows.append({
            "tool_name":    name,
            "calls":        c,
            "successes":    s,
            "errors":       e,
            "error_rate":   e / c * 100 if c else 0.0,
            "success_rate": s / c * 100 if c else 0.0,
            "questions":    len(tool_qs[name]),
        })
    return pd.DataFrame(rows).sort_values("calls", ascending=False).reset_index(drop=True)


def successful_tool_counts(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        n_success = sum(
            1
            for step in rec.get("tu_rounds", [])
            for t in step.get("tools", [])
            if not t.get("error") and _is_real_output(t.get("output"))
        )
        rows.append({
            "id":       rec.get("_id", ""),
            "group":    rec.get("_group", "Unknown"),
            "subgroup": rec.get("_subgroup", ""),
            "n_successful_tools": n_success,
        })
    return pd.DataFrame(rows)


def build_group_tool_matrix(
    records: list[dict],
    group_key: str = "_group",
    top_n_tools: int = 20,
) -> pd.DataFrame:
    """DataFrame [group × tool_name] with call counts."""
    counts = defaultdict(lambda: defaultdict(int))
    for rec in records:
        grp = rec.get(group_key, "Unknown")
        for step in rec.get("tu_rounds", []):
            for t in step.get("tools", []):
                counts[grp][t.get("tool_name", "unknown")] += 1

    df = pd.DataFrame(counts).fillna(0).T
    if df.empty:
        return df
    top_tools = df.sum(axis=0).nlargest(top_n_tools).index
    return df[top_tools]


# ── plotting ───────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def plot_top10_calls(stats: pd.DataFrame, out_path: Path):
    top = stats.nlargest(10, "calls").iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top["tool_name"], top["calls"], color="#4C72B0", alpha=0.85)
    for bar, (_, row) in zip(bars, top.iterrows()):
        ax.text(bar.get_width() + top["calls"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{row['calls']} calls  |  {row['error_rate']:.0f}% err",
                va="center", fontsize=8, color="#333")
    ax.set_xlabel("Total calls")
    ax.set_title("Top 10 Tools by Call Count\n(error rate annotated)", fontweight="bold")
    ax.set_xlim(right=top["calls"].max() * 1.35)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_top10_errors(stats: pd.DataFrame, out_path: Path):
    top = stats[stats["errors"] > 0].nlargest(10, "errors").iloc[::-1]
    if top.empty:
        print("  No tool errors found — skipping error plot.")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top["tool_name"], top["errors"], color="#C44E52", alpha=0.85)
    for bar, (_, row) in zip(bars, top.iterrows()):
        ax.text(bar.get_width() + top["errors"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{row['errors']} / {row['calls']}  ({row['error_rate']:.0f}%)",
                va="center", fontsize=8, color="#333")
    ax.set_xlabel("Error count")
    ax.set_title("Top 10 Tools by Error Count\n(errors / total calls annotated)", fontweight="bold")
    ax.set_xlim(right=top["errors"].max() * 1.35)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_top10_successes(stats: pd.DataFrame, out_path: Path):
    top = stats[stats["successes"] > 0].nlargest(10, "successes").iloc[::-1]
    if top.empty:
        print("  No successful tool executions found — skipping.")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top["tool_name"], top["successes"], color="#55A868", alpha=0.85)
    for bar, (_, row) in zip(bars, top.iterrows()):
        ax.text(bar.get_width() + top["successes"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{row['successes']} / {row['calls']}  ({row['success_rate']:.0f}%)",
                va="center", fontsize=8, color="#333")
    ax.set_xlabel("Successful executions")
    ax.set_title("Top 10 Tools by Successful Execution\n(successes / total calls annotated)",
                 fontweight="bold")
    ax.set_xlim(right=top["successes"].max() * 1.45)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_heatmap(matrix: pd.DataFrame, out_path: Path, title: str, row_label: str):
    if matrix.empty:
        print(f"  Empty matrix for {title} — skipping.")
        return
    short_cols = [c.replace("_", " ") for c in matrix.columns]
    fig, ax = plt.subplots(figsize=(max(12, len(matrix.columns) * 0.8),
                                    max(4,  len(matrix.index)  * 0.45)))
    norm = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0) * 100
    im = ax.imshow(norm.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="% of group's tool calls (row-normalised)")
    ax.set_xticks(range(len(short_cols)))
    ax.set_xticklabels(short_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)
    for i, row in enumerate(matrix.index):
        for j, col in enumerate(matrix.columns):
            v = int(matrix.loc[row, col])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=7,
                        color="black" if norm.values[i, j] < 60 else "white")
    ax.set_ylabel(row_label)
    ax.set_title(f"{title}\n(colour = % of row calls; numbers = raw count)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_avg_successful_tools(sc: pd.DataFrame, out_path: Path, group_label: str = "Group"):
    overall_avg = sc["n_successful_tools"].mean()
    by_grp = (
        sc.groupby("group")["n_successful_tools"]
        .agg(avg="mean", n="count")
        .reset_index()
        .sort_values("avg")
    )
    labels = list(by_grp["group"]) + ["Aggregate"]
    values = list(by_grp["avg"])   + [overall_avg]
    colors = ["#4C72B0"] * len(by_grp) + ["#C44E52"]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.4)))
    bars = ax.barh(labels, values, color=colors, alpha=0.85)
    for bar, val, (_, row) in zip(bars[:-1], values[:-1], by_grp.iterrows()):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}  (n={int(row['n'])})", va="center", fontsize=8, color="#333")
    ax.text(overall_avg + 0.02, bars[-1].get_y() + bars[-1].get_height() / 2,
            f"{overall_avg:.2f}  (n={len(sc)})", va="center", fontsize=8, color="#333")
    ax.set_xlabel("Avg. successful tool calls per question")
    ax.set_title(f"Avg. Successful Tool Calls per Question\nby {group_label}  (red = overall)",
                 fontweight="bold")
    ax.set_xlim(right=max(values) * 1.45)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse ToolUni logs for MedXpertQA or MedPerturb."
    )
    parser.add_argument("--logs-dir", default=None,
                        help="Directory containing detailed_*.json log files "
                             "(default: auto-selected based on --mode)")
    parser.add_argument("--mode", default="auto",
                        choices=["auto", "medxpertqa", "medperturb"],
                        help="Log format (default: auto-detect)")
    parser.add_argument("--subgroup", default="all",
                        help="Filter by subgroup: perturbation name (medperturb) or "
                             "question_type (medxpertqa). Use 'all' for no filter.")
    parser.add_argument("--top-tools", type=int, default=20,
                        help="Number of top tools to show in heatmaps (default: 20)")
    parser.add_argument("--meta-jsonl", default=str(JSONL_PATH),
                        help="Path to MedXpertQA metadata JSONL (medxpertqa mode only)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for plots and CSVs "
                             "(default: scripts/<mode>/)")
    args = parser.parse_args()

    # ── resolve paths ──────────────────────────────────────────────────────────
    logs_dir = Path(args.logs_dir) if args.logs_dir else None

    mode = args.mode
    if mode == "auto":
        if logs_dir is None:
            # Can't auto-detect without a dir; default to medxpertqa
            mode = "medxpertqa"
        else:
            mode = _detect_mode(logs_dir)
        print(f"  Auto-detected mode: {mode}")

    if logs_dir is None:
        logs_dir = DEFAULT_MEDPERTURB_LOGS if mode == "medperturb" else DEFAULT_MEDXPERTQA_LOGS

    out_dir = Path(args.out_dir) if args.out_dir else \
              Path(__file__).resolve().parent.parent / "plots" /"tooluni"/ mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── labels ─────────────────────────────────────────────────────────────────
    group_label    = "Dataset"      if mode == "medperturb" else "Body System"
    subgroup_label = "Perturbation" if mode == "medperturb" else "Question Type"

    # ── load metadata (medxpertqa only) ────────────────────────────────────────
    meta = {}
    if mode == "medxpertqa":
        jsonl_path = Path(args.meta_jsonl)
        print(f"Loading MedXpertQA metadata from {jsonl_path}…")
        meta = load_medxpertqa_meta(jsonl_path)
        if not meta:
            print("  WARNING: metadata not found — body system breakdown unavailable")

    # ── load logs ──────────────────────────────────────────────────────────────
    print(f"Loading logs from {logs_dir}  (mode={mode}, subgroup={args.subgroup})…")
    records = load_logs(logs_dir, mode=mode, subgroup_filter=args.subgroup, meta=meta)
    print(f"  Loaded {len(records)} log files")
    if not records:
        print("  No logs found — check --logs-dir and --mode.")
        return

    # ── print subgroup breakdown ───────────────────────────────────────────────
    subgroup_counts = defaultdict(int)
    for r in records:
        subgroup_counts[r["_subgroup"]] += 1
    print(f"\n── {subgroup_label} breakdown ──────────────────────────────")
    for sg, n in sorted(subgroup_counts.items()):
        print(f"  {sg:25s}  {n:4d}")

    # ── aggregate stats ────────────────────────────────────────────────────────
    print("\nAggregating tool stats…")
    stats = aggregate_tool_stats(records)
    print(f"  {len(stats)} unique tools found")

    print("\n── Top 10 by call count ─────────────────────────────")
    print(stats[["tool_name", "calls", "errors", "error_rate", "questions"]]
          .head(10).to_string(index=False))

    print("\n── Top 10 by error count ────────────────────────────")
    err_df = stats[stats["errors"] > 0].nlargest(10, "errors")
    print(err_df[["tool_name", "errors", "calls", "error_rate"]].to_string(index=False)
          if not err_df.empty else "  (none)")

    print("\n── Top 10 by successful execution ──────────────────")
    suc_df = stats[stats["successes"] > 0].nlargest(10, "successes")
    print(suc_df[["tool_name", "successes", "calls", "success_rate"]].to_string(index=False)
          if not suc_df.empty else "  (none)")

    sc = successful_tool_counts(records)
    overall_avg = sc["n_successful_tools"].mean()
    print(f"\n── Avg successful tool calls per question: {overall_avg:.2f}  (n={len(sc)}) ──")
    by_grp = (
        sc.groupby("group")["n_successful_tools"]
        .agg(avg="mean", n="count")
        .reset_index()
        .sort_values("avg", ascending=False)
    )
    print(by_grp.rename(columns={"group": group_label}).to_string(index=False))

    # ── plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    prefix = "tooluni"
    plot_top10_calls(    stats, out_dir / f"{prefix}_top10_calls.png")
    plot_top10_errors(   stats, out_dir / f"{prefix}_top10_errors.png")
    plot_top10_successes(stats, out_dir / f"{prefix}_top10_successes.png")
    plot_avg_successful_tools(sc, out_dir / f"{prefix}_avg_successful_tools.png",
                              group_label=group_label)

    # Group heatmap (body_system / dataset)
    grp_matrix = build_group_tool_matrix(records, "_group", args.top_tools)
    plot_heatmap(grp_matrix,
                 out_dir / f"{prefix}_{group_label.lower().replace(' ','_')}_heatmap.png",
                 title=f"Tool Usage by {group_label}",
                 row_label=group_label)

    # Subgroup heatmap (question_type / perturbation)
    sub_matrix = build_group_tool_matrix(records, "_subgroup", args.top_tools)
    plot_heatmap(sub_matrix,
                 out_dir / f"{prefix}_{subgroup_label.lower().replace(' ','_')}_heatmap.png",
                 title=f"Tool Usage by {subgroup_label}",
                 row_label=subgroup_label)

    # ── CSVs ───────────────────────────────────────────────────────────────────
    stats_csv = out_dir / f"{prefix}_tool_stats.csv"
    stats.to_csv(stats_csv, index=False)
    print(f"  Full stats saved: {stats_csv}")

    sc_csv = out_dir / f"{prefix}_successful_tool_counts.csv"
    sc.to_csv(sc_csv, index=False)
    print(f"  Per-question counts: {sc_csv}")


if __name__ == "__main__":
    main()
