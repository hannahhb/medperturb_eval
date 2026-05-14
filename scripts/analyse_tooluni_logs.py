"""
Analyse ToolUni tool usage from per-question detailed JSON logs.

Outputs:
  1. Top-10 tools by call count (+ error rate per tool)
  2. Top-10 tools by error count
  3. Heatmap: tool usage clustered by question body system

Usage:
    python scripts/analyse_tooluni_logs.py
    python scripts/analyse_tooluni_logs.py --logs-dir logs/medxpertqa/tooluni_llama70b3_3
    python scripts/analyse_tooluni_logs.py --question-type baseline   # baseline | gender_swap | both
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT.parent / "data"
LOGS_DIR  = ROOT / "logs" / "medxpertqa" / "tooluni_llama70b3_3"
JSONL_PATH = DATA_DIR / "medxpertqa_text.jsonl"
PLOTS_DIR = Path(__file__).resolve().parent / "medxpertqa"
PLOTS_DIR.mkdir(exist_ok=True)


# ── load ──────────────────────────────────────────────────────────────────────

def load_logs(logs_dir: Path, question_type: str = "both") -> list[dict]:
    """Load all detailed JSON log files, optionally filtered by question_type."""
    records = []
    for f in sorted(logs_dir.glob("detailed_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        qt = d.get("question_type", "")
        if question_type != "both" and qt != question_type:
            continue
        records.append(d)
    return records


def load_meta(jsonl_path: Path) -> dict[str, dict]:
    """Return {id: {body_system, medical_task, q_type}} from the MedXpertQA JSONL."""
    meta = {}
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        meta[row["id"]] = {
            "body_system":  row.get("body_system", "Unknown"),
            "medical_task": row.get("medical_task", "Unknown"),
            "q_type":       row.get("question_type", "Unknown"),
        }
    return meta


# ── aggregate tool stats ──────────────────────────────────────────────────────

def aggregate_tool_stats(records: list[dict]) -> pd.DataFrame:
    """
    Returns one row per tool with columns:
      tool_name, calls, errors, error_rate, questions
    """
    tool_calls  = defaultdict(int)
    tool_errors = defaultdict(int)
    tool_qs     = defaultdict(set)   # unique question ids

    for rec in records:
        qid = rec.get("id", "")
        for step in rec.get("tu_rounds", []):
            for t in step.get("tools", []):
                name = t.get("tool_name", "unknown")
                tool_calls[name]  += 1
                tool_qs[name].add(qid)
                if t.get("error"):
                    tool_errors[name] += 1

    rows = []
    for name in tool_calls:
        c = tool_calls[name]
        e = tool_errors[name]
        rows.append({
            "tool_name":  name,
            "calls":      c,
            "errors":     e,
            "error_rate": e / c * 100 if c else 0.0,
            "questions":  len(tool_qs[name]),
        })
    return pd.DataFrame(rows).sort_values("calls", ascending=False).reset_index(drop=True)


# ── body-system × tool matrix ─────────────────────────────────────────────────

def build_body_tool_matrix(
    records: list[dict],
    meta: dict[str, dict],
    top_n_tools: int = 20,
) -> pd.DataFrame:
    """
    Returns a DataFrame [body_system × tool_name] with call counts.
    Only the top_n_tools most-called tools are included.
    """
    counts = defaultdict(lambda: defaultdict(int))

    for rec in records:
        qid  = rec.get("id", "")
        sys  = meta.get(qid, {}).get("body_system", "Unknown")
        for step in rec.get("tu_rounds", []):
            for t in step.get("tools", []):
                name = t.get("tool_name", "unknown")
                counts[sys][name] += 1

    df = pd.DataFrame(counts).fillna(0).T   # rows = body_system, cols = tools
    df.index.name = "body_system"

    # Keep only top-N tools by total calls
    top_tools = df.sum(axis=0).nlargest(top_n_tools).index
    df = df[top_tools]

    return df


# ── plotting ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def plot_top10_calls(stats: pd.DataFrame, out_path: Path):
    top = stats.nlargest(10, "calls").iloc[::-1]   # ascending for barh

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top["tool_name"], top["calls"], color="#4C72B0", alpha=0.85)

    # Annotate with error rate
    for bar, (_, row) in zip(bars, top.iterrows()):
        ax.text(
            bar.get_width() + max(top["calls"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{row['calls']} calls  |  {row['error_rate']:.0f}% err",
            va="center", fontsize=8, color="#333",
        )

    ax.set_xlabel("Total calls")
    ax.set_title("Top 10 Tools by Call Count\n(error rate annotated)", fontweight="bold")
    ax.set_xlim(right=top["calls"].max() * 1.35)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_top10_errors(stats: pd.DataFrame, out_path: Path):
    top = stats[stats["errors"] > 0].nlargest(10, "errors").iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top["tool_name"], top["errors"], color="#C44E52", alpha=0.85)

    for bar, (_, row) in zip(bars, top.iterrows()):
        ax.text(
            bar.get_width() + max(top["errors"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{row['errors']} / {row['calls']}  ({row['error_rate']:.0f}%)",
            va="center", fontsize=8, color="#333",
        )

    ax.set_xlabel("Error count")
    ax.set_title("Top 10 Tools by Error Count\n(errors / total calls annotated)", fontweight="bold")
    ax.set_xlim(right=top["errors"].max() * 1.35)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_body_tool_heatmap(matrix: pd.DataFrame, out_path: Path):
    """Heatmap: rows = body systems, columns = tools, values = call counts."""
    # Shorten tool names for readability
    short_cols = [c.replace("_", " ") for c in matrix.columns]

    fig, ax = plt.subplots(figsize=(max(12, len(matrix.columns) * 0.8),
                                    max(5,  len(matrix.index)  * 0.5)))

    # Normalise per row so colour shows relative focus, not raw count
    norm_matrix = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0) * 100

    im = ax.imshow(norm_matrix.values, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="% of questions' tool calls (row-normalised)")

    ax.set_xticks(range(len(short_cols)))
    ax.set_xticklabels(short_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)

    # Annotate cells with raw count (only non-zero)
    for i, sys in enumerate(matrix.index):
        for j, tool in enumerate(matrix.columns):
            v = int(matrix.loc[sys, tool])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=7, color="black" if norm_matrix.values[i, j] < 60 else "white")

    ax.set_title("Tool Usage by Body System\n(colour = % of row calls; numbers = raw count)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default=str(LOGS_DIR),
                        help="Directory containing detailed_*.json log files")
    parser.add_argument("--question-type", default="both",
                        choices=["baseline", "gender_swap", "both"],
                        help="Which question types to include (default: both)")
    parser.add_argument("--top-tools", type=int, default=20,
                        help="Number of top tools to show in heatmap (default: 20)")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)

    print(f"Loading logs from {logs_dir} (question_type={args.question_type})…")
    records = load_logs(logs_dir, question_type=args.question_type)
    print(f"  Loaded {len(records)} log files")

    print("Loading MedXpertQA metadata…")
    meta = load_meta(JSONL_PATH) if JSONL_PATH.exists() else {}
    if not meta:
        print("  WARNING: metadata JSONL not found — body system breakdown unavailable")

    print("Aggregating tool stats…")
    stats = aggregate_tool_stats(records)
    print(f"  {len(stats)} unique tools found")

    print("\n── Top 10 by call count ─────────────────────────────")
    print(stats[["tool_name", "calls", "errors", "error_rate", "questions"]]
          .head(10).to_string(index=False))

    print("\n── Top 10 by error count ────────────────────────────")
    print(stats[stats["errors"] > 0]
          .nlargest(10, "errors")[["tool_name", "errors", "calls", "error_rate"]]
          .to_string(index=False))

    print("\nGenerating plots…")
    plot_top10_calls(stats,  PLOTS_DIR / "tooluni_top10_calls.png")
    plot_top10_errors(stats, PLOTS_DIR / "tooluni_top10_errors.png")

    if meta:
        matrix = build_body_tool_matrix(records, meta, top_n_tools=args.top_tools)
        plot_body_tool_heatmap(matrix, PLOTS_DIR / "tooluni_body_system_heatmap.png")

    # Save full stats table
    csv_path = PLOTS_DIR / "tooluni_tool_stats.csv"
    stats.to_csv(csv_path, index=False)
    print(f"  Full stats saved: {csv_path}")


if __name__ == "__main__":
    main()
