"""
Re-extract answer letters from existing MedXpertQA results.csv using
the fixed _extract_letter logic, without re-running any inference.

Usage:
    python scripts/reextract_letters.py \
        --results output/medxpertqa/tooluni_qwen235b/results.csv \
        --questions /Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/medxpertqa_text.jsonl

    # dry run — print changes without saving
    python scripts/reextract_letters.py --results ... --dry-run
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# ── same _extract_letter as run_medxpertqa.py ─────────────────────────────────

def _extract_letter(text: str, valid: set) -> str:
    for pattern in [
        r"the answer is\s*[:\(]?\s*([A-J])(?:\)|\.|\s|$)",
        r"correct answer[:\s]+\(?([A-J])\)?",
        r"Therefore[^.]*\b([A-J])\b",
        r"answer\s*:\s*\(?([A-J])\)?",
        r"\*{1,2}([A-J])\*{1,2}",
    ]:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if m.group(1).upper() in valid:
                return m.group(1).upper()

    matches = [m for m in re.finditer(r'\b([A-J])\b', text) if m.group(1) in valid]
    if matches:
        return matches[-1].group(1)
    return "?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",  default="/Users/hannah_mac/Documents/rmit/rmit_hons_y4/medperturb_eval/output/medxpertqa/tooluni_qwen235b_mt-Basic_Science/results.csv", help="Path to results.csv")
    parser.add_argument("--questions", default="/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data/medxpertqa_text.jsonl",
                        help="Path to medxpertqa_text.jsonl (for options lookup)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print changes without saving")
    args = parser.parse_args()

    # Load questions for options
    questions = {}
    with open(args.questions, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            questions[q["id"]] = q

    df = pd.read_csv(args.results)
    print(f"Loaded {len(df)} rows from {args.results}")

    changed = 0
    for i, row in df.iterrows():
        qid      = str(row["id"])
        reasoning = str(row.get("reasoning", "") or "")
        old_letter = str(row.get("model_answer_letter", "?")).strip()

        q = questions.get(qid, {})
        options = q.get("options", {}) or {}
        valid = set(options.keys()) if options else set("ABCDEFGHIJ")

        new_letter = _extract_letter(reasoning, valid)

        if new_letter != old_letter:
            changed += 1
            correct = str(row.get("correct_label","")).strip()
            old_correct = 1 if old_letter == correct else 0
            new_correct = 1 if new_letter == correct else 0
            print(f"  {qid}: {old_letter}→{new_letter}  gold={correct}  "
                  f"({'✓' if new_correct else '✗'} was {'✓' if old_correct else '✗'})")
            df.at[i, "model_answer_letter"] = new_letter
            df.at[i, "correct"] = new_correct if "correct" in df.columns else None

    print(f"\nChanged: {changed}/{len(df)}")
    if "correct" in df.columns:
        print(f"Accuracy after:  {df['correct'].mean()*100:.2f}%  ({int(df['correct'].sum())}/{len(df)})")

    if not args.dry_run:
        df.to_csv(args.results, index=False)
        print(f"Saved → {args.results}")
    else:
        print("Dry run — not saved.")


if __name__ == "__main__":
    main()
