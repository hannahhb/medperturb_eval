"""
Gender perturbation for MedXpertQA Text test set.

Call 1 — classify (single digit, plain text):
   2  no gendered language
  -1  gender-specific condition (swap nonsensical)
   0  gender incidental — answer letter unchanged after swap
   1  gender clinically relevant — answer letter would change

Call 2 — swap question text (only if label 0 or 1)
Call 3 — swap options      (only if label 0 or 1)

Usage:
  python perturb/gender_perturb_medxpertqa.py [--limit N] [--no-resume]
"""
from __future__ import annotations
import argparse, csv, json, logging, os, re, sys, time
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from config import ModelConfig, ModelType, AdvancedConfig, MODELS
from models.bedrock import BedrockChatModel

DATA_DIR    = Path(__file__).resolve().parent.parent.parent / "data"
INPUT_JSONL = DATA_DIR / "medxpertqa_text.jsonl"
OUTPUT_CSV  = DATA_DIR / "medxpertqa_gender_perturb.csv"
MODEL_NAME  = "llama70b3_3"
RATE_DELAY  = 0.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── prompts ───────────────────────────────────────────────────────────────────

CLASSIFY_PROMPT = """\
Medical MCQ below. Reply with ONE digit + short reason.

QUESTION: {question}
OPTIONS:
{options_block}
CORRECT ANSWER: {label}

Focus only on the PRIMARY PATIENT's gender. Ignore procedural/contextual
terms that imply sex indirectly (e.g. "cesarean section" refers to the
mother, not the infant patient; "newborn/infant/child/patient" are ungendered).
Explicit gender words: male, female, man, woman, boy, girl, he, she, his,
her, him, Mr, Mrs, Ms, husband, wife, mother, father, son, daughter.

 2  None of the explicit words above describe the patient → no gender present.
-1  Explicit gender present AND the vignette is sex-specific → swap produces a
    medically nonsensical narrative. This includes:
    • Sex-specific diagnoses: ovarian/prostate/cervical/testicular cancer,
      pregnancy, menopause, PCOS, hysterectomy, erectile dysfunction, etc.
    • Sex-specific symptoms central to the case: menses/menstrual irregularity/
      amenorrhea, galactorrhea in a reproductive context, "hoping to conceive"
      framed with female reproductive symptoms.
    Judge the PRESENTATION, not just the treatment — if the swap makes the
    clinical narrative internally inconsistent, use -1.
 0  Explicit gender present but incidental → correct answer letter unchanged by swap.
 1  Explicit gender present AND clinically relevant → different answer letter after swap.

Reply on two lines only:
LABEL: <digit>
REASON: <one phrase, max 32 chars>"""

SWAP_Q_PROMPT = """\
Swap the patient gender in this question. Rules: male↔female, man↔woman, \
he↔she, his↔her, Mr↔Ms, husband↔wife, father↔mother, son↔daughter, etc. \
For female-only modifiers with no male equivalent, drop the modifier entirely: \
"postmenopausal woman" → "man" (NOT "postmenopausal man"). \
Keep all else identical.

{question}

Return ONLY the rewritten question."""

SWAP_OPTS_PROMPT = """\
Swap the patient gender in each option (male↔female, man↔woman, etc.). \
Drop female-only modifiers with no male equivalent: \
"postmenopausal woman" → "man" (NOT "postmenopausal man"). \
Keep all else identical.

{options_lines}

Return one option per line as: LETTER: text"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _opts_block(opts: Dict[str, str]) -> str:
    return "\n".join(f"({k}) {v}" for k, v in opts.items())

def _opts_lines(opts: Dict[str, str]) -> str:
    return "\n".join(f"{k}: {v}" for k, v in opts.items())

def _truncate(s: str, n: int = 32) -> str:
    return s[:n] if len(s) <= n else s[:n-1] + "…"

def make_model(max_tokens: int) -> BedrockChatModel:
    mc = ModelConfig(
        name=MODEL_NAME, type=ModelType.BEDROCK,
        bedrock_model_id=MODELS[MODEL_NAME],
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        max_tokens=max_tokens, temperature=0.0,
    )
    return BedrockChatModel(mc, AdvancedConfig())


# ── LLM calls ─────────────────────────────────────────────────────────────────

def classify(model, question, options, label) -> Tuple[int, str]:
    prompt = CLASSIFY_PROMPT.format(
        question=question, options_block=_opts_block(options), label=label)
    raw = model.generate(prompt=prompt).reasoning.strip()
    lbl_m = re.search(r"LABEL\s*:\s*(-?\d)", raw)
    rsn_m = re.search(r"REASON\s*:\s*(.+)", raw)
    lbl = int(lbl_m.group(1)) if lbl_m else 2
    rsn = _truncate(rsn_m.group(1).strip() if rsn_m else raw[:32])
    return lbl, rsn


def swap_question(model, question: str) -> str:
    prompt = SWAP_Q_PROMPT.format(question=question)
    return model.generate(prompt=prompt).reasoning.strip()


def swap_options(model, options: Dict[str, str]) -> Dict[str, str]:
    prompt = SWAP_OPTS_PROMPT.format(options_lines=_opts_lines(options))
    raw = model.generate(prompt=prompt).reasoning.strip()
    result = {}
    for line in raw.splitlines():
        m = re.match(r"^([A-J])\s*:\s*(.+)$", line.strip())
        if m:
            result[m.group(1)] = m.group(2).strip()
    # fall back to original for any missing keys
    return {k: result.get(k, v) for k, v in options.items()}


# ── CSV helpers ───────────────────────────────────────────────────────────────

FIELDS = ["id", "original_question", "original_label",
          "gender_swap_perturbed", "gender_swap_options", "change", "reasoning"]

def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {r["id"] for r in csv.DictReader(f)}

def save_row(path: Path, row: dict):
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",     default=str(INPUT_JSONL))
    ap.add_argument("--output",    default=str(OUTPUT_CSV))
    ap.add_argument("--limit",     type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    output_path = Path(args.output)
    done = set() if args.no_resume else load_done(output_path)
    remaining = [r for r in rows if r["id"] not in done]
    log.info("Loaded %d  |  to process: %d", len(rows), len(remaining))

    c_model = make_model(64)     # classify — tiny response
    s_model = make_model(1024)   # swap — full question text

    counts = {k: 0 for k in ("neutral", "specific", "no_change", "changes", "err")}

    for row in tqdm(remaining, desc="Perturbing", unit="q"):
        qid, question, options, label = row["id"], row["question"], row["options"], row["label"]

        try:
            lbl, reasoning = classify(c_model, question, options, label)
            time.sleep(RATE_DELAY)

            if lbl in (2, -1):
                save_row(output_path, dict(
                    id=qid, original_question=question, original_label=label,
                    gender_swap_perturbed="", gender_swap_options="",
                    change=lbl, reasoning=reasoning,
                ))
                tag = "NEUTRAL" if lbl == 2 else "SPECIFIC"
                counts["neutral" if lbl == 2 else "specific"] += 1
            else:
                swapped_q = swap_question(s_model, question);  time.sleep(RATE_DELAY)
                swapped_o = swap_options(s_model, options);     time.sleep(RATE_DELAY)
                save_row(output_path, dict(
                    id=qid, original_question=question, original_label=label,
                    gender_swap_perturbed=swapped_q,
                    gender_swap_options=json.dumps(swapped_o),
                    change=lbl, reasoning=reasoning,
                ))
                tag = "CHANGES" if lbl == 1 else "no-change"
                counts["changes" if lbl == 1 else "no_change"] += 1

            tqdm.write(f"  {qid} → {tag} | {reasoning}")

        except Exception as exc:
            tqdm.write(f"  ERROR {qid}: {exc}")
            save_row(output_path, dict(
                id=qid, original_question=question, original_label=label,
                gender_swap_perturbed="", gender_swap_options="",
                change=-1, reasoning=_truncate(f"ERR:{exc}"),
            ))
            counts["err"] += 1

    log.info("Done. %s", counts)


if __name__ == "__main__":
    main()
