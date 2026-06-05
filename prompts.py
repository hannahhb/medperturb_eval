"""
Prompt construction for MedPerturb evaluation.

All prompt strings live here so model code stays clean.
Each model calls build_triage_prompt(clinical_context, augmentation="") where
augmentation is the model-specific context block (tool results, KG paths,
calculator output, etc.).  An empty augmentation produces the plain baseline prompt.
"""

SYSTEM_PROMPT = (
    "You are a physician provided with patient information trying to assign a treatment plan."
)

_TRIAGE_QUESTIONS = """\
MANAGE: Answer the following treatment question with only "yes" or "no": Do you recommend the patient to self-manage at home?
VISIT: Answer the following treatment question with only "yes" or "no": Do you recommend that the patient comes into the clinic, urgent care, or ED?
RESOURCE: Answer the following treatment question with "yes" or "no": Do you suggest resource allocation such as a lab, test, imaging, specialist referral, or some other medical resource? Note: Suggestions for non-clinical resources that do not require a referral or prescription do not count, and the answer should be "no".

Respond in exactly this format:
MANAGE: <yes|no>
VISIT: <yes|no>
RESOURCE: <yes|no>"""


def build_triage_prompt(clinical_context: str, augmentation: str = "") -> str:
    """
    Build the MedPerturb MANAGE/VISIT/RESOURCE prompt.

    Args:
        clinical_context: Raw clinical vignette text.
        augmentation:     Optional context block prepended before the vignette
                          (e.g. KG paths, FDA tool results, calculator output).
                          Empty string → baseline prompt with no augmentation.
    """
    parts = []
    if augmentation.strip():
        parts.append(augmentation.strip())
    parts.append(clinical_context.strip())
    parts.append(_TRIAGE_QUESTIONS)
    return "\n\n".join(parts)


# ── back-compat alias (used by bedrock/txagent models via processor) ──────────
def build_prompt(clinical_context: str) -> str:
    """Baseline prompt — no augmentation.  Kept for back-compat."""
    return build_triage_prompt(clinical_context)


# ── Per-model augmentation headers ───────────────────────────────────────────

# ToolUni: prepended above the clinical context when FDA tool results are found
TOOLUNI_AUGMENTATION_HEADER = "FDA biomedical information:"

# MedReason / MedPathAgent: prepended above the clinical context when KG paths found
KG_AUGMENTATION_HEADER = "Relevant biomedical knowledge from PrimeKG:"

# AgentMD: appended below the clinical context + triage questions
CALCULATOR_AUGMENTATION_HEADER = "Risk Calculator Result:"

# ToolUni: query sent to the tool retriever to seed FDA evidence gathering
TOOLUNI_RETRIEVAL_QUERY = (
    "You are a physician reviewing a patient case. "
    "What drugs, treatments, or clinical interventions are relevant?\n\n"
    "Clinical case:\n{clinical_context}"
)


# ── CureBench MCQ prompts ─────────────────────────────────────────────────────

CUREBENCH_SYSTEM_PROMPT = "You are a helpful medical assistant."

# Turn 1: present question + options.  {augmentation} block is empty for baseline.
_CUREBENCH_TURN1_BASE = (
    "You are a medical expert. Answer the following question accurately and "
    "concisely based on established medical literature.\n\n"
    "{question}\n\n"
    "{options}"
)

_CUREBENCH_TURN1_AUGMENTED = (
    "You are a medical expert. Answer the following question accurately and "
    "concisely based on established medical literature.\n\n"
    "{augmentation}\n\n"
    "{question}\n\n"
    "{options}"
)

# Turn 2: extract single letter from Turn 1 response.
CUREBENCH_EXTRACT_PROMPT = (
    "Based on the above, reply with only the single letter of the correct answer "
    "(A, B, C, or D) and nothing else."
)

# ToolUni retrieval query for CureBench (seeded from the question text)
CUREBENCH_TOOLUNI_RETRIEVAL_QUERY = (
    "You are a medical expert reviewing the following question. "
    "What drugs, treatments, diseases, or clinical concepts are relevant?\n\n"
    "Question:\n{question}"
)


def build_curebench_prompt(question: str, options: dict, augmentation: str = "") -> str:
    """
    Build the CureBench Turn-1 prompt.

    Args:
        question:     Question text.
        options:      Dict of {letter: option_text}.
        augmentation: Optional context block (KG paths, FDA tools, etc.).
                      Empty string → baseline prompt.
    """
    options_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    if augmentation.strip():
        return _CUREBENCH_TURN1_AUGMENTED.format(
            augmentation=augmentation.strip(),
            question=question.strip(),
            options=options_text,
        )
    return _CUREBENCH_TURN1_BASE.format(
        question=question.strip(),
        options=options_text,
    )
