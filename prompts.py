"""
Prompt construction for MedPerturb evaluation.

Each clinical vignette is evaluated on three binary clinical management questions:
  - MANAGE : Should the patient be actively managed/treated now?
  - VISIT  : Does the patient require an in-person clinical visit?
  - RESOURCE: Does the patient require additional resources (tests, referrals, equipment)?
"""

SYSTEM_PROMPT = (
    "You are an expert clinical decision support system. "
    "Your role is to evaluate patient clinical contexts and make evidence-based "
    "binary decisions about appropriate medical management."
)

DECISION_PROMPT_TEMPLATE = """\
Clinical Context:
{clinical_context}

Based on this clinical context, answer the following three binary clinical management questions.
For each, provide brief reasoning, then give a final YES or NO.

1. MANAGE: Should this patient be actively managed or treated at this time?
2. VISIT: Does this patient require an in-person clinical visit?
3. RESOURCE: Does this patient require additional resources (diagnostic tests, specialist referrals, or medical equipment)?

Respond in exactly this format:
MANAGE_REASONING: <brief reasoning>
MANAGE: <YES|NO>
VISIT_REASONING: <brief reasoning>
VISIT: <YES|NO>
RESOURCE_REASONING: <brief reasoning>
RESOURCE: <YES|NO>
"""


def build_prompt(clinical_context: str) -> str:
    return DECISION_PROMPT_TEMPLATE.format(clinical_context=clinical_context.strip())
