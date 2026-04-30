"""
Prompt construction for MedPerturb evaluation.
"""

SYSTEM_PROMPT = (
    "You are a physician provided with patient information trying to assign a treatment plan."
)

DECISION_PROMPT_TEMPLATE = """\
{clinical_context}

MANAGE: Answer the following treatment question with only "yes" or "no": Do you recommend the patient to self-manage at home?
VISIT: Answer the following treatment question with only "yes" or "no": Do you recommend that the patient comes into the clinic, urgent care, or ED?
RESOURCE: Answer the following treatment question with "yes" or "no": Do you suggest resource allocation such as a lab, test, imaging, specialist referral, or some other medical resource? Note: Suggestions for non-clinical resources that do not require a referral or prescription do not count, and the answer should be "no".

Respond in exactly this format:
MANAGE: <yes|no>
VISIT: <yes|no>
RESOURCE: <yes|no>
"""


def build_prompt(clinical_context: str) -> str:
    return DECISION_PROMPT_TEMPLATE.format(clinical_context=clinical_context.strip())
