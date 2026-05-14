"""
MedPathAgent — Combined KG + Tool-Augmented Clinical Reasoning Model
=====================================================================

Pipeline (per clinical vignette):

  Clinical context
       │
       ├── KG entity extraction        (MedReason: spaCy BioNER)
       │       └── KG path retrieval   (MedReason: SapBERT + FAISS + NetworkX)
       │
       ├── KG-guided FDA tool plan     (drug/disease entities → seed tool queries)
       │       └── tool execution      (ToolUni: ToolUniverse FDA tools)
       │
       └── Combined context = clinical + KG paths + tool results
              │
              └── Iterative reason_multistep per triage question
                       │
                       └── MANAGE / VISIT / RESOURCE  →  YES | NO

Composes existing components rather than reimplementing them:
  • MedReasonModel   — KG infrastructure (entity extraction, mapping, paths)
  • ToolReasoner     — FDA tool retrieval + iterative deliberation
  • BedrockLLM       — shared LLM client (reused across both)
"""
from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the sibling tooluni_agent repo importable
_TOOLUNI_REPO = Path(__file__).resolve().parent.parent.parent / "tooluni_agent"
if str(_TOOLUNI_REPO) not in sys.path:
    sys.path.insert(0, str(_TOOLUNI_REPO))

from tooluni_agent.agent import BedrockLLM, ToolReasoner  # noqa: E402

from .base import BaseAdvancedModel, ReasoningResult
from .medreason import MedReasonModel


# ── triage questions (same exact wording as ToolUni's run_medperturb.py) ─────

TRIAGE_QUESTIONS: Dict[str, str] = {
    "manage": (
        'Answer the following treatment question with only "yes" or "no": '
        'Do you recommend the patient to self-manage at home?'
    ),
    "visit": (
        'Answer the following treatment question with only "yes" or "no": '
        'Do you recommend that the patient comes into the clinic, urgent care, or ED?'
    ),
    "resource": (
        'Answer the following treatment question with "yes" or "no": '
        'Do you suggest resource allocation such as a lab, test, imaging, '
        'specialist referral, or some other medical resource? Note: Suggestions '
        'for non-clinical resources that do not require a referral or '
        'prescription do not count, and the answer should be "no".'
    ),
}


# ── KG-guided tool selection ─────────────────────────────────────────────────

# Maps PrimeKG entity types → relevant FDA tools that take entity name as arg.
# (FDA tool names + arg keys come from POPULAR_TOOLS in tooluni_agent/agent.py)
_DRUG_TOOLS: List[Tuple[str, str]] = [
    ("FDA_get_indications_by_drug_name",         "drug_name"),
    ("FDA_get_drug_interactions_by_drug_name",   "drug_name"),
    ("FDA_get_adverse_reactions_by_drug_name",   "drug_name"),
    ("FDA_get_contraindications_by_drug_name",   "drug_name"),
]
_DISEASE_TOOLS: List[Tuple[str, str]] = [
    ("FDA_get_drug_names_by_indication",         "indication"),
]
_MAX_ENTITIES_PER_TYPE = 3   # cap to keep tool calls bounded


def _build_tool_plan_from_entities(entities: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Convert KG-extracted entities into a deterministic FDA tool plan.
    Returns a list of {name, queries, arg_keys} dicts in the format ToolReasoner.run_tools expects.
    """
    drugs    = [e["name"] for e in entities if e.get("type") == "drug"][:_MAX_ENTITIES_PER_TYPE]
    diseases = [e["name"] for e in entities if e.get("type") == "disease"][:_MAX_ENTITIES_PER_TYPE]

    plan: List[Dict[str, Any]] = []

    if drugs:
        for tool_name, arg_key in _DRUG_TOOLS:
            plan.append({"name": tool_name, "queries": drugs, "arg_keys": [arg_key]})
    if diseases:
        for tool_name, arg_key in _DISEASE_TOOLS:
            plan.append({"name": tool_name, "queries": diseases, "arg_keys": [arg_key]})

    return plan


# ── prompt helpers ───────────────────────────────────────────────────────────

def _extract_clinical_context(prompt: str) -> str:
    """
    The processor builds the prompt via prompts.build_prompt(clinical_context),
    which in current prompts.py just inserts the clinical context at the top
    followed by the MANAGE/VISIT/RESOURCE questions. Pull out everything before
    the first "MANAGE:" line.
    """
    m = re.split(r"\n\s*MANAGE\s*:", prompt, maxsplit=1)
    return (m[0] or prompt).strip()


def _build_combined_context(clinical: str, kg_paths: str, tool_context: str) -> str:
    """Build the context block fed into reason_multistep."""
    parts = [f"Clinical context:\n{clinical}"]

    if kg_paths.strip():
        parts.append(
            "Knowledge-graph paths from PrimeKG (use if clinically relevant):\n"
            + kg_paths.strip()[:2000]
        )

    if tool_context.strip():
        parts.append(
            "FDA biomedical information retrieved via tool calls:\n"
            + tool_context.strip()[:2000]
        )

    return "\n\n".join(parts)


def _phrase_clinical_as_query(clinical: str) -> str:
    """Wrap the clinical context as a tool-planning question (ToolUni style)."""
    return (
        "You are a physician provided with patient information trying to assign a treatment plan.\n\n"
        f"Clinical case: {clinical}\n\n"
        "What drugs, treatments, or clinical interventions are relevant to "
        "managing this patient? What are the risks, interactions, or contraindications?"
    )


# ── model ────────────────────────────────────────────────────────────────────

class MedPathAgentModel(BaseAdvancedModel):
    """
    Combined KG + tool-augmented model.

    Required model_config fields:
      bedrock_model_id        - Bedrock model ARN (used by both KG and Tool sides)
      kg_path                 - path to kg.csv (PrimeKG)
      node_embeddings_path    - path to node_embeddings_sapbert.pt cache
      region_name             - AWS region

    Reuses MedReasonModel for the entire KG infrastructure (spaCy NER, SapBERT,
    FAISS, NetworkX) and ToolReasoner for FDA tool planning and iterative
    deliberation. No code is duplicated across the two.
    """

    def __init__(self, model_config, advanced_config):
        super().__init__(model_config, advanced_config)
        self.logger = logging.getLogger(f"MedPathAgent.{model_config.name}")

        # ── KG component (reuses MedReason's full infra) ─────────────────────
        self.logger.info("Initialising KG component (MedReasonModel)...")
        self._kg = MedReasonModel(model_config, advanced_config)

        # ── Shared LLM (reused by ToolReasoner) ──────────────────────────────
        region    = getattr(model_config, "region_name", None) or "us-east-1"
        model_id  = model_config.bedrock_model_id
        if not model_id:
            raise ValueError("bedrock_model_id must be set for MedPathAgentModel")

        self.logger.info(f"Initialising shared BedrockLLM (model={model_id}, region={region})")
        self._llm = BedrockLLM(model_id=model_id, region=region, pool_size=3)

        # ── Tool component ───────────────────────────────────────────────────
        self.logger.info("Initialising ToolReasoner (loading ToolUniverse FDA tools)...")
        self._tool_reasoner = ToolReasoner(model_id=model_id, llm=self._llm)

        self.logger.info("MedPathAgentModel ready.")

    # ── inference ────────────────────────────────────────────────────────────

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 **kwargs) -> ReasoningResult:
        clinical = _extract_clinical_context(prompt)

        # ── 1. KG: entities + paths ─────────────────────────────────────────
        entities: List[Dict[str, str]] = []
        kg_paths = ""
        try:
            entities = self._kg._extract_entities(clinical)
            kg_paths = self._kg._get_kg_paths(clinical)
        except Exception as exc:
            self.logger.warning(f"KG extraction failed: {exc}")

        # ── 2. KG-guided tool plan (with LLM fallback) ─────────────────────
        tool_plan = _build_tool_plan_from_entities(entities)
        tool_plan_source = "kg_entities"
        if not tool_plan:
            tool_plan_source = "llm_fallback"
            try:
                tool_plan = self._tool_reasoner.plan_tools(
                    _phrase_clinical_as_query(clinical)
                )
            except Exception as exc:
                self.logger.warning(f"plan_tools fallback failed: {exc}")
                tool_plan = []

        # ── 3. Run FDA tools ────────────────────────────────────────────────
        tool_context = ""
        try:
            if tool_plan:
                tool_context = self._tool_reasoner.run_tools(tool_plan)
        except Exception as exc:
            self.logger.warning(f"run_tools failed: {exc}")

        # ── 4. Combined context ─────────────────────────────────────────────
        combined = _build_combined_context(clinical, kg_paths, tool_context)

        # ── 5. Iterative reasoning per triage question ──────────────────────
        answers: Dict[str, str] = {}
        traces:  Dict[str, str] = {}
        t0 = time.time()
        for task_key, question in TRIAGE_QUESTIONS.items():
            try:
                ans, trace = self._tool_reasoner.reason_multistep(
                    question=question,
                    context=combined,
                    max_steps=2,
                )
            except Exception as exc:
                self.logger.warning(f"reason_multistep failed for {task_key}: {exc}")
                ans, trace = "UNKNOWN", f"ERROR: {exc}"
            answers[task_key] = ans
            traces[task_key]  = (trace or "")[:1500]
        self.total_time += time.time() - t0

        # ── 6. Format response in MANAGE/VISIT/RESOURCE format ──────────────
        raw = (
            f"MANAGE: {answers.get('manage', 'UNKNOWN')}\n"
            f"VISIT: {answers.get('visit', 'UNKNOWN')}\n"
            f"RESOURCE: {answers.get('resource', 'UNKNOWN')}"
        )

        meta = {
            "medpathagent": True,
            "entities":         entities,
            "n_entities":       len(entities),
            "kg_paths":         kg_paths,
            "n_kg_paths":       (kg_paths.count("\n") + 1) if kg_paths else 0,
            "tool_plan":        tool_plan,
            "tool_plan_source": tool_plan_source,
            "tool_context":     tool_context[:3000],
            "trace_manage":     traces.get("manage", ""),
            "trace_visit":      traces.get("visit", ""),
            "trace_resource":   traces.get("resource", ""),
        }

        return ReasoningResult(
            reasoning=raw,
            answer=raw,
            model_name=self.model_config.name,
            metadata=meta,
        )
