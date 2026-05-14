"""
ToolUni adapter for MedPerturb and MedXpertQA.

For MedPerturb (MANAGE/VISIT/RESOURCE):
  1. Extract clinical context from the prompt.
  2. Run ToolReasoner.run_multistep() to gather FDA tool evidence.
  3. Run ToolReasoner.reason_multistep() once per triage question.
  4. Return "MANAGE: YES\nVISIT: NO\nRESOURCE: YES".

For MedXpertQA MCQ (two-turn CoT via run_medxpertqa.py):
  Turn 1  ("Let's think step by step"):
    - Run FDA tool retrieval on the question text.
    - Inject tool context above the question, then call LLM for reasoning.
    - Return full metadata (tu_rounds, tool_context) for the log.
  Turn 2  (single-letter extraction):
    - No tools needed — direct LLM call only.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

from .base import BaseAdvancedModel, ReasoningResult

# ── import ToolUni from sibling repo ─────────────────────────────────────────

_TOOLUNI_REPO = Path(__file__).resolve().parent.parent.parent / "tooluni_agent"
if str(_TOOLUNI_REPO) not in sys.path:
    sys.path.insert(0, str(_TOOLUNI_REPO))

try:
    from tooluni_agent.agent import BedrockLLM, ToolReasoner
    _TOOLUNI_AVAILABLE = True
except ImportError as _e:
    _TOOLUNI_AVAILABLE = False
    _TOOLUNI_IMPORT_ERROR = str(_e)


# ── triage question templates ─────────────────────────────────────────────────

_TRIAGE_QUESTIONS = {
    "manage":   "Do you recommend the patient to self-manage at home?",
    "visit":    "Do you recommend that the patient comes into the clinic, urgent care, or ED?",
    "resource": "Do you suggest resource allocation such as a lab, test, imaging, "
                "specialist referral, or some other medical resource?",
}

# Regex to detect MedPerturb-style prompts
_MANAGE_RE = re.compile(r"\bMANAGE\s*:", re.IGNORECASE)
# Regex to detect MCQ Turn-1 prompts ("Let's think step by step")
_TURN1_RE = re.compile(r"Let'?s think step by step", re.IGNORECASE)
# Regex to detect the Turn-2 extraction prompt (overrides Turn-1 detection)
_TURN2_RE = re.compile(
    r"reply with the single letter of the correct answer",
    re.IGNORECASE,
)


class ToolUniModel(BaseAdvancedModel):
    """
    ToolUni (FDA tool retrieval + iterative reasoning) wrapped as a MedPerturb model.

    Works for both:
    - MedPerturb: prompt contains MANAGE/VISIT/RESOURCE block → full pipeline
    - MedXpertQA: plain MCQ prompt → direct LLM call (two-turn CoT handled by caller)
    """

    def __init__(self, model_config, advanced_config):
        if not _TOOLUNI_AVAILABLE:
            raise ImportError(
                f"tooluni_agent package not found at {_TOOLUNI_REPO}. "
                f"Original error: {_TOOLUNI_IMPORT_ERROR}"
            )
        super().__init__(model_config, advanced_config)
        self.logger = logging.getLogger(f"ToolUni.{model_config.name}")

        model_id = model_config.bedrock_model_id or os.environ.get(
            "TOOLUNI_MODEL_ID", "us.meta.llama3-3-70b-instruct-v1:0"
        )
        region = model_config.region_name or os.environ.get("AWS_REGION", "us-east-1")

        self._llm = BedrockLLM(model_id=model_id, region=region, pool_size=3)

        # ToolRAG dense retrieval (optional — set use_toolrag=True in model_config)
        use_toolrag     = getattr(model_config, "use_toolrag", False)
        toolrag_model   = getattr(model_config, "toolrag_model",
                                  "mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B")
        toolrag_device  = getattr(model_config, "toolrag_device", "cpu")
        toolrag_cache   = getattr(model_config, "toolrag_cache_dir", None)

        self._tool_reasoner = ToolReasoner(
            model_id=model_id,
            llm=self._llm,
            use_toolrag=use_toolrag,
            toolrag_model=toolrag_model,
            toolrag_device=toolrag_device,
            toolrag_cache_dir=toolrag_cache,
        )

        self.logger.info(
            "ToolUniModel ready  model_id=%s  region=%s  toolrag=%s",
            model_id, region, use_toolrag,
        )

    # ── public interface ──────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> ReasoningResult:
        """
        Dispatch:
        - MedPerturb prompt (contains "MANAGE:")            → _generate_triage()
        - MCQ Turn 1 ("Let's think step by step")           → _generate_mcq_turn1()
        - MCQ Turn 2 (letter extraction, long combined txt) → _generate_direct()
        """
        if _MANAGE_RE.search(prompt):
            return self._generate_triage(prompt, system_prompt)
        # Turn-2 check BEFORE Turn-1: the Turn-2 prompt embeds the Turn-1 text
        # (which contains "Let's think step by step"), so we must detect Turn 2
        # first to avoid re-running expensive tool retrieval for letter extraction.
        if _TURN2_RE.search(prompt):
            return self._generate_direct(prompt, system_prompt)
        if _TURN1_RE.search(prompt):
            return self._generate_mcq_turn1(prompt, system_prompt)
        return self._generate_direct(prompt, system_prompt)

    # ── MedPerturb pipeline ───────────────────────────────────────────────────

    def _generate_triage(
        self,
        prompt: str,
        system_prompt: Optional[str],
    ) -> ReasoningResult:
        """Full ToolUni pipeline: tool retrieval → iterative YES/NO reasoning."""

        clinical = _extract_clinical_context(prompt)
        self.logger.debug("Clinical context (%d chars)", len(clinical))

        # Step 1 — gather FDA evidence (inline loop so we can track tools per step)
        tool_query = (
            "You are a physician reviewing a patient case. "
            "What drugs, treatments, or clinical interventions are relevant?\n\n"
            f"Clinical case:\n{clinical}"
        )
        tool_context, retrieval_rounds = self._run_multistep_tracked(
            tool_query, max_steps=3
        )
        self.logger.debug(
            "Tool context (%d chars, %d retrieval steps)", len(tool_context), len(retrieval_rounds)
        )

        # Step 2 — build combined context
        combined = f"Clinical context:\n{clinical}"
        if tool_context.strip():
            combined += f"\n\nFDA biomedical information:\n{tool_context[:2500]}"

        # Step 3 — iterative YES/NO per triage question
        answers: dict[str, str] = {}
        traces: dict[str, str] = {}
        for key, question in _TRIAGE_QUESTIONS.items():
            ans, trace = self._tool_reasoner.reason_multistep(
                question=question,
                context=combined,
                max_steps=2,
            )
            answers[key] = ans or "NO"
            traces[key]  = trace[:1500]
            self.logger.debug("%s → %s", key.upper(), answers[key])

        raw = (
            f"MANAGE: {answers['manage']}\n"
            f"VISIT: {answers['visit']}\n"
            f"RESOURCE: {answers['resource']}"
        )

        return ReasoningResult(
            reasoning=raw,
            answer=raw,
            model_name=self.model_config.name,
            metadata={
                "tooluni": True,
                "tool_context":    tool_context[:3000],
                "tu_rounds":       retrieval_rounds,   # [{step, plan, tool_context}]
                "trace_manage":    traces["manage"],
                "trace_visit":     traces["visit"],
                "trace_resource":  traces["resource"],
            },
        )

    # ── Tracked multistep retrieval ───────────────────────────────────────────

    def _run_multistep_tracked(
        self, question: str, max_steps: int = 3
    ) -> "tuple[str, list[dict]]":
        """
        Replicates ToolReasoner.run_multistep() but records per-step tool usage.

        Returns:
            tool_context  (str)  — accumulated FDA tool results
            rounds        (list) — one dict per step:
                {
                  "step":         int,
                  "plan":         [{"tool_name": str, "queries": [...]}],
                  "tool_context": str   (output from this step only),
                  "status":       str   ("COMPLETE" | "CONTINUE" | "no_plan"),
                }
        """
        from tooluni_agent.agent import STEP_PROMPT  # noqa: PLC0415

        accumulated: list[str] = []
        rounds: list[dict] = []
        last_reasoning = ""

        for step in range(1, max_steps + 1):
            # Build plan — on step > 1 focus on NEEDS from prior reasoning
            if step == 1:
                plan = self._tool_reasoner.plan_tools(question)
            else:
                needs = self._tool_reasoner._extract_field(last_reasoning, "NEEDS")
                if not needs:
                    break
                plan = self._tool_reasoner.plan_tools(f"{question}\nFocus on: {needs}")

            if not plan:
                rounds.append({"step": step, "plan": [], "tools": [],
                               "tool_context": "", "status": "no_plan"})
                break

            # Detailed: list of {tool_name, arguments, output, cached, error}
            tool_details, step_context = self._tool_reasoner.run_tools_detailed(
                plan, summarise=True, question=question
            )
            norm = " ".join(step_context.split())
            if norm and step_context not in accumulated:
                accumulated.append(step_context)

            combined = "\n\n".join(accumulated) or "No relevant context retrieved."

            last_reasoning = self._tool_reasoner._llm_chat(
                STEP_PROMPT.format(
                    question=question,
                    step=step,
                    max_steps=max_steps,
                    context=combined,
                ),
                max_tokens=128,
            )

            status = self._tool_reasoner._extract_field(last_reasoning, "STATUS").upper()

            # Compact plan summary (legacy field — preserved for back-compat)
            plan_summary = [
                {"tool_name": d["tool_name"], "arguments": d["arguments"]}
                for d in tool_details
            ]
            # Per-tool output: each entry has the raw output snippet too,
            # truncated so the JSON log doesn't blow up.
            tool_log = [
                {
                    "tool_name": d["tool_name"],
                    "arguments": d["arguments"],
                    "output":    (d["output"] or "")[:1500],
                    "cached":    d["cached"],
                    "error":     d["error"],
                }
                for d in tool_details
            ]
            rounds.append({
                "step":         step,
                "plan":         plan_summary,
                "tools":        tool_log,                # NEW: per-tool output
                "tool_context": step_context[:2000],     # combined (legacy)
                "status":       status or "CONTINUE",
            })

            if status == "COMPLETE":
                break

        return "\n\n".join(accumulated), rounds

    # ── MedXpertQA Turn 1 — tool-augmented reasoning ─────────────────────────

    def _generate_mcq_turn1(
        self,
        prompt: str,
        system_prompt: Optional[str],
    ) -> ReasoningResult:
        """
        MCQ Turn 1: retrieve FDA evidence for the question, inject it above
        the original prompt, then ask the LLM to reason step by step.
        Full tool metadata is returned so the log is populated.
        """
        # Extract just the question text for the tool query
        # Prompt format: "Q: {question}\n\nOptions:\n...\n\nA: Let's think step by step."
        q_match = re.search(r"^Q:\s*(.+?)(?:\n\nOptions:|\Z)", prompt, re.DOTALL)
        question_text = q_match.group(1).strip() if q_match else prompt

        # Retrieve FDA evidence — single pass is plenty for an MCQ; multi-step
        # was reserved for clinical triage. Halves the LLM round-trips.
        tool_context, retrieval_rounds = self._run_multistep_tracked(
            question_text, max_steps=1
        )
        self.logger.debug(
            "MCQ Turn1 tool context (%d chars, %d steps)", len(tool_context), len(retrieval_rounds)
        )

        # Augment prompt with tool evidence (inserted before the question)
        if tool_context.strip():
            augmented_prompt = (
                f"FDA biomedical information:\n{tool_context[:2000]}\n\n"
                f"{prompt}"
            )
        else:
            augmented_prompt = prompt

        text = self._llm.chat(
            augmented_prompt,
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens,
            system_prompt=system_prompt,
        )
        return ReasoningResult(
            reasoning=text or "(no response)",
            answer=text or "(no response)",
            model_name=self.model_config.name,
            metadata={
                "tooluni":      True,
                "tool_context": tool_context[:3000],
                "tu_rounds":    retrieval_rounds,
                "final_prompt": augmented_prompt,
            },
        )

    # ── MedXpertQA Turn 2 / generic call ─────────────────────────────────────

    def _generate_direct(
        self,
        prompt: str,
        system_prompt: Optional[str],
    ) -> ReasoningResult:
        """Direct LLM call — used for MCQ Turn 2 (single-letter extraction)."""
        text = self._llm.chat(
            prompt,
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens,
            system_prompt=system_prompt,
        )
        return ReasoningResult(
            reasoning=text or "(no response)",
            answer=text or "(no response)",
            model_name=self.model_config.name,
            metadata={"tooluni": True},
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_clinical_context(prompt: str) -> str:
    """Strip the MANAGE/VISIT/RESOURCE instruction block, return just the clinical text."""
    # The MedPerturb DECISION_PROMPT_TEMPLATE ends the clinical context with a blank
    # line before the first "MANAGE:" question.
    m = re.search(r"(.+?)\n\s*\n\s*MANAGE\s*:", prompt, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: return everything
    return prompt.strip()
