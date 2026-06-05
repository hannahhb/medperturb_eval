"""
ToolUni adapter — pure augmentation engine.

generate(prompt, system_prompt=None, retrieval_query=None, **kwargs)
  1. Retrieves FDA tool context using retrieval_query (defaults to prompt)
  2. Prepends augmentation to prompt
  3. Calls Bedrock LLM
  4. Returns raw text in ReasoningResult

All dataset-specific logic (triage questions, MCQ prompts, two-turn dispatch)
lives in the caller (processor.py, run_curebench.py, etc.).
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

# Cap concurrent tool pipelines (DrugBank XML is lazy-loaded once, then cached)
_PIPELINE_CONCURRENCY = int(os.environ.get("TOOLUNI_CONCURRENCY", "4"))
_pipeline_semaphore = threading.Semaphore(_PIPELINE_CONCURRENCY)

from .base import BaseAdvancedModel, ReasoningResult
from prompts import TOOLUNI_AUGMENTATION_HEADER

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


class ToolUniModel(BaseAdvancedModel):
    """
    ToolUni (FDA tool retrieval) wrapped as a pure augmentation engine.

    generate(prompt, system_prompt=None, retrieval_query=None, **kwargs)
      retrieval_query — text used to seed the FDA tool retrieval; defaults to prompt.
                        Callers may pass the raw clinical context or question text
                        to avoid feeding triage-question boilerplate into the planner.
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
        region = model_config.region_name or os.environ.get("AWS_REGION", "us-east-2")

        self._llm = BedrockLLM(model_id=model_id, region=region, pool_size=3)

        use_toolrag    = getattr(model_config, "use_toolrag", False)
        toolrag_model  = getattr(model_config, "toolrag_model",
                                 "mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B")
        toolrag_device = getattr(model_config, "toolrag_device", "cpu")
        toolrag_cache  = getattr(model_config, "toolrag_cache_dir", None)

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
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        retrieval_query: Optional[str] = None,
        **kwargs,
    ) -> ReasoningResult:
        """
        Augment prompt with FDA tool results, then call LLM.

        Args:
            prompt:           Full pre-built prompt the LLM will receive (after
                              augmentation is prepended).
            system_prompt:    Optional system message.
            retrieval_query:  Text used to seed FDA tool retrieval.  Defaults to
                              prompt when not supplied.  Pass the raw clinical
                              vignette or question text for cleaner retrieval.
        """
        prompt = prompt or ""
        query  = retrieval_query or prompt

        # ── cached tool rounds ───────────────────────────────────────────────
        # Priority 1: caller passes raw tu_rounds list (from results CSV)
        # Priority 2: load from per-row JSON log file (legacy --tooluni-log-dir)
        # If neither is available, run the full tool pipeline live.
        cached_rounds = kwargs.get("cached_tool_rounds")  # list[dict] or None
        tool_context  = ""
        rounds: list  = []

        if not cached_rounds:
            cached_log_dir = kwargs.get("tooluni_log_dir")
            row_id         = kwargs.get("row_id")
            if cached_log_dir and row_id:
                import json as _json
                log_file = os.path.join(cached_log_dir, f"tool_{row_id}.json")
                if os.path.exists(log_file):
                    try:
                        with open(log_file) as f:
                            log_data = _json.load(f)
                        cached_rounds = log_data.get("tu_rounds", []) or None
                        self.logger.debug("Loaded cached rounds from log for %s", row_id)
                    except Exception as exc:
                        self.logger.warning("Failed to load tool log %s: %s", log_file, exc)

        if cached_rounds:
            # Raw tool outputs available — re-run summarise + reasoning fresh
            tool_context, rounds = self._summarise_cached(
                cached_rounds=cached_rounds,
                question=query,
                fallback_tool_context=kwargs.get("cached_tool_context_fallback", ""),
            )
        else:
            with _pipeline_semaphore:
                tool_context, rounds = self._run_multistep_tracked(query, max_steps=1)

        if tool_context.strip():
            augmentation = f"{TOOLUNI_AUGMENTATION_HEADER}\n{tool_context[:2500]}"
            full_prompt  = f"{augmentation}\n\n{prompt}"
        else:
            augmentation = ""
            full_prompt  = prompt

        text = self._llm.chat(
            full_prompt,
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
                "tu_rounds":    rounds,
                "final_prompt": full_prompt,
            },
        )

    # ── Cached-path summarise + reasoning ────────────────────────────────────

    def _summarise_cached(
        self,
        cached_rounds: list,
        question: str,
        fallback_tool_context: str = "",
    ) -> "tuple[str, list[dict]]":
        """
        Re-run summarise + STEP_PROMPT reasoning on cached raw tool outputs.

        Pipeline:
          1. Reconstruct raw combined tool output from cached_rounds[*].tools[*].output
          2. Call summarise_tool_context() fresh (new model summarises the raw outputs)
          3. Call STEP_PROMPT reasoning fresh on the fresh summary
          4. Return the fresh summary as tool_context + updated rounds

        The expensive FDA API calls are NOT repeated.
        """
        from tooluni_agent.agent import STEP_PROMPT  # noqa: PLC0415

        # 1. Reconstruct raw combined output from cached per-tool outputs
        raw_parts = []
        for r in cached_rounds:
            for t in r.get("tools", []):
                raw_output = t.get("output", "")
                if raw_output and raw_output not in raw_parts:
                    raw_parts.append(raw_output)
        raw_combined = "\n\n".join(raw_parts)

        if not raw_combined.strip():
            if fallback_tool_context:
                # No raw outputs (tool calls failed) — use cached summary directly
                # and skip re-summarise; still re-run STEP_PROMPT fresh.
                self.logger.debug("_summarise_cached: no raw outputs, using fallback tool_context")
                tool_context = fallback_tool_context
            else:
                self.logger.warning("_summarise_cached: no raw tool outputs and no fallback — skipping augmentation")
                return "", cached_rounds
        else:
            # 2. Re-run summarise fresh with the new model
            fresh_summary = self._tool_reasoner.summarise_tool_context(
                raw_combined, question=question
            )
            tool_context = fresh_summary or raw_combined

        # 3. Re-run STEP_PROMPT reasoning fresh on the fresh summary
        step      = 1
        max_steps = max(len(cached_rounds), 1)

        last_reasoning = self._tool_reasoner._llm_chat(
            STEP_PROMPT.format(
                question=question,
                step=step,
                max_steps=max_steps,
                context=tool_context,
            ),
            max_tokens=128,
        )

        status = self._tool_reasoner._extract_field(last_reasoning, "STATUS").upper()
        answer = self._tool_reasoner._extract_field(last_reasoning, "ANSWER").strip().upper()

        # 4. Rebuild rounds — preserve tool metadata, replace reasoning fields
        new_rounds = []
        for i, r in enumerate(cached_rounds):
            updated = dict(r)
            if i == 0:
                updated["tool_context"]     = tool_context[:2000]
                updated["last_reasoning"]   = last_reasoning
                updated["status"]           = status or "COMPLETE"
                updated["answer"]           = answer
                updated["summarise_cached"] = True
            new_rounds.append(updated)

        if not new_rounds:
            new_rounds = [{
                "step": 1, "plan": [], "tools": [],
                "tool_context":     tool_context[:2000],
                "status":           status or "COMPLETE",
                "answer":           answer,
                "last_reasoning":   last_reasoning,
                "summarise_cached": True,
            }]

        return tool_context, new_rounds

    # ── Tracked multistep retrieval ───────────────────────────────────────────

    def _run_multistep_tracked(
        self, question: str, max_steps: int = 2
    ) -> "tuple[str, list[dict]]":
        """
        Run tool retrieval for up to max_steps, recording per-step details.

        Returns:
            tool_context  (str)        — accumulated FDA tool results
            rounds        (list[dict]) — per-step metadata for logging
        """
        from tooluni_agent.agent import STEP_PROMPT  # noqa: PLC0415

        accumulated: list[str] = []
        rounds: list[dict] = []
        last_reasoning = ""
        combined = ""

        for step in range(1, max_steps + 1):
            if step == 1:
                plan = self._tool_reasoner.plan_tools(question)
            else:
                needs = self._tool_reasoner._extract_field(last_reasoning, "NEEDS")
                followup = question
                if combined.strip() and combined != "No relevant context retrieved.":
                    followup += f"\n\nContext already retrieved:\n{combined[:800]}"
                if needs:
                    followup += f"\n\nStill needed: {needs}"
                plan = self._tool_reasoner.plan_tools(followup)

            if not plan:
                rounds.append({"step": step, "plan": [], "tools": [],
                               "tool_context": "", "status": "no_plan",
                               "answer": "", "last_reasoning": ""})
                break

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
            answer = self._tool_reasoner._extract_field(last_reasoning, "ANSWER").strip().upper()

            rounds.append({
                "step":           step,
                "plan":           [{"tool_name": d["tool_name"], "arguments": d["arguments"]}
                                   for d in tool_details],
                "tools":          [{"tool_name": d["tool_name"], "arguments": d["arguments"],
                                    "output": (d["output"] or "")[:1500],
                                    "cached": d["cached"], "error": d["error"]}
                                   for d in tool_details],
                "tool_context":   step_context[:2000],
                "status":         status or "CONTINUE",
                "answer":         answer,
                "last_reasoning": last_reasoning,
            })

            if status in ("END", "COMPLETE"):
                break

        return "\n\n".join(accumulated), rounds
