"""
MedPathAgent — KG-Guided Tool Retrieval (pure augmentation engine)
==================================================================

Pipeline per question:

  1. Accept pre-loaded ToolUni tool_context (raw string from results CSV).

  2. Plan additional tools using the KG paths as a retrieval query.
     Deduplicate against tool names already present in the cached context.

  3. Execute only the novel tools (summarise=False to get raw outputs).

  4. Combine cached raw text + novel raw outputs.

  5. Single summarise_tool_context() call over the full combined output.

  6. Prepend KG paths + summary to the caller's prompt, call LLM once.

generate(prompt, system_prompt=None, retrieval_query=None,
         kg_paths="", cached_tool_context="", **kwargs)
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Optional

_TOOLUNI_REPO = Path(__file__).resolve().parent.parent.parent / "tooluni_agent"
if str(_TOOLUNI_REPO) not in sys.path:
    sys.path.insert(0, str(_TOOLUNI_REPO))

from tooluni_agent.agent import BedrockLLM, ToolReasoner  # noqa: E402

from .base import BaseAdvancedModel, ReasoningResult
from prompts import TOOLUNI_AUGMENTATION_HEADER

_PIPELINE_CONCURRENCY = int(os.environ.get("TOOLUNI_CONCURRENCY", "4"))
_pipeline_semaphore = threading.Semaphore(_PIPELINE_CONCURRENCY)

_KG_PATHS_HEADER = "Knowledge-graph paths (PrimeKG):"


class MedPathAgentModel(BaseAdvancedModel):
    """
    KG-path-guided tool retrieval — pure augmentation engine.

    Takes a pre-loaded ToolUni tool_context string, finds what additional
    tools the KG paths suggest, runs only the novel ones, then summarises
    everything together in a single pass before passing to the LLM.
    """

    def __init__(self, model_config, advanced_config):
        super().__init__(model_config, advanced_config)
        self.logger = logging.getLogger(f"MedPathAgent.{model_config.name}")

        region   = getattr(model_config, "region_name", None) or os.environ.get("AWS_REGION", "us-east-2")
        model_id = model_config.bedrock_model_id
        if not model_id:
            raise ValueError("bedrock_model_id must be set for MedPathAgentModel")

        use_toolrag    = getattr(model_config, "use_toolrag", False)
        toolrag_model  = getattr(model_config, "toolrag_model",
                                 "mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B")
        toolrag_device = getattr(model_config, "toolrag_device", "cpu")
        toolrag_cache  = getattr(model_config, "toolrag_cache_dir", None)

        self._llm = BedrockLLM(model_id=model_id, region=region, pool_size=3)
        self._tool_reasoner = ToolReasoner(
            model_id=model_id,
            llm=self._llm,
            use_toolrag=use_toolrag,
            toolrag_model=toolrag_model,
            toolrag_device=toolrag_device,
            toolrag_cache_dir=toolrag_cache,
        )
        self.logger.info(
            "MedPathAgentModel ready  model=%s  region=%s", model_id, region
        )

    # ── public interface ──────────────────────────────────────────────────────

    def generate(
        self,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        retrieval_query: Optional[str] = None,
        kg_paths: str = "",
        cached_tool_context: str = "",
        **kwargs,
    ) -> ReasoningResult:
        prompt = prompt or ""
        query  = retrieval_query or prompt

        with _pipeline_semaphore:
            tool_context, rounds, n_novel = self._retrieve(
                query, kg_paths, cached_tool_context
            )

        parts = []
        if kg_paths.strip():
            parts.append(f"{_KG_PATHS_HEADER}\n{kg_paths.strip()[:2000]}")
        if tool_context.strip():
            parts.append(f"{TOOLUNI_AUGMENTATION_HEADER}\n{tool_context[:2500]}")
        augmentation = "\n\n".join(parts)

        full_prompt = f"{augmentation}\n\n{prompt}" if augmentation else prompt

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
                "medpathagent":       True,
                "kg_paths":           kg_paths,
                "cached_tool_context": cached_tool_context[:1000],
                "tool_context":       tool_context[:3000],
                "n_novel_tools":      n_novel,
                "tu_rounds":          rounds,
                "final_prompt":       full_prompt,
            },
        )

    # ── retrieval pipeline ────────────────────────────────────────────────────

    def _retrieve(
        self,
        query: str,
        kg_paths: str,
        cached_tool_context: str,
    ) -> "tuple[str, list, int]":
        """
        Returns:
            tool_context  — single summarised string over all tool outputs
            rounds        — list of round dicts for logging
            n_novel       — number of new tools run from KG-path plan
        """
        # ── Step 1: extract tool names already covered by the cached context ──
        # Use the "[FDA_get_xxx]" prefixes present in raw ToolUni outputs
        already_called = set(re.findall(r'\[([A-Za-z_]+)\]', cached_tool_context))
        self.logger.info(
            "MPA: cached context %d chars, tools already covered: %s",
            len(cached_tool_context), already_called or "(none)",
        )

        # ── Step 2: plan novel tools from KG paths ────────────────────────────
        novel_plan  = []
        novel_rounds: list[dict] = []
        if kg_paths.strip():
            kg_query = (
                "Retrieve FDA drug safety information for the following "
                "biomedical entities and relationships:\n\n"
                f"{kg_paths.strip()[:1000]}\n\n"
                "Focus on indications, contraindications, interactions, "
                "adverse reactions, and dosing."
            )
            raw_plan = self._tool_reasoner.plan_tools(kg_query) or []
            for item in raw_plan:
                name = item.get("name") or item.get("tool_name", "")
                if name and name not in already_called:
                    novel_plan.append(item)
                    already_called.add(name)

            self.logger.info(
                "MPA: KG plan=%d tools, novel (not in cache)=%d",
                len(raw_plan), len(novel_plan),
            )

        # ── Step 3: run only novel tools, get raw outputs ─────────────────────
        novel_raw: list[str] = []
        if novel_plan:
            novel_details, _ = self._tool_reasoner.run_tools_detailed(
                novel_plan, summarise=False, question=query
            )
            novel_rounds = [{
                "step":  "kg_novel",
                "label": "kg_novel",
                "plan":  [{"tool_name": d["tool_name"], "arguments": d["arguments"]}
                          for d in novel_details],
                "tools": [{"tool_name": d["tool_name"], "arguments": d["arguments"],
                           "output": (d["output"] or "")[:1500],
                           "cached": d["cached"], "error": d["error"]}
                          for d in novel_details],
            }]
            novel_raw = [
                d["output"] for d in novel_details if (d.get("output") or "").strip()
            ]
            self.logger.info("MPA: %d novel tools ran, %d returned output",
                             len(novel_plan), len(novel_raw))

        # ── Step 4: combine cached text + novel raw outputs ───────────────────
        parts = []
        if cached_tool_context.strip():
            parts.append(cached_tool_context.strip())
        parts.extend(novel_raw)
        combined_raw = "\n\n".join(parts)

        # ── Step 5: single summarise pass ─────────────────────────────────────
        if combined_raw.strip():
            tool_context = self._tool_reasoner.summarise_tool_context(
                combined_raw, question=query
            )
            self.logger.info("MPA: summarised %d chars → %d chars",
                             len(combined_raw), len(tool_context))
        else:
            tool_context = ""

        return tool_context, novel_rounds, len(novel_raw)
