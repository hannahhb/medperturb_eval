from __future__ import annotations
import os
import time
import asyncio
from typing import Any, Dict, List, Optional

from .base import BaseAdvancedModel, ReasoningResult

from medperturb_eval.models.txagent import TxAgent


class TxAgentModel(BaseAdvancedModel):
    """
    BaseAdvancedModel adapter over TxAgent.

    Supports two backends controlled by model_config.backend:
      - "huggingface"  : loads the model locally via vLLM (requires GPU + vLLM)
      - "bedrock"      : routes LLM calls to AWS Bedrock (no local GPU needed)

    For the "bedrock" backend the tool RAG is also done via Bedrock
    (ToolRAGModelBedrock), so no SentenceTransformer download is required.
    """

    def __init__(self, model_config, advanced_config):
        super().__init__(model_config, advanced_config)
        self.agent: Optional[TxAgent] = None
        self._load_model()

    # ── model initialisation ──────────────────────────────────────────────────

    def _load_model(self):
        backend = getattr(self.model_config, "backend", "huggingface").lower()
        use_bedrock = backend == "bedrock"

        os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
        if not use_bedrock:
            os.environ.setdefault("VLLM_USE_V1", "0")

        rag_model = getattr(
            self.model_config, "rag_model", "mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B"
        )

        self.logger.info(
            f"Initializing TxAgent | backend={backend} "
            f"LLM={self.model_config.checkpoint} RAG={rag_model}"
        )

        common_kwargs = dict(
            enable_summary=getattr(self.model_config, "enable_summary", False),
            enable_entity_awareness=self.model_config.enable_entity_awareness,
            force_finish=True,
        )

        if use_bedrock:
            self.agent = TxAgent(
                model_name=self.model_config.checkpoint,   # ignored at runtime; bedrock_model_id is used
                rag_model_name=rag_model,
                use_bedrock=True,
                bedrock_model_id=self.model_config.bedrock_model_id,
                bedrock_region=getattr(self.model_config, "region_name", None)
                    or os.environ.get("AWS_REGION", "us-east-1"),
                bedrock_rag_model_id=self.model_config.bedrock_model_id,  # same model handles RAG
                device_id="cpu",   # no GPU needed
                **common_kwargs,
            )
            self.device = "cpu"
        else:
            device_id = getattr(self.model_config, "device_id", 0) or 0
            self.agent = TxAgent(
                model_name=self.model_config.checkpoint,
                rag_model_name=rag_model,
                use_bedrock=False,
                device_id=device_id,
                **common_kwargs,
            )
            self.device = f"cuda:{device_id}"

        self.agent.init_model()
        self.logger.info(f"TxAgent ready (backend={backend})")

    # ── inference ─────────────────────────────────────────────────────────────

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> ReasoningResult:
        assert self.agent is not None, "TxAgent not initialized"

        temperature = kwargs.get("temperature", self.model_config.temperature)
        max_new_tokens = kwargs.get("max_tokens", self.model_config.max_tokens)
        max_token = getattr(self.model_config, "max_token", 90240)
        call_agent = getattr(self.model_config, "call_agent", False)
        max_round = getattr(self.model_config, "max_round", 20)

        t0 = time.time()
        text = self.agent.run_multistep_agent(
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_token=max_token,
            call_agent=call_agent,
            max_round=max_round,
        )

        meta = {
            "txagent": True,
            "backend": getattr(self.model_config, "backend", "huggingface"),
            "tools": getattr(self.agent, "last_used_tools", None),
            "tool_log_path": getattr(self.agent, "last_question_log_path", None),
        }

        self.total_time += time.time() - t0

        return ReasoningResult(
            reasoning=str(text).strip(),
            answer=str(text).strip(),
            model_name=self.model_config.name,
            metadata=meta,
        )

    # ── batch helpers ─────────────────────────────────────────────────────────

    async def generate_batch_async(
        self, batch_requests: List[Dict[str, Any]], max_concurrency: int = 1
    ) -> List[ReasoningResult]:
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(req):
            async with sem:
                return await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.generate(**req)
                )

        return await asyncio.gather(*[asyncio.create_task(_one(r)) for r in batch_requests])

    def generate_batch_sync(
        self, batch_requests: List[Dict[str, Any]], **_
    ) -> List[ReasoningResult]:
        return [self.generate(**req) for req in batch_requests]
