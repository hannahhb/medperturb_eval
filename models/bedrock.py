import json
import time
import itertools
import threading
import concurrent.futures
import asyncio
import re
from typing import Tuple, List, Dict, Any

import boto3
from botocore.config import Config as BotoConfig

from .base import BaseAdvancedModel, ReasoningResult
from prompts import SYSTEM_PROMPT


def _extract_text_from_blocks(blocks: list) -> str:
    parts = []
    for b in blocks:
        if isinstance(b, dict) and "reasoningContent" in b:
            rt = (b.get("reasoningContent") or {}).get("reasoningText") or {}
            t = rt.get("text", "")
            if t:
                parts.append(t)
        if isinstance(b, dict) and "text" in b:
            t = b.get("text", "")
            if t:
                parts.append(t)
    return "\n".join(parts).strip()


class BedrockChatModel(BaseAdvancedModel):
    """AWS Bedrock Converse API adapter with connection pooling."""

    def __init__(self, model_config, advanced_config):
        super().__init__(model_config, advanced_config)

        max_connections = getattr(advanced_config, "bedrock_max_connections", 50)
        self.client_pool_size = getattr(advanced_config, "bedrock_client_pool_size", 3)
        self.clients = []

        for _ in range(self.client_pool_size):
            client = boto3.client(
                "bedrock-runtime",
                region_name=self.model_config.region_name,
                config=BotoConfig(
                    retries={"max_attempts": 10, "mode": "adaptive"},
                    max_pool_connections=max(1, max_connections // self.client_pool_size),
                    connect_timeout=30,
                    read_timeout=300,
                ),
            )
            self.clients.append(client)

        self._client_lock = threading.Lock()
        self.client_cycle = itertools.cycle(self.clients)
        self.model_id = self.model_config.bedrock_model_id
        if not self.model_id:
            raise ValueError("bedrock_model_id must be set for Bedrock models")

        self.token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def get_client(self):
        with self._client_lock:
            return next(self.client_cycle)

    def generate(self, prompt: str = None, system_prompt: str = None, **kwargs) -> ReasoningResult:
        start = time.time()
        client = self.get_client()

        temperature = kwargs.get("temperature", self.model_config.temperature)
        max_tokens = kwargs.get("max_tokens", self.model_config.max_tokens)

        if prompt is None:
            prompt = ""

        converse_params = {
            "modelId": self.model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "maxTokens": int(max_tokens),
                "temperature": float(temperature),
                "topP": 0.9,
            },
        }
        if system_prompt:
            converse_params["system"] = [{"text": system_prompt}]

        resp = None
        for attempt in range(3):
            try:
                resp = client.converse(**converse_params)
                break
            except Exception:
                time.sleep(2 ** attempt)
        if resp is None:
            raise RuntimeError("Bedrock converse failed after retries.")

        blocks = (resp.get("output", {}) or {}).get("message", {}).get("content", []) or []
        full_text = _extract_text_from_blocks(blocks)

        input_tokens, output_tokens = self._extract_token_usage(resp)
        total_tokens = input_tokens + output_tokens
        self.token_usage["input_tokens"] += input_tokens
        self.token_usage["output_tokens"] += output_tokens
        self.token_usage["total_tokens"] += total_tokens
        self.total_tokens += total_tokens
        self.total_time += time.time() - start

        metadata = {
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        }
        return ReasoningResult(
            reasoning=full_text or "(no response)",
            answer=full_text or "(no response)",
            model_name=self.model_config.name,
            metadata=metadata,
        )

    @staticmethod
    def _extract_token_usage(response: Dict[str, Any]) -> Tuple[int, int]:
        usage = response.get("usage") or {}
        meta = response.get("metadata") or {}
        input_tokens = (
            (usage.get("inputTokens") if isinstance(usage, dict) else 0)
            or (usage.get("input_tokens") if isinstance(usage, dict) else 0)
            or (meta.get("inputTokens") if isinstance(meta, dict) else 0)
            or 0
        )
        output_tokens = (
            (usage.get("outputTokens") if isinstance(usage, dict) else 0)
            or (usage.get("output_tokens") if isinstance(usage, dict) else 0)
            or (meta.get("outputTokens") if isinstance(meta, dict) else 0)
            or 0
        )
        return int(input_tokens or 0), int(output_tokens or 0)

    def generate_batch_sync(
        self, requests: List[Dict[str, Any]], max_workers: int = 10
    ) -> List[ReasoningResult]:
        def _run(req):
            try:
                return self.generate(**req)
            except Exception as e:
                return ReasoningResult(
                    reasoning=f"Error: {e}",
                    answer="Error",
                    model_name=self.model_config.name,
                )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max_workers, len(requests))
        ) as executor:
            futures = {executor.submit(_run, req): i for i, req in enumerate(requests)}
            results = [None] * len(requests)
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    async def generate_batch_async(
        self, requests: List[Dict[str, Any]], max_concurrent: int = 20
    ) -> List[ReasoningResult]:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _one(req):
            async with semaphore:
                return await asyncio.to_thread(self.generate, **req)

        return await asyncio.gather(*[_one(r) for r in requests])
