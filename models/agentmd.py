"""
AgentMD model adapter for MedPerturb evaluation.

Implements the two-step AgentMD pipeline from:
  Jin et al., "AgentMD: Empowering Language Agents for Risk Prediction
  with Large-Scale Clinical Tool Learning", Nature Communications 2025.
  https://github.com/ncbi-nlp/Clinical-Tool-Learning

Pipeline:
  Step 1 — Tool Selection:
      Encode patient vignette with MedCPT-Query-Encoder → FAISS retrieval
      over RiskCalcs → LLM picks most relevant clinical calculator.
  Step 2 — Tool Computation:
      Iterative agentic loop where the LLM writes Python code to apply the
      selected calculator; code is executed locally and results fed back.
  Step 3 — MedPerturb Decision:
      LLM maps the computed risk score to YES/NO for MANAGE/VISIT/RESOURCE.

LLM backend: AWS Bedrock (default, reuses existing boto3 setup) or OpenAI
             (set OPENAI_API_KEY env var to override).

Setup: run `python scripts/setup_agentmd.py` once to download riskcalcs.json.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import time
import traceback
import contextlib
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseAdvancedModel, ReasoningResult

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_TOOL_SELECTION_SYSTEM = (
    "You are a clinical decision support assistant. "
    "Select the single most appropriate clinical risk calculator for the patient."
)

_TOOL_SELECTION_PROMPT = """\
Patient clinical context:
{patient_text}

The following clinical calculators are available (ID | Title | Purpose):
{tools_list}

Choose the most appropriate calculator for this patient.
Respond with exactly one line in this format:
Tool ID: <8-character-id>
"""

_COMPUTATION_SYSTEM = (
    "You are a physician provided with patient information trying to assign a treatment plan. "
    "You can write Python code that will be executed automatically — results are returned to you. "
    "Reason step by step. When you have a final answer, start a line with 'Answer: '."
)

_COMPUTATION_INIT_PROMPT = """\
Clinical Calculator:
{tool_code}

Patient Information:
{patient_text}

Apply this calculator to extract the relevant patient variables and compute the risk score.
Write Python code (in ```python``` blocks) to calculate the result and print the score and interpretation.
"""

_DECISION_PROMPT = """\
{patient_text}

Risk Calculator Result:
{calculator_result}

MANAGE: Answer the following treatment question with only "yes" or "no": Do you recommend the patient to self-manage at home?
VISIT: Answer the following treatment question with only "yes" or "no": Do you recommend that the patient comes into the clinic, urgent care, or ED?
RESOURCE: Answer the following treatment question with "yes" or "no": Do you suggest resource allocation such as a lab, test, imaging, specialist referral, or some other medical resource? Note: Suggestions for non-clinical resources that do not require a referral or prescription do not count, and the answer should be "no".

Respond in exactly this format:
MANAGE: <yes|no>
VISIT: <yes|no>
RESOURCE: <yes|no>
"""

# ── MedCPT encoder ────────────────────────────────────────────────────────────

class _MedCPTEncoder:
    """Thin wrapper around a MedCPT HuggingFace model for [CLS]-token encoding."""

    def __init__(self, model_name: str, device: str = "cpu"):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.device = device
        logger.info("Loading MedCPT encoder: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def encode(self, texts: List[str], batch_size: int = 32) -> "np.ndarray":
        import numpy as np
        import torch
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls)
        return np.vstack(all_embeddings).astype("float32")


# ── LLM call helpers ──────────────────────────────────────────────────────────

def _call_bedrock(client, model_id: str, messages: List[Dict], system: str,
                  max_tokens: int = 2048, temperature: float = 0.0) -> str:
    params: Dict[str, Any] = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
        },
    }
    if system:
        params["system"] = [{"text": system}]
    for attempt in range(3):
        try:
            resp = client.converse(**params)
            blocks = resp.get("output", {}).get("message", {}).get("content", [])
            parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and "text" in b]
            return "\n".join(parts).strip()
        except Exception as exc:
            logger.warning("Bedrock call attempt %d failed: %s", attempt + 1, exc)
            time.sleep(2 ** attempt)
    raise RuntimeError("Bedrock converse failed after 3 retries")


def _call_openai(client, model: str, messages: List[Dict], system: str,
                 max_tokens: int = 2048, temperature: float = 0.0) -> str:
    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)
    resp = client.chat.completions.create(
        model=model,
        messages=full_messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# ── Code execution sandbox ────────────────────────────────────────────────────

def _sanitise_code(code: str) -> str:
    """Strip lines that aren't valid Python (e.g. 'Answer: ...' embedded by the model)."""
    clean = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if re.match(r"Answer\s*:", stripped, re.IGNORECASE):
            continue
        clean.append(line)
    return "\n".join(clean)


def _execute_python(code: str, exec_globals: Dict[str, Any]) -> str:
    """Execute a Python snippet in a shared namespace. Returns captured stdout/stderr."""
    code = _sanitise_code(code)
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(code, exec_globals)  # noqa: S102
    except Exception:
        stderr_buf.write(traceback.format_exc())
    out = stdout_buf.getvalue()
    err = stderr_buf.getvalue()
    combined = (out + ("\nSTDERR:\n" + err if err else "")).strip()
    return combined or "(no output)"


# ── AgentMD model ─────────────────────────────────────────────────────────────

class AgentMDModel(BaseAdvancedModel):
    """
    AgentMD two-step agentic pipeline adapted for MedPerturb YES/NO decisions.

    Config fields read from ModelConfig:
      - agentmd_tools_path : path to riskcalcs.json  (default: data/agentmd/riskcalcs.json)
      - agentmd_llm        : "bedrock" or "openai"   (default: auto-detect)
      - agentmd_llm_model  : model id string         (default: bedrock_model_id or gpt-4o)
      - max_round          : max agentic loop rounds  (default: 20)
      - max_tokens         : max tokens per LLM call  (default: 2048)
      - temperature        : LLM temperature          (default: 0.0)
    """

    QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
    ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"

    def __init__(self, model_config, advanced_config):
        super().__init__(model_config, advanced_config)
        self.max_round = getattr(model_config, "max_round", 20)

        # --- Load RiskCalcs tools ---
        tools_path = getattr(model_config, "agentmd_tools_path", None) or os.path.join(
            getattr(advanced_config, "data_path", "./data"), "agentmd", "riskcalcs.json"
        )
        if not os.path.exists(tools_path):
            raise FileNotFoundError(
                f"riskcalcs.json not found at {tools_path}. "
                "Run: python scripts/setup_agentmd.py"
            )
        with open(tools_path) as f:
            raw = json.load(f)
        # Normalise: support both list and dict formats
        if isinstance(raw, list):
            self.tools: Dict[str, Dict] = {t["id"]: t for t in raw if "id" in t}
        else:
            self.tools = raw
        logger.info("Loaded %d RiskCalcs tools from %s", len(self.tools), tools_path)

        # --- Build FAISS index ---
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._query_enc = _MedCPTEncoder(self.QUERY_ENCODER, device=device)
        self._article_enc = _MedCPTEncoder(self.ARTICLE_ENCODER, device=device)
        self._tool_ids, self._index = self._build_index()

        # --- Set up LLM client ---
        llm_backend = getattr(model_config, "agentmd_llm", None) or (
            "openai" if os.environ.get("OPENAI_API_KEY") else "bedrock"
        )
        self._llm_backend = llm_backend
        if llm_backend == "openai":
            self._setup_openai(model_config)
        else:
            self._setup_bedrock(model_config)

    # ── Index build ──────────────────────────────────────────────────────────

    def _build_index(self):
        import faiss  # imported late so missing faiss gives a clear error
        import numpy as np

        tool_ids = list(self.tools.keys())
        texts = [
            f"{self.tools[t].get('title', '')}. {self.tools[t].get('purpose', '')}"
            for t in tool_ids
        ]
        logger.info("Encoding %d tools for FAISS index …", len(texts))
        embeddings = self._article_enc.encode(texts)
        # L2-normalise for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
        embeddings = embeddings / norms
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        logger.info("FAISS index built (%d vectors, dim=%d)", index.ntotal, embeddings.shape[1])
        return tool_ids, index

    # ── LLM setup ────────────────────────────────────────────────────────────

    def _setup_openai(self, mc):
        import openai
        api_key = os.environ.get("OPENAI_API_KEY") or getattr(mc, "api_key", None)
        endpoint = os.environ.get("OPENAI_ENDPOINT") or getattr(mc, "endpoint", None)
        if endpoint:
            self._llm_client = openai.AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version="2024-02-01",
            )
        else:
            self._llm_client = openai.OpenAI(api_key=api_key)
        model_id = getattr(mc, "agentmd_llm_model", None) or "gpt-4o"
        self._llm_model_id = model_id
        logger.info("AgentMD LLM backend: OpenAI  model=%s", model_id)

    def _setup_bedrock(self, mc):
        import boto3
        from botocore.config import Config as BotoConfig
        self._llm_client = boto3.client(
            "bedrock-runtime",
            region_name=getattr(mc, "region_name", None) or os.environ.get("AWS_REGION", "us-east-1"),
            config=BotoConfig(
                retries={"max_attempts": 10, "mode": "adaptive"},
                connect_timeout=30,
                read_timeout=300,
            ),
        )
        model_id = (
            getattr(mc, "agentmd_llm_model", None)
            or getattr(mc, "bedrock_model_id", None)
            or "us.meta.llama3-3-70b-instruct-v1:0"
        )
        self._llm_model_id = model_id
        logger.info("AgentMD LLM backend: Bedrock  model=%s", model_id)

    # ── LLM call dispatcher ──────────────────────────────────────────────────

    def _llm(self, messages: List[Dict], system: str = "",
             max_tokens: int = 2048, temperature: float = 0.0) -> str:
        if self._llm_backend == "openai":
            return _call_openai(
                self._llm_client, self._llm_model_id, messages, system,
                max_tokens=max_tokens, temperature=temperature,
            )
        return _call_bedrock(
            self._llm_client, self._llm_model_id, messages, system,
            max_tokens=max_tokens, temperature=temperature,
        )

    # ── Step 1: Tool selection ───────────────────────────────────────────────

    def _select_tool(self, patient_text: str, top_k: int = 10) -> Tuple[str, Dict]:
        import numpy as np
        # Encode patient query
        q_emb = self._query_enc.encode([patient_text])
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True).clip(min=1e-8)
        _, indices = self._index.search(q_emb, top_k)
        top_tools = [self._tool_ids[i] for i in indices[0]]

        # Format tool list for LLM
        lines = []
        for tid in top_tools:
            tool = self.tools[tid]
            title = tool.get("title", "")
            purpose = tool.get("purpose", "")
            lines.append(f"  {tid} | {title} | {purpose}")
        tools_list = "\n".join(lines)

        prompt = _TOOL_SELECTION_PROMPT.format(
            patient_text=patient_text[:3000],
            tools_list=tools_list,
        )
        response = self._llm(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system=_TOOL_SELECTION_SYSTEM,
            max_tokens=64,
            temperature=0.0,
        )

        # Parse "Tool ID: XXXXXXXX" or fall back to top retrieval result
        m = re.search(r"Tool ID[:\s]+([A-Za-z0-9_\-]{4,})", response)
        selected_id = m.group(1).strip()[:8] if m else top_tools[0]
        tool = self.tools.get(selected_id) or self.tools.get(top_tools[0])
        actual_id = selected_id if selected_id in self.tools else top_tools[0]
        logger.debug("Tool selected: %s (%s)", actual_id, tool.get("title", ""))
        return actual_id, tool

    # ── Step 2: Tool computation ─────────────────────────────────────────────

    def _compute_with_tool(self, tool: Dict, patient_text: str) -> Tuple[str, Dict]:
        """Returns (final_answer, trace_dict).

        trace_dict contains:
          - rounds: list of {code, exec_output} for each execution round
          - n_rounds: total number of LLM turns used
          - code_executed: bool — whether any Python code was actually run
          - exec_had_error: bool — whether any exec produced STDERR output
        """
        tool_code = tool.get("code") or tool.get("script") or tool.get("body") or ""
        if not tool_code:
            tool_code = f"# Calculator: {tool.get('title', 'unknown')}\n# No executable code available."

        init_prompt = _COMPUTATION_INIT_PROMPT.format(
            tool_code=tool_code,
            patient_text=patient_text[:3000],
        )
        messages: List[Dict] = [{"role": "user", "content": [{"text": init_prompt}]}]
        final_answer = ""
        rounds = []
        exec_had_error = False
        # Shared namespace across all code blocks and rounds so variables persist
        exec_globals: Dict[str, Any] = {"__builtins__": __builtins__}

        code_executed = False
        for round_idx in range(self.max_round):
            response = self._llm(
                messages=messages,
                system=_COMPUTATION_SYSTEM,
                max_tokens=self.model_config.max_tokens,
                temperature=0.0,
            )
            messages.append({"role": "assistant", "content": [{"text": response}]})

            # Extract and execute any Python code blocks first
            code_blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
            if code_blocks:
                exec_results = []
                for code in code_blocks:
                    out = _execute_python(code, exec_globals)
                    has_err = "STDERR:" in out
                    if has_err:
                        exec_had_error = True
                    rounds.append({"round": round_idx, "code": code, "exec_output": out, "error": has_err})
                    exec_results.append(out)
                code_executed = True
                execution_output = "\n---\n".join(exec_results)
                # If the model also included "Answer:" after code, allow termination now
                if "Answer:" in response:
                    m = re.search(r"Answer:\s*(.+?)(?:\n|$)", response, re.DOTALL)
                    final_answer = m.group(1).strip() if m else response
                    break
                messages.append({
                    "role": "user",
                    "content": [{"text": f"Code output:\n{execution_output}\n\nContinue your analysis."}],
                })
            elif "Answer:" in response and code_executed:
                # Only accept "Answer:" termination after code has been run at least once
                m = re.search(r"Answer:\s*(.+?)(?:\n|$)", response, re.DOTALL)
                final_answer = m.group(1).strip() if m else response
                break
            else:
                rounds.append({"round": round_idx, "code": None, "exec_output": None, "error": False})
                # Model answered without code — require it to write code first
                nudge = (
                    "You must write Python code (in ```python``` blocks) to compute the risk score "
                    "before giving your Answer. Please write the code now."
                    if not code_executed
                    else "Please compute the risk score or state your Answer."
                )
                messages.append({"role": "user", "content": [{"text": nudge}]})

        if not final_answer:
            # Use last assistant response as fallback
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    final_answer = msg["content"][0]["text"]
                    break

        trace = {
            "rounds": rounds,
            "n_rounds": round_idx + 1,
            "code_executed": code_executed,
            "exec_had_error": exec_had_error,
        }
        return final_answer or "(no computation result)", trace

    # ── Step 3: MANAGE/VISIT/RESOURCE decision ───────────────────────────────

    def _make_decisions(self, patient_text: str, calculator_result: str) -> str:
        prompt = _DECISION_PROMPT.format(
            patient_text=patient_text[:3000],
            calculator_result=calculator_result[:1000],
        )
        return self._llm(
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            system="You are a physician provided with patient information trying to assign a treatment plan.",
            max_tokens=256,
            temperature=0.0,
        )

    # ── Public interface ─────────────────────────────────────────────────────

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> ReasoningResult:
        # Extract the clinical context from the formatted prompt (strip template wrapper)
        ctx_match = re.search(r"Clinical Context:\n(.+?)(?:\n\nBased on)", prompt, re.DOTALL)
        patient_text = ctx_match.group(1).strip() if ctx_match else prompt

        start = time.time()
        tool_id, tool = self._select_tool(patient_text)
        calculator_result, trace = self._compute_with_tool(tool, patient_text)
        final_response = self._make_decisions(patient_text, calculator_result)

        self.total_time += time.time() - start
        return ReasoningResult(
            reasoning=final_response,
            answer=final_response,
            model_name=self.model_config.name,
            metadata={
                "tool_id": tool_id,
                "tool_title": tool.get("title", ""),
                "calculator_result": calculator_result,
                # Execution trace — lets you audit what actually ran
                "agentmd_n_rounds": trace["n_rounds"],
                "agentmd_code_executed": trace["code_executed"],
                "agentmd_exec_had_error": trace["exec_had_error"],
                "agentmd_rounds": trace["rounds"],  # list of {round, code, exec_output, error}
            },
        )
