"""
MedPerturb Processor

Loads the MedPerturb dataset from HuggingFace, runs each clinical vignette
through the configured model (TxAgent or AWS Bedrock), and saves predictions
as a CSV that matches the MedPerturb data.csv schema.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from threading import Lock
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from config import AdvancedConfig
from factory import ModelFactory
from prompts import build_prompt, SYSTEM_PROMPT
from scripts.logging_setup import (
    discover_processed_ids,
    load_cached_rows,
    save_detail,
    save_csv,
    append_row_csv,
)


# ── answer extraction ────────────────────────────────────────────────────────

def _extract_yes_no(text: str, label: str) -> str:
    """Extract YES/NO for a labelled field (MANAGE, VISIT, RESOURCE)."""
    pattern = rf"{label}\s*:\s*(YES|NO)\b"
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: look for bare YES/NO anywhere near the label
    block = text[max(0, text.upper().find(label)) :]
    m2 = re.search(r"\b(YES|NO)\b", block, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return "UNKNOWN"


def _extract_reasoning(text: str, label: str) -> str:
    """Extract the LABEL_REASONING line."""
    pattern = rf"{label}_REASONING\s*:\s*(.+?)(?=\n[A-Z_]+\s*:|$)"
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_response(raw: str) -> Dict[str, str]:
    return {
        "manage": _extract_yes_no(raw, "MANAGE"),
        "visit": _extract_yes_no(raw, "VISIT"),
        "resource": _extract_yes_no(raw, "RESOURCE"),
        "manage_reasoning": _extract_reasoning(raw, "MANAGE"),
        "visit_reasoning": _extract_reasoning(raw, "VISIT"),
        "resource_reasoning": _extract_reasoning(raw, "RESOURCE"),
        "raw_response": raw,
    }


# ── row ID helper ────────────────────────────────────────────────────────────

def _row_id(context_id: str, perturbation: str) -> str:
    """Unique stable key for a (context_id, perturbation) pair."""
    safe_pert = str(perturbation).replace(" ", "_")
    return f"{context_id}__{safe_pert}"


# ── processor ────────────────────────────────────────────────────────────────

class MedPerturbProcessor:
    def __init__(self, cfg: AdvancedConfig):
        self.cfg = cfg
        self.logger = logging.getLogger("MedPerturbProcessor")
        os.makedirs(cfg.output_path, exist_ok=True)
        os.makedirs(cfg.log_path, exist_ok=True)
        self.write_lock = Lock()

        primary_cfg = cfg.model_configs[0]
        self.model = ModelFactory.create_model(primary_cfg, cfg)
        self.model_name = primary_cfg.name

    # ── data loading ──────────────────────────────────────────────────────────

    def load_dataset(self) -> pd.DataFrame:
        url = self.cfg.hf_dataset_url
        self.logger.info(f"Loading MedPerturb from: {url}")
        try:
            df = pd.read_csv(url)
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

        # Filter to requested source datasets
        if self.cfg.datasets:
            df = df[df["dataset"].isin(self.cfg.datasets)]

        # Filter to requested perturbations
        if self.cfg.perturbations:
            df = df[df["perturbation"].isin(self.cfg.perturbations)]

        self.logger.info(
            f"Loaded {len(df)} rows covering datasets: "
            f"{df['dataset'].unique().tolist()} | perturbations: "
            f"{df['perturbation'].unique().tolist()}"
        )
        return df.reset_index(drop=True)

    # ── processing ────────────────────────────────────────────────────────────

    def process(self, resume: bool = True) -> pd.DataFrame:
        df = self.load_dataset()

        # Slice for parallel runs
        n = len(df)
        start_idx = int(round(n * self.cfg.slice_start))
        end_idx = int(round(n * self.cfg.slice_end))
        if n > 0 and start_idx == end_idx:
            end_idx = min(n, start_idx + 1)
        df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        self.logger.info(f"Slice {start_idx}:{end_idx} → {len(df)} rows")

        if self.cfg.debug_mode and self.cfg.sample_size:
            df = df.head(int(self.cfg.sample_size))
            self.logger.info(f"Debug mode: {len(df)} rows")

        # Resume: skip already-processed rows
        cached_ids = set()
        cached_rows: Dict[str, Dict] = {}
        if resume:
            cached_ids = discover_processed_ids(self.cfg.log_path)
            cached_rows = load_cached_rows(cached_ids, self.cfg.log_path)
            self.logger.info(f"Resume: {len(cached_rows)} cached rows found")

        results: List[Dict] = list(cached_rows.values())
        remaining = [
            row
            for _, row in df.iterrows()
            if _row_id(str(row["context_id"]), str(row["perturbation"])) not in cached_ids
        ]

        if remaining:
            pbar = tqdm(remaining, desc="Processing vignettes")
            for row in pbar:
                rid = _row_id(str(row["context_id"]), str(row["perturbation"]))
                try:
                    result = self._process_row(row, rid)
                    results.append(result)
                    if self.cfg.save_intermediate:
                        save_detail(self.cfg.log_path, rid, result, self.write_lock)
                    append_row_csv(result, self.cfg.output_path, "results.csv")
                except Exception as e:
                    self.logger.error(f"Failed on {rid}: {e}")
                    import traceback
                    results.append(self._error_row(row, rid, traceback.format_exc()))
                time.sleep(self.cfg.rate_limit_delay)
            pbar.close()

        out_df = pd.DataFrame(results)
        self.logger.info(f"Saved {len(results)} results to {self.cfg.output_path}/results.csv")
        return out_df

    def _process_row(self, row: pd.Series, rid: str) -> Dict:
        clinical_context = str(row.get("clinical_context", ""))
        prompt = build_prompt(clinical_context)

        result = self.model.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )

        parsed = parse_response(result.reasoning)

        base = {
            "row_id": rid,
            "context_id": str(row.get("context_id", "")),
            "perturbation": str(row.get("perturbation", "")),
            "model": self.model_name,
            # Preserve MedPerturb metadata columns if present
            "dataset": row.get("dataset", ""),
            "original_gender": row.get("original_gender", ""),
            "age": row.get("age", ""),
            "gendered_condition": row.get("gendered_condition", ""),
            # Gold standard labels (from data.csv)
            "gold_standard_manage": row.get("gold_standard_manage", ""),
            "gold_standard_visit": row.get("gold_standard_visit", ""),
            "gold_standard_resource": row.get("gold_standard_resource", ""),
        }
        # Model predictions in MedPerturb-compatible naming: <model>_manage etc.
        base[f"{self.model_name}_manage"] = parsed["manage"]
        base[f"{self.model_name}_visit"] = parsed["visit"]
        base[f"{self.model_name}_resource"] = parsed["resource"]
        base[f"{self.model_name}_manage_reasoning"] = parsed["manage_reasoning"]
        base[f"{self.model_name}_visit_reasoning"] = parsed["visit_reasoning"]
        base[f"{self.model_name}_resource_reasoning"] = parsed["resource_reasoning"]
        base["raw_response"] = parsed["raw_response"]

        # Model-specific metadata
        meta = getattr(result, "metadata", None) or {}
        if meta.get("txagent"):
            base["txagent_tools"] = json.dumps(meta.get("tools"), default=str)
            base["txagent_tool_log"] = meta.get("tool_log_path", "")
        if meta.get("medpathagent"):
            base["kg_paths"] = meta.get("kg_paths", "")
        if "tool_id" in meta:
            base["agentmd_tool_id"] = meta.get("tool_id", "")
            base["agentmd_tool_title"] = meta.get("tool_title", "")
            base["agentmd_calculator_result"] = meta.get("calculator_result", "")
            # Audit columns: did code actually run, did it error, how many rounds?
            base["agentmd_n_rounds"] = meta.get("agentmd_n_rounds", "")
            base["agentmd_code_executed"] = meta.get("agentmd_code_executed", "")
            base["agentmd_exec_had_error"] = meta.get("agentmd_exec_had_error", "")
            # All generated code blocks concatenated (one per round, separated by markers)
            rounds = meta.get("agentmd_rounds", [])
            code_blocks = [
                f"# --- round {r['round']} ---\n{r['code']}"
                for r in rounds if r.get("code")
            ]
            base["agentmd_generated_code"] = "\n\n".join(code_blocks)
            # Full per-round trace (code + exec output) in per-row detail JSON
            base["agentmd_rounds"] = json.dumps(rounds, default=str)

        return base

    @staticmethod
    def _error_row(row: pd.Series, rid: str, traceback_str: str) -> Dict:
        return {
            "row_id": rid,
            "context_id": str(row.get("context_id", "")),
            "perturbation": str(row.get("perturbation", "")),
            "model": "error",
            "raw_response": f"Processing error:\n{traceback_str}",
        }
