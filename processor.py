"""
MedPerturb Processor

Loads the MedPerturb dataset from HuggingFace, runs each clinical vignette
through the configured model (TxAgent or AWS Bedrock), and saves predictions
as a CSV that matches the MedPerturb data.csv schema.
"""
from __future__ import annotations

import concurrent.futures
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
from prompts import build_triage_prompt, SYSTEM_PROMPT
from scripts.logging_setup import (
    discover_processed_ids,
    load_cached_rows,
    save_detail,
    save_csv,
    append_row_csv,
)


# ── tool log ─────────────────────────────────────────────────────────────────

def _save_tool_log(
    log_path: str,
    row_id: str,
    perturbation: str,
    meta: dict,
    lock: Optional[Lock] = None,
) -> None:
    """
    Write a per-row tool-detail JSON to logs/<model>/run_0/tool_<row_id>.json.
    Mirrors run_medxpertqa._save_tool_log — one file per vignette.
    """
    if meta.get("tooluni"):
        detail = {
            "row_id":         row_id,
            "perturbation":   perturbation,
            "tu_rounds":      meta.get("tu_rounds", []),
            "tool_context":   meta.get("tool_context", ""),
            "trace_manage":   meta.get("trace_manage", ""),
            "trace_visit":    meta.get("trace_visit", ""),
            "trace_resource": meta.get("trace_resource", ""),
            "final_prompt":   meta.get("final_prompt", ""),
        }
    elif meta.get("medpathagent"):
        detail = {
            "row_id":             row_id,
            "perturbation":       perturbation,
            "entities":           meta.get("entities", []),
            "kg_paths":           meta.get("kg_paths", ""),
            "tool_plan":          meta.get("tool_plan", []),
            "tool_plan_source":   meta.get("tool_plan_source", ""),
            "tool_context":       meta.get("tool_context", ""),
            "combined_context":   meta.get("combined_context", ""),
            "tu_rounds":          meta.get("tu_rounds", []),
            "trace_manage":       meta.get("trace_manage", ""),
            "trace_visit":        meta.get("trace_visit", ""),
            "trace_resource":     meta.get("trace_resource", ""),
        }
    elif "tool_id" in meta:  # AgentMD
        detail = {
            "row_id":                 row_id,
            "perturbation":           perturbation,
            "tool_id":                meta.get("tool_id", ""),
            "tool_title":             meta.get("tool_title", ""),
            "calculator_result":      meta.get("calculator_result", ""),
            "agentmd_n_rounds":       meta.get("agentmd_n_rounds", 0),
            "agentmd_code_executed":  meta.get("agentmd_code_executed", False),
            "agentmd_exec_had_error": meta.get("agentmd_exec_had_error", False),
            "agentmd_rounds":         meta.get("agentmd_rounds", []),
            "final_prompt":           meta.get("final_prompt", ""),
        }
    else:
        return

    os.makedirs(log_path, exist_ok=True)
    fname = os.path.join(log_path, f"tool_{row_id}.json")
    if lock:
        with lock:
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(detail, f, indent=2, default=str)
    else:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2, default=str)


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

        # Optional: ToolUni log directory for MedPathAgent cached_dual mode
        self.tooluni_log_dir: Optional[str] = getattr(cfg, "tooluni_log_path", None)

        # Optional: row_id → tool_context lookup for cached ToolUni runs
        # Loaded from a prior ToolUni results CSV via cfg.tooluni_cache_csv_path
        # row_id → (rounds, fallback_tool_context) for cached ToolUni runs
        self.tool_rounds_lookup: Dict[str, list] = {}
        self.tool_context_fallback: Dict[str, str] = {}
        tooluni_cache_csv = getattr(cfg, "tooluni_cache_csv_path", None)
        if tooluni_cache_csv and os.path.exists(tooluni_cache_csv) and os.path.getsize(tooluni_cache_csv) > 0:
            tu = pd.read_csv(tooluni_cache_csv)
            if "tu_rounds" in tu.columns and "row_id" in tu.columns:
                for _, row in tu[tu["tu_rounds"].notna()].iterrows():
                    rid = row["row_id"]
                    try:
                        rounds = json.loads(row["tu_rounds"])
                        if rounds:
                            self.tool_rounds_lookup[rid] = rounds
                    except Exception:
                        pass
                    # store tu_tool_context as fallback for rows with no raw outputs
                    if "tu_tool_context" in tu.columns and pd.notna(row.get("tu_tool_context", "")):
                        ctx = str(row["tu_tool_context"]).strip()
                        if ctx:
                            self.tool_context_fallback[rid] = ctx
                self.logger.info(
                    "Loaded %d tu_rounds + %d fallback contexts from %s",
                    len(self.tool_rounds_lookup), len(self.tool_context_fallback), tooluni_cache_csv
                )

        # Optional: row_id → kg_paths lookup for MedPathAgent
        # Loaded from a prior MedReason results CSV via cfg.medreason_csv_path
        self.kg_paths_lookup: Dict[str, str] = {}
        medreason_csv = getattr(cfg, "medreason_csv_path", None)
        if medreason_csv and os.path.exists(medreason_csv):
            mr = pd.read_csv(medreason_csv)
            if "kg_paths" in mr.columns and "row_id" in mr.columns:
                self.kg_paths_lookup = (
                    mr[mr["kg_paths"].notna() & (mr["kg_paths"].str.strip() != "")]
                    .set_index("row_id")["kg_paths"]
                    .to_dict()
                )
                self.logger.info(
                    "Loaded %d kg_paths entries from %s", len(self.kg_paths_lookup), medreason_csv
                )

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

        # MedPathAgent: exclude colorful_tone (medreason never ran on it so
        # there are no kg_paths to guide retrieval and it's out of scope).
        if self.kg_paths_lookup:
            before = len(remaining)
            remaining = [
                row for row in remaining
                if str(row.get("perturbation", "")) != "colorful_tone"
            ]
            self.logger.info(
                "MedPathAgent: excluded colorful_tone, %d → %d rows remaining",
                before, len(remaining),
            )

        # cached_dual: skip rows with no kg_paths — they would add nothing over
        # the existing ToolUni result (same tool context, no KG signal).
        mpa_mode = getattr(
            self.cfg.model_configs[0] if self.cfg.model_configs else None,
            "mpa_retrieval_mode", ""
        )
        if mpa_mode == "cached_dual":
            before = len(remaining)
            remaining = [
                row for row in remaining
                if self.kg_paths_lookup.get(
                    _row_id(str(row["context_id"]), str(row["perturbation"])), ""
                ).strip()
            ]
            self.logger.info(
                "cached_dual: excluded %d rows without kg_paths, %d → %d remaining",
                before - len(remaining), before, len(remaining),
            )

        if remaining:
            n_workers = getattr(self.cfg, "max_workers", 1)

            # Pre-compute all KG paths in one GPU batch before parallel Bedrock calls
            # Only applies to MedReason (model has precompute_kg_paths method)
            model = getattr(self, "model", None)
            if model is not None and hasattr(model, "precompute_kg_paths") and \
               not getattr(self.cfg, "cached_paths_only", False):
                contexts = [
                    str(row.get("clinical_context", row.get("prompt", "")))
                    for row in remaining
                ]
                model.precompute_kg_paths(contexts)

            if n_workers > 1:
                results += self._process_parallel(remaining, n_workers)
            else:
                results += self._process_sequential(remaining)

        out_df = pd.DataFrame(results)

        # cached_dual: fill in rows that were skipped (no kg_paths) by copying
        # the corresponding ToolUni results, renaming prediction columns so the
        # output CSV is complete for all row IDs.
        tooluni_csv = getattr(self.cfg, "tooluni_results_csv", None)
        if mpa_mode == "cached_dual" and tooluni_csv and os.path.exists(tooluni_csv):
            out_df = self._fill_from_tooluni(out_df, tooluni_csv)

        self.logger.info(f"Saved {len(out_df)} results to {self.cfg.output_path}/results.csv")
        return out_df

    def _fill_from_tooluni(self, out_df: pd.DataFrame, tooluni_csv: str) -> pd.DataFrame:
        """
        Copy rows from a ToolUni results CSV that are missing in out_df.
        Prediction columns (anything ending in _manage, _visit, _resource and
        their _reasoning siblings) are renamed from the ToolUni model name to
        the current MedPathAgent model name.  All other columns are kept as-is.
        """
        tu = pd.read_csv(tooluni_csv, low_memory=False)

        present_ids = set(out_df["row_id"]) if "row_id" in out_df.columns else set()
        missing = tu[~tu["row_id"].isin(present_ids)].copy()

        if missing.empty:
            self.logger.info("cached_dual fill: no missing rows in ToolUni CSV")
            return out_df

        # Detect the ToolUni model-name prefix by looking for a *_manage column
        tu_model = None
        for col in tu.columns:
            if col.endswith("_manage") and not col.startswith("gold"):
                tu_model = col[: -len("_manage")]
                break

        if tu_model:
            suffixes = ["_manage", "_visit", "_resource",
                        "_manage_reasoning", "_visit_reasoning", "_resource_reasoning"]
            rename = {
                f"{tu_model}{s}": f"{self.model_name}{s}"
                for s in suffixes
                if f"{tu_model}{s}" in missing.columns
            }
            missing = missing.rename(columns=rename)
            self.logger.info(
                "cached_dual fill: copying %d rows from ToolUni (src model=%s → dst model=%s)",
                len(missing), tu_model, self.model_name,
            )
        else:
            self.logger.warning(
                "cached_dual fill: could not detect ToolUni model name; "
                "prediction columns will not be renamed"
            )

        filled = pd.concat([out_df, missing], ignore_index=True, sort=False)

        # Persist the filled CSV
        results_path = os.path.join(self.cfg.output_path, "results.csv")
        filled.to_csv(results_path, index=False)
        return filled

    def _process_sequential(self, rows) -> List[Dict]:
        results = []
        pbar = tqdm(rows, desc="Processing vignettes")
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
        return results

    def _process_parallel(self, rows, n_workers: int) -> List[Dict]:
        results = []
        pbar = tqdm(total=len(rows), desc=f"Processing vignettes ({n_workers} workers)")

        def _work(row):
            rid = _row_id(str(row["context_id"]), str(row["perturbation"]))
            try:
                result = self._process_row(row, rid)
                if self.cfg.save_intermediate:
                    save_detail(self.cfg.log_path, rid, result, self.write_lock)
                append_row_csv(result, self.cfg.output_path, "results.csv")
                return result
            except Exception as e:
                self.logger.error(f"Failed on {rid}: {e}")
                import traceback
                return self._error_row(row, rid, traceback.format_exc())

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_work, row): row for row in rows}
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())
                pbar.update(1)

        pbar.close()
        return results

    def _process_row(self, row: pd.Series, rid: str) -> Dict:
        clinical_context = str(row.get("clinical_context", ""))

        # Pass pre-extracted kg_paths if available (used by MedPathAgent)
        kg_paths = self.kg_paths_lookup.get(rid, "")

        # Build full triage prompt here so models stay dataset-agnostic
        triage_prompt = build_triage_prompt(clinical_context)

        result = self.model.generate(
            prompt=triage_prompt,
            system_prompt=SYSTEM_PROMPT,
            retrieval_query=clinical_context,   # clean text for NER / tool retrieval
            kg_paths=kg_paths,
            row_id=rid,
            tooluni_log_dir=self.tooluni_log_dir,
            cached_tool_rounds=self.tool_rounds_lookup.get(rid, None),
            cached_tool_context_fallback=self.tool_context_fallback.get(rid, ""),
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
        if meta.get("medreason"):
            base["kg_paths"] = meta.get("kg_paths", "")
        if meta.get("tooluni"):
            base["tu_tool_context"]   = meta.get("tool_context", "")
            base["tu_trace_manage"]   = meta.get("trace_manage", "")
            base["tu_trace_visit"]    = meta.get("trace_visit", "")
            base["tu_trace_resource"] = meta.get("trace_resource", "")
            base["tu_rounds"]         = json.dumps(meta.get("tu_rounds", []), default=str)
        if meta.get("kgrank"):
            base["kgrank_entities"]   = json.dumps(meta.get("entities", []), default=str)
            base["kgrank_n_entities"] = meta.get("n_entities", 0)
            base["kgrank_triplets"]   = meta.get("triplets_text", "")
            base["kgrank_n_triplets"] = meta.get("n_triplets", 0)
            base["kgrank_method"]     = meta.get("kgrank_method", "")
        if meta.get("medpathagent"):
            # NEW combined MedPathAgent: KG entities + paths + FDA tool results
            base["mpa_entities"]         = json.dumps(meta.get("entities", []), default=str)
            base["mpa_n_entities"]       = meta.get("n_entities", 0)
            base["mpa_kg_paths"]         = meta.get("kg_paths", "")
            base["mpa_n_kg_paths"]       = meta.get("n_kg_paths", 0)
            base["mpa_tool_plan"]        = json.dumps(meta.get("tool_plan", []), default=str)
            base["mpa_tool_plan_source"] = meta.get("tool_plan_source", "")
            base["mpa_tool_context"]     = meta.get("tool_context", "")
            base["mpa_tu_rounds"]        = json.dumps(meta.get("tu_rounds", []), default=str)
            base["mpa_trace_manage"]     = meta.get("trace_manage", "")
            base["mpa_trace_visit"]      = meta.get("trace_visit", "")
            base["mpa_trace_resource"]   = meta.get("trace_resource", "")
        # Write separate tool-detail JSON (mirrors run_medxpertqa._save_tool_log)
        if meta.get("tooluni") or meta.get("medpathagent") or "tool_id" in meta:
            _save_tool_log(self.cfg.log_path, rid, base.get("perturbation", ""), meta, self.write_lock)

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
