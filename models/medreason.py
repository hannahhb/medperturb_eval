"""
MedReason model adapter for medperturb_eval.

Self-contained KG-augmented reasoning model:
  1. Extract biomedical entities from clinical vignette (spaCy NER)
  2. Map entities to PrimeKG nodes via SapBERT cosine similarity + FAISS
  3. Find shortest paths between mapped entity pairs in the KG (networkx)
  4. Augment the clinical decision prompt with paths
  5. Call AWS Bedrock LLM → YES/NO decisions for MANAGE/VISIT/RESOURCE

(Previously named "MedPathAgent" — renamed because the new combined
KG + tool-augmented model is the proper "MedPathAgent".)
"""
from __future__ import annotations

import itertools
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import boto3
import faiss
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from botocore.config import Config as BotoConfig
from transformers import AutoModel, AutoTokenizer

from .base import BaseAdvancedModel, ReasoningResult

try:
    import spacy
except ImportError:
    spacy = None


# ── constants ─────────────────────────────────────────────────────────────────

_SPACY_MODELS = ["en_ner_bc5cdr_md", "en_ner_bionlp13cg_md"]

_SPACY_TO_PRIMEKG: Dict[str, str] = {
    "DISEASE": "disease",
    "CHEMICAL": "drug",
    "SIMPLE_CHEMICAL": "drug",
    "GENE_OR_GENE_PRODUCT": "gene/protein",
    "AMINO_ACID": "gene/protein",
    "ANATOMICAL_SYSTEM": "anatomy",
    "TISSUE": "anatomy",
    "CELL": "anatomy",
    "CELLULAR_COMPONENT": "cellular_component",
    "DEVELOPING_ANATOMICAL_STRUCTURE": "anatomy",
    "MULTI-TISSUE_STRUCTURE": "anatomy",
    "ORGAN": "anatomy",
    "ORGANISM": "anatomy",
    "ORGANISM_SUBDIVISION": "anatomy",
    "ORGANISM_SUBSTANCE": "anatomy",
    "IMMATERIAL_ANATOMICAL_ENTITY": "anatomy",
    "PATHOLOGICAL_FORMATION": "effect/phenotype",
}


# ── KG graph helpers ──────────────────────────────────────────────────────────

def _build_graph(triples: List) -> nx.Graph:
    G = nx.Graph()
    for x_name, relation, y_name in triples:
        G.add_edge(str(x_name).lower(), str(y_name).lower(), relation=relation)
    return G


def _build_node_index(
    primekg: pd.DataFrame,
    nodeemb_dict: Dict[str, Any],
    node_name_dict: Dict[str, List[str]],
) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    node_list: List[str] = []
    emb_rows: List[np.ndarray] = []
    for t, emb_for_type in nodeemb_dict.items():
        names = node_name_dict.get(t)
        if names is None:
            names = primekg.query(f'x_type == "{t}"')["x_name"].unique().tolist()
        n = min(len(names), emb_for_type.shape[0])
        for i in range(n):
            node_list.append(names[i])
            row = emb_for_type[i]
            if isinstance(row, torch.Tensor):
                row = row.detach().cpu().numpy()
            emb_rows.append(np.asarray(row, dtype=np.float32))
    if not emb_rows:
        return [], np.empty((0, 1), dtype=np.float32), {}
    mat = np.stack(emb_rows, axis=0).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mat_norm = (mat / norms).astype(np.float32)
    name_to_idx = {n: i for i, n in enumerate(node_list)}
    return node_list, mat_norm, name_to_idx


def _faiss_knn(
    query_name: str,
    name_to_idx: Dict[str, int],
    emb_norm: np.ndarray,
    index: faiss.Index,
    node_list: List[str],
    k: int = 200,
) -> List[str]:
    idx = name_to_idx.get(query_name)
    if idx is None:
        return []
    q = emb_norm[idx].reshape(1, -1).astype(np.float32)
    _, I = index.search(q, k)
    return [node_list[i] for i in I[0] if i < len(node_list)]


def _find_paths(
    G: nx.Graph,
    name_to_idx: Dict[str, int],
    emb_norm: np.ndarray,
    index: faiss.Index,
    node_list: List[str],
    src: str,
    dst: str,
    k_each: int = 200,
) -> List[Tuple]:
    src_knn = _faiss_knn(src.lower(), name_to_idx, emb_norm, index, node_list, k=k_each)
    dst_knn = _faiss_knn(dst.lower(), name_to_idx, emb_norm, index, node_list, k=k_each)
    nodes = set(src_knn) | set(dst_knn) | {src.lower(), dst.lower()}
    subG = G.subgraph(nodes).copy()
    paths = []
    try:
        for path in nx.all_shortest_paths(subG, src.lower(), dst.lower()):
            rels = [
                G.get_edge_data(u, v, {}).get("relation")
                for u, v in zip(path[:-1], path[1:])
            ]
            paths.append((path, rels))
    except Exception:
        pass
    return paths


def _serialize_paths(path_data: List, max_paths: int = 50) -> str:
    lines = []
    for item in path_data[:max_paths]:
        if isinstance(item, tuple) and len(item) == 2:
            nodes, rels = item
            parts = []
            for i, node in enumerate(nodes):
                parts.append(node)
                if i < len(rels) and rels[i] and rels[i] != "parent-child":
                    parts.append(str(rels[i]))
            lines.append(" -> ".join(parts))
        elif isinstance(item, list):
            lines.append(" -> ".join(item))
    return "\n".join(lines)


# ── Bedrock helper ────────────────────────────────────────────────────────────

def _bedrock_call(
    client,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    params: Dict[str, Any] = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
        "inferenceConfig": {
            "maxTokens": int(max_tokens),
            "temperature": float(temperature),
            "topP": 0.9,
        },
    }
    if system_prompt:
        params["system"] = [{"text": system_prompt}]
    resp = client.converse(**params)
    blocks = ((resp.get("output") or {}).get("message") or {}).get("content") or []
    parts = [b.get("text", "") for b in blocks if "text" in b]
    return "\n".join(parts).strip()


# ── model ─────────────────────────────────────────────────────────────────────

class MedReasonModel(BaseAdvancedModel):
    """
    KG-augmented clinical decision model for MedPerturb.

    Required model_config fields:
      bedrock_model_id        - Bedrock model ARN
      kg_path                 - path to kg.csv (PrimeKG)
      node_embeddings_path    - path to node_embeddings_sapbert.pt cache
      region_name             - AWS region (default: AWS_REGION env or us-east-1)

    Optional:
      sapbert_model           - HuggingFace model name for SapBERT (default below)
      device                  - torch device string (default: auto-detect)
    """

    _SAPBERT_DEFAULT = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

    def __init__(self, model_config, advanced_config):
        super().__init__(model_config, advanced_config)
        self.logger = logging.getLogger(f"MedReason.{model_config.name}")

        self._load_kg(model_config)
        self._load_sapbert(model_config)
        self._load_spacy()
        self._init_bedrock(model_config, advanced_config)
        self.logger.info("MedReasonModel ready.")

    # ── init helpers ──────────────────────────────────────────────────────────

    def _load_kg(self, mc):
        kg_path = getattr(mc, "kg_path", None)
        emb_path = getattr(mc, "node_embeddings_path", None)
        if not kg_path or not os.path.exists(kg_path):
            raise FileNotFoundError(f"kg_path not found: {kg_path!r}")
        if not emb_path or not os.path.exists(emb_path):
            raise FileNotFoundError(f"node_embeddings_path not found: {emb_path!r}")

        self.logger.info(f"Loading PrimeKG from {kg_path}")
        primekg = pd.read_csv(kg_path, low_memory=False)
        self._G = _build_graph(
            primekg[["x_name", "display_relation", "y_name"]].values.tolist()
        )
        self.logger.info(
            f"KG: {self._G.number_of_nodes()} nodes, {self._G.number_of_edges()} edges"
        )

        self.logger.info(f"Loading node embeddings from {emb_path}")
        data = torch.load(emb_path, map_location="cpu", weights_only=False)
        nodeemb_dict = data.get("nodeemb_dict", data)
        node_name_dict = data.get("node_name_dict", {})
        self._node_list, self._emb_norm, self._name_to_idx = _build_node_index(
            primekg, nodeemb_dict, node_name_dict
        )
        d = self._emb_norm.shape[1]
        self._faiss_index = faiss.IndexFlatIP(d)
        self._faiss_index.add(self._emb_norm)
        self.logger.info(f"FAISS index ready: {len(self._node_list)} nodes, dim={d}")

    def _load_sapbert(self, mc):
        sapbert_name = getattr(mc, "sapbert_model", self._SAPBERT_DEFAULT)
        dev_str = getattr(mc, "device", "cpu")
        if torch.cuda.is_available() and "cuda" in dev_str:
            self._emb_device = torch.device(dev_str)
        elif torch.backends.mps.is_available():
            self._emb_device = torch.device("mps")
        else:
            self._emb_device = torch.device("cpu")
        self.logger.info(f"Loading SapBERT ({sapbert_name}) on {self._emb_device}")
        self._tokenizer = AutoTokenizer.from_pretrained(sapbert_name)
        self._sapbert = AutoModel.from_pretrained(sapbert_name).to(self._emb_device)
        self._sapbert.eval()

    def _load_spacy(self):
        self._nlps = []
        if spacy is None:
            self.logger.warning("spaCy not installed; entity extraction disabled.")
            return
        for nm in _SPACY_MODELS:
            try:
                self._nlps.append(spacy.load(nm))
                self.logger.info(f"Loaded spaCy model: {nm}")
            except Exception:
                self.logger.warning(f"Could not load spaCy model {nm!r}; skipping.")
        if not self._nlps:
            self.logger.warning("No spaCy NER models available.")

    def _init_bedrock(self, mc, cfg):
        region = (
            getattr(mc, "region_name", None)
            or os.environ.get("AWS_REGION", "us-east-1")
        )
        boto_cfg = BotoConfig(
            retries={"max_attempts": 5, "mode": "adaptive"},
            connect_timeout=20,
            read_timeout=180,
        )
        pool_size = getattr(cfg, "bedrock_client_pool_size", 3)
        self._clients = [
            boto3.client("bedrock-runtime", region_name=region, config=boto_cfg)
            for _ in range(max(1, pool_size))
        ]
        self._client_cycle = itertools.cycle(self._clients)
        self._lock = threading.Lock()
        self._model_id = mc.bedrock_model_id
        if not self._model_id:
            raise ValueError("bedrock_model_id must be set for MedReasonModel")
        self.logger.info(f"Bedrock client ready (model={self._model_id}, region={region})")

    # ── entity extraction & KG path finding ──────────────────────────────────

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        if not self._nlps:
            return []
        seen: set = set()
        entities = []
        for nlp in self._nlps:
            for ent in nlp(text).ents:
                key = (ent.text.lower(), ent.label_)
                if key in seen:
                    continue
                seen.add(key)
                primekg_type = _SPACY_TO_PRIMEKG.get(ent.label_)
                if primekg_type:
                    entities.append({"name": ent.text, "type": primekg_type})
        return entities

    def _embed_name(self, name: str) -> np.ndarray:
        inputs = self._tokenizer(
            [name], padding=True, truncation=True, return_tensors="pt"
        ).to(self._emb_device)
        with torch.no_grad():
            out = self._sapbert(**inputs).last_hidden_state[:, 0, :]
        emb = F.normalize(out, p=2, dim=-1).cpu().numpy().astype(np.float32)
        return emb

    def _map_to_kg_node(self, entity: Dict[str, str], k: int = 100) -> Optional[str]:
        name = entity.get("name", "")
        if not name:
            return None
        q = self._embed_name(name)
        _, I = self._faiss_index.search(q, k)
        candidates = [self._node_list[i] for i in I[0] if i < len(self._node_list)]
        for c in candidates:
            if c.lower() == name.lower():
                return c
        return candidates[0] if candidates else None

    def _get_kg_paths(self, clinical_context: str, max_pairs: int = 10) -> str:
        entities = self._extract_entities(clinical_context)
        if not entities:
            return ""
        mapped: List[str] = []
        seen_nodes: set = set()
        for ent in entities:
            node = self._map_to_kg_node(ent)
            if node and node.lower() not in seen_nodes:
                mapped.append(node)
                seen_nodes.add(node.lower())
        if len(mapped) < 2:
            return ""
        all_paths: List = []
        pairs = [
            (mapped[i], mapped[j])
            for i in range(len(mapped))
            for j in range(i + 1, len(mapped))
        ]
        for src, dst in pairs[:max_pairs]:
            all_paths.extend(
                _find_paths(
                    self._G,
                    self._name_to_idx,
                    self._emb_norm,
                    self._faiss_index,
                    self._node_list,
                    src,
                    dst,
                )
            )
        return _serialize_paths(all_paths)

    # ── inference ─────────────────────────────────────────────────────────────

    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> ReasoningResult:
        temperature = kwargs.get("temperature", self.model_config.temperature)
        max_tokens = kwargs.get("max_tokens", self.model_config.max_tokens)

        # Pull clinical context out of the already-formatted prompt
        m = re.search(r"Clinical Context:\s*\n(.+?)\n\nBased on", prompt, re.DOTALL)
        clinical_context = m.group(1).strip() if m else prompt

        paths_text = ""
        try:
            paths_text = self._get_kg_paths(clinical_context)
        except Exception as exc:
            self.logger.warning(f"KG path extraction failed: {exc}")

        if paths_text:
            augmented_prompt = (
                "Relevant biomedical knowledge from PrimeKG:\n"
                + paths_text
                + "\n\n"
                + prompt
            )
        else:
            augmented_prompt = prompt

        with self._lock:
            client = next(self._client_cycle)

        t0 = time.time()
        raw = ""
        for attempt in range(3):
            try:
                raw = _bedrock_call(
                    client,
                    self._model_id,
                    system_prompt=system_prompt or "",
                    user_prompt=augmented_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                self.logger.warning(f"Bedrock attempt {attempt+1} failed: {exc}")
                time.sleep(2 ** attempt)

        self.total_time += time.time() - t0

        meta = {
            "medreason": True,
            "kg_paths_found": bool(paths_text),
            "n_paths": paths_text.count("\n") + 1 if paths_text else 0,
            "kg_paths": paths_text,
        }
        return ReasoningResult(
            reasoning=raw,
            answer=raw,
            model_name=self.model_config.name,
            metadata=meta,
        )
