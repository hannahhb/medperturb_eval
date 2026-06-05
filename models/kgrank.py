"""
KGRank — triplet-based KG augmentation with MMR + optional MedCPT re-ranking.

Based on: "KGRank: Enhancing Large Language Models for Medical QA with
Knowledge Graphs and Ranking Techniques" (adapted to PrimeKG + SapBERT).

Pipeline
--------
1. spaCy NER  →  medical entities
2. Type-restricted SapBERT cosine similarity  →  KG node mapping
3. One-hop triplet retrieval from PrimeKG  →  (entity, relation, neighbour)
4. Ranking  (choose via model_config.kgrank_method):
     'similarity' — top-p by cosine(question, triplet)
     'mmr'        — MMR diversity-aware ranking (default)
     'rerank'     — similarity pre-filter → MedCPT cross-encoder re-rank
5. Top-p triplets prepended to prompt  →  LLM answer

The KG infrastructure (spaCy, SapBERT, PrimeKG graph, per-type embeddings)
is identical to MedReason.  Entity extraction and node mapping are shared
via imports from medreason.py.
"""
from __future__ import annotations

import itertools
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import boto3
import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
from botocore.config import Config as BotoConfig
from transformers import AutoModel, AutoTokenizer

from .base import BaseAdvancedModel, ReasoningResult
# Reuse shared graph builder, spaCy constants, and bedrock helper from medreason
from .medreason import (
    _build_graph,
    _SPACY_MODELS,
    _SPACY_TO_PRIMEKG,
    _bedrock_call,
    KG_AUGMENTATION_HEADER,
)
from prompts import SYSTEM_PROMPT

try:
    import spacy as _spacy
except ImportError:
    _spacy = None

# Optional: MedCPT cross-encoder for re-ranking
try:
    from transformers import AutoTokenizer as _CETokenizer, AutoModelForSequenceClassification as _CEModel
    _MEDCPT_AVAILABLE = True
except ImportError:
    _MEDCPT_AVAILABLE = False


class KGRankModel(BaseAdvancedModel):
    """
    KGRank: one-hop triplet retrieval + MMR/similarity/MedCPT ranking.

    model_config fields
    -------------------
    bedrock_model_id      — Bedrock LLM ARN
    kg_path               — path to kg.csv (PrimeKG)
    node_embeddings_path  — path to node_embeddings_sapbert.pt
    region_name           — AWS region
    kgrank_method         — 'mmr' (default) | 'similarity' | 'rerank'
    kgrank_top_p          — number of triplets to keep (default 10)
    kgrank_mmr_w_base     — MMR base weight (default 0.5)
    kgrank_mmr_delta      — MMR incremental weight per selected triplet (default 0.1)
    kgrank_rerank_top_n   — candidate pool for MedCPT re-ranking (default 50)
    sapbert_model         — HuggingFace SapBERT model name
    medcpt_model          — HuggingFace MedCPT cross-encoder name
    """

    _SAPBERT_DEFAULT = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    _MEDCPT_DEFAULT  = "ncats/MedCPT-Cross-Encoder"

    def __init__(self, model_config, advanced_config):
        super().__init__(model_config, advanced_config)
        self.logger = logging.getLogger(f"KGRank.{model_config.name}")

        self.method    = getattr(model_config, "kgrank_method",    "mmr")
        self.top_p     = getattr(model_config, "kgrank_top_p",     10)
        self.mmr_w     = getattr(model_config, "kgrank_mmr_w_base", 0.5)
        self.mmr_delta = getattr(model_config, "kgrank_mmr_delta",  0.1)
        self.rerank_n  = getattr(model_config, "kgrank_rerank_top_n", 50)

        self._load_kg(model_config)
        self._load_sapbert(model_config)
        self._load_spacy()
        self._load_medcpt(model_config)
        self._init_bedrock(model_config, advanced_config)

        self.logger.info(
            "KGRankModel ready  method=%s  top_p=%d", self.method, self.top_p
        )

    # ── init ──────────────────────────────────────────────────────────────────

    def _load_kg(self, mc):
        kg_path  = getattr(mc, "kg_path", None)
        emb_path = getattr(mc, "node_embeddings_path", None)
        if not kg_path or not os.path.exists(kg_path):
            raise FileNotFoundError(f"kg_path not found: {kg_path!r}")
        if not emb_path or not os.path.exists(emb_path):
            raise FileNotFoundError(f"node_embeddings_path not found: {emb_path!r}")

        self.logger.info("Loading PrimeKG from %s", kg_path)
        primekg   = pd.read_csv(kg_path, low_memory=False)
        self._G   = _build_graph(
            primekg[["x_name", "display_relation", "y_name"]].values.tolist()
        )
        self.logger.info(
            "KG: %d nodes, %d edges", self._G.number_of_nodes(), self._G.number_of_edges()
        )

        self.logger.info("Loading node embeddings from %s", emb_path)
        data = torch.load(emb_path, map_location="cpu", weights_only=False)
        nodeemb_dict   = data.get("nodeemb_dict", data)
        node_name_dict = data.get("node_name_dict", {})
        self._nodeemb_dict   = nodeemb_dict
        self._node_name_dict = node_name_dict
        self.logger.info(
            "Embeddings: %d types, %d nodes",
            len(nodeemb_dict),
            sum(len(v) for v in node_name_dict.values()),
        )

    def _load_sapbert(self, mc):
        name    = getattr(mc, "sapbert_model", self._SAPBERT_DEFAULT)
        dev_str = getattr(mc, "device", "cpu")
        if torch.cuda.is_available() and "cuda" in dev_str:
            self._dev = torch.device(dev_str)
        elif torch.backends.mps.is_available():
            self._dev = torch.device("mps")
        else:
            self._dev = torch.device("cpu")
        self.logger.info("Loading SapBERT (%s) on %s", name, self._dev)
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._sapbert   = AutoModel.from_pretrained(name).to(self._dev)
        self._sapbert.eval()

    def _load_spacy(self):
        self._nlps = []
        if _spacy is None:
            self.logger.warning("spaCy not installed — NER disabled")
            return
        for nm in _SPACY_MODELS:
            try:
                self._nlps.append(_spacy.load(nm))
                self.logger.info("Loaded spaCy model: %s", nm)
            except Exception:
                self.logger.warning("Could not load spaCy model %r — skipping", nm)

    def _load_medcpt(self, mc):
        self._ce_tokenizer = None
        self._ce_model     = None
        if self.method != "rerank":
            return
        if not _MEDCPT_AVAILABLE:
            self.logger.warning("transformers not available for MedCPT — falling back to similarity")
            self.method = "similarity"
            return
        name = getattr(mc, "medcpt_model", self._MEDCPT_DEFAULT)
        try:
            self.logger.info("Loading MedCPT cross-encoder (%s)", name)
            self._ce_tokenizer = _CETokenizer.from_pretrained(name)
            self._ce_model     = _CEModel.from_pretrained(name).to(self._dev)
            self._ce_model.eval()
            self.logger.info("MedCPT cross-encoder ready")
        except Exception as e:
            self.logger.warning("Could not load MedCPT (%s) — falling back to MMR: %s", name, e)
            self.method = "mmr"

    def _init_bedrock(self, mc, cfg):
        region   = getattr(mc, "region_name", None) or os.environ.get("AWS_REGION", "us-east-2")
        boto_cfg = BotoConfig(
            retries={"max_attempts": 5, "mode": "adaptive"},
            connect_timeout=20, read_timeout=180,
        )
        pool = getattr(cfg, "bedrock_client_pool_size", 3)
        self._clients      = [boto3.client("bedrock-runtime", region_name=region, config=boto_cfg)
                               for _ in range(max(1, pool))]
        self._client_cycle = itertools.cycle(self._clients)
        self._lock         = threading.Lock()
        self._model_id     = mc.bedrock_model_id
        if not self._model_id:
            raise ValueError("bedrock_model_id must be set for KGRankModel")
        self.logger.info("Bedrock ready (model=%s, region=%s)", self._model_id, region)

    # ── entity extraction & KG mapping (identical to MedReason) ───────────────

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
                pk_type = _SPACY_TO_PRIMEKG.get(ent.label_)
                if pk_type:
                    entities.append({"name": ent.text, "type": pk_type})
        return entities

    def _map_to_kg_node(
        self,
        entity: Dict[str, str],
        k: int = 200,
        filter_threshold: float = 0.4,
    ) -> Optional[str]:
        """Type-restricted SapBERT mapping — identical to MedReason."""
        name  = entity.get("name", "")
        etype = entity.get("type", "")
        if not name or etype not in self._nodeemb_dict:
            return None
        node_names = self._node_name_dict.get(etype, [])
        if not node_names:
            return None

        node_embs      = self._nodeemb_dict[etype].to(self._dev)
        node_embs_norm = F.normalize(node_embs, p=2, dim=-1)

        inputs = self._tokenizer(
            [name], padding=True, truncation=True, return_tensors="pt"
        ).to(self._dev)
        with torch.no_grad():
            out = self._sapbert(**inputs).last_hidden_state[:, 0, :]
        emb_norm = F.normalize(out, p=2, dim=-1)

        sims = (emb_norm @ node_embs_norm.T).squeeze(0).cpu()
        vals, idxs = torch.topk(sims, min(k, sims.size(0)))
        top1_sim = float(vals[0].item()) if vals.numel() > 0 else 0.0

        for idx in idxs.tolist():
            if node_names[idx].lower() == name.lower():
                return node_names[idx]

        if top1_sim < filter_threshold:
            return None
        return node_names[idxs[0].item()]

    # ── one-hop triplet retrieval ──────────────────────────────────────────────

    def _get_one_hop_triplets(
        self, mapped_nodes: List[str]
    ) -> List[Tuple[str, str, str]]:
        """
        Return all (entity, relation, neighbour) triplets reachable in one hop
        from any mapped entity node.  Deduplicates by (e, r, n).
        """
        triplets: List[Tuple[str, str, str]] = []
        seen: set = set()
        for node in mapped_nodes:
            n = node.lower()
            if n not in self._G:
                continue
            for nbr in self._G.neighbors(n):
                rel = self._G.get_edge_data(n, nbr, {}).get("relation", "related")
                if rel == "parent-child":
                    continue            # skip uninformative hierarchy edges
                key = (n, rel, nbr)
                if key not in seen:
                    triplets.append(key)
                    seen.add(key)
        return triplets

    # ── SapBERT text encoding (batched) ───────────────────────────────────────

    def _encode(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
        """Encode a list of strings with SapBERT; returns (N, d) normalised tensor."""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=64, return_tensors="pt",
            ).to(self._dev)
            with torch.no_grad():
                out = self._sapbert(**inputs).last_hidden_state[:, 0, :]
            all_embs.append(F.normalize(out, p=2, dim=-1).cpu())
        return torch.cat(all_embs, dim=0)   # (N, d)

    @staticmethod
    def _triplet_text(t: Tuple[str, str, str]) -> str:
        e, r, n = t
        return f"{e} {r} {n}"

    # ── ranking methods ───────────────────────────────────────────────────────

    def _rank_similarity(
        self,
        q_emb: torch.Tensor,              # (1, d)
        triplets: List[Tuple[str, str, str]],
    ) -> List[Tuple[str, str, str]]:
        """Top-p triplets by cosine similarity to question embedding."""
        texts  = [self._triplet_text(t) for t in triplets]
        t_embs = self._encode(texts)                       # (N, d)
        sims   = (q_emb @ t_embs.T).squeeze(0)            # (N,)
        top_i  = sims.topk(min(self.top_p, len(triplets))).indices.tolist()
        return [triplets[i] for i in top_i]

    def _rank_mmr(
        self,
        q_emb: torch.Tensor,
        triplets: List[Tuple[str, str, str]],
    ) -> List[Tuple[str, str, str]]:
        """
        MMR ranking: iteratively pick the triplet that maximises
            score = sim(q, t) - w * sim(t, t_already_selected)
        where w grows with each selection to penalise redundancy.
        """
        texts  = [self._triplet_text(t) for t in triplets]
        t_embs = self._encode(texts)                       # (N, d)
        q_sims = (q_emb @ t_embs.T).squeeze(0)            # (N,)

        selected:      List[Tuple] = []
        sel_embs:      List[torch.Tensor] = []
        remaining_idx: List[int] = list(range(len(triplets)))

        for n_sel in range(min(self.top_p, len(triplets))):
            w = self.mmr_w + self.mmr_delta * n_sel

            if not sel_embs:
                # First pick: highest similarity to question
                best = max(remaining_idx, key=lambda i: q_sims[i].item())
            else:
                sel_mat = torch.stack(sel_embs)            # (n_sel, d)
                best, best_score = None, float("-inf")
                for i in remaining_idx:
                    redundancy = (t_embs[i] @ sel_mat.T).max().item()
                    score = q_sims[i].item() - w * redundancy
                    if score > best_score:
                        best_score, best = score, i

            selected.append(triplets[best])
            sel_embs.append(t_embs[best])
            remaining_idx.remove(best)

        return selected

    def _rank_rerank(
        self,
        question: str,
        triplets: List[Tuple[str, str, str]],
    ) -> List[Tuple[str, str, str]]:
        """
        Two-stage: similarity pre-filter to top rerank_n candidates,
        then MedCPT cross-encoder scores and re-orders.
        """
        q_emb      = self._encode([question])
        candidates = self._rank_similarity(q_emb, triplets)   # already top_p

        # Expand candidate pool for cross-encoder
        top_n = min(self.rerank_n, len(triplets))
        texts  = [self._triplet_text(t) for t in triplets]
        t_embs = self._encode(texts)
        q_sims = (q_emb @ t_embs.T).squeeze(0)
        pool_i = q_sims.topk(top_n).indices.tolist()
        pool   = [triplets[i] for i in pool_i]

        if self._ce_model is None:
            return pool[:self.top_p]

        pair_texts = [self._triplet_text(t) for t in pool]
        enc = self._ce_tokenizer(
            [question] * len(pair_texts),
            pair_texts,
            padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        ).to(self._dev)
        with torch.no_grad():
            ce_scores = self._ce_model(**enc).logits.squeeze(-1).cpu()

        sorted_i = ce_scores.argsort(descending=True).tolist()
        return [pool[i] for i in sorted_i[: self.top_p]]

    # ── public interface ──────────────────────────────────────────────────────

    def generate(
        self,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        retrieval_query: Optional[str] = None,
        **kwargs,
    ) -> ReasoningResult:
        prompt = prompt or ""
        query  = retrieval_query or prompt

        augmentation, meta = self._augment(query)
        full_prompt = f"{augmentation}\n\n{prompt}" if augmentation.strip() else prompt

        temperature = kwargs.get("temperature", self.model_config.temperature)
        max_tokens  = kwargs.get("max_tokens",  self.model_config.max_tokens)

        with self._lock:
            client = next(self._client_cycle)

        t0  = time.time()
        raw = ""
        for attempt in range(3):
            try:
                raw = _bedrock_call(
                    client, self._model_id,
                    system_prompt=system_prompt or SYSTEM_PROMPT,
                    user_prompt=full_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                self.logger.warning("Bedrock attempt %d failed: %s", attempt + 1, exc)
                time.sleep(2 ** attempt)

        self.total_time += time.time() - t0
        return ReasoningResult(
            reasoning=raw,
            answer=raw,
            model_name=self.model_config.name,
            metadata=meta,
        )

    def _augment(self, retrieval_text: str) -> Tuple[str, dict]:
        """
        Full KGRank pipeline: NER → KG mapping → one-hop triplets → ranking.
        Returns (augmentation_string, metadata_dict).
        """
        entities = self._extract_entities(retrieval_text)
        if not entities:
            return "", {"kgrank": True, "triplets": [], "n_triplets_raw": 0}

        mapped: List[str] = []
        seen: set = set()
        for ent in entities:
            node = self._map_to_kg_node(ent)
            if node and node.lower() not in seen:
                mapped.append(node)
                seen.add(node.lower())

        if not mapped:
            return "", {"kgrank": True, "triplets": [], "n_triplets_raw": 0}

        # One-hop retrieval
        triplets = self._get_one_hop_triplets(mapped)
        n_raw    = len(triplets)
        self.logger.debug("One-hop: %d triplets from %d entities", n_raw, len(mapped))

        if not triplets:
            return "", {"kgrank": True, "triplets": [], "n_triplets_raw": 0}

        # Ranking
        q_emb = self._encode([retrieval_text])   # (1, d)

        if self.method == "rerank":
            ranked = self._rank_rerank(retrieval_text, triplets)
        elif self.method == "similarity":
            ranked = self._rank_similarity(q_emb, triplets)
        else:                                     # default: mmr
            ranked = self._rank_mmr(q_emb, triplets)

        augmentation = self._serialize_triplets(ranked)
        return augmentation, {
            "kgrank":         True,
            "method":         self.method,
            "n_triplets_raw": n_raw,
            "n_triplets_top": len(ranked),
            "triplets":       ranked,
            "entities":       mapped,
        }

    @staticmethod
    def _serialize_triplets(triplets: List[Tuple[str, str, str]]) -> str:
        """
        Format top-p triplets as readable lines for the LLM prompt.

            (nicotine, side effect, anxiety)
            (nicotine, synergistic interaction, caffeine)
            ...
        """
        if not triplets:
            return ""
        lines = [f"({e}, {r}, {n})" for e, r, n in triplets]
        return KG_AUGMENTATION_HEADER + "\n" + "\n".join(lines)
