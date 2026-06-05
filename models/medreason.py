"""
MedReason model adapter for medperturb_eval.

Self-contained KG-augmented reasoning model:
  1. Extract biomedical entities from clinical vignette / question (spaCy NER)
  2. Map entities to PrimeKG nodes via type-restricted SapBERT cosine similarity
  3. Personalised PageRank (QA-GNN style) seeded on all entity nodes →
     query-focused working subgraph (replaces per-pair KNN-ball approach)
  4. Shortest paths within PPR subgraph serialised as incident encoding
  5. Augment prompt with paths, call AWS Bedrock LLM

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
import networkx as nx
import pandas as pd
import torch
import torch.nn.functional as F
from botocore.config import Config as BotoConfig
from transformers import AutoModel, AutoTokenizer

from .base import BaseAdvancedModel, ReasoningResult
from prompts import SYSTEM_PROMPT, KG_AUGMENTATION_HEADER

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



def _ppr_subgraph(
    G: nx.Graph,
    seed_nodes: List[str],
    top_k: int = 300,
    alpha: float = 0.85,
    bfs_hops: int = 2,
    max_neighbourhood: int = 15000,
    hub_degree: int = 1000,
) -> nx.Graph:
    """
    Personalised PageRank (QA-GNN style) seeded on all extracted entity nodes.

    Rather than taking the union of two KNN balls (one per endpoint), PPR
    propagates relevance from *all* seeds simultaneously.  Nodes that sit on
    paths between multiple seeds — clinically meaningful intermediates — score
    high; generic hub nodes that are reachable but unrelated to the query score
    low and are pruned away.

    Steps
    -----
    1. BFS up to `bfs_hops` hops from all seeds → local neighbourhood
       (keeps the graph tractable; full PrimeKG pagerank is too slow per query)
    2. Run PPR on the local subgraph with uniform seed personalisation
    3. Return the subgraph induced by the top_k highest-PPR nodes
       (seed nodes are always included)
    """
    seeds = [n for n in seed_nodes if n in G]
    if not seeds:
        return G.subgraph([])

    # 1. BFS neighbourhood — bounded.
    #    Skip fanning out through generic hub nodes (degree > hub_degree) and
    #    stop once the neighbourhood hits max_neighbourhood. Without these
    #    guards a 3-hop BFS in PrimeKG can pull in most of the graph through
    #    hubs, making the subgraph copy + pagerank pathologically slow.
    seed_set = set(seeds)
    neighbourhood: set = set(seeds)
    frontier: set = set(seeds)
    for _ in range(bfs_hops):
        if len(neighbourhood) >= max_neighbourhood:
            break
        nxt = set()
        for node in frontier:
            if node not in seed_set and G.degree(node) > hub_degree:
                continue  # don't expand through generic hubs
            nxt.update(G.neighbors(node))
        nxt -= neighbourhood
        neighbourhood |= nxt
        frontier = nxt

    local_G = G.subgraph(neighbourhood).copy()

    # 2. PPR on local subgraph
    weight = 1.0 / len(seeds)
    personalization = {n: weight for n in seeds if n in local_G}
    if not personalization:
        return local_G

    scores = nx.pagerank(
        local_G, alpha=alpha, personalization=personalization,
        max_iter=100, tol=1e-4,
    )

    # 3. Top-k nodes (seeds always kept)
    top_nodes = set(sorted(scores, key=scores.get, reverse=True)[:top_k])
    top_nodes |= set(seeds)
    return G.subgraph(top_nodes).copy()


def _serialize_paths(path_data: List, max_paths: int = 50) -> str:
    """
    Arrow encoding: each path serialised as node -> relation -> node -> ...

    Example output:
        fosaprepitant -> synergistic interaction -> aprepitant -> side effect -> nausea
    """
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

        # Skip expensive KG/SapBERT/spaCy loading when all paths are pre-cached
        self._cached_paths_only = getattr(advanced_config, "cached_paths_only", False)
        if self._cached_paths_only:
            self.logger.info("cached_paths_only=True — skipping KG/SapBERT/spaCy load.")
            self._G = None
            self._nodeemb_norm = {}
            self._nlp = None
        else:
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

        # Per-type tensors for type-restricted entity mapping (matches MedPathAgent)
        self._nodeemb_dict   = nodeemb_dict    # {type: Tensor(N, d)}
        self._node_name_dict = node_name_dict  # {type: [name, ...]}
        self.logger.info(
            "Node embeddings loaded: %d types, %d total nodes",
            len(nodeemb_dict),
            sum(len(v) for v in node_name_dict.values()),
        )

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

        # Precompute L2-normalised per-type node embeddings on device ONCE.
        # Previously _map_to_kg_node re-transferred + re-normalised the full
        # per-type tensor (tens of thousands of rows) on every entity lookup —
        # the dominant cost in KG-path extraction. Cache it here instead.
        self._nodeemb_norm: Dict[str, torch.Tensor] = {}
        for etype, tensor in self._nodeemb_dict.items():
            t = tensor.to(self._emb_device, dtype=torch.float32)
            self._nodeemb_norm[etype] = F.normalize(t, p=2, dim=-1)
        self.logger.info("Pre-normalised node embeddings for %d types",
                         len(self._nodeemb_norm))

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
            or os.environ.get("AWS_REGION", "us-east-2")
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

    def _embed_batch(self, names: List[str]) -> torch.Tensor:
        """
        Embed a batch of entity name strings in one SapBERT forward pass.
        Returns L2-normalised embeddings of shape (len(names), d) on CPU.
        """
        inputs = self._tokenizer(
            names, padding=True, truncation=True,
            max_length=64, return_tensors="pt"
        ).to(self._emb_device)
        with torch.no_grad():
            out = self._sapbert(**inputs).last_hidden_state[:, 0, :]  # (B, d)
        return F.normalize(out, p=2, dim=-1).cpu()

    def _map_entities_batch(
        self,
        entities: List[Dict[str, str]],
        k: int = 200,
        filter_threshold: float = 0.2,
    ) -> List[Optional[str]]:
        """
        Map a list of entities to KG nodes in ONE batched SapBERT call.
        Returns a list of node names (or None) aligned with entities.
        Far more GPU-efficient than calling _map_to_kg_node per entity.
        """
        if not entities:
            return []

        names  = [e.get("name", "") for e in entities]
        etypes = [e.get("type", "") for e in entities]

        # Single batched forward pass for all entity names
        valid_mask = [bool(n and t in self._nodeemb_norm) for n, t in zip(names, etypes)]
        valid_names = [n for n, v in zip(names, valid_mask) if v]

        if not valid_names:
            return [None] * len(entities)

        all_embs = self._embed_batch(valid_names)   # (n_valid, d)

        results: List[Optional[str]] = []
        valid_idx = 0
        for name, etype, is_valid in zip(names, etypes, valid_mask):
            if not is_valid:
                results.append(None)
                continue

            emb_norm = all_embs[valid_idx].unsqueeze(0).to(self._emb_device)  # (1, d)
            valid_idx += 1

            node_names     = self._node_name_dict.get(etype, [])
            node_embs_norm = self._nodeemb_norm[etype]                          # (N, d)

            sims     = (emb_norm @ node_embs_norm.T).squeeze(0).cpu()
            vals, idxs = torch.topk(sims, min(k, sims.size(0)))
            top1_sim = float(vals[0].item()) if vals.numel() > 0 else 0.0

            # Perfect name match — always accept
            matched = None
            for idx in idxs.tolist():
                if node_names[idx].lower() == name.lower():
                    matched = node_names[idx]
                    break
            if matched is None and top1_sim >= filter_threshold:
                matched = node_names[idxs[0].item()]
            results.append(matched)

        return results

    def _map_to_kg_node(
        self,
        entity: Dict[str, str],
        k: int = 200,
        filter_threshold: float = 0.2,
    ) -> Optional[str]:
        """Single-entity wrapper — calls the batched version internally."""
        results = self._map_entities_batch([entity], k=k,
                                            filter_threshold=filter_threshold)
        return results[0] if results else None

    def _map_to_kg_node_legacy(
        self,
        entity: Dict[str, str],
        k: int = 200,
        filter_threshold: float = 0.2,
    ) -> Optional[str]:
        """Original single-entity implementation (kept for reference)."""
        name  = entity.get("name", "")
        etype = entity.get("type", "")
        if not name or etype not in self._nodeemb_norm:
            return None

        node_names = self._node_name_dict.get(etype, [])
        if not node_names:
            return None

        node_embs_norm = self._nodeemb_norm[etype]

        inputs = self._tokenizer(
            [name], padding=True, truncation=True, return_tensors="pt"
        ).to(self._emb_device)
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

    def precompute_kg_paths(
        self,
        contexts: List[str],
        batch_size: int = 256,
    ) -> None:
        """
        Pre-compute KG paths for a list of clinical contexts in one GPU pass.

        All entity names across all contexts are embedded together in large
        batches → maximum GPU utilisation. Results are stored in
        self._path_cache keyed by clinical context string.

        Call this once before parallel Bedrock inference so that generate()
        hits the cache and never calls SapBERT per-row.
        """
        if not hasattr(self, "_path_cache"):
            self._path_cache: Dict[str, str] = {}

        # Deduplicate contexts
        unique_contexts = list(dict.fromkeys(c for c in contexts if c and c not in self._path_cache))
        if not unique_contexts:
            return

        self.logger.info("Precomputing KG paths for %d unique contexts...", len(unique_contexts))

        # Step 1: extract all entities from all contexts
        # Store as (context_idx, entity_dict) pairs
        all_entries: List[tuple] = []   # (ctx_idx, entity)
        ctx_entities: List[List] = []  # per-context entity list

        for ci, ctx in enumerate(unique_contexts):
            ents = self._extract_entities(ctx)
            ctx_entities.append(ents)
            for e in ents:
                if e.get("name") and e.get("type") in self._nodeemb_norm:
                    all_entries.append((ci, e))

        self.logger.info("  Total entities to embed: %d", len(all_entries))

        if not all_entries:
            for ctx in unique_contexts:
                self._path_cache[ctx] = ""
            return

        # Step 2: batch SapBERT — one giant forward pass
        from tqdm import tqdm as _tqdm
        all_names   = [e["name"] for _, e in all_entries]
        all_embs    = []
        n_batches   = (len(all_names) + batch_size - 1) // batch_size
        with _tqdm(total=len(all_names), desc="SapBERT embedding", unit="ent") as pbar:
            for i in range(0, len(all_names), batch_size):
                chunk = all_names[i:i + batch_size]
                embs  = self._embed_batch(chunk)
                all_embs.append(embs)
                pbar.update(len(chunk))
        all_embs = torch.cat(all_embs, dim=0)  # (total_entities, d)

        # Step 3: KG node matching — distribute embeddings back to contexts
        ctx_mapped: List[List[str]] = [[] for _ in unique_contexts]
        ctx_seen:   List[set]       = [set() for _ in unique_contexts]

        for emb, (ci, entity) in _tqdm(
            zip(all_embs, all_entries),
            total=len(all_entries),
            desc="KG node matching",
            unit="ent",
        ):
            etype      = entity["type"]
            name       = entity["name"]
            node_names = self._node_name_dict.get(etype, [])
            if not node_names:
                continue

            emb_norm = F.normalize(emb.unsqueeze(0), p=2, dim=-1).to(self._emb_device)
            node_embs_norm = self._nodeemb_norm[etype]
            sims = (emb_norm @ node_embs_norm.T).squeeze(0).cpu()
            vals, idxs = torch.topk(sims, min(200, sims.size(0)))
            top1_sim = float(vals[0].item()) if vals.numel() > 0 else 0.0

            matched = None
            for idx in idxs.tolist():
                if node_names[idx].lower() == name.lower():
                    matched = node_names[idx]
                    break
            if matched is None and top1_sim >= 0.2:
                matched = node_names[idxs[0].item()]

            if matched and matched.lower() not in ctx_seen[ci]:
                ctx_mapped[ci].append(matched)
                ctx_seen[ci].add(matched.lower())

        # Step 4: KG path search per context (CPU, fast)
        for ci, ctx in _tqdm(
            enumerate(unique_contexts),
            total=len(unique_contexts),
            desc="KG path search",
            unit="ctx",
        ):
            mapped = ctx_mapped[ci]
            if len(mapped) < 2:
                self._path_cache[ctx] = ""
                continue

            seed_nodes = [n.lower() for n in mapped]
            subG = _ppr_subgraph(self._G, seed_nodes, top_k=500, alpha=0.85, bfs_hops=3)

            pairs = [
                (mapped[i], mapped[j])
                for i in range(len(mapped))
                for j in range(i + 1, len(mapped))
            ]
            all_paths = []
            for src, dst in pairs[:20]:
                if len(all_paths) >= 3:
                    break
                try:
                    path = nx.shortest_path(subG, src.lower(), dst.lower())
                    if len(path) - 1 <= 6:
                        rels = [
                            self._G.get_edge_data(u, v, {}).get("relation")
                            for u, v in zip(path[:-1], path[1:])
                        ]
                        all_paths.append((path, rels))
                except Exception:
                    pass

            self._path_cache[ctx] = _serialize_paths(all_paths, max_paths=3)

        self.logger.info(
            "  Precompute done. %d/%d contexts have KG paths.",
            sum(1 for v in self._path_cache.values() if v),
            len(unique_contexts),
        )

    def _get_kg_paths(
        self,
        clinical_context: str,
        max_pairs: int = 20,
        max_hops: int = 6,
        ppr_top_k: int = 500,
        ppr_alpha: float = 0.85,
        ppr_bfs_hops: int = 3,
    ) -> str:
        """
        Extract KG paths using Personalised PageRank (QA-GNN style).

        PPR is seeded on all extracted entity nodes simultaneously.  The
        top-k PPR-scored nodes form one focused working subgraph for the
        entire question — much cleaner than the union of per-pair KNN balls.
        """
        # Check precomputed cache first
        if hasattr(self, "_path_cache") and clinical_context in self._path_cache:
            return self._path_cache[clinical_context]

        entities = self._extract_entities(clinical_context)
        if not entities:
            return ""

        # Batch all entity embeddings in one SapBERT forward pass
        mapped: List[str] = []
        seen_nodes: set = set()
        for node in self._map_entities_batch(entities):
            if node and node.lower() not in seen_nodes:
                mapped.append(node)
                seen_nodes.add(node.lower())

        if len(mapped) < 2:
            return ""

        # One PPR pass seeded on all entity nodes → focused subgraph
        seed_nodes = [n.lower() for n in mapped]
        subG = _ppr_subgraph(
            self._G, seed_nodes,
            top_k=ppr_top_k, alpha=ppr_alpha, bfs_hops=ppr_bfs_hops,
        )

        # Find shortest paths between each entity pair within the PPR subgraph
        pairs = [
            (mapped[i], mapped[j])
            for i in range(len(mapped))
            for j in range(i + 1, len(mapped))
        ]
        all_paths: List = []
        for src, dst in pairs[:max_pairs]:
            if len(all_paths) >= 3:        # only first 3 are serialised anyway
                break
            try:
                # single shortest path per pair (not all tied paths)
                path = nx.shortest_path(subG, src.lower(), dst.lower())
                if len(path) - 1 <= max_hops:
                    rels = [
                        self._G.get_edge_data(u, v, {}).get("relation")
                        for u, v in zip(path[:-1], path[1:])
                    ]
                    all_paths.append((path, rels))
            except Exception:
                pass

        return _serialize_paths(all_paths, max_paths=3)

    # ── inference ─────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        retrieval_query: Optional[str] = None,
        **kwargs,
    ) -> ReasoningResult:
        """
        Augment prompt with KG paths, then call Bedrock LLM.

        Args:
            prompt:           Full pre-built prompt (triage, MCQ, etc.).
            system_prompt:    Optional system message.
            retrieval_query:  Text used for NER / KG path extraction.
                              Defaults to prompt when not supplied.  Pass the
                              raw clinical context or question text to avoid
                              feeding boilerplate text through NER.
        """
        temperature = kwargs.get("temperature", self.model_config.temperature)
        max_tokens  = kwargs.get("max_tokens",  self.model_config.max_tokens)

        prompt = prompt or ""
        query  = retrieval_query or prompt
        cot_mode    = kwargs.get("cot_mode", False)
        cached_paths = kwargs.get("kg_paths", "")
        augmentation = ""

        if cot_mode:
            full_prompt = self._build_cot_prompt(query, prompt)
        elif cached_paths:
            # Reuse pre-extracted KG paths — skip expensive NER/PPR/SapBERT
            augmentation = f"{KG_AUGMENTATION_HEADER}\n{cached_paths}"
            full_prompt  = f"{augmentation}\n\n{prompt}"
        elif self._cached_paths_only:
            # No cached path for this row — run without KG augmentation
            self.logger.debug("cached_paths_only: no path for this row, running without KG.")
            full_prompt = prompt
        else:
            augmentation = self._augment(query)
            full_prompt  = f"{augmentation}\n\n{prompt}" if augmentation.strip() else prompt

        with self._lock:
            client = next(self._client_cycle)

        t0 = time.time()
        raw = ""
        for attempt in range(3):
            try:
                raw = _bedrock_call(
                    client,
                    self._model_id,
                    system_prompt=system_prompt or SYSTEM_PROMPT,
                    user_prompt=full_prompt,
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

        paths_text = augmentation.removeprefix(f"{KG_AUGMENTATION_HEADER}\n").strip() \
                     if augmentation else ""
        return ReasoningResult(
            reasoning=raw,
            answer=raw,
            model_name=self.model_config.name,
            metadata={
                "medreason":      True,
                "kg_paths_found": bool(paths_text),
                "n_paths":        paths_text.count("\n") + 1 if paths_text else 0,
                "kg_paths":       paths_text,
            },
        )

    def _augment(self, retrieval_text: str) -> str:
        """Extract KG paths from retrieval_text and return formatted augmentation, or ''."""
        paths_text = ""
        try:
            paths_text = self._get_kg_paths(retrieval_text)
        except Exception as exc:
            self.logger.warning(f"KG path extraction failed: {exc}")
        if not paths_text:
            return ""
        return f"{KG_AUGMENTATION_HEADER}\n{paths_text}"

    def _build_cot_prompt(self, query: str, original_prompt: str) -> str:
        """
        Entity-anchored chain-of-thought prompt.

        Replaces static path prefix with an explicit reasoning scaffold:
          Step 1 — entity extraction result
          Step 2 — KG relationship summary per entity pair
          Step 3 — structured differential built from paths
          Step 4 — the original question prompt

        Falls back to original_prompt if no paths are found.
        """
        paths_text = ""
        try:
            paths_text = self._get_kg_paths(query)
        except Exception as exc:
            self.logger.warning(f"CoT KG extraction failed: {exc}")

        if not paths_text:
            return original_prompt

        scaffold = self._build_cot_scaffold(paths_text)
        return (
            f"{scaffold}\n\n"
            f"Now use the reasoning above to answer the following:\n\n"
            f"{original_prompt}"
        )

    @staticmethod
    def _build_cot_scaffold(paths_text: str) -> str:
        """
        Convert incident-encoded KG paths into a structured CoT scaffold.

        Input format (incident encoding):
            metformin: type 2 diabetes (indication), lactic acidosis (side effect)
            type 2 diabetes: metformin (indication), insulin (indication)

        Output:
            Step 1 — Entities identified from the clinical vignette:
            • metformin (drug)
            • type 2 diabetes (disease)

            Step 2 — Knowledge-graph relationships:
            • metformin → indication → type 2 diabetes
            • metformin → side effect → lactic acidosis
            • type 2 diabetes → indication → insulin

            Step 3 — Structured reasoning:
            Given that metformin is indicated for type 2 diabetes and is
            associated with lactic acidosis as a side effect, and that
            type 2 diabetes is also treated with insulin...
        """
        # Parse incident encoding → directed edge list
        edges: List[Tuple[str, str, str]] = []
        seen_nodes: List[str] = []
        for line in paths_text.strip().splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            node, rest = line.split(":", 1)
            node = node.strip()
            if node and node not in seen_nodes:
                seen_nodes.append(node)
            for neighbour_rel in rest.split(","):
                m = re.match(r"\s*(.+?)\s*\((.+?)\)\s*$", neighbour_rel.strip())
                if m:
                    neighbour, relation = m.group(1).strip(), m.group(2).strip()
                    edges.append((node, relation, neighbour))

        if not edges:
            return f"Knowledge-graph context:\n{paths_text}"

        # Step 1: entities
        step1_lines = "\n".join(f"  • {n}" for n in seen_nodes[:10])

        # Step 2: relationships (deduplicated)
        seen_edges: set = set()
        step2_lines_list = []
        for src, rel, dst in edges:
            key = (src, rel, dst)
            if key not in seen_edges:
                seen_edges.add(key)
                step2_lines_list.append(f"  • {src} → {rel} → {dst}")
        step2_lines = "\n".join(step2_lines_list[:20])

        # Step 3: natural-language differential
        # Group by source node to build "Given that X is [rel] to Y and [rel] to Z..."
        from collections import defaultdict
        node_rels: Dict[str, List[Tuple[str,str]]] = defaultdict(list)
        for src, rel, dst in edges:
            node_rels[src].append((rel, dst))

        differential_parts = []
        for node, rels in list(node_rels.items())[:6]:
            clauses = [f"{rel} {dst}" for rel, dst in rels[:3]]
            differential_parts.append(
                f"{node} is {'; '.join(clauses)}"
            )
        step3 = "Given that " + ", and that ".join(differential_parts) + "."

        return (
            "Step 1 — Entities identified from the clinical vignette:\n"
            f"{step1_lines}\n\n"
            "Step 2 — Knowledge-graph relationships:\n"
            f"{step2_lines}\n\n"
            "Step 3 — Structured reasoning:\n"
            f"{step3}"
        )
