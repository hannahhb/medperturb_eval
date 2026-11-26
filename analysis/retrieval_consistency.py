"""
Retrieval Consistency Analysis
================================
Measures how stable MRA (KG paths) and TUA (tool sets) retrieval is
across perturbation types, compared to the baseline clinical vignette.

Inclusion rule: both-non-empty — only questions where BOTH the baseline
and the perturbed variant produced non-empty retrieval are included.

Metrics:
  - Jaccard similarity of retrieved sets (paths or tool names)
  - Exact match rate (sets are identical)
  - TUA content Jaccard: word-set Jaccard on tu_tool_context free text
  - TUA cosine (TF-IDF): TF-IDF vector cosine similarity on retrieved text
  - TUA cosine (bio): PubMedBERT mean-pooled embedding cosine similarity

Usage:
    python analysis/retrieval_consistency.py
    python analysis/retrieval_consistency.py --run 0        # single run (default)
    python analysis/retrieval_consistency.py --run all      # average across run_0/1/2
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

# ── biomedical sentence embedding model ────────────────────────────────────────
# pritamdeka/S-PubMedBert-MS-MARCO: PubMedBERT fine-tuned with contrastive loss
# for semantic sentence similarity (SBERT-style). Avoids the anisotropy problem
# of raw BERT models (which produce near-1.0 cosine for all pairs).
# Requires: pip install transformers torch
BIO_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
_bio_tokenizer = None
_bio_model     = None


def _load_bio_model():
    global _bio_tokenizer, _bio_model
    if _bio_model is not None:
        return _bio_tokenizer, _bio_model
    from transformers import AutoTokenizer, AutoModel
    print(f"  [bio] loading {BIO_MODEL_NAME} …", flush=True)
    _bio_tokenizer = AutoTokenizer.from_pretrained(BIO_MODEL_NAME)
    _bio_model     = AutoModel.from_pretrained(BIO_MODEL_NAME)
    _bio_model.eval()
    print("  [bio] model ready", flush=True)
    return _bio_tokenizer, _bio_model


def _mean_pool(model_output, attention_mask):
    """Mean-pool last hidden states over non-padding tokens."""
    import torch
    token_embeddings = model_output.last_hidden_state          # (B, L, H)
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def _bio_embed(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """
    Encode texts with S-PubMedBert-MS-MARCO (SBERT-style fine-tuned model).
    Returns (n, 768) float32 array of L2-normalised embeddings.
    Mean-centres the corpus before normalising to further reduce anisotropy.
    """
    import torch
    tokenizer, model = _load_bio_model()
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**enc)
        vecs = _mean_pool(out, enc["attention_mask"]).numpy()
        all_vecs.append(vecs)

    embeddings = np.vstack(all_vecs)                          # (N, H)
    # mean-centre: subtracts corpus mean to counteract residual anisotropy
    embeddings -= embeddings.mean(axis=0, keepdims=True)
    # L2-normalise so dot product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
    return embeddings / norms

# ── paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT  = Path(__file__).resolve().parent.parent
LOGS_ROOT  = REPO_ROOT / "logs"
OUT_DIR    = REPO_ROOT / "analysis"

PERTS      = ["gender_swap", "uncertain_tone", "summary"]
BACKBONES  = ["llama8b3_1", "llama70b3_3", "qwen235b"]
BACKBONE_LABEL = {
    "llama8b3_1":  "Llama-8B",
    "llama70b3_3": "Llama-70B",
    "qwen235b":    "Qwen-235B",
}

# ── loaders ────────────────────────────────────────────────────────────────────

def load_mra_paths(logs_dir: Path) -> dict[str, dict[str, frozenset]]:
    """Returns {context_id: {perturbation: frozenset(path_strings)}}"""
    data: dict[str, dict] = defaultdict(dict)
    for f in sorted(logs_dir.glob("detailed_*.json")):
        d = json.loads(f.read_text())
        cid  = str(d.get("context_id", ""))
        pert = d.get("perturbation", "unknown")
        raw  = d.get("kg_paths", "") or ""
        paths = frozenset(
            p.strip() for p in str(raw).split("\n")
            if p.strip() and p.strip().lower() != "nan"
        )
        data[cid][pert] = paths
    return dict(data)


def _parse_tu_rounds(d: dict) -> list:
    """Parse tu_rounds from a log entry. Returns [] if missing, non-list, or
    unparseable JSON string — treats all such cases as empty/erroneous."""
    rounds = d.get("tu_rounds", [])
    if isinstance(rounds, str):
        try:
            rounds = json.loads(rounds)
        except Exception:
            return []
    if not isinstance(rounds, list):
        return []
    return rounds


def _tu_rounds_have_tools(d: dict) -> bool:
    """Return True only if tu_rounds contains at least one valid tool call.
    Used to gate all TUA loaders so that empty/erroneous rounds → empty retrieval."""
    rounds = _parse_tu_rounds(d)
    return any(
        t.get("tool_name")
        for rnd in rounds
        for t in (rnd.get("tools") or [])
    )


def load_tua_tools(logs_dir: Path) -> dict[str, dict[str, frozenset]]:
    """Returns {context_id: {perturbation: frozenset(tool_names)}}.
    Empty frozenset when tu_rounds is empty or erroneous."""
    data: dict[str, dict] = defaultdict(dict)
    for f in sorted(logs_dir.glob("detailed_*.json")):
        d = json.loads(f.read_text())
        cid  = str(d.get("context_id", ""))
        pert = d.get("perturbation", "unknown")
        rounds = _parse_tu_rounds(d)
        tools = frozenset(
            t.get("tool_name", "")
            for rnd in rounds
            for t in (rnd.get("tools") or [])
            if t.get("tool_name")
        )
        data[cid][pert] = tools
    return dict(data)


# stopwords to exclude from content Jaccard (common English + clinical filler)
_STOPWORDS = frozenset("""
a an the and or but if in on at to of for with by from is are was were be been
being have has had do does did will would could should may might must can shall
this that these those it its i we you he she they them their our your his her
as up out about into through over after before between no not also more some
all any each such than when where who which what how very just now then only
both so well very
""".split())


def _text_to_wordset(text: str) -> frozenset:
    """Lowercase, tokenise, drop stopwords and single-char tokens."""
    if not text or str(text).strip().lower() in ("nan", "none", ""):
        return frozenset()
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return frozenset(w for w in words if len(w) > 2 and w not in _STOPWORDS)


def load_tua_context(logs_dir: Path) -> dict[str, dict[str, frozenset]]:
    """Returns {context_id: {perturbation: frozenset(content_words)}}.
    Returns empty frozenset when tu_rounds is empty or erroneous, so that
    coverage is consistent with load_tua_tools (both gated on tu_rounds)."""
    data: dict[str, dict] = defaultdict(dict)
    for f in sorted(logs_dir.glob("detailed_*.json")):
        d = json.loads(f.read_text())
        cid  = str(d.get("context_id", ""))
        pert = d.get("perturbation", "unknown")
        if not _tu_rounds_have_tools(d):
            data[cid][pert] = frozenset()
            continue
        ctx  = d.get("tu_tool_context", "") or ""
        data[cid][pert] = _text_to_wordset(ctx)
    return dict(data)


def load_tua_context_raw(logs_dir: Path) -> dict[str, dict[str, str]]:
    """Returns {context_id: {perturbation: raw_text}} for TF-IDF / bio cosine.
    Returns empty string when tu_rounds is empty or erroneous."""
    data: dict[str, dict] = defaultdict(dict)
    for f in sorted(logs_dir.glob("detailed_*.json")):
        d = json.loads(f.read_text())
        cid  = str(d.get("context_id", ""))
        pert = d.get("perturbation", "unknown")
        if not _tu_rounds_have_tools(d):
            data[cid][pert] = ""
            continue
        ctx  = str(d.get("tu_tool_context", "") or "").strip()
        data[cid][pert] = ctx
    return dict(data)


def load_mra_paths_raw(logs_dir: Path) -> dict[str, dict[str, str]]:
    """Returns {context_id: {perturbation: raw_kg_paths_text}} for bio cosine."""
    data: dict[str, dict] = defaultdict(dict)
    for f in sorted(logs_dir.glob("detailed_*.json")):
        d = json.loads(f.read_text())
        cid  = str(d.get("context_id", ""))
        pert = d.get("perturbation", "unknown")
        raw  = str(d.get("kg_paths", "") or "").strip()
        # normalise "nan" sentinel to empty string
        data[cid][pert] = "" if raw.lower() == "nan" else raw
    return dict(data)


def compute_cosine_consistency(raw_data: dict, run_label: str = "") -> pd.DataFrame:
    """
    Fit a TF-IDF vectorizer over all tu_tool_context texts, then compute
    cosine similarity between baseline and each perturbation per context_id.
    Inclusion rule: both-non-empty (same as compute_consistency).
    """
    # collect all non-empty texts to fit the vectorizer
    all_texts = [
        text
        for pm in raw_data.values()
        for text in pm.values()
        if text and text.lower() not in ("nan", "none")
    ]
    if not all_texts:
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=2,           # ignore very rare terms
        max_df=0.95,        # ignore near-universal terms
        ngram_range=(1, 2), # unigrams + bigrams for richer signal
        sublinear_tf=True,  # log-scale TF to dampen high-freq terms
    )
    vectorizer.fit(all_texts)

    rows = []
    for cid, pm in raw_data.items():
        ref_text = pm.get("baseline", "")
        if not ref_text or ref_text.lower() in ("nan", "none"):
            continue
        ref_vec = vectorizer.transform([ref_text])

        for pert in PERTS:
            pert_text = pm.get(pert, "")
            if not pert_text or pert_text.lower() in ("nan", "none"):
                continue
            pert_vec = vectorizer.transform([pert_text])
            cos_sim  = float(sk_cosine(ref_vec, pert_vec)[0, 0])
            rows.append({
                "context_id":   cid,
                "perturbation": pert,
                "cosine":       cos_sim,
                "run":          run_label,
            })
    return pd.DataFrame(rows)


def compute_bio_cosine_consistency(raw_data: dict, run_label: str = "") -> pd.DataFrame:
    """
    Compute cosine similarity between baseline and perturbed tu_tool_context
    using PubMedBERT mean-pooled embeddings (L2-normalised → dot = cosine).
    Inclusion rule: both-non-empty, same as other metrics.
    """
    # gather all (cid, pert, text) triples where both baseline + pert are non-empty
    pairs: list[tuple[str, str, str, str]] = []  # (cid, pert, base_text, pert_text)
    for cid, pm in raw_data.items():
        ref = pm.get("baseline", "")
        if not ref or ref.lower() in ("nan", "none"):
            continue
        for pert in PERTS:
            pt = pm.get(pert, "")
            if pt and pt.lower() not in ("nan", "none"):
                pairs.append((cid, pert, ref, pt))

    if not pairs:
        return pd.DataFrame()

    # Embed every text without deduplication so that:
    # (a) mean-centering in _bio_embed reflects the true corpus distribution, and
    # (b) every valid pair is represented independently.
    # Layout: first n entries = baseline texts, next n = perturbed texts.
    n          = len(pairs)
    base_texts = [b for _, _, b, _ in pairs]
    pert_texts = [p for _, _, _, p in pairs]
    print(f"  [bio] embedding {n} pairs ({2 * n} texts, no dedup) …", flush=True)
    embeddings = _bio_embed(base_texts + pert_texts)   # (2n, 768) L2-normalised

    rows = []
    for i, (cid, pert, _, _) in enumerate(pairs):
        b_vec = embeddings[i]
        p_vec = embeddings[n + i]
        cos   = float(np.dot(b_vec, p_vec))   # == cosine (L2-normalised)
        rows.append({
            "context_id":   cid,
            "perturbation": pert,
            "cosine":       cos,
            "run":          run_label,
        })
    return pd.DataFrame(rows)


# ── metrics ────────────────────────────────────────────────────────────────────

def jaccard(a: frozenset, b: frozenset) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def compute_consistency(data: dict, run_label: str = "") -> pd.DataFrame:
    """
    For each (context_id, perturbation) pair where BOTH baseline and perturbed
    retrieval are non-empty, compute Jaccard and exact match vs baseline.
    """
    rows = []
    for cid, pm in data.items():
        if "baseline" not in pm:
            continue
        ref = pm["baseline"]
        if len(ref) == 0:
            continue                     # baseline empty — skip
        for pert in PERTS:
            if pert not in pm:
                continue
            p = pm[pert]
            if len(p) == 0:
                continue                 # perturbed empty — skip
            rows.append({
                "context_id":  cid,
                "perturbation": pert,
                "exact_match":  int(ref == p),
                "jaccard":      jaccard(ref, p),
                "run":          run_label,
            })
    return pd.DataFrame(rows)


# ── per-backbone driver ────────────────────────────────────────────────────────

def backbone_consistency(
    backbone: str,
    method: str,          # "mra" or "tua"
    run_spec: str = "0",  # "0", "1", "2", or "all"
) -> pd.DataFrame:
    """
    Returns a DataFrame averaged over requested runs with columns:
      perturbation, jaccard (mean), exact_match (mean), n
    """
    is_cosine = method in ("tua_cosine", "tua_cosine_bio", "mra_cosine_bio")

    if method == "mra":
        log_base = LOGS_ROOT / "medreason" / backbone / "oncqa_askdocs"
        loader   = load_mra_paths
    elif method == "mra_cosine_bio":
        log_base = LOGS_ROOT / "medreason" / backbone / "oncqa_askdocs"
        loader   = load_mra_paths_raw
    elif method == "tua_content":
        log_base = LOGS_ROOT / "tooluni" / backbone / "oncqa_askdocs"
        loader   = load_tua_context
    elif method in ("tua_cosine", "tua_cosine_bio"):
        log_base = LOGS_ROOT / "tooluni" / backbone / "oncqa_askdocs"
        loader   = load_tua_context_raw
    else:
        log_base = LOGS_ROOT / "tooluni" / backbone / "oncqa_askdocs"
        loader   = load_tua_tools

    if not log_base.exists():
        print(f"  [MISSING] {log_base}")
        return pd.DataFrame()

    runs = (
        [int(run_spec)] if run_spec != "all"
        else [int(d.name.replace("run_", ""))
              for d in sorted(log_base.iterdir())
              if d.is_dir() and d.name.startswith("run_")]
    )

    all_dfs = []
    for r in runs:
        run_dir = log_base / f"run_{r}"
        if not run_dir.exists():
            continue
        data = loader(run_dir)
        if method in ("tua_cosine_bio", "mra_cosine_bio"):
            df = compute_bio_cosine_consistency(data, run_label=str(r))
        elif method == "tua_cosine":
            df = compute_cosine_consistency(data, run_label=str(r))
        else:
            df = compute_consistency(data, run_label=str(r))
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    if is_cosine:
        per_case = (
            combined
            .groupby(["context_id", "perturbation"])[["cosine"]]
            .mean()
            .reset_index()
        )
        summary = (
            per_case
            .groupby("perturbation")
            .agg(
                cosine_mean=("cosine", "mean"),
                n          =("cosine", "count"),
            )
            .reindex(PERTS)
            .reset_index()
        )
        # add dummy cols so downstream pivot doesn't break
        summary["jaccard_mean"] = np.nan
        summary["exact_pct"]    = np.nan
    else:
        per_case = (
            combined
            .groupby(["context_id", "perturbation"])[["exact_match", "jaccard"]]
            .mean()
            .reset_index()
        )
        summary = (
            per_case
            .groupby("perturbation")
            .agg(
                jaccard_mean=("jaccard",      "mean"),
                exact_pct   =("exact_match",  lambda x: x.mean() * 100),
                n           =("jaccard",      "count"),
            )
            .reindex(PERTS)
            .reset_index()
        )
        summary["cosine_mean"] = np.nan

    summary["backbone"] = BACKBONE_LABEL.get(backbone, backbone)
    summary["method"]   = method.upper()
    return summary


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="0",
                    help="Which run(s) to use: 0, 1, 2, or 'all' (default: 0)")
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    args = ap.parse_args()

    all_rows = []
    for bb in args.backbones:
        label = BACKBONE_LABEL.get(bb, bb)
        print(f"\n{label}")
        print("-" * 60)
        for method in ("mra", "mra_cosine_bio", "tua", "tua_content", "tua_cosine", "tua_cosine_bio"):
            df = backbone_consistency(bb, method, run_spec=args.run)
            if df.empty:
                print(f"  {method.upper()}: no data")
                continue
            all_rows.append(df)
            for _, row in df.iterrows():
                mlabel = method.upper().replace("_", " ")
                if method in ("tua_cosine", "tua_cosine_bio", "mra_cosine_bio"):
                    print(f"  {mlabel:<14}  {row['perturbation']:<18} "
                          f"cosine={row['cosine_mean']:.3f}  "
                          f"(n={int(row['n'])})")
                else:
                    print(f"  {mlabel:<12}  {row['perturbation']:<18} "
                          f"jaccard={row['jaccard_mean']:.3f}  "
                          f"exact={row['exact_pct']:.1f}%  "
                          f"(n={int(row['n'])})")

    if not all_rows:
        print("No results."); return

    out_df = pd.concat(all_rows, ignore_index=True)

    # ── pivot table: one row per perturbation, columns = backbone × method ──
    print("\n\n" + "=" * 70)
    print("SUMMARY TABLE (Jaccard | Exact%)")
    print("=" * 70)
    pivot = out_df.pivot_table(
        index="perturbation",
        columns=["method", "backbone"],
        values=["jaccard_mean", "exact_pct"],
        aggfunc="first",
    ).reindex(PERTS)

    print(pivot.round(3).to_string())

    # ── save ────────────────────────────────────────────────────────────────
    out_csv = OUT_DIR / f"retrieval_consistency_run{args.run}.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()
