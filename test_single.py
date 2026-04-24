"""
Run TxAgent on a single MedPerturb question.

LLM backend : AWS Bedrock  (us.meta.llama3-3-70b-instruct-v1:0)
RAG backend : HuggingFace  (mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B)

Usage:
    python test_single.py
    python test_single.py --idx 42          # pick a different row
    python test_single.py --perturbation gender_swap
    python test_single.py --local data/medperturb_data.csv
"""
import argparse
import os
import sys

# Make sure TxAgent source is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TxAgent_tool_tracking"))

import pandas as pd
from txagent import TxAgent
from txagent.toolrag import ToolRAGModel

from medperturb_eval.prompts import build_prompt, SYSTEM_PROMPT


BEDROCK_MODEL_ID = "us.meta.llama3-3-70b-instruct-v1:0"
HF_RAG_MODEL     = "mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B"
HF_DATASET_URL   = "https://huggingface.co/datasets/abinitha/MedPerturb/resolve/main/data.csv"


def load_row(local_path, idx, perturbation):
    src = local_path or HF_DATASET_URL
    print(f"Loading dataset from: {src}")
    df = pd.read_csv(src)
    if perturbation:
        df = df[df["perturbation"] == perturbation].reset_index(drop=True)
    if idx >= len(df):
        raise IndexError(f"Row index {idx} out of range (dataset has {len(df)} rows after filtering)")
    row = df.iloc[idx]
    return row


def build_agent(region: str, rag_device: str) -> TxAgent:
    """
    Construct TxAgent with:
      - main LLM  → AWS Bedrock (Llama 70B)
      - RAG model → HuggingFace SentenceTransformer (local GPU)
    """
    print(f"\n[1/3] Creating TxAgent (Bedrock LLM + HuggingFace RAG) ...")
    agent = TxAgent(
        model_name=BEDROCK_MODEL_ID,       # passed as model_name but overridden by bedrock_model_id
        rag_model_name=HF_RAG_MODEL,       # will be overridden below with HF model
        use_bedrock=True,
        bedrock_model_id=BEDROCK_MODEL_ID,
        bedrock_region=region,
        enable_summary=False,
        enable_entity_awareness=False,
        force_finish=True,
        init_rag_num=0,
        step_rag_num=10,
    )

    # Override the Bedrock-based RAG with the HuggingFace embedding model.
    # This must happen BEFORE init_model() so load_tool_desc_embedding() uses it.
    print(f"[2/3] Swapping RAG to HuggingFace: {HF_RAG_MODEL} on {rag_device}")
    agent.rag_model = ToolRAGModel(HF_RAG_MODEL, device=rag_device)

    print(f"[3/3] Loading tools and computing embeddings ...")
    agent.init_model()  # load_models() → Bedrock client; load_tool_desc_embedding() → HF RAG

    return agent


def run(args):
    row = load_row(args.local, args.idx, args.perturbation)

    print(f"\n{'='*60}")
    print(f"  context_id  : {row['context_id']}")
    print(f"  perturbation: {row['perturbation']}")
    print(f"  dataset     : {row['dataset']}")
    print(f"  gold manage : {row.get('gold_standard_manage', 'N/A')}")
    print(f"  gold visit  : {row.get('gold_standard_visit', 'N/A')}")
    print(f"  gold resource:{row.get('gold_standard_resource', 'N/A')}")
    print(f"{'='*60}\n")
    print("Clinical context:")
    print(row["clinical_context"])
    print()

    region     = os.environ.get("AWS_REGION", "us-east-1")
    rag_device = f"cuda:{args.rag_device}" if args.rag_device is not None else "cpu"

    agent = build_agent(region, rag_device)

    prompt = build_prompt(row["clinical_context"])

    print(f"\n{'='*60}")
    print("Running TxAgent ...")
    print(f"{'='*60}\n")

    answer = agent.run_multistep_agent(
        prompt,
        temperature=0.3,
        max_new_tokens=2048,
        max_token=91000,
        max_round=20,
    )

    print(f"\n{'='*60}")
    print("FINAL ANSWER")
    print(f"{'='*60}")
    print(answer)

    if agent.last_used_tools:
        print(f"\n--- Tools used ({len(agent.last_used_tools)}) ---")
        for t in agent.last_used_tools:
            print(f"  [{t.get('stage','?')}] {t.get('name','?')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx",          type=int,   default=0,
                        help="Row index into (filtered) dataset (default: 0)")
    parser.add_argument("--perturbation", type=str,   default="baseline",
                        help="Filter to this perturbation type (default: baseline)")
    parser.add_argument("--local",        type=str,   default=None,
                        help="Path to local data.csv (skips HuggingFace download)")
    parser.add_argument("--rag_device",   type=int,   default=None,
                        help="CUDA device index for the HF RAG model (default: cpu). Pass 0, 1, etc. to use GPU.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
