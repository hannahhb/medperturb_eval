from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel, Field
import os


class ModelType(Enum):
    BEDROCK = "bedrock"
    TX_AGENT = "txagent"
    MEDREASON = "medreason"            # KG-augmented reasoning (was MEDPATH_AGENT)
    MEDPATH_AGENT = "medpathagent"     # NEW: combined KG + tool-augmented model
    KGRANK        = "kgrank"           # KGRank: one-hop triplet retrieval with MMR/similarity/MedCPT ranking
    AGENT_MD = "agentmd"
    TOOLUNI = "tooluni"                # FDA tool retrieval + iterative YES/NO reasoning


MODELS = {
    # AWS Bedrock models
    "llama70b3_3": "us.meta.llama3-3-70b-instruct-v1:0",
    "llama8b3_1": "us.meta.llama3-1-8b-instruct-v1:0",
    "gpt_oss_120b": "openai.gpt-oss-120b-1:0",
    "deepseekr1": "us.deepseek.r1-v1:0",
    "qwen235b": "qwen.qwen3-235b-a22b-2507-v1:0",
}


@dataclass
class ModelConfig:
    name: str
    type: ModelType
    weight: float = 1.0
    max_tokens: int = 2048
    temperature: float = 0.2
    device: str = "cuda"
    checkpoint: Optional[str] = None
    api_key: Optional[str] = None
    endpoint: Optional[str] = None

    region_name: Optional[str] = None
    bedrock_model_id: Optional[str] = None

    # TxAgent extras
    rag_model: Optional[str] = None
    enable_summary: bool = False
    call_agent: bool = False
    max_round: int = 20
    max_token: int = 90240
    enable_entity_awareness: bool = False
    backend: str = "huggingface"   # "huggingface" (vLLM local) | "bedrock" (AWS)
    device_id: int = 0             # GPU device index for huggingface backend

    # ToolUni / ToolRAG extras
    use_toolrag: bool = False
    toolrag_model: str = "mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B"
    toolrag_device: str = "cpu"          # "cpu" | "mps" (Apple Silicon) | "cuda:0"
                                         # NOTE: MPS shares unified RAM with CPU — no memory benefit on Apple Silicon
    toolrag_cache_dir: Optional[str] = None   # defaults to tooluni_agent package dir

    # MedPathAgent extras
    kg_path: Optional[str] = None
    node_embeddings_path: Optional[str] = None
    sapbert_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    mpa_retrieval_mode: str = "dual"   # "dual" | "paths_only"

    # AgentMD extras
    agentmd_tools_path: Optional[str] = None   # path to riskcalcs.json
    agentmd_llm: Optional[str] = None          # "openai" | "bedrock" (auto-detected if None)
    agentmd_llm_model: Optional[str] = None    # overrides bedrock_model_id / openai model


DATA_DIR = "/Users/hannah_mac/Documents/rmit/rmit_hons_y4/data"


class AdvancedConfig(BaseModel):
    # Dataset
    hf_dataset_url: str = f"{DATA_DIR}/medperturb_data.csv"
    datasets: List[str] = Field(
        default_factory=lambda: ["oncqa", "askdocs"]
    )
    perturbations: List[str] = Field(
        default_factory=lambda: [
            "baseline", 
            "summary", 
            # "colorful_tone",
            "uncertain_tone",
            "gender_swap", 
            # "gender_removal",
        ]
    )

    # Paths
    data_path: str = DATA_DIR
    output_path: str = "./output/"
    log_path: str = "./logs/"

    # Model / run settings
    primary_model: str = "llama70b3_3"
    model_configs: List[ModelConfig] = Field(default_factory=list)

    # Bedrock concurrency
    bedrock_max_connections: int = 50
    bedrock_client_pool_size: int = 3
    max_workers: int = 8
    rate_limit_delay: float = 0.5
    max_retries: int = 3

    # Dataset slicing for parallel runs
    slice_start: float = 0.0  # proportion [0, 1]
    slice_end: float = 1.0

    # Debug
    debug_mode: bool = False
    sample_size: Optional[int] = 20

    # Resume
    save_intermediate: bool = True

    # MedPathAgent / MedReason: path to a results CSV with a 'kg_paths' column.
    # When set for MEDREASON mode, paths are reused and KG/SapBERT are not loaded.
    medreason_csv_path: Optional[str] = None

    # When True, MedReasonModel skips KG/SapBERT/spaCy loading entirely and
    # relies solely on pre-cached kg_paths from medreason_csv_path.
    cached_paths_only: bool = False

    # ToolUni cached mode: path to a prior ToolUni results CSV.
    # Loads tu_tool_context per row_id so the FDA tool pipeline is skipped;
    # only the summarise + final-answer LLM steps are re-run.
    tooluni_cache_csv_path: Optional[str] = None

    # MedPathAgent cached_dual mode: directory containing ToolUni log JSONs
    # (tool_<row_id>.json).  Set via --tooluni-log-dir.
    tooluni_log_path: Optional[str] = None

    # MedPathAgent cached_dual mode: path to the ToolUni results CSV.
    # Rows missing from the MedPathAgent output (no kg_paths) are copied from
    # here with prediction columns renamed to the MedPathAgent model name.
    # Set via --tooluni-results-csv.
    tooluni_results_csv: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


def get_default_config(
    start: float = 0.0,
    end: float = 1.0,
    max_tokens: int = 2048,
    name: str = "txagent_hf_oncqa_askdocs",
    mode: str = "TXAGENT",
) -> AdvancedConfig:
    """
    mode options:
      "AWS"            - AWS Bedrock standalone (no TxAgent)
      "TXAGENT"        - TxAgent with local HuggingFace/vLLM backend (GPU required)
      "TXAGENT_BEDROCK"- TxAgent with AWS Bedrock as the LLM backend (no GPU needed)
    """
    cfg = AdvancedConfig()
    cfg.slice_start = start
    cfg.slice_end = end
    cfg.primary_model = name

    run_id = "run_0"
    cfg.output_path = os.path.join("./output", name, run_id)
    cfg.log_path = os.path.join("./logs", name, run_id)

    if mode == "TXAGENT":
        # Default MedPerturb experiment: local TxAgent on oncqa + askdocs.
        model_cfg = ModelConfig(
            name="txagent",
            type=ModelType.TX_AGENT,
            backend="huggingface",
            checkpoint="mims-harvard/TxAgent-T1-Llama-3.1-8B",
            rag_model="mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B",
            max_tokens=max_tokens,
            temperature=0.3,
            call_agent=False,
            max_round=20,
            max_token=91000,
            enable_summary=False,
            enable_entity_awareness=True,
            device_id=0,
        )
        cfg.primary_model = "txagent"

    elif mode == "TXAGENT_BEDROCK":
        # TxAgent logic (tool calling, multi-step reasoning) but LLM calls go
        # to AWS Bedrock — no GPU or vLLM required.
        bedrock_model_id = MODELS.get(name, name)  # accept raw ARN too
        model_cfg = ModelConfig(
            name=f"txagent_bedrock_{name}",
            type=ModelType.TX_AGENT,
            backend="bedrock",
            checkpoint=bedrock_model_id,    # passed to TxAgent as model_name
            bedrock_model_id=bedrock_model_id,
            region_name=os.environ.get("AWS_REGION", "us-east-2"),
            rag_model=bedrock_model_id,     # ToolRAGModelBedrock uses same model
            max_tokens=max_tokens,
            temperature=0.3,
            call_agent=False,
            max_round=20,
            max_token=91000,
            enable_summary=False,
            enable_entity_awareness=False,  # spaCy NER not needed for Bedrock mode
        )
        cfg.primary_model = f"txagent_bedrock/{name}"

    elif mode == "AWS":
        # Plain Bedrock — no TxAgent tool-calling loop
        model_id = MODELS[name]
        model_cfg = ModelConfig(
            name=name,
            type=ModelType.BEDROCK,
            backend="bedrock",
            weight=2.0,
            region_name=os.environ.get("AWS_REGION", "us-east-2"),
            bedrock_model_id=model_id,
            max_tokens=max_tokens,
        )
        cfg.primary_model = f"baseline/{name}"


    elif mode == "MEDREASON":
        # MedReason: KG-augmented reasoning + Bedrock LLM, no GPU required.
        # Requires kg_path and node_embeddings_path to be set via CLI or overridden after this call.
        # (Previously named "MEDPATHAGENT" — renamed because the new combined model is now MedPathAgent.)
        bedrock_model_id = MODELS.get(name, name)
        model_cfg = ModelConfig(
            name=f"medreason_{name}",
            type=ModelType.MEDREASON,
            backend="medreason",
            checkpoint=bedrock_model_id,
            bedrock_model_id=bedrock_model_id,
            region_name=os.environ.get("AWS_REGION", "us-east-2"),
            kg_path=f"{DATA_DIR}/kg.csv",
            node_embeddings_path=f"{DATA_DIR}/node_embeddings_sapbert.pt",
            max_tokens=max_tokens,
            temperature=0.3,
        )
        cfg.primary_model = f"medreason/{name}"

    elif mode == "MEDPATHAGENT":
        # MedPathAgent: NEW combined KG + tool-augmented model.
        # Uses MedReason's KG infrastructure (PrimeKG, spaCy NER, SapBERT) to extract
        # entities and KG paths, then uses ToolUni's FDA tools (ToolUniverse) for
        # factual retrieval. Both signals feed into iterative reason_multistep
        # decisions for MANAGE/VISIT/RESOURCE.
        bedrock_model_id = MODELS.get(name, name)
        model_cfg = ModelConfig(
            name=f"medpathagent_{name}",
            type=ModelType.MEDPATH_AGENT,
            backend="medpathagent",
            checkpoint=bedrock_model_id,
            bedrock_model_id=bedrock_model_id,
            region_name=os.environ.get("AWS_REGION", "us-east-2"),
            kg_path=f"{DATA_DIR}/kg.csv",
            node_embeddings_path=f"{DATA_DIR}/node_embeddings_sapbert.pt",
            max_tokens=max_tokens,
            temperature=0.3,
        )
        cfg.primary_model = f"medpathagent/{name}"

    elif mode == "AGENTMD":
        # AgentMD: MedCPT tool retrieval + iterative code execution + Bedrock/OpenAI LLM.
        # Requires riskcalcs.json (run: python scripts/setup_agentmd.py)
        bedrock_model_id = MODELS.get(name, name)
        model_cfg = ModelConfig(
            name=f"agentmd_{name}",
            type=ModelType.AGENT_MD,
            backend="agentmd",
            checkpoint=bedrock_model_id,
            bedrock_model_id=bedrock_model_id,
            region_name=os.environ.get("AWS_REGION", "us-east-2"),
            agentmd_tools_path=os.path.join(DATA_DIR, "agentmd", "riskcalcs.json"),
            agentmd_llm=None,          # auto-detect from env (openai key → OpenAI, else Bedrock)
            agentmd_llm_model=bedrock_model_id,
            max_round=20,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        cfg.primary_model = f"agentmd/{name}"

    elif mode == "TOOLUNI":
        # ToolUni: FDA tool retrieval (ToolUniverse) + iterative YES/NO reasoning.
        # For MedPerturb: runs full tool-gathering + reason_multistep pipeline.
        # For MedXpertQA MCQ: falls back to a direct LLM call (two-turn CoT handled by caller).
        # Set use_toolrag=True to use dense ToolRAG retrieval instead of keyword matching.
        bedrock_model_id = MODELS.get(name, name)
        model_cfg = ModelConfig(
            name=f"tooluni_{name}",
            type=ModelType.TOOLUNI,
            backend="tooluni",
            bedrock_model_id=bedrock_model_id,
            region_name=os.environ.get("AWS_REGION", "us-east-2"),
            max_tokens=max_tokens,
            temperature=0.0,
            use_toolrag=True,                                           # flip to True to enable
            toolrag_model="mims-harvard/ToolRAG-T1-GTE-Qwen2-1.5B",
            toolrag_device="cpu",                                        # MPS shares unified RAM — no benefit
            toolrag_cache_dir=DATA_DIR,
        )
        cfg.primary_model = f"tooluni/{name}"

    elif mode == "KGRANK":
        # KGRank: one-hop KG triplet retrieval with MMR/similarity/MedCPT re-ranking + Bedrock LLM.
        bedrock_model_id = MODELS.get(name, name)
        model_cfg = ModelConfig(
            name=f"kgrank_{name}",
            type=ModelType.KGRANK,
            backend="kgrank",
            checkpoint=bedrock_model_id,
            bedrock_model_id=bedrock_model_id,
            region_name=os.environ.get("AWS_REGION", "us-east-2"),
            kg_path=f"{DATA_DIR}/kg.csv",
            node_embeddings_path=f"{DATA_DIR}/node_embeddings_sapbert.pt",
            max_tokens=max_tokens,
            temperature=0.3,
            kgrank_method="mmr",          # "similarity" | "mmr" | "rerank"
            kgrank_top_p=20,              # top-p triplets to keep
            kgrank_mmr_w_base=0.5,        # MMR diversity weight base
            kgrank_mmr_delta=0.05,        # MMR weight increment per selected
            kgrank_rerank_top_n=50,       # MedCPT rerank candidate pool
            medcpt_model="ncats/MedCPT-Cross-Encoder",
        )
        cfg.primary_model = f"kgrank/{name}"

    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. "
            "Supported: AWS, TXAGENT, TXAGENT_BEDROCK, MEDREASON, MEDPATHAGENT, AGENTMD, TOOLUNI, KGRANK"
        )

    cfg.model_configs = [model_cfg]
    return cfg
