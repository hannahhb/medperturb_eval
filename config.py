from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel, Field
import os


class ModelType(Enum):
    BEDROCK = "bedrock"
    TX_AGENT = "txagent"


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


class AdvancedConfig(BaseModel):
    # Dataset
    hf_dataset_url: str = "https://huggingface.co/datasets/abinitha/MedPerturb/resolve/main/data.csv"
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

    # Output paths
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
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            rag_model=bedrock_model_id,     # ToolRAGModelBedrock uses same model
            max_tokens=max_tokens,
            temperature=0.3,
            call_agent=False,
            max_round=20,
            max_token=91000,
            enable_summary=False,
            enable_entity_awareness=False,  # spaCy NER not needed for Bedrock mode
        )
        cfg.primary_model = f"txagent_bedrock_{name}"

    elif mode == "AWS":
        # Plain Bedrock — no TxAgent tool-calling loop
        model_id = MODELS[name]
        model_cfg = ModelConfig(
            name=name,
            type=ModelType.BEDROCK,
            backend="bedrock",
            weight=2.0,
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            bedrock_model_id=model_id,
            max_tokens=max_tokens,
        )

    else:
        raise ValueError(f"Unknown mode: {mode!r}. Supported: AWS, TXAGENT, TXAGENT_BEDROCK")

    cfg.model_configs = [model_cfg]
    return cfg
