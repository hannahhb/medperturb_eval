"""
MedPerturb TxAgent experiment runner.

Default experiment:
  - model: TxAgent local HuggingFace backend
  - datasets: oncqa + askdocs
  - perturbations: baseline, summary, uncertain_tone, gender_swap

Examples:
  python run.py
  python run.py --datasets oncqa askdocs --perturbations baseline gender_swap
  python run.py --local-csv data/medperturb_data.csv --device 1
  python run.py --start 0.0 --end 0.5
"""
import argparse

from scripts.logging_setup import setup_logging
from config import get_default_config
from processor import MedPerturbProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MedPerturb TxAgent experiment")
    parser.add_argument(
        "--local-csv",
        default=None,
        help="Optional local CSV path. If omitted, the HuggingFace MedPerturb URL is used.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["oncqa", "askdocs"],
        help="Datasets to include (default: oncqa askdocs).",
    )
    parser.add_argument(
        "--perturbations",
        nargs="+",
        default=["baseline", "summary", "uncertain_tone", "gender_swap"],
        help="Perturbations to include.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index for the local TxAgent backend (default: 0).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max generation tokens per call.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Slice start proportion [0, 1] for parallel execution.",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=1.0,
        help="Slice end proportion [0, 1] for parallel execution.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: process a small sample.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10,
        help="Sample size used when --debug is set.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resume and reprocess all rows.",
    )
    parser.add_argument(
        "--mode",
        default="MEDPATHAGENT",
        choices=["TXAGENT", "TXAGENT_BEDROCK", "AWS", "MEDPATHAGENT", "AGENTMD"],
        help="Runner mode (default: MEDPATHAGENT). AGENTMD requires setup_agentmd.py to be run first.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Model name key from MODELS dict (used by TXAGENT_BEDROCK, AWS, MEDPATHAGENT modes).",
    )
    parser.add_argument(
        "--kg-path",
        default=None,
        help="Path to kg.csv (PrimeKG). Required for MEDPATHAGENT mode.",
    )
    parser.add_argument(
        "--emb-path",
        default=None,
        help="Path to node_embeddings_sapbert.pt cache. Required for MEDPATHAGENT mode.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logger = setup_logging()
    logger.info("Starting MedPerturb experiment (mode=%s)", args.mode)

    default_name = {
        "TXAGENT": "txagent_hf_oncqa_askdocs",
        "TXAGENT_BEDROCK": "llama70b3_3",
        "AWS": "llama70b3_3",
        "MEDPATHAGENT": "llama70b3_3",
        "AGENTMD": "llama70b3_3",
    }[args.mode]
    name = args.name or default_name

    cfg = get_default_config(
        start=args.start,
        end=args.end,
        max_tokens=args.max_tokens,
        name=name,
        mode=args.mode,
    )

    if args.local_csv:
        cfg.hf_dataset_url = args.local_csv
    cfg.datasets = args.datasets
    cfg.perturbations = args.perturbations
    if args.mode == "TXAGENT":
        cfg.model_configs[0].device_id = args.device
    if args.mode == "MEDPATHAGENT":
        mc = cfg.model_configs[0]
        if args.kg_path:
            mc.kg_path = args.kg_path
        if args.emb_path:
            mc.node_embeddings_path = args.emb_path

    run_name = "_".join(cfg.datasets)
    cfg.output_path = f"./output/{cfg.primary_model}/{run_name}/run_0"
    cfg.log_path = f"./logs/{cfg.primary_model}/{run_name}/run_0"

    if args.debug:
        cfg.debug_mode = True
        cfg.sample_size = args.sample_size

    mc = cfg.model_configs[0]
    print(f"\n{'='*60}")
    print(f"  MedPerturb Experiment  [{args.mode}]")
    print(f"  Model        : {mc.checkpoint or mc.name}")
    print(f"  Backend      : {mc.backend}")
    print(f"  Datasets     : {', '.join(cfg.datasets)}")
    print(f"  Perturbations: {', '.join(cfg.perturbations)}")
    if args.mode == "TXAGENT":
        print(f"  Device       : cuda:{mc.device_id}")
    if args.mode == "MEDPATHAGENT":
        print(f"  KG path      : {mc.kg_path}")
        print(f"  Embeddings   : {mc.node_embeddings_path}")
    if args.mode == "AGENTMD":
        print(f"  Tools path   : {mc.agentmd_tools_path}")
        print(f"  LLM backend  : {mc.agentmd_llm or 'auto (OpenAI if key set, else Bedrock)'}")
        print(f"  LLM model    : {mc.agentmd_llm_model}")
    print(f"  Slice        : {args.start:.2f} - {args.end:.2f}")
    print(f"  Output       : {cfg.output_path}")
    print(f"{'='*60}\n")

    processor = MedPerturbProcessor(cfg)
    df = processor.process(resume=not args.no_resume)

    print(f"\n{'='*60}")
    print(f"  DONE  -  {len(df)} rows processed")
    print(f"  Results: {cfg.output_path}/results.csv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
