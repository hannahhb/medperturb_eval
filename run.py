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
import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be",
                        category=UserWarning)

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
        default=["gender_swap", "baseline", "summary", "uncertain_tone", "colorful_tone"],
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
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker threads (default: 1). "
             "For TOOLUNI, the pipeline semaphore caps concurrent tool pipelines "
             "regardless of worker count — set TOOLUNI_CONCURRENCY env var to tune.",
    )
    parser.add_argument(
        "--mode",
        default="MEDREASON",
        choices=["TXAGENT", "TXAGENT_BEDROCK", "AWS", "MEDREASON", "MEDPATHAGENT", "AGENTMD", "TOOLUNI", "KGRANK"],
        help=(
            "Runner mode (default: MEDREASON). "
            "MEDREASON = KG-augmented reasoning. "
            "MEDPATHAGENT = combined KG + tool-augmented model. "
            "TOOLUNI = FDA tool retrieval + iterative YES/NO reasoning. "
            "KGRANK = one-hop KG triplet retrieval with MMR/similarity/MedCPT ranking. "
            "AGENTMD requires setup_agentmd.py to be run first."
        ),
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Model name key from MODELS dict (used by TXAGENT_BEDROCK, AWS, MEDREASON, MEDPATHAGENT modes).",
    )
    parser.add_argument(
        "--kg-path",
        default=None,
        help="Path to kg.csv (PrimeKG). Required for MEDREASON mode.",
    )
    parser.add_argument(
        "--emb-path",
        default=None,
        help="Path to node_embeddings_sapbert.pt cache. Required for MEDREASON mode.",
    )
    parser.add_argument(
        "--medreason-csv",
        default=None,
        help=(
            "Path to a MedReason results CSV containing a 'kg_paths' column. "
            "Required for MEDPATHAGENT mode. Only rows with non-empty kg_paths are processed."
        ),
    )
    parser.add_argument(
        "--mpa-mode",
        default="dual",
        choices=["dual", "cached_dual", "paths_only"],
        help=(
            "MedPathAgent retrieval mode (default: dual). "
            "'dual' = retrieve from clinical context + KG paths separately (two live API calls). "
            "'cached_dual' = read clinical-context tool output from existing ToolUni logs "
            "(requires --tooluni-log-dir), run paths retrieval live, deduplicate and merge. "
            "'paths_only' = retrieve from KG paths text only (ablation)."
        ),
    )
    parser.add_argument(
        "--tooluni-log-dir",
        default=None,
        help=(
            "Path to directory containing ToolUni log JSONs (tool_<row_id>.json). "
            "Required when --mpa-mode cached_dual is set."
        ),
    )
    parser.add_argument(
        "--tooluni-results-csv",
        default=None,
        help=(
            "Path to a ToolUni results CSV.  In cached_dual mode, rows that are "
            "skipped (no kg_paths) are copied from this CSV into the MedPathAgent "
            "output so the results file is complete for all row IDs."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent runs (default: 1). "
             "Each run is saved to run_0, run_1, ... sub-directories.",
    )
    parser.add_argument(
        "--tooluni-cache-csv",
        default=None,
        help=(
            "Path to a prior ToolUni results CSV (must have tu_tool_context and row_id columns). "
            "When set for TOOLUNI mode, FDA tool calls are skipped; only the summarise "
            "and final-answer LLM steps are re-run fresh. Use for multi-run variance estimation."
        ),
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
        "MEDREASON": "llama70b3_3",
        "MEDPATHAGENT": "llama70b3_3",
        "AGENTMD": "llama70b3_3",
        "TOOLUNI": "llama70b3_3",
        "KGRANK":  "llama70b3_3",
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
    if args.mode in ("MEDREASON", "KGRANK"):
        mc = cfg.model_configs[0]
        if args.kg_path:
            mc.kg_path = args.kg_path
        if args.emb_path:
            mc.node_embeddings_path = args.emb_path
        if args.medreason_csv:
            # Reuse pre-extracted KG paths — skip KG/SapBERT/spaCy loading entirely
            cfg.medreason_csv_path = args.medreason_csv
            cfg.cached_paths_only = True

    if args.mode == "TOOLUNI":
        if args.tooluni_log_dir:
            cfg.tooluni_log_path = args.tooluni_log_dir
        if args.tooluni_cache_csv:
            cfg.tooluni_cache_csv_path = args.tooluni_cache_csv

    if args.mode == "MEDPATHAGENT":
        if not args.medreason_csv:
            raise ValueError("--medreason-csv is required for MEDPATHAGENT mode")
        cfg.medreason_csv_path = args.medreason_csv
        cfg.model_configs[0].mpa_retrieval_mode = args.mpa_mode
        if args.mpa_mode == "cached_dual":
            if not args.tooluni_log_dir:
                raise ValueError(
                    "--tooluni-log-dir is required when --mpa-mode cached_dual is set"
                )
            cfg.tooluni_log_path = args.tooluni_log_dir
            if args.tooluni_results_csv:
                cfg.tooluni_results_csv = args.tooluni_results_csv

    run_name = "_".join(cfg.datasets)
    cfg.max_workers = args.workers

    if args.debug:
        cfg.debug_mode = True
        cfg.sample_size = args.sample_size

    for run_idx in range(args.runs):
        cfg.output_path = f"./output/{cfg.primary_model}/{run_name}/run_{run_idx}"
        cfg.log_path    = f"./logs/{cfg.primary_model}/{run_name}/run_{run_idx}"

        mc = cfg.model_configs[0]
        print(f"\n{'='*60}")
        print(f"  MedPerturb Experiment  [{args.mode}]  run {run_idx + 1}/{args.runs}")
        print(f"  Model        : {mc.checkpoint or mc.name}")
        print(f"  Backend      : {mc.backend}")
        print(f"  Datasets     : {', '.join(cfg.datasets)}")
        print(f"  Perturbations: {', '.join(cfg.perturbations)}")
        print(f"  Temperature  : {mc.temperature}")
        if args.mode == "TXAGENT":
            print(f"  Device       : cuda:{mc.device_id}")
        if args.mode == "MEDREASON":
            if args.medreason_csv:
                print(f"  Cached paths : {args.medreason_csv}  (KG extraction skipped)")
            else:
                print(f"  KG path      : {mc.kg_path}")
                print(f"  Embeddings   : {mc.node_embeddings_path}")
        if args.mode == "MEDPATHAGENT":
            print(f"  MedReason CSV: {args.medreason_csv}")
            print(f"  Retrieval    : {args.mpa_mode}")
            if args.mpa_mode == "cached_dual":
                print(f"  ToolUni logs : {args.tooluni_log_dir}")
                if args.tooluni_results_csv:
                    print(f"  ToolUni CSV  : {args.tooluni_results_csv}")
        if args.mode == "AGENTMD":
            print(f"  Tools path   : {mc.agentmd_tools_path}")
            print(f"  LLM backend  : {mc.agentmd_llm or 'auto (OpenAI if key set, else Bedrock)'}")
            print(f"  LLM model    : {mc.agentmd_llm_model}")
        if args.mode == "TOOLUNI":
            print(f"  ToolRAG      : {mc.use_toolrag}")
            if mc.use_toolrag:
                print(f"  ToolRAG model: {mc.toolrag_model}")
                print(f"  ToolRAG dev  : {mc.toolrag_device}")
            if args.tooluni_cache_csv:
                print(f"  Cached tools : {args.tooluni_cache_csv}  (FDA pipeline skipped, summarise+answer re-run)")
            elif args.tooluni_log_dir:
                print(f"  Cached tools : {args.tooluni_log_dir}  (tool pipeline skipped)")
        print(f"  Workers      : {args.workers}")
        print(f"  Slice        : {args.start:.2f} - {args.end:.2f}")
        print(f"  Output       : {cfg.output_path}")
        print(f"{'='*60}\n")

        processor = MedPerturbProcessor(cfg)
        df = processor.process(resume=not args.no_resume)

        print(f"\n{'='*60}")
        print(f"  DONE run {run_idx + 1}/{args.runs}  -  {len(df)} rows processed")
        print(f"  Results: {cfg.output_path}/results.csv")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
