"""CLI entry point.

Usage:
  python -m equitas --config <yaml>                          # run experiment
  python -m equitas batch --config <yaml>                    # batch API mode
  python -m equitas analyze regime-map --results-dir <dir>   # regime analysis
  python -m equitas report --results-dir <dir>               # JSON report
"""
import sys

from dotenv import load_dotenv

load_dotenv()

from equitas.experiments.runner import main as run_main
from equitas.analysis_cli import main as analyze_main
from equitas.report_cli import main as report_main


def batch_main() -> None:
    """Run experiment via OpenAI Batch API (3-stage pipeline)."""
    import argparse

    # Remove 'batch' from argv so argparse sees --config
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    parser = argparse.ArgumentParser(description="Equitas batch pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing stage results (skip completed stages)")
    args = parser.parse_args()

    from equitas.config import load_config, validate_config

    config = load_config(args.config)
    validate_config(config)

    if config.experiment_type == "sweep_fh":
        from equitas.batch.pipeline_fh import FHBatchSweepPipeline
        pipeline = FHBatchSweepPipeline(config)
    else:
        from equitas.batch.pipeline import BatchSweepPipeline
        pipeline = BatchSweepPipeline(config)

    pipeline.run(resume=args.resume)


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "analyze":
        analyze_main()
    elif len(sys.argv) >= 2 and sys.argv[1] == "report":
        report_main()
    elif len(sys.argv) >= 2 and sys.argv[1] == "batch":
        batch_main()
    else:
        run_main()


if __name__ == "__main__":
    main()
