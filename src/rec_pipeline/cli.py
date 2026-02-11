from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rec_pipeline.config import load_settings
from rec_pipeline.ingestion import IngestionProcessor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rec",
        description="Local-first long-form recording transcription and summarization",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to a custom .env file (defaults to .env in the current working directory)",
    )

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the processing pipeline")
    run_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory with recordings",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts"),
        help="Output directory where run artifacts are created",
    )
    run_parser.add_argument(
        "--run-name",
        default="default",
        help="Run folder name under --output (re-use for resumable runs)",
    )
    run_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop processing on first ingestion error",
    )
    run_parser.add_argument(
        "--lang",
        default=None,
        help="Language override for transcription (defaults to REC_DEFAULT_LANG)",
    )
    run_parser.set_defaults(handler=_handle_run)

    return parser


def _handle_run(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file)
    language = args.lang or settings.default_language
    run_dir = args.output / args.run_name
    processor = IngestionProcessor(fail_fast=args.fail_fast or not settings.continue_on_error)
    ingestion_result = processor.process_directory(args.input, run_dir)
    print(f"Ingestion completed. run_dir={run_dir}")
    print(
        "Ingestion summary: "
        f"files={len(ingestion_result.files)} "
        f"normalized={ingestion_result.normalized_count} "
        f"skipped={ingestion_result.skipped_count} "
        f"errors={len(ingestion_result.errors)}"
    )
    print(
        f"Pipeline scaffold ready. asr_provider={settings.asr_provider} "
        f"summary_provider={settings.summary_provider} lang={language}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0

    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
