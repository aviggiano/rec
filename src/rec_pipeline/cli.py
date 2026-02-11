from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rec_pipeline.config import load_settings


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
        "--lang",
        default=None,
        help="Language override for transcription (defaults to REC_DEFAULT_LANG)",
    )
    run_parser.set_defaults(handler=_handle_run)

    return parser


def _handle_run(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file)
    language = args.lang or settings.default_language
    print(
        f"Pipeline scaffold ready. input={args.input} asr_provider={settings.asr_provider} "
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
