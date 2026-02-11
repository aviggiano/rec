from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rec_pipeline.asr import ASRError, ASRPipeline, FasterWhisperTranscriber
from rec_pipeline.config import load_settings
from rec_pipeline.ingestion import IngestionProcessor
from rec_pipeline.summary import SummaryError, SummaryPipeline
from rec_pipeline.transcript import TranscriptArtifactBuilder, TranscriptError


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
    run_parser.add_argument(
        "--asr-model-size",
        default=None,
        help="faster-whisper model size (default from REC_ASR_MODEL_SIZE)",
    )
    run_parser.add_argument(
        "--asr-device",
        default=None,
        help="ASR device for faster-whisper (default from REC_ASR_DEVICE)",
    )
    run_parser.add_argument(
        "--asr-compute-type",
        default=None,
        help="ASR compute type for faster-whisper (default from REC_ASR_COMPUTE_TYPE)",
    )
    run_parser.add_argument(
        "--asr-beam-size",
        type=int,
        default=None,
        help="ASR decoding beam size (default from REC_ASR_BEAM_SIZE)",
    )
    run_parser.add_argument(
        "--asr-vad-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable VAD filter for ASR (default from REC_ASR_VAD_FILTER)",
    )
    run_parser.add_argument(
        "--asr-max-retries",
        type=int,
        default=None,
        help="Number of retries per file on transient ASR failures",
    )
    run_parser.add_argument(
        "--summary-local-backend",
        default=None,
        help="Local summary backend: ollama, llamacpp, or heuristic",
    )
    run_parser.add_argument(
        "--summary-model",
        default=None,
        help="Model name for local summary backends that require one",
    )
    run_parser.add_argument(
        "--summary-max-chunk-tokens",
        type=int,
        default=None,
        help="Maximum estimated tokens per transcript chunk for summarization",
    )
    run_parser.add_argument(
        "--summary-max-chunk-seconds",
        type=int,
        default=None,
        help="Maximum transcript seconds per summarization chunk",
    )
    run_parser.set_defaults(handler=_handle_run)

    return parser


def _handle_run(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file)
    language = args.lang or settings.default_language
    run_dir = args.output / args.run_name
    should_fail_fast = args.fail_fast or not settings.continue_on_error
    processor = IngestionProcessor(fail_fast=should_fail_fast)
    ingestion_result = processor.process_directory(args.input, run_dir)
    print(f"Ingestion completed. run_dir={run_dir}")
    print(
        "Ingestion summary: "
        f"files={len(ingestion_result.files)} "
        f"normalized={ingestion_result.normalized_count} "
        f"skipped={ingestion_result.skipped_count} "
        f"errors={len(ingestion_result.errors)}"
    )
    transcriber = FasterWhisperTranscriber(
        model_size=args.asr_model_size or settings.asr_model_size,
        device=args.asr_device or settings.asr_device,
        compute_type=args.asr_compute_type or settings.asr_compute_type,
    )
    asr_pipeline = ASRPipeline(
        transcriber=transcriber,
        beam_size=args.asr_beam_size or settings.asr_beam_size,
        vad_filter=(
            settings.asr_vad_filter if args.asr_vad_filter is None else bool(args.asr_vad_filter)
        ),
        max_retries=args.asr_max_retries or settings.asr_max_retries,
        retry_backoff_sec=0.5,
        fail_fast=should_fail_fast,
    )
    try:
        asr_result = asr_pipeline.transcribe_run(run_dir=run_dir, language=language)
    except ASRError as exc:
        print(f"ASR stage failed: {exc}")
        return 1
    print(
        "ASR summary: "
        f"files={len(asr_result.files)} "
        f"transcribed={asr_result.transcribed_count} "
        f"skipped={asr_result.skipped_count} "
        f"errors={len(asr_result.errors)}"
    )
    transcript_builder = TranscriptArtifactBuilder()
    try:
        transcript_result = transcript_builder.build(run_dir=run_dir, language=language)
    except TranscriptError as exc:
        print(f"Transcript assembly failed: {exc}")
        return 1
    print(
        "Transcript summary: "
        f"segments={transcript_result.segment_count} "
        f"generated={len(transcript_result.generated_files)} "
        f"skipped={len(transcript_result.skipped_files)}"
    )
    summary_pipeline = SummaryPipeline(
        model_backend=args.summary_local_backend or settings.summary_local_backend,
        model_name=args.summary_model or settings.summary_model_name,
        ollama_base_url=settings.ollama_base_url,
        llamacpp_server_url=settings.llamacpp_server_url,
        max_chunk_tokens=args.summary_max_chunk_tokens or settings.summary_max_chunk_tokens,
        max_chunk_seconds=args.summary_max_chunk_seconds or settings.summary_max_chunk_seconds,
        fail_fast=should_fail_fast,
    )
    try:
        summary_result = summary_pipeline.build(run_dir=run_dir, language=settings.output_language)
    except SummaryError as exc:
        print(f"Summarization failed: {exc}")
        return 1
    print(
        "Summary stage: "
        f"generated={len(summary_result.generated_files)} "
        f"skipped={len(summary_result.skipped_files)}"
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
