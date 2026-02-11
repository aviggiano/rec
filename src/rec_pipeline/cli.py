from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from rec_pipeline.asr import ASRError, ASRPipeline, FasterWhisperTranscriber, Transcriber
from rec_pipeline.config import load_settings
from rec_pipeline.diarization import (
    DiarizationError,
    PyannoteDiarizer,
    SpeakerDiarizationPipeline,
)
from rec_pipeline.evaluation import EvaluationError, default_thresholds, evaluate_dataset
from rec_pipeline.ingestion import IngestionProcessor
from rec_pipeline.providers import (
    ExternalASRTranscriber,
    ExternalSummaryModel,
    ProviderConfigError,
    is_external_provider,
    resolve_provider_key,
    validate_provider_configuration,
)
from rec_pipeline.summary import SummaryError, SummaryModel, SummaryPipeline
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
        "--asr-provider",
        choices=["local", "openai", "deepgram", "groq"],
        default=None,
        help="ASR provider override (defaults to REC_ASR_PROVIDER)",
    )
    run_parser.add_argument(
        "--summary-provider",
        choices=["local", "openai", "deepgram", "groq"],
        default=None,
        help="Summary provider override (defaults to REC_SUMMARY_PROVIDER)",
    )
    run_parser.add_argument(
        "--external-fallback-to-local",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fallback to local providers when external providers fail",
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
    run_parser.add_argument(
        "--diarization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable local speaker diarization stage",
    )
    run_parser.add_argument(
        "--diarization-model",
        default=None,
        help="pyannote diarization model name",
    )
    run_parser.add_argument(
        "--diarization-export-speakers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable per-speaker text export files",
    )
    run_parser.set_defaults(handler=_handle_run)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run quality evaluation harness")
    evaluate_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/datasets/pt_noisy_subset.json"),
        help="Path to evaluation dataset JSON",
    )
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/reports/latest"),
        help="Directory for evaluation report artifacts",
    )
    evaluate_parser.add_argument("--max-wer", type=float, default=None, help="Maximum allowed WER")
    evaluate_parser.add_argument("--max-cer", type=float, default=None, help="Maximum allowed CER")
    evaluate_parser.add_argument(
        "--max-der-proxy",
        type=float,
        default=None,
        help="Maximum allowed DER proxy",
    )
    evaluate_parser.add_argument(
        "--min-traceability-coverage",
        type=float,
        default=None,
        help="Minimum required summary traceability coverage",
    )
    evaluate_parser.add_argument(
        "--blocking-gates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat quality gate violations as blocking failures",
    )
    evaluate_parser.set_defaults(handler=_handle_evaluate)

    return parser


def _handle_run(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file)
    language = args.lang or settings.default_language
    run_dir = args.output / args.run_name
    should_fail_fast = args.fail_fast or not settings.continue_on_error
    external_fallback_to_local = (
        settings.external_fallback_to_local
        if args.external_fallback_to_local is None
        else bool(args.external_fallback_to_local)
    )
    requested_asr_provider = args.asr_provider or settings.asr_provider
    requested_summary_provider = args.summary_provider or settings.summary_provider

    try:
        validate_provider_configuration(
            asr_provider=requested_asr_provider,
            summary_provider=requested_summary_provider,
            openai_api_key=settings.openai_api_key,
            deepgram_api_key=settings.deepgram_api_key,
            groq_api_key=settings.groq_api_key,
        )
    except ProviderConfigError as exc:
        print(f"Provider configuration error: {exc}")
        return 1

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

    def build_local_transcriber() -> FasterWhisperTranscriber:
        return FasterWhisperTranscriber(
            model_size=args.asr_model_size or settings.asr_model_size,
            device=args.asr_device or settings.asr_device,
            compute_type=args.asr_compute_type or settings.asr_compute_type,
        )

    beam_size = args.asr_beam_size or settings.asr_beam_size
    vad_filter = (
        settings.asr_vad_filter if args.asr_vad_filter is None else bool(args.asr_vad_filter)
    )
    asr_max_retries = (
        args.asr_max_retries if args.asr_max_retries is not None else settings.asr_max_retries
    )

    asr_transcriber: Transcriber
    if requested_asr_provider == "local":
        asr_transcriber = build_local_transcriber()
    else:
        provider_key = resolve_provider_key(
            provider_name=requested_asr_provider,
            openai_api_key=settings.openai_api_key,
            deepgram_api_key=settings.deepgram_api_key,
            groq_api_key=settings.groq_api_key,
        )
        if provider_key is None:
            print(f"Missing API key for ASR provider '{requested_asr_provider}'")
            return 1
        asr_transcriber = ExternalASRTranscriber(
            provider_name=requested_asr_provider,
            api_key=provider_key,
            model_name=args.asr_model_size or settings.asr_model_size,
            timeout_sec=settings.provider_timeout_sec,
            max_retries=settings.provider_max_retries,
            retry_base_delay_sec=settings.provider_retry_base_delay_sec,
        )

    def build_asr_pipeline(transcriber: Transcriber) -> ASRPipeline:
        return ASRPipeline(
            transcriber=transcriber,
            beam_size=beam_size,
            vad_filter=vad_filter,
            max_retries=asr_max_retries,
            retry_backoff_sec=0.5,
            fail_fast=should_fail_fast,
        )

    asr_pipeline = build_asr_pipeline(asr_transcriber)
    effective_asr_provider = requested_asr_provider
    asr_fallback_used = False
    try:
        asr_result = asr_pipeline.transcribe_run(run_dir=run_dir, language=language)
    except ASRError as exc:
        if is_external_provider(requested_asr_provider) and external_fallback_to_local:
            print(f"ASR external provider failed ({exc}). Falling back to local ASR provider.")
            asr_pipeline = build_asr_pipeline(build_local_transcriber())
            effective_asr_provider = "local"
            asr_fallback_used = True
            try:
                asr_result = asr_pipeline.transcribe_run(run_dir=run_dir, language=language)
            except ASRError as fallback_exc:
                print(f"ASR stage failed after fallback: {fallback_exc}")
                return 1
        else:
            print(f"ASR stage failed: {exc}")
            return 1
    if (
        is_external_provider(requested_asr_provider)
        and external_fallback_to_local
        and getattr(asr_result, "errors", [])
        and effective_asr_provider != "local"
    ):
        print("ASR external provider returned errors. Re-running with local ASR fallback.")
        asr_pipeline = build_asr_pipeline(build_local_transcriber())
        effective_asr_provider = "local"
        asr_fallback_used = True
        try:
            asr_result = asr_pipeline.transcribe_run(run_dir=run_dir, language=language)
        except ASRError as fallback_exc:
            print(f"ASR stage failed after fallback retry: {fallback_exc}")
            return 1
    print(
        "ASR summary: "
        f"files={len(asr_result.files)} "
        f"transcribed={asr_result.transcribed_count} "
        f"skipped={asr_result.skipped_count} "
        f"errors={len(asr_result.errors)}"
    )
    diarization_enabled = (
        settings.diarization_enabled if args.diarization is None else bool(args.diarization)
    )
    if diarization_enabled:
        diarizer = PyannoteDiarizer(
            model_name=args.diarization_model or settings.diarization_model_name,
            hf_token=settings.huggingface_token,
        )
        diarization_pipeline = SpeakerDiarizationPipeline(
            diarizer=diarizer,
            fail_fast=should_fail_fast,
        )
        try:
            diarization_result = diarization_pipeline.run(
                run_dir=run_dir,
                enabled=True,
                export_per_speaker=(
                    settings.diarization_export_speakers
                    if args.diarization_export_speakers is None
                    else bool(args.diarization_export_speakers)
                ),
            )
        except DiarizationError as exc:
            print(f"Diarization failed: {exc}")
            return 1
        print(
            "Diarization summary: "
            f"processed={diarization_result.processed_files} "
            f"skipped={diarization_result.skipped_files} "
            f"segments={diarization_result.labeled_segments} "
            f"errors={len(diarization_result.errors)} "
            f"speaker_exports={len(diarization_result.speaker_exports)}"
        )
    else:
        print("Diarization summary: disabled")
    transcript_builder = TranscriptArtifactBuilder()
    try:
        transcript_result = transcript_builder.build(
            run_dir=run_dir,
            language=language,
            prefer_diarized=diarization_enabled,
        )
    except TranscriptError as exc:
        print(f"Transcript assembly failed: {exc}")
        return 1
    print(
        "Transcript summary: "
        f"segments={transcript_result.segment_count} "
        f"generated={len(transcript_result.generated_files)} "
        f"skipped={len(transcript_result.skipped_files)}"
    )
    summary_model_name = args.summary_model or settings.summary_model_name
    summary_model_override = None
    if is_external_provider(requested_summary_provider):
        provider_key = resolve_provider_key(
            provider_name=requested_summary_provider,
            openai_api_key=settings.openai_api_key,
            deepgram_api_key=settings.deepgram_api_key,
            groq_api_key=settings.groq_api_key,
        )
        if provider_key is None:
            print(f"Missing API key for summary provider '{requested_summary_provider}'")
            return 1
        summary_model_override = ExternalSummaryModel(
            provider_name=requested_summary_provider,
            api_key=provider_key,
            model_name=summary_model_name,
            timeout_sec=settings.provider_timeout_sec,
            max_retries=settings.provider_max_retries,
            retry_base_delay_sec=settings.provider_retry_base_delay_sec,
        )

    def build_summary_pipeline(*, model_override: SummaryModel | None) -> SummaryPipeline:
        return SummaryPipeline(
            model_backend=args.summary_local_backend or settings.summary_local_backend,
            model_name=summary_model_name,
            ollama_base_url=settings.ollama_base_url,
            llamacpp_server_url=settings.llamacpp_server_url,
            max_chunk_tokens=args.summary_max_chunk_tokens or settings.summary_max_chunk_tokens,
            max_chunk_seconds=args.summary_max_chunk_seconds or settings.summary_max_chunk_seconds,
            fail_fast=should_fail_fast,
            model_override=model_override,
        )

    effective_summary_provider = requested_summary_provider
    summary_fallback_used = False
    summary_pipeline = build_summary_pipeline(model_override=summary_model_override)
    try:
        summary_result = summary_pipeline.build(run_dir=run_dir, language=settings.output_language)
    except SummaryError as exc:
        if is_external_provider(requested_summary_provider) and external_fallback_to_local:
            print(
                f"Summary external provider failed ({exc}). Falling back to local summary provider."
            )
            effective_summary_provider = "local"
            summary_fallback_used = True
            summary_pipeline = build_summary_pipeline(model_override=None)
            try:
                summary_result = summary_pipeline.build(
                    run_dir=run_dir,
                    language=settings.output_language,
                )
            except SummaryError as fallback_exc:
                print(f"Summarization failed after fallback: {fallback_exc}")
                return 1
        else:
            print(f"Summarization failed: {exc}")
            return 1
    print(
        "Summary stage: "
        f"generated={len(summary_result.generated_files)} "
        f"skipped={len(summary_result.skipped_files)}"
    )
    run_metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "asr_provider_requested": requested_asr_provider,
        "asr_provider_effective": effective_asr_provider,
        "summary_provider_requested": requested_summary_provider,
        "summary_provider_effective": effective_summary_provider,
        "external_fallback_to_local": external_fallback_to_local,
        "asr_fallback_used": asr_fallback_used,
        "summary_fallback_used": summary_fallback_used,
        "diarization_enabled": diarization_enabled,
    }
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Pipeline scaffold ready. asr_provider={effective_asr_provider} "
        f"summary_provider={effective_summary_provider} lang={language}"
    )
    return 0


def _handle_evaluate(args: argparse.Namespace) -> int:
    thresholds = default_thresholds(blocking_gates=bool(args.blocking_gates))
    if args.max_wer is not None:
        thresholds = thresholds.__class__(
            max_wer=args.max_wer,
            max_cer=thresholds.max_cer,
            max_der_proxy=thresholds.max_der_proxy,
            min_traceability_coverage=thresholds.min_traceability_coverage,
            blocking_gates=thresholds.blocking_gates,
        )
    if args.max_cer is not None:
        thresholds = thresholds.__class__(
            max_wer=thresholds.max_wer,
            max_cer=args.max_cer,
            max_der_proxy=thresholds.max_der_proxy,
            min_traceability_coverage=thresholds.min_traceability_coverage,
            blocking_gates=thresholds.blocking_gates,
        )
    if args.max_der_proxy is not None:
        thresholds = thresholds.__class__(
            max_wer=thresholds.max_wer,
            max_cer=thresholds.max_cer,
            max_der_proxy=args.max_der_proxy,
            min_traceability_coverage=thresholds.min_traceability_coverage,
            blocking_gates=thresholds.blocking_gates,
        )
    if args.min_traceability_coverage is not None:
        thresholds = thresholds.__class__(
            max_wer=thresholds.max_wer,
            max_cer=thresholds.max_cer,
            max_der_proxy=thresholds.max_der_proxy,
            min_traceability_coverage=args.min_traceability_coverage,
            blocking_gates=thresholds.blocking_gates,
        )

    try:
        report = evaluate_dataset(
            dataset_path=args.dataset,
            output_dir=args.output,
            thresholds=thresholds,
        )
    except EvaluationError as exc:
        print(f"Evaluation failed: {exc}")
        return 1

    print(f"Evaluation complete. output={args.output}")
    print(
        "Evaluation summary: "
        f"scenarios={report.scenario_count} "
        f"long_multifile={report.long_multifile_scenario_count} "
        f"gate_status={report.gate_status} "
        f"violations={len(report.violations)}"
    )
    if report.gate_status == "failed":
        return 1
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
