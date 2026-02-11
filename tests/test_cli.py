from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import rec_pipeline.cli as cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cli_help_smoke() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "rec_pipeline.cli", "--help"],  # noqa: S607
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "rec" in completed.stdout
    assert "run" in completed.stdout


def test_cli_respects_zero_asr_max_retries_override(monkeypatch: object, tmp_path: Path) -> None:
    captured: dict[str, int] = {}

    settings = SimpleNamespace(
        default_language="pt",
        continue_on_error=True,
        asr_provider="local",
        summary_provider="local",
        asr_model_size="small",
        asr_device="auto",
        asr_compute_type="int8",
        asr_beam_size=5,
        asr_vad_filter=True,
        asr_max_retries=3,
        diarization_enabled=False,
        diarization_model_name="pyannote/speaker-diarization-3.1",
        huggingface_token=None,
        diarization_export_speakers=True,
        summary_local_backend="heuristic",
        summary_model_name="unused",
        summary_max_chunk_tokens=100,
        summary_max_chunk_seconds=60,
        output_language="pt",
        external_fallback_to_local=True,
        openai_api_key=None,
        deepgram_api_key=None,
        groq_api_key=None,
        provider_timeout_sec=30,
        provider_max_retries=2,
        provider_retry_base_delay_sec=0.1,
        ollama_base_url="http://localhost:11434",
        llamacpp_server_url="http://localhost:8080",
    )

    class StubIngestionProcessor:
        def __init__(self, *, fail_fast: bool) -> None:
            del fail_fast

        def process_directory(self, input_dir: Path, run_dir: Path) -> SimpleNamespace:
            del input_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "ingestion_manifest.json").write_text('{"files": []}', encoding="utf-8")
            return SimpleNamespace(files=[], normalized_count=0, skipped_count=0, errors=[])

    class StubASRPipeline:
        def __init__(
            self,
            *,
            transcriber: object,
            beam_size: int,
            vad_filter: bool,
            max_retries: int,
            retry_backoff_sec: float,
            fail_fast: bool,
        ) -> None:
            del transcriber, beam_size, vad_filter, retry_backoff_sec, fail_fast
            captured["max_retries"] = max_retries

        def transcribe_run(self, *, run_dir: Path, language: str) -> SimpleNamespace:
            del language
            asr_dir = run_dir / "asr"
            asr_dir.mkdir(parents=True, exist_ok=True)
            (asr_dir / "raw_transcript_segments.json").write_text(
                '{"files": [], "errors": [], "error_count": 0}',
                encoding="utf-8",
            )
            return SimpleNamespace(files=[], transcribed_count=0, skipped_count=0, errors=[])

    class StubTranscriptArtifactBuilder:
        def build(
            self,
            *,
            run_dir: Path,
            language: str,
            prefer_diarized: bool | None = None,
        ) -> SimpleNamespace:
            del language, prefer_diarized
            artifacts = run_dir / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / "transcript.json").write_text(
                '{"segments": [], "schema_version": "1.0", "language": "pt", "segment_count": 0}',
                encoding="utf-8",
            )
            return SimpleNamespace(segment_count=0, generated_files=[], skipped_files=[])

    class StubSummaryPipeline:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def build(self, *, run_dir: Path, language: str) -> SimpleNamespace:
            del run_dir, language
            return SimpleNamespace(generated_files=[], skipped_files=[])

    monkeypatch.setattr(cli, "load_settings", lambda env_file: settings)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "IngestionProcessor", StubIngestionProcessor)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "FasterWhisperTranscriber", lambda **kwargs: object())  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "ASRPipeline", StubASRPipeline)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "TranscriptArtifactBuilder", StubTranscriptArtifactBuilder)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "SummaryPipeline", StubSummaryPipeline)  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "validate_provider_configuration", lambda **kwargs: None)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        cli,
        "_run_preflight_checks",
        lambda *, requested_asr_provider: None,
    )  # type: ignore[attr-defined]
    monkeypatch.setattr(
        cli,
        "_prepare_input_directory",
        lambda *, input_path, run_dir: cli.PreparedInput(
            directory=input_path.resolve(),
            source_path=input_path.resolve(),
            source_type="directory",
            archive_extracted=False,
        ),
    )  # type: ignore[attr-defined]

    input_dir = tmp_path / "input"
    input_dir.mkdir()

    args = argparse.Namespace(
        env_file=None,
        input=input_dir,
        output=tmp_path / "output",
        run_name="demo",
        fail_fast=False,
        lang=None,
        asr_provider=None,
        summary_provider=None,
        external_fallback_to_local=None,
        asr_model_size=None,
        asr_device=None,
        asr_compute_type=None,
        asr_beam_size=None,
        asr_vad_filter=None,
        asr_max_retries=0,
        summary_local_backend=None,
        summary_model=None,
        summary_max_chunk_tokens=None,
        summary_max_chunk_seconds=None,
        diarization=None,
        diarization_model=None,
        diarization_export_speakers=None,
    )

    code = cli._handle_run(args)

    assert code == 0
    assert captured["max_retries"] == 0


def test_prepare_input_directory_accepts_directory(tmp_path: Path) -> None:
    input_dir = tmp_path / "recordings"
    input_dir.mkdir(parents=True)
    run_dir = tmp_path / "artifacts" / "demo"

    prepared = cli._prepare_input_directory(input_path=input_dir, run_dir=run_dir)

    assert prepared.source_type == "directory"
    assert prepared.directory == input_dir.resolve()
    assert prepared.archive_extracted is False


def test_prepare_input_directory_extracts_and_reuses_zip(tmp_path: Path) -> None:
    archive_path = tmp_path / "recordings.zip"
    run_dir = tmp_path / "artifacts" / "demo"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("recordings/a.wav", "audio-a")
        archive.writestr("recordings/b.wav", "audio-b")

    first = cli._prepare_input_directory(input_path=archive_path, run_dir=run_dir)
    assert first.source_type == "archive"
    assert first.archive_extracted is True
    assert (first.directory / "recordings" / "a.wav").read_text(encoding="utf-8") == "audio-a"

    marker = first.directory / "reused.marker"
    marker.write_text("keep", encoding="utf-8")
    second = cli._prepare_input_directory(input_path=archive_path, run_dir=run_dir)

    assert second.source_type == "archive"
    assert second.archive_extracted is False
    assert second.directory == first.directory
    assert marker.read_text(encoding="utf-8") == "keep"


def test_prepare_input_directory_rejects_unsupported_file(tmp_path: Path) -> None:
    unsupported = tmp_path / "recordings.tar"
    unsupported.write_text("archive", encoding="utf-8")

    with pytest.raises(cli.InputPreparationError, match="directory or a .zip archive"):
        cli._prepare_input_directory(
            input_path=unsupported, run_dir=tmp_path / "artifacts" / "demo"
        )


def test_run_preflight_checks_fails_without_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli.shutil,
        "which",
        lambda binary_name: None if binary_name == "ffmpeg" else f"/usr/bin/{binary_name}",
    )

    with pytest.raises(cli.RunPreflightError, match="ffmpeg"):
        cli._run_preflight_checks(requested_asr_provider="local")


def test_run_preflight_checks_fails_without_local_asr_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda binary_name: f"/usr/bin/{binary_name}")
    monkeypatch.setattr(cli, "_is_local_asr_runtime_available", lambda: False)

    with pytest.raises(cli.RunPreflightError, match="faster-whisper"):
        cli._run_preflight_checks(requested_asr_provider="local")
