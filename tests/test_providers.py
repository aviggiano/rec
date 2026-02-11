from __future__ import annotations

import json
from pathlib import Path

import pytest

from rec_pipeline.asr import ASRPipeline
from rec_pipeline.providers import (
    ExternalASRTranscriber,
    ExternalSummaryModel,
    ProviderConfigError,
    ProviderRequestError,
    call_with_retry,
    validate_provider_configuration,
)
from rec_pipeline.summary import SummaryPipeline
from rec_pipeline.transcript import TranscriptArtifactBuilder


def test_provider_validation_allows_local_mode_without_keys() -> None:
    validate_provider_configuration(
        asr_provider="local",
        summary_provider="local",
        openai_api_key=None,
        deepgram_api_key=None,
        groq_api_key=None,
    )


def test_provider_validation_fails_for_missing_key() -> None:
    with pytest.raises(ProviderConfigError):
        validate_provider_configuration(
            asr_provider="openai",
            summary_provider="local",
            openai_api_key=None,
            deepgram_api_key=None,
            groq_api_key=None,
        )


def test_retry_with_backoff_retries_transient_errors() -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def flaky_operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ProviderRequestError("rate limited", status_code=429)
        return "ok"

    result = call_with_retry(
        flaky_operation,
        max_retries=3,
        base_delay_sec=0.1,
        sleep_fn=sleeps.append,
    )

    assert result == "ok"
    assert attempts["count"] == 3
    assert sleeps == [0.1, 0.2]


def test_external_provider_adapters_with_mocked_responses(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "external"
    normalized_dir = run_dir / "normalized"
    normalized_dir.mkdir(parents=True)

    normalized_file = normalized_dir / "0000_clip.wav"
    normalized_file.write_bytes(b"audio")

    ingestion_manifest = {
        "files": [
            {
                "source_name": "clip.wav",
                "normalized_name": "0000_clip.wav",
                "normalized_path": str(normalized_file.resolve()),
                "offset_start_sec": 0.0,
            }
        ]
    }
    (run_dir / "ingestion_manifest.json").write_text(
        json.dumps(ingestion_manifest),
        encoding="utf-8",
    )

    def fake_request(
        url: str, payload: dict[str, object], api_key: str, timeout_sec: int
    ) -> dict[str, object]:
        del payload, api_key, timeout_sec
        if "audio/transcriptions" in url:
            return {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.2,
                        "text": "fala externa",
                        "confidence": 0.8,
                    }
                ]
            }
        return {"text": "Resumo externo com citacao."}

    transcriber = ExternalASRTranscriber(
        provider_name="openai",
        api_key="sk-test-value",
        model_name="whisper-1",
        timeout_sec=30,
        max_retries=2,
        retry_base_delay_sec=0.01,
        request_fn=fake_request,
    )
    asr_pipeline = ASRPipeline(
        transcriber=transcriber,
        beam_size=5,
        vad_filter=True,
        max_retries=2,
        retry_backoff_sec=0,
        fail_fast=True,
    )

    asr_result = asr_pipeline.transcribe_run(run_dir=run_dir, language="pt")

    assert asr_result.transcribed_count == 1
    assert asr_result.files[0].segments[0].text == "fala externa"

    transcript_result = TranscriptArtifactBuilder().build(run_dir=run_dir, language="pt")
    assert transcript_result.json_path.exists()

    summary_model = ExternalSummaryModel(
        provider_name="openai",
        api_key="sk-test-value",
        model_name="gpt-4o-mini",
        timeout_sec=30,
        max_retries=2,
        retry_base_delay_sec=0.01,
        request_fn=fake_request,
    )
    summary_pipeline = SummaryPipeline(
        model_backend="heuristic",
        model_name="unused",
        ollama_base_url="http://localhost:11434",
        llamacpp_server_url="http://localhost:8080",
        max_chunk_tokens=100,
        max_chunk_seconds=60,
        fail_fast=True,
        model_override=summary_model,
    )

    summary_result = summary_pipeline.build(run_dir=run_dir, language="pt")

    assert summary_result.json_path.exists()
    payload = json.loads(summary_result.json_path.read_text(encoding="utf-8"))
    assert payload["overview"]["text"]
