from __future__ import annotations

import json
from pathlib import Path

import pytest

from rec_pipeline.asr import (
    ASRError,
    ASRPipeline,
    ASRValidationError,
    TranscriptSegment,
    validate_segments,
)


class StubTranscriber:
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        vad_filter: bool,
        beam_size: int,
    ) -> list[TranscriptSegment]:
        del audio_path, language, vad_filter, beam_size
        return [
            TranscriptSegment(
                segment_id=0,
                start_sec=0.0,
                end_sec=1.2,
                text="ola mundo",
                avg_logprob=-0.15,
                confidence=0.92,
                no_speech_prob=0.05,
            )
        ]


class InvalidTranscriber:
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        vad_filter: bool,
        beam_size: int,
    ) -> list[TranscriptSegment]:
        del audio_path, language, vad_filter, beam_size
        return [
            TranscriptSegment(
                segment_id=0,
                start_sec=1.0,
                end_sec=0.1,
                text="invalid",
                avg_logprob=None,
                confidence=None,
                no_speech_prob=None,
            )
        ]


def test_validate_segments_enforces_monotonic_timestamps() -> None:
    validate_segments(
        [
            TranscriptSegment(0, 0.0, 1.0, "a", None, None, None),
            TranscriptSegment(1, 1.1, 2.0, "b", None, None, None),
        ]
    )

    with pytest.raises(ASRValidationError):
        validate_segments([TranscriptSegment(0, 2.0, 1.0, "broken", None, None, None)])


def test_asr_pipeline_generates_non_empty_raw_segments(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo"
    normalized_dir = run_dir / "normalized"
    normalized_dir.mkdir(parents=True)

    normalized_file = normalized_dir / "0000_clip.wav"
    normalized_file.write_bytes(b"normalized-audio")

    ingestion_manifest = {
        "files": [
            {
                "source_name": "clip.wav",
                "normalized_name": "0000_clip.wav",
                "normalized_path": str(normalized_file.resolve()),
            }
        ]
    }
    (run_dir / "ingestion_manifest.json").write_text(
        json.dumps(ingestion_manifest), encoding="utf-8"
    )

    pipeline = ASRPipeline(
        transcriber=StubTranscriber(),
        beam_size=5,
        vad_filter=True,
        max_retries=2,
        retry_backoff_sec=0,
        fail_fast=True,
    )

    result = pipeline.transcribe_run(run_dir=run_dir, language="pt")

    assert result.transcribed_count == 1
    assert result.skipped_count == 0
    assert len(result.files) == 1
    assert len(result.files[0].segments) == 1
    assert result.files[0].segments[0].text == "ola mundo"

    raw_segments_file = run_dir / "asr" / "raw" / "0000_clip.segments.json"
    assert raw_segments_file.exists()

    payload = json.loads(raw_segments_file.read_text(encoding="utf-8"))
    assert payload["segments"]


def test_asr_pipeline_surfaces_invalid_segments(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "invalid"
    normalized_dir = run_dir / "normalized"
    normalized_dir.mkdir(parents=True)
    normalized_file = normalized_dir / "0000_clip.wav"
    normalized_file.write_bytes(b"normalized-audio")

    ingestion_manifest = {
        "files": [
            {
                "source_name": "clip.wav",
                "normalized_name": "0000_clip.wav",
                "normalized_path": str(normalized_file.resolve()),
            }
        ]
    }
    (run_dir / "ingestion_manifest.json").write_text(
        json.dumps(ingestion_manifest), encoding="utf-8"
    )

    pipeline = ASRPipeline(
        transcriber=InvalidTranscriber(),
        beam_size=5,
        vad_filter=True,
        max_retries=1,
        retry_backoff_sec=0,
        fail_fast=True,
    )

    with pytest.raises(ASRError):
        pipeline.transcribe_run(run_dir=run_dir, language="pt")
