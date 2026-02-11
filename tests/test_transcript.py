from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rec_pipeline.transcript import (
    AssembledSegment,
    TranscriptArtifactBuilder,
    TranscriptDocument,
    TranscriptSchemaError,
    assemble_transcript,
    build_transcript_payload,
    format_srt_timestamp,
    validate_transcript_payload,
)


def _write_manifests(run_dir: Path) -> None:
    (run_dir / "asr").mkdir(parents=True, exist_ok=True)
    ingestion_manifest = {
        "files": [
            {
                "source_name": "a.wav",
                "normalized_name": "0000_a.wav",
                "offset_start_sec": 0.0,
            },
            {
                "source_name": "b.wav",
                "normalized_name": "0001_b.wav",
                "offset_start_sec": 5.0,
            },
        ]
    }
    (run_dir / "ingestion_manifest.json").write_text(
        json.dumps(ingestion_manifest), encoding="utf-8"
    )

    asr_payload = {
        "files": [
            {
                "source_name": "a.wav",
                "normalized_name": "0000_a.wav",
                "segments": [
                    {
                        "start_sec": 0.5,
                        "end_sec": 1.4,
                        "text": "primeiro",
                        "avg_logprob": -0.2,
                        "confidence": 0.8,
                        "no_speech_prob": 0.1,
                    }
                ],
            },
            {
                "source_name": "b.wav",
                "normalized_name": "0001_b.wav",
                "segments": [
                    {
                        "start_sec": 0.2,
                        "end_sec": 0.9,
                        "text": "segundo",
                        "avg_logprob": -0.1,
                        "confidence": 0.9,
                        "no_speech_prob": 0.05,
                    }
                ],
            },
        ]
    }
    (run_dir / "asr" / "raw_transcript_segments.json").write_text(
        json.dumps(asr_payload), encoding="utf-8"
    )


def test_timeline_stitching_and_srt_format(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "timeline"
    _write_manifests(run_dir)

    document = assemble_transcript(run_dir, language="pt")

    assert len(document.segments) == 2
    assert [segment.absolute_start_sec for segment in document.segments] == [0.5, 5.2]
    assert [segment.relative_start_sec for segment in document.segments] == [0.5, 0.2]
    assert format_srt_timestamp(5.2) == "00:00:05,200"


def test_transcript_json_schema_validation() -> None:
    document = TranscriptDocument(
        language="pt",
        segments=[
            AssembledSegment(
                segment_id=0,
                source_name="a.wav",
                normalized_name="0000_a.wav",
                text="texto",
                relative_start_sec=0.0,
                relative_end_sec=1.0,
                absolute_start_sec=0.0,
                absolute_end_sec=1.0,
                speaker=None,
                avg_logprob=-0.1,
                confidence=0.9,
                no_speech_prob=0.05,
            )
        ],
    )

    payload = build_transcript_payload(document)
    validate_transcript_payload(payload)

    invalid_payload = build_transcript_payload(document)
    invalid_payload["segments"][0].pop("timing")

    with pytest.raises(TranscriptSchemaError):
        validate_transcript_payload(invalid_payload)


def test_partial_run_recovery_preserves_existing_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "recovery"
    _write_manifests(run_dir)

    builder = TranscriptArtifactBuilder()
    first = builder.build(run_dir=run_dir, language="pt")

    assert len(first.generated_files) == 3
    initial_txt = first.transcript_path.read_text(encoding="utf-8")

    first.srt_path.unlink()

    second = builder.build(run_dir=run_dir, language="pt")

    assert "transcript.srt" in second.generated_files
    assert "transcript.txt" in second.skipped_files
    assert "transcript.json" in second.skipped_files
    assert second.transcript_path.read_text(encoding="utf-8") == initial_txt


def test_transcript_builder_regenerates_all_artifacts_when_segments_change(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "regen"
    _write_manifests(run_dir)

    builder = TranscriptArtifactBuilder()
    first = builder.build(run_dir=run_dir, language="pt")
    assert set(first.generated_files) == {"transcript.txt", "transcript.srt", "transcript.json"}

    asr_payload = json.loads(
        (run_dir / "asr" / "raw_transcript_segments.json").read_text(encoding="utf-8")
    )
    asr_payload["files"][0]["segments"].append(
        {
            "start_sec": 1.5,
            "end_sec": 2.0,
            "text": "novo trecho",
            "avg_logprob": -0.1,
            "confidence": 0.9,
            "no_speech_prob": 0.05,
        }
    )
    (run_dir / "asr" / "raw_transcript_segments.json").write_text(
        json.dumps(asr_payload),
        encoding="utf-8",
    )

    second = builder.build(run_dir=run_dir, language="pt")

    assert set(second.generated_files) == {"transcript.txt", "transcript.srt", "transcript.json"}
    payload = json.loads(second.json_path.read_text(encoding="utf-8"))
    assert payload["segment_count"] == 3


def test_timeline_allows_overlapping_segments(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "overlap"
    (run_dir / "asr").mkdir(parents=True, exist_ok=True)
    (run_dir / "ingestion_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "source_name": "a.wav",
                        "normalized_name": "0000_a.wav",
                        "offset_start_sec": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "asr" / "raw_transcript_segments.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "source_name": "a.wav",
                        "normalized_name": "0000_a.wav",
                        "segments": [
                            {"start_sec": 0.0, "end_sec": 1.0, "text": "um"},
                            {"start_sec": 0.9, "end_sec": 2.0, "text": "dois"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    document = assemble_transcript(run_dir, language="pt")

    assert len(document.segments) == 2
    assert document.segments[1].absolute_start_sec == 0.9


def test_transcript_prefers_raw_when_diarization_disabled(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "prefer-raw"
    (run_dir / "asr").mkdir(parents=True, exist_ok=True)
    (run_dir / "ingestion_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "source_name": "a.wav",
                        "normalized_name": "0000_a.wav",
                        "offset_start_sec": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    raw_payload = {
        "files": [
            {
                "source_name": "a.wav",
                "normalized_name": "0000_a.wav",
                "segments": [{"start_sec": 0.0, "end_sec": 1.0, "text": "raw"}],
            }
        ]
    }
    diarized_payload = {
        "files": [
            {
                "source_name": "a.wav",
                "normalized_name": "0000_a.wav",
                "segments": [
                    {"start_sec": 0.0, "end_sec": 1.0, "text": "diarized", "speaker": "S0"}
                ],
            }
        ]
    }
    (run_dir / "asr" / "raw_transcript_segments.json").write_text(
        json.dumps(raw_payload),
        encoding="utf-8",
    )
    (run_dir / "asr" / "speaker_transcript_segments.json").write_text(
        json.dumps(diarized_payload),
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"diarization_enabled": False}),
        encoding="utf-8",
    )

    document = assemble_transcript(run_dir, language="pt", prefer_diarized=False)

    assert document.segments[0].text == "raw"


def test_transcript_uses_raw_when_diarized_payload_is_stale(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "stale-diarized"
    (run_dir / "asr").mkdir(parents=True, exist_ok=True)
    (run_dir / "ingestion_manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "source_name": "a.wav",
                        "normalized_name": "0000_a.wav",
                        "offset_start_sec": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    raw_path = run_dir / "asr" / "raw_transcript_segments.json"
    diarized_path = run_dir / "asr" / "speaker_transcript_segments.json"
    raw_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "source_name": "a.wav",
                        "normalized_name": "0000_a.wav",
                        "segments": [{"start_sec": 0.0, "end_sec": 1.0, "text": "fresh raw"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    diarized_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "source_name": "a.wav",
                        "normalized_name": "0000_a.wav",
                        "segments": [{"start_sec": 0.0, "end_sec": 1.0, "text": "stale diarized"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    old_time = raw_path.stat().st_mtime - 10
    os.utime(diarized_path, (old_time, old_time))

    document = assemble_transcript(run_dir, language="pt", prefer_diarized=True)

    assert document.segments[0].text == "fresh raw"
