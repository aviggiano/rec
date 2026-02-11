from __future__ import annotations

import json
from pathlib import Path

from rec_pipeline.diarization import (
    DiarizationTurn,
    SpeakerDiarizationPipeline,
    merge_segments_with_speakers,
)


class StubDiarizer:
    def __init__(self, turns: list[DiarizationTurn]) -> None:
        self._turns = turns

    def diarize(self, audio_path: Path) -> list[DiarizationTurn]:
        del audio_path
        return self._turns


class FailingDiarizer:
    def diarize(self, audio_path: Path) -> list[DiarizationTurn]:
        del audio_path
        raise RuntimeError("diarization failed")


def test_merge_overlap_assigns_expected_speaker() -> None:
    segments = [
        {"start_sec": 0.0, "end_sec": 2.0, "text": "ola"},
        {"start_sec": 2.0, "end_sec": 4.0, "text": "mundo"},
    ]
    turns = [
        DiarizationTurn(start_sec=0.0, end_sec=2.0, speaker="SPEAKER_00"),
        DiarizationTurn(start_sec=2.0, end_sec=4.0, speaker="SPEAKER_01"),
    ]

    merged = merge_segments_with_speakers(segments, turns)

    assert [segment["speaker"] for segment in merged] == ["SPEAKER_00", "SPEAKER_01"]


def test_merge_short_turns_chooses_max_overlap() -> None:
    segments = [{"start_sec": 0.0, "end_sec": 2.0, "text": "fala"}]
    turns = [
        DiarizationTurn(start_sec=0.0, end_sec=0.2, speaker="SPEAKER_00"),
        DiarizationTurn(start_sec=0.2, end_sec=1.9, speaker="SPEAKER_01"),
    ]

    merged = merge_segments_with_speakers(segments, turns)

    assert merged[0]["speaker"] == "SPEAKER_01"


def test_merge_silence_gap_defaults_to_unknown() -> None:
    segments = [{"start_sec": 5.0, "end_sec": 6.0, "text": "sem voz"}]
    turns = [DiarizationTurn(start_sec=0.0, end_sec=1.0, speaker="SPEAKER_00")]

    merged = merge_segments_with_speakers(segments, turns)

    assert merged[0]["speaker"] == "UNKNOWN"


def test_diarization_pipeline_emits_speaker_labels_and_exports(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo"
    normalized_dir = run_dir / "normalized"
    normalized_dir.mkdir(parents=True)
    normalized_file = normalized_dir / "0000_clip.wav"
    normalized_file.write_bytes(b"audio")

    raw_payload = {
        "files": [
            {
                "source_name": "clip.wav",
                "normalized_name": "0000_clip.wav",
                "normalized_path": str(normalized_file.resolve()),
                "segments": [
                    {"start_sec": 0.0, "end_sec": 1.0, "text": "bom dia"},
                    {"start_sec": 1.0, "end_sec": 2.0, "text": "tudo bem"},
                ],
            }
        ]
    }
    (run_dir / "asr").mkdir(parents=True)
    (run_dir / "asr" / "raw_transcript_segments.json").write_text(
        json.dumps(raw_payload),
        encoding="utf-8",
    )

    ingestion_manifest = {
        "files": [
            {
                "normalized_name": "0000_clip.wav",
                "offset_start_sec": 10.0,
            }
        ]
    }
    (run_dir / "ingestion_manifest.json").write_text(
        json.dumps(ingestion_manifest),
        encoding="utf-8",
    )

    pipeline = SpeakerDiarizationPipeline(
        diarizer=StubDiarizer(
            [
                DiarizationTurn(start_sec=0.0, end_sec=2.0, speaker="SPEAKER_00"),
            ]
        ),
        fail_fast=True,
    )

    result = pipeline.run(run_dir=run_dir, enabled=True, export_per_speaker=True)

    assert result.processed_files == 1
    assert result.labeled_segments == 2

    aggregate = json.loads(
        (run_dir / "asr" / "speaker_transcript_segments.json").read_text(encoding="utf-8")
    )
    speaker_values = [segment["speaker"] for segment in aggregate["files"][0]["segments"]]
    assert speaker_values == ["SPEAKER_00", "SPEAKER_00"]

    speaker_export = run_dir / "artifacts" / "speakers" / "speaker_SPEAKER_00.txt"
    assert speaker_export.exists()
    content = speaker_export.read_text(encoding="utf-8")
    assert "[00:00:10.000]" in content


def test_diarization_failure_can_continue_with_unknown_labels(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "fallback"
    normalized_dir = run_dir / "normalized"
    normalized_dir.mkdir(parents=True)
    normalized_file = normalized_dir / "0000_clip.wav"
    normalized_file.write_bytes(b"audio")

    raw_payload = {
        "files": [
            {
                "source_name": "clip.wav",
                "normalized_name": "0000_clip.wav",
                "normalized_path": str(normalized_file.resolve()),
                "segments": [
                    {"start_sec": 0.0, "end_sec": 1.0, "text": "bom dia"},
                ],
            }
        ]
    }
    (run_dir / "asr").mkdir(parents=True)
    (run_dir / "asr" / "raw_transcript_segments.json").write_text(
        json.dumps(raw_payload),
        encoding="utf-8",
    )

    pipeline = SpeakerDiarizationPipeline(diarizer=FailingDiarizer(), fail_fast=False)

    result = pipeline.run(run_dir=run_dir, enabled=True, export_per_speaker=False)

    assert result.errors
    aggregate = json.loads(
        (run_dir / "asr" / "speaker_transcript_segments.json").read_text(encoding="utf-8")
    )
    assert aggregate["files"][0]["segments"][0]["speaker"] == "UNKNOWN"
    assert not (run_dir / "checkpoints" / "0000_clip.diarization.done").exists()
