from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


class TranscriptError(RuntimeError):
    """Raised when transcript assembly fails."""


class TranscriptSchemaError(TranscriptError):
    """Raised when transcript JSON schema validation fails."""


@dataclass(frozen=True)
class AssembledSegment:
    segment_id: int
    source_name: str
    normalized_name: str
    text: str
    relative_start_sec: float
    relative_end_sec: float
    absolute_start_sec: float
    absolute_end_sec: float
    speaker: str | None
    avg_logprob: float | None
    confidence: float | None
    no_speech_prob: float | None


@dataclass(frozen=True)
class TranscriptDocument:
    language: str
    segments: list[AssembledSegment]


@dataclass(frozen=True)
class TranscriptArtifactResult:
    transcript_path: Path
    srt_path: Path
    json_path: Path
    manifest_path: Path
    generated_files: list[str]
    skipped_files: list[str]
    segment_count: int


def _parse_ingestion_offsets(run_dir: Path) -> dict[str, float]:
    ingestion_manifest_path = run_dir / "ingestion_manifest.json"
    if not ingestion_manifest_path.exists():
        raise TranscriptError(f"Missing ingestion manifest: {ingestion_manifest_path}")

    payload = json.loads(ingestion_manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files", [])
    if not isinstance(files, list):
        raise TranscriptError("Malformed ingestion manifest: files must be a list")

    offsets: dict[str, float] = {}
    for file_payload in files:
        normalized_name = str(file_payload.get("normalized_name", ""))
        offset_value = file_payload.get("offset_start_sec")
        if not normalized_name or not isinstance(offset_value, (float, int)):
            continue
        offsets[normalized_name] = float(offset_value)

    return offsets


def _load_asr_payload(run_dir: Path) -> dict[str, object]:
    diarized_asr_path = run_dir / "asr" / "speaker_transcript_segments.json"
    asr_payload_path = (
        diarized_asr_path
        if diarized_asr_path.exists()
        else run_dir / "asr" / "raw_transcript_segments.json"
    )
    if not asr_payload_path.exists():
        raise TranscriptError(f"Missing ASR aggregate payload: {asr_payload_path}")
    payload = json.loads(asr_payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TranscriptError("Malformed ASR payload: expected object")
    return payload


def assemble_transcript(run_dir: Path, *, language: str) -> TranscriptDocument:
    offsets = _parse_ingestion_offsets(run_dir)
    asr_payload = _load_asr_payload(run_dir)

    files_payload = asr_payload.get("files", [])
    if not isinstance(files_payload, list):
        raise TranscriptError("Malformed ASR payload: files must be a list")

    ordered_files = sorted(
        files_payload,
        key=lambda item: str(item.get("normalized_name", "")),
    )

    assembled: list[AssembledSegment] = []
    next_id = 0

    for file_payload in ordered_files:
        normalized_name = str(file_payload.get("normalized_name", ""))
        source_name = str(file_payload.get("source_name", normalized_name))
        offset_start = offsets.get(normalized_name, 0.0)

        segments_payload = file_payload.get("segments", [])
        if not isinstance(segments_payload, list):
            raise TranscriptError(
                f"Malformed ASR file payload for {normalized_name}: segments is not list"
            )

        for segment_payload in segments_payload:
            relative_start = float(segment_payload.get("start_sec", 0.0))
            relative_end = float(segment_payload.get("end_sec", 0.0))
            absolute_start = offset_start + relative_start
            absolute_end = offset_start + relative_end

            if relative_start >= relative_end:
                raise TranscriptError(
                    "Invalid relative timestamps in "
                    f"{normalized_name}: {relative_start} >= {relative_end}"
                )
            if absolute_start >= absolute_end:
                raise TranscriptError(
                    "Invalid absolute timestamps in "
                    f"{normalized_name}: {absolute_start} >= {absolute_end}"
                )

            assembled.append(
                AssembledSegment(
                    segment_id=next_id,
                    source_name=source_name,
                    normalized_name=normalized_name,
                    text=str(segment_payload.get("text", "")).strip(),
                    relative_start_sec=relative_start,
                    relative_end_sec=relative_end,
                    absolute_start_sec=absolute_start,
                    absolute_end_sec=absolute_end,
                    speaker=_optional_str(segment_payload.get("speaker")),
                    avg_logprob=_optional_float(segment_payload.get("avg_logprob")),
                    confidence=_optional_float(segment_payload.get("confidence")),
                    no_speech_prob=_optional_float(segment_payload.get("no_speech_prob")),
                )
            )
            next_id += 1

    assembled.sort(
        key=lambda item: (item.absolute_start_sec, item.absolute_end_sec, item.segment_id)
    )

    previous_end = -1.0
    for segment in assembled:
        if segment.absolute_start_sec < previous_end:
            raise TranscriptError(
                "Absolute timeline is non-monotonic after stitching "
                f"at segment {segment.segment_id}: {segment.absolute_start_sec} < {previous_end}"
            )
        previous_end = segment.absolute_end_sec

    return TranscriptDocument(language=language, segments=assembled)


def build_transcript_payload(document: TranscriptDocument) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "language": document.language,
        "segment_count": len(document.segments),
        "segments": [
            {
                "id": segment.segment_id,
                "source_name": segment.source_name,
                "normalized_name": segment.normalized_name,
                "speaker": segment.speaker,
                "text": segment.text,
                "timing": {
                    "relative_start_sec": segment.relative_start_sec,
                    "relative_end_sec": segment.relative_end_sec,
                    "absolute_start_sec": segment.absolute_start_sec,
                    "absolute_end_sec": segment.absolute_end_sec,
                },
                "metrics": {
                    "avg_logprob": segment.avg_logprob,
                    "confidence": segment.confidence,
                    "no_speech_prob": segment.no_speech_prob,
                },
            }
            for segment in document.segments
        ],
    }


def validate_transcript_payload(payload: dict[str, object]) -> None:
    required_top_level = {"schema_version", "language", "segment_count", "segments"}
    missing = [key for key in required_top_level if key not in payload]
    if missing:
        raise TranscriptSchemaError(
            f"Missing required top-level keys: {', '.join(sorted(missing))}"
        )

    segments_payload = payload["segments"]
    if not isinstance(segments_payload, list):
        raise TranscriptSchemaError("'segments' must be a list")

    for segment_payload in segments_payload:
        if not isinstance(segment_payload, dict):
            raise TranscriptSchemaError("Each segment entry must be an object")

        for key in ["id", "source_name", "normalized_name", "text", "timing", "metrics"]:
            if key not in segment_payload:
                raise TranscriptSchemaError(f"Segment missing key: {key}")

        timing = segment_payload["timing"]
        if not isinstance(timing, dict):
            raise TranscriptSchemaError("Segment timing must be an object")

        timing_keys = [
            "relative_start_sec",
            "relative_end_sec",
            "absolute_start_sec",
            "absolute_end_sec",
        ]
        for key in timing_keys:
            value = timing.get(key)
            if not isinstance(value, (int, float)):
                raise TranscriptSchemaError(f"Segment timing '{key}' must be numeric")

        relative_start = float(timing["relative_start_sec"])
        relative_end = float(timing["relative_end_sec"])
        absolute_start = float(timing["absolute_start_sec"])
        absolute_end = float(timing["absolute_end_sec"])

        if relative_start >= relative_end:
            raise TranscriptSchemaError(
                f"Invalid relative timing in segment: {relative_start} >= {relative_end}"
            )
        if absolute_start >= absolute_end:
            raise TranscriptSchemaError(
                f"Invalid absolute timing in segment: {absolute_start} >= {absolute_end}"
            )


def format_srt_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    secs = milliseconds // 1_000
    milliseconds -= secs * 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def format_text_timestamp(seconds: float) -> str:
    total_millis = int(round(seconds * 1000))
    hours = total_millis // 3_600_000
    total_millis -= hours * 3_600_000
    minutes = total_millis // 60_000
    total_millis -= minutes * 60_000
    secs = total_millis // 1_000
    millis = total_millis - secs * 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def write_transcript_txt(document: TranscriptDocument, path: Path) -> None:
    lines: list[str] = []
    for segment in document.segments:
        speaker_prefix = f"[{segment.speaker}] " if segment.speaker else ""
        lines.append(
            f"[{format_text_timestamp(segment.absolute_start_sec)}] "
            f"{speaker_prefix}{segment.text}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_transcript_srt(document: TranscriptDocument, path: Path) -> None:
    lines: list[str] = []
    for index, segment in enumerate(document.segments, start=1):
        lines.extend(
            [
                str(index),
                (
                    f"{format_srt_timestamp(segment.absolute_start_sec)} --> "
                    f"{format_srt_timestamp(segment.absolute_end_sec)}"
                ),
                segment.text or " ",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_transcript_json(document: TranscriptDocument, path: Path) -> None:
    payload = build_transcript_payload(document)
    validate_transcript_payload(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


class TranscriptArtifactBuilder:
    def build(self, *, run_dir: Path, language: str) -> TranscriptArtifactResult:
        artifacts_dir = run_dir / "artifacts"
        checkpoints_dir = run_dir / "checkpoints"
        logs_dir = run_dir / "logs"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "transcript.jsonl"

        document = assemble_transcript(run_dir, language=language)

        transcript_path = artifacts_dir / "transcript.txt"
        srt_path = artifacts_dir / "transcript.srt"
        json_path = artifacts_dir / "transcript.json"
        manifest_path = artifacts_dir / "transcript_manifest.json"

        jobs = [
            (
                "transcript.txt",
                transcript_path,
                checkpoints_dir / "transcript.txt.done",
                write_transcript_txt,
            ),
            (
                "transcript.srt",
                srt_path,
                checkpoints_dir / "transcript.srt.done",
                write_transcript_srt,
            ),
            (
                "transcript.json",
                json_path,
                checkpoints_dir / "transcript.json.done",
                write_transcript_json,
            ),
        ]

        generated_files: list[str] = []
        skipped_files: list[str] = []

        for label, output_path, checkpoint_path, writer in jobs:
            if checkpoint_path.exists() and output_path.exists():
                skipped_files.append(label)
                self._log(log_path, "transcript_skip_checkpoint", file=label)
                continue
            writer(document, output_path)
            checkpoint_path.write_text("ok\n", encoding="utf-8")
            generated_files.append(label)
            self._log(log_path, "transcript_generated", file=label)

        manifest_payload = {
            "language": language,
            "segment_count": len(document.segments),
            "generated_files": generated_files,
            "skipped_files": skipped_files,
            "artifacts": {
                "transcript_txt": str(transcript_path),
                "transcript_srt": str(srt_path),
                "transcript_json": str(json_path),
            },
            "segments": [asdict(segment) for segment in document.segments],
        }
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        return TranscriptArtifactResult(
            transcript_path=transcript_path,
            srt_path=srt_path,
            json_path=json_path,
            manifest_path=manifest_path,
            generated_files=generated_files,
            skipped_files=skipped_files,
            segment_count=len(document.segments),
        )

    def _log(self, log_path: Path, event: str, **payload: object) -> None:
        entry = {"event": event, "payload": payload}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
