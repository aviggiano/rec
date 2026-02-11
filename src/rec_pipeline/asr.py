from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


class ASRError(RuntimeError):
    """Raised when transcription fails."""


class ASRValidationError(ASRError):
    """Raised when transcript segments do not pass validation rules."""


@dataclass(frozen=True)
class TranscriptSegment:
    segment_id: int
    start_sec: float
    end_sec: float
    text: str
    avg_logprob: float | None
    confidence: float | None
    no_speech_prob: float | None


@dataclass(frozen=True)
class FileTranscript:
    source_name: str
    normalized_name: str
    normalized_path: str
    language: str
    segments: list[TranscriptSegment]


@dataclass(frozen=True)
class ASRRunResult:
    files: list[FileTranscript]
    errors: list[str]
    transcribed_count: int
    skipped_count: int


class Transcriber(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        vad_filter: bool,
        beam_size: int,
    ) -> list[TranscriptSegment]: ...


class FasterWhisperTranscriber:
    def __init__(
        self,
        *,
        model_size: str,
        device: str,
        compute_type: str,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: object | None = None

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except (
            ImportError
        ) as exc:  # pragma: no cover - exercised in environments without optional deps
            raise ASRError(
                "faster-whisper is not installed. "
                "Install optional deps with: pip install -e '.[asr]'"
            ) from exc

        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        vad_filter: bool,
        beam_size: int,
    ) -> list[TranscriptSegment]:
        model = self._get_model()
        segments, _info = model.transcribe(  # type: ignore[attr-defined]
            str(audio_path),
            language=language,
            vad_filter=vad_filter,
            beam_size=beam_size,
            word_timestamps=True,
        )

        converted: list[TranscriptSegment] = []
        for segment in segments:
            words = segment.words if hasattr(segment, "words") else None
            confidence_values = [
                float(word.probability)
                for word in words or []
                if getattr(word, "probability", None) is not None
            ]
            confidence = (
                sum(confidence_values) / len(confidence_values) if confidence_values else None
            )
            converted.append(
                TranscriptSegment(
                    segment_id=int(segment.id) if hasattr(segment, "id") else len(converted),
                    start_sec=float(segment.start),
                    end_sec=float(segment.end),
                    text=str(segment.text).strip(),
                    avg_logprob=(
                        _optional_float(segment.avg_logprob)
                        if hasattr(segment, "avg_logprob")
                        else None
                    ),
                    confidence=confidence,
                    no_speech_prob=(
                        _optional_float(segment.no_speech_prob)
                        if hasattr(segment, "no_speech_prob")
                        else None
                    ),
                )
            )

        return converted


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def validate_segments(segments: Sequence[TranscriptSegment]) -> None:
    previous_start = -1.0
    previous_end = -1.0

    for segment in segments:
        if segment.start_sec >= segment.end_sec:
            raise ASRValidationError(
                f"Invalid segment timing (start >= end): {segment.start_sec} >= {segment.end_sec}"
            )
        if segment.start_sec < previous_start:
            raise ASRValidationError(
                f"Non-monotonic segment start time: {segment.start_sec} < {previous_start}"
            )
        if segment.end_sec < previous_end:
            raise ASRValidationError(
                f"Non-monotonic segment end time: {segment.end_sec} < {previous_end}"
            )

        previous_start = segment.start_sec
        previous_end = segment.end_sec


class ASRPipeline:
    def __init__(
        self,
        *,
        transcriber: Transcriber,
        beam_size: int,
        vad_filter: bool,
        max_retries: int,
        retry_backoff_sec: float,
        fail_fast: bool,
    ) -> None:
        self._transcriber = transcriber
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._max_retries = max_retries
        self._retry_backoff_sec = retry_backoff_sec
        self._fail_fast = fail_fast

    def transcribe_run(self, *, run_dir: Path, language: str) -> ASRRunResult:
        manifest_path = run_dir / "ingestion_manifest.json"
        if not manifest_path.exists():
            raise ASRError(f"Missing ingestion manifest: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files_payload = manifest.get("files", [])
        if not isinstance(files_payload, list):
            raise ASRError("Malformed ingestion manifest: 'files' must be a list")

        asr_dir = run_dir / "asr"
        raw_dir = asr_dir / "raw"
        logs_dir = run_dir / "logs"
        checkpoints_dir = run_dir / "checkpoints"
        raw_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        stage_log = logs_dir / "asr.jsonl"

        file_results: list[FileTranscript] = []
        errors: list[str] = []
        transcribed_count = 0
        skipped_count = 0

        for file_payload in files_payload:
            normalized_name = str(file_payload.get("normalized_name", ""))
            normalized_path = Path(str(file_payload.get("normalized_path", ""))).resolve()
            source_name = str(file_payload.get("source_name", normalized_name))

            if not normalized_name or not normalized_path:
                message = f"Malformed ingestion file payload: {file_payload}"
                errors.append(message)
                self._log(stage_log, "asr_error", message=message)
                if self._fail_fast:
                    raise ASRError(message)
                continue

            transcript_file_path = raw_dir / f"{Path(normalized_name).stem}.segments.json"
            checkpoint_path = checkpoints_dir / f"{Path(normalized_name).stem}.asr.done"

            if checkpoint_path.exists() and transcript_file_path.exists():
                transcript_payload = json.loads(transcript_file_path.read_text(encoding="utf-8"))
                segments_payload = transcript_payload.get("segments", [])
                segments = [TranscriptSegment(**segment) for segment in segments_payload]
                file_results.append(
                    FileTranscript(
                        source_name=source_name,
                        normalized_name=normalized_name,
                        normalized_path=str(normalized_path),
                        language=str(transcript_payload.get("language", language)),
                        segments=segments,
                    )
                )
                skipped_count += 1
                self._log(
                    stage_log,
                    "asr_skip_checkpoint",
                    normalized_name=normalized_name,
                    transcript_file=str(transcript_file_path),
                )
                continue

            try:
                self._log(
                    stage_log,
                    "asr_start",
                    normalized_name=normalized_name,
                    normalized_path=str(normalized_path),
                )
                segments = self._transcribe_with_retries(
                    normalized_path,
                    language=language,
                )
                validate_segments(segments)
                file_transcript = FileTranscript(
                    source_name=source_name,
                    normalized_name=normalized_name,
                    normalized_path=str(normalized_path),
                    language=language,
                    segments=segments,
                )
                transcript_file_path.write_text(
                    json.dumps(
                        {
                            "source_name": source_name,
                            "normalized_name": normalized_name,
                            "normalized_path": str(normalized_path),
                            "language": language,
                            "segments": [asdict(segment) for segment in segments],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                checkpoint_path.write_text("ok\n", encoding="utf-8")
                file_results.append(file_transcript)
                transcribed_count += 1
                self._log(
                    stage_log,
                    "asr_done",
                    normalized_name=normalized_name,
                    segment_count=len(segments),
                )
            except Exception as exc:  # noqa: BLE001
                message = f"ASR failed for {normalized_name}: {exc}"
                errors.append(message)
                self._log(
                    stage_log,
                    "asr_error",
                    normalized_name=normalized_name,
                    error=str(exc),
                )
                if self._fail_fast:
                    raise ASRError(message) from exc

        aggregate_path = asr_dir / "raw_transcript_segments.json"
        aggregate_path.write_text(
            json.dumps(
                {
                    "language": language,
                    "files": [
                        {
                            "source_name": item.source_name,
                            "normalized_name": item.normalized_name,
                            "normalized_path": item.normalized_path,
                            "segment_count": len(item.segments),
                            "segments": [asdict(segment) for segment in item.segments],
                        }
                        for item in file_results
                    ],
                    "error_count": len(errors),
                    "errors": errors,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        return ASRRunResult(
            files=file_results,
            errors=errors,
            transcribed_count=transcribed_count,
            skipped_count=skipped_count,
        )

    def _transcribe_with_retries(
        self, audio_path: Path, *, language: str
    ) -> list[TranscriptSegment]:
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._transcriber.transcribe(
                    audio_path,
                    language=language,
                    vad_filter=self._vad_filter,
                    beam_size=self._beam_size,
                )
            except Exception as exc:  # noqa: BLE001
                if attempt >= self._max_retries:
                    raise ASRError(
                        f"Transcription failed after {attempt} attempts: {audio_path}"
                    ) from exc
                sleep_for = self._retry_backoff_sec * (2 ** (attempt - 1))
                time.sleep(sleep_for)

    def _log(self, log_path: Path, event: str, **payload: object) -> None:
        entry = {
            "event": event,
            "payload": payload,
            "time": time.time(),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
