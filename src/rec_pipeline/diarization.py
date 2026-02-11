from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class DiarizationError(RuntimeError):
    """Raised when speaker diarization fails."""


@dataclass(frozen=True)
class DiarizationTurn:
    start_sec: float
    end_sec: float
    speaker: str


@dataclass(frozen=True)
class DiarizationResult:
    processed_files: int
    skipped_files: int
    labeled_segments: int
    errors: list[str]
    speaker_exports: list[str]


class Diarizer(Protocol):
    def diarize(self, audio_path: Path) -> list[DiarizationTurn]: ...


class PyannoteDiarizer:
    def __init__(self, *, model_name: str, hf_token: str | None) -> None:
        self._model_name = model_name
        self._hf_token = hf_token
        self._pipeline: object | None = None

    def _get_pipeline(self) -> object:
        if self._pipeline is not None:
            return self._pipeline

        if not self._hf_token:
            raise DiarizationError(
                "HUGGINGFACE_TOKEN is required for pyannote diarization model downloads. "
                "Generate a token at https://huggingface.co/settings/tokens"
            )

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:  # pragma: no cover - optional dependency in CI
            raise DiarizationError(
                "pyannote.audio is not installed. "
                "Install optional deps with: pip install -e '.[diarization]'"
            ) from exc

        self._pipeline = Pipeline.from_pretrained(self._model_name, use_auth_token=self._hf_token)
        return self._pipeline

    def diarize(self, audio_path: Path) -> list[DiarizationTurn]:
        pipeline = self._get_pipeline()
        diarization = pipeline(str(audio_path))  # type: ignore[operator]
        turns: list[DiarizationTurn] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(
                DiarizationTurn(
                    start_sec=float(turn.start),
                    end_sec=float(turn.end),
                    speaker=str(speaker),
                )
            )
        return sorted(turns, key=lambda value: (value.start_sec, value.end_sec, value.speaker))


def merge_segments_with_speakers(
    segments_payload: list[dict[str, object]],
    turns: list[DiarizationTurn],
    *,
    unknown_speaker: str = "UNKNOWN",
) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []

    for segment in segments_payload:
        start = _coerce_float(segment.get("start_sec"))
        end = _coerce_float(segment.get("end_sec"))

        best_speaker = unknown_speaker
        best_overlap = 0.0

        for turn in turns:
            overlap = min(end, turn.end_sec) - max(start, turn.start_sec)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker

        labeled = dict(segment)
        labeled["speaker"] = best_speaker if best_overlap > 0.0 else unknown_speaker
        merged.append(labeled)

    return merged


def _load_ingestion_offsets(run_dir: Path) -> dict[str, float]:
    manifest_path = run_dir / "ingestion_manifest.json"
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_payload = payload.get("files", [])
    if not isinstance(files_payload, list):
        return {}

    offsets: dict[str, float] = {}
    for item in files_payload:
        normalized_name = str(item.get("normalized_name", ""))
        if not normalized_name:
            continue
        offset = item.get("offset_start_sec")
        if isinstance(offset, (int, float)):
            offsets[normalized_name] = float(offset)
    return offsets


def _format_timestamp(seconds: float) -> str:
    total_millis = int(round(seconds * 1000))
    hours = total_millis // 3_600_000
    total_millis -= hours * 3_600_000
    minutes = total_millis // 60_000
    total_millis -= minutes * 60_000
    secs = total_millis // 1_000
    millis = total_millis - secs * 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    return cleaned or "unknown"


def _coerce_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


class SpeakerDiarizationPipeline:
    def __init__(
        self,
        *,
        diarizer: Diarizer,
        fail_fast: bool,
        unknown_speaker: str = "UNKNOWN",
    ) -> None:
        self._diarizer = diarizer
        self._fail_fast = fail_fast
        self._unknown_speaker = unknown_speaker

    def run(
        self,
        *,
        run_dir: Path,
        enabled: bool,
        export_per_speaker: bool,
    ) -> DiarizationResult:
        if not enabled:
            return DiarizationResult(
                processed_files=0,
                skipped_files=0,
                labeled_segments=0,
                errors=[],
                speaker_exports=[],
            )

        asr_aggregate_path = run_dir / "asr" / "raw_transcript_segments.json"
        if not asr_aggregate_path.exists():
            raise DiarizationError(f"Missing ASR aggregate payload: {asr_aggregate_path}")

        payload = json.loads(asr_aggregate_path.read_text(encoding="utf-8"))
        files_payload = payload.get("files", [])
        if not isinstance(files_payload, list):
            raise DiarizationError("Malformed ASR payload: files must be a list")

        diarized_dir = run_dir / "asr" / "diarized"
        checkpoints_dir = run_dir / "checkpoints"
        logs_dir = run_dir / "logs"
        diarized_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "diarization.jsonl"

        offsets = _load_ingestion_offsets(run_dir)
        speaker_lines: dict[str, list[str]] = defaultdict(list)

        merged_files: list[dict[str, object]] = []
        errors: list[str] = []
        processed_files = 0
        skipped_files = 0
        labeled_segments = 0

        ordered_files = sorted(
            files_payload, key=lambda value: str(value.get("normalized_name", ""))
        )

        for file_payload in ordered_files:
            normalized_name = str(file_payload.get("normalized_name", ""))
            normalized_path = Path(str(file_payload.get("normalized_path", ""))).resolve()
            source_name = str(file_payload.get("source_name", normalized_name))
            segments_payload = file_payload.get("segments", [])
            if not isinstance(segments_payload, list):
                continue

            output_path = diarized_dir / f"{Path(normalized_name).stem}.segments.json"
            checkpoint_path = checkpoints_dir / f"{Path(normalized_name).stem}.diarization.done"

            if checkpoint_path.exists() and output_path.exists():
                existing = json.loads(output_path.read_text(encoding="utf-8"))
                merged_files.append(existing)
                skipped_files += 1
                self._log(log_path, "diarization_skip_checkpoint", file=normalized_name)
                self._collect_speaker_lines(
                    speaker_lines=speaker_lines,
                    normalized_name=normalized_name,
                    segments=existing.get("segments", []),
                    file_offset=offsets.get(normalized_name, 0.0),
                )
                continue

            try:
                turns = self._diarizer.diarize(normalized_path)
                merged_segments = merge_segments_with_speakers(
                    [dict(segment) for segment in segments_payload],
                    turns,
                    unknown_speaker=self._unknown_speaker,
                )
                processed_files += 1
                self._log(log_path, "diarization_done", file=normalized_name, turn_count=len(turns))
            except Exception as exc:  # noqa: BLE001
                message = f"Diarization failed for {normalized_name}: {exc}"
                errors.append(message)
                self._log(log_path, "diarization_error", file=normalized_name, error=str(exc))
                if self._fail_fast:
                    raise DiarizationError(message) from exc
                merged_segments = merge_segments_with_speakers(
                    [dict(segment) for segment in segments_payload],
                    [],
                    unknown_speaker=self._unknown_speaker,
                )

            labeled_segments += len(merged_segments)
            merged_payload: dict[str, object] = {
                "source_name": source_name,
                "normalized_name": normalized_name,
                "normalized_path": str(normalized_path),
                "segments": merged_segments,
            }
            output_path.write_text(
                json.dumps(merged_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            checkpoint_path.write_text("ok\n", encoding="utf-8")
            merged_files.append(merged_payload)

            self._collect_speaker_lines(
                speaker_lines=speaker_lines,
                normalized_name=normalized_name,
                segments=merged_segments,
                file_offset=offsets.get(normalized_name, 0.0),
            )

        aggregate_path = run_dir / "asr" / "speaker_transcript_segments.json"
        aggregate_path.write_text(
            json.dumps(
                {
                    "files": merged_files,
                    "error_count": len(errors),
                    "errors": errors,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        speaker_exports: list[str] = []
        if export_per_speaker:
            speaker_exports = self._write_speaker_exports(
                run_dir=run_dir, speaker_lines=speaker_lines
            )

        return DiarizationResult(
            processed_files=processed_files,
            skipped_files=skipped_files,
            labeled_segments=labeled_segments,
            errors=errors,
            speaker_exports=speaker_exports,
        )

    def _collect_speaker_lines(
        self,
        *,
        speaker_lines: dict[str, list[str]],
        normalized_name: str,
        segments: object,
        file_offset: float,
    ) -> None:
        del normalized_name
        if not isinstance(segments, list):
            return

        for segment in segments:
            if not isinstance(segment, dict):
                continue
            speaker = str(segment.get("speaker", self._unknown_speaker))
            text = str(segment.get("text", "")).strip()
            start_sec = _coerce_float(segment.get("start_sec"))
            absolute_start = file_offset + start_sec
            line = f"[{_format_timestamp(absolute_start)}] {text}"
            speaker_lines[speaker].append(line)

    def _write_speaker_exports(
        self,
        *,
        run_dir: Path,
        speaker_lines: dict[str, list[str]],
    ) -> list[str]:
        output_dir = run_dir / "artifacts" / "speakers"
        output_dir.mkdir(parents=True, exist_ok=True)

        exported: list[str] = []
        for speaker, lines in sorted(speaker_lines.items(), key=lambda item: item[0]):
            filename = f"speaker_{_safe_filename(speaker)}.txt"
            path = output_dir / filename
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            exported.append(str(path))
        return exported

    def _log(self, log_path: Path, event: str, **payload: object) -> None:
        entry = {"event": event, "payload": payload}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
