from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
}


class IngestionError(RuntimeError):
    """Raised when ingestion fails and fail-fast mode is enabled."""


@dataclass(frozen=True)
class AudioProbe:
    duration_sec: float
    sample_rate_hz: int | None
    channels: int | None
    codec_name: str | None


@dataclass(frozen=True)
class AudioFileMetadata:
    index: int
    source_path: str
    source_name: str
    normalized_name: str
    normalized_path: str
    duration_sec: float
    offset_start_sec: float
    offset_end_sec: float
    sample_rate_hz: int | None
    channels: int | None
    codec_name: str | None
    status: str

    @classmethod
    def from_json(cls, metadata_path: Path) -> AudioFileMetadata:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return cls(**payload)

    def to_json(self, metadata_path: Path) -> None:
        metadata_path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class IngestionResult:
    files: list[AudioFileMetadata]
    errors: list[str]
    normalized_count: int
    skipped_count: int


ProbeFn = Callable[[Path], AudioProbe]
NormalizeFn = Callable[[Path, Path], None]


def sort_audio_paths(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda value: value.as_posix().lower())


def compute_offsets(durations_sec: list[float]) -> list[float]:
    offset = 0.0
    offsets: list[float] = []
    for duration in durations_sec:
        offsets.append(offset)
        offset += duration
    return offsets


def scan_input_files(input_dir: Path, *, excluded_dirs: list[Path] | None = None) -> list[Path]:
    excluded = [path.resolve() for path in (excluded_dirs or [])]
    return sort_audio_paths(
        [
            path
            for path in input_dir.rglob("*")
            if path.is_file() and not _is_excluded_path(path.resolve(), excluded)
        ]
    )


def _is_excluded_path(path: Path, excluded_dirs: list[Path]) -> bool:
    return any(path.is_relative_to(excluded_dir) for excluded_dir in excluded_dirs)


def probe_audio_ffprobe(audio_path: Path) -> AudioProbe:
    completed = subprocess.run(  # noqa: S603
        [
            "ffprobe",  # noqa: S607
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-print_format",
            "json",
            str(audio_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise IngestionError(f"ffprobe failed for {audio_path}: {completed.stderr.strip()}")

    payload = json.loads(completed.stdout)
    format_payload = payload.get("format", {})
    duration_raw = format_payload.get("duration")
    if duration_raw is None:
        raise IngestionError(f"ffprobe did not return duration for {audio_path}")

    duration_sec = float(duration_raw)
    streams = payload.get("streams", [])
    audio_stream: dict[str, object] = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        {},
    )

    sample_rate_value = audio_stream.get("sample_rate")
    sample_rate_hz: int | None
    if isinstance(sample_rate_value, (int, str)):
        sample_rate_hz = int(sample_rate_value)
    else:
        sample_rate_hz = None

    channels_value = audio_stream.get("channels")
    channels: int | None
    if isinstance(channels_value, (int, str)):
        channels = int(channels_value)
    else:
        channels = None

    codec_name_value = audio_stream.get("codec_name")
    codec_name = str(codec_name_value) if codec_name_value else None

    return AudioProbe(
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        codec_name=codec_name,
    )


def normalize_audio_ffmpeg(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(  # noqa: S603
        [
            "ffmpeg",  # noqa: S607
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise IngestionError(
            f"ffmpeg normalize failed for {source_path}: {completed.stderr.strip()}"
        )


def _slugify(filename_stem: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", filename_stem.strip().lower()).strip("-")
    return normalized or "audio"


def _log_stage(log_path: Path, event: str, **payload: object) -> None:
    entry: dict[str, object] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "event": event,
        **payload,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


@dataclass
class _PreparedFile:
    index: int
    source_path: Path
    source_name: str
    normalized_name: str
    normalized_path: Path
    metadata_path: Path
    checkpoint_path: Path
    duration_sec: float
    sample_rate_hz: int | None
    channels: int | None
    codec_name: str | None
    status: str


class IngestionProcessor:
    def __init__(
        self,
        *,
        probe_fn: ProbeFn = probe_audio_ffprobe,
        normalize_fn: NormalizeFn = normalize_audio_ffmpeg,
        fail_fast: bool = False,
    ) -> None:
        self._probe_fn = probe_fn
        self._normalize_fn = normalize_fn
        self._fail_fast = fail_fast

    def process_directory(self, input_dir: Path, run_dir: Path) -> IngestionResult:
        input_dir = input_dir.resolve()
        run_dir = run_dir.resolve()
        normalized_dir = run_dir / "normalized"
        metadata_dir = run_dir / "metadata"
        checkpoints_dir = run_dir / "checkpoints"
        logs_dir = run_dir / "logs"
        stage_log_path = logs_dir / "stages.jsonl"

        normalized_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        candidate_files = scan_input_files(input_dir, excluded_dirs=[run_dir])
        prepared_files: list[_PreparedFile] = []
        errors: list[str] = []
        normalized_count = 0
        skipped_count = 0

        valid_audio_files: list[Path] = []
        for source_path in candidate_files:
            extension = source_path.suffix.lower()
            if extension not in SUPPORTED_EXTENSIONS:
                message = f"Unsupported file format: {source_path}"
                errors.append(message)
                _log_stage(stage_log_path, "ingestion_unsupported", source_path=str(source_path))
                if self._fail_fast:
                    raise IngestionError(message)
                continue
            valid_audio_files.append(source_path)

        for index, source_path in enumerate(valid_audio_files):
            slug = _slugify(source_path.stem)
            normalized_name = f"{index:04d}_{slug}.wav"
            normalized_path = normalized_dir / normalized_name
            metadata_path = metadata_dir / f"{index:04d}_{slug}.json"
            checkpoint_path = checkpoints_dir / f"{index:04d}_{slug}.normalize.done"

            try:
                if checkpoint_path.exists() and metadata_path.exists() and normalized_path.exists():
                    existing = AudioFileMetadata.from_json(metadata_path)
                    prepared_files.append(
                        _PreparedFile(
                            index=index,
                            source_path=source_path,
                            source_name=source_path.name,
                            normalized_name=normalized_name,
                            normalized_path=normalized_path,
                            metadata_path=metadata_path,
                            checkpoint_path=checkpoint_path,
                            duration_sec=existing.duration_sec,
                            sample_rate_hz=existing.sample_rate_hz,
                            channels=existing.channels,
                            codec_name=existing.codec_name,
                            status="skipped",
                        )
                    )
                    skipped_count += 1
                    _log_stage(
                        stage_log_path,
                        "normalize_skip_checkpoint",
                        source_path=str(source_path),
                        normalized_path=str(normalized_path),
                    )
                    continue

                _log_stage(stage_log_path, "probe_start", source_path=str(source_path))
                probe = self._probe_fn(source_path)
                _log_stage(
                    stage_log_path,
                    "probe_done",
                    source_path=str(source_path),
                    duration_sec=probe.duration_sec,
                )

                _log_stage(
                    stage_log_path,
                    "normalize_start",
                    source_path=str(source_path),
                    normalized_path=str(normalized_path),
                )
                self._normalize_fn(source_path, normalized_path)
                _log_stage(
                    stage_log_path,
                    "normalize_done",
                    source_path=str(source_path),
                    normalized_path=str(normalized_path),
                )

                prepared_files.append(
                    _PreparedFile(
                        index=index,
                        source_path=source_path,
                        source_name=source_path.name,
                        normalized_name=normalized_name,
                        normalized_path=normalized_path,
                        metadata_path=metadata_path,
                        checkpoint_path=checkpoint_path,
                        duration_sec=probe.duration_sec,
                        sample_rate_hz=probe.sample_rate_hz,
                        channels=probe.channels,
                        codec_name=probe.codec_name,
                        status="normalized",
                    )
                )
                normalized_count += 1
            except Exception as exc:  # noqa: BLE001
                message = f"Failed to ingest {source_path}: {exc}"
                errors.append(message)
                _log_stage(
                    stage_log_path,
                    "ingestion_error",
                    source_path=str(source_path),
                    error=str(exc),
                )
                if self._fail_fast:
                    raise IngestionError(message) from exc

        offsets = compute_offsets([entry.duration_sec for entry in prepared_files])
        artifacts: list[AudioFileMetadata] = []
        for entry, offset in zip(prepared_files, offsets, strict=True):
            metadata = AudioFileMetadata(
                index=entry.index,
                source_path=str(entry.source_path),
                source_name=entry.source_name,
                normalized_name=entry.normalized_name,
                normalized_path=str(entry.normalized_path),
                duration_sec=entry.duration_sec,
                offset_start_sec=offset,
                offset_end_sec=offset + entry.duration_sec,
                sample_rate_hz=entry.sample_rate_hz,
                channels=entry.channels,
                codec_name=entry.codec_name,
                status=entry.status,
            )
            metadata.to_json(entry.metadata_path)
            entry.checkpoint_path.write_text("ok\n", encoding="utf-8")
            artifacts.append(metadata)

        manifest_path = run_dir / "ingestion_manifest.json"
        manifest = {
            "input_dir": str(input_dir),
            "run_dir": str(run_dir),
            "files": [asdict(item) for item in artifacts],
            "normalized_count": normalized_count,
            "skipped_count": skipped_count,
            "error_count": len(errors),
            "errors": errors,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        return IngestionResult(
            files=artifacts,
            errors=errors,
            normalized_count=normalized_count,
            skipped_count=skipped_count,
        )
