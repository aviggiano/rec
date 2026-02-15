from __future__ import annotations

from pathlib import Path

from rec_pipeline.ingestion import (
    AudioProbe,
    IngestionProcessor,
    compute_offsets,
    scan_input_files,
    sort_audio_paths,
)


def test_sort_audio_paths_is_deterministic() -> None:
    paths = [
        Path("nested/B.mp3"),
        Path("a.wav"),
        Path("nested/a.wav"),
        Path("A.flac"),
    ]

    sorted_paths = sort_audio_paths(paths)

    assert [path.as_posix() for path in sorted_paths] == [
        "A.flac",
        "a.wav",
        "nested/a.wav",
        "nested/B.mp3",
    ]


def test_compute_offsets() -> None:
    assert compute_offsets([1.5, 2.0, 0.5]) == [0.0, 1.5, 3.5]


def test_ingestion_resume_behavior(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    first_audio = input_dir / "b.mp3"
    second_audio = input_dir / "a.wav"
    first_audio.write_bytes(b"fake-mp3")
    second_audio.write_bytes(b"fake-wav")

    durations = {
        first_audio.resolve(): 2.0,
        second_audio.resolve(): 1.0,
    }
    calls = {"normalize": 0, "probe": 0}

    def fake_probe(audio_path: Path) -> AudioProbe:
        calls["probe"] += 1
        return AudioProbe(
            duration_sec=durations[audio_path.resolve()],
            sample_rate_hz=44100,
            channels=2,
            codec_name="pcm_s16le",
        )

    def fake_normalize(source_path: Path, destination_path: Path) -> None:
        del source_path
        calls["normalize"] += 1
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"normalized")

    processor = IngestionProcessor(probe_fn=fake_probe, normalize_fn=fake_normalize, fail_fast=True)
    run_dir = tmp_path / "runs" / "demo"

    first_run = processor.process_directory(input_dir=input_dir, run_dir=run_dir)

    assert first_run.normalized_count == 2
    assert first_run.skipped_count == 0
    assert calls["probe"] == 2
    assert calls["normalize"] == 2
    assert [item.normalized_name for item in first_run.files] == ["0000_a.wav", "0001_b.wav"]
    assert [item.offset_start_sec for item in first_run.files] == [0.0, 1.0]

    second_run = processor.process_directory(input_dir=input_dir, run_dir=run_dir)

    assert second_run.normalized_count == 0
    assert second_run.skipped_count == 2
    assert calls["probe"] == 2
    assert calls["normalize"] == 2
    assert (run_dir / "checkpoints" / "0000_a.normalize.done").exists()
    assert (run_dir / "checkpoints" / "0001_b.normalize.done").exists()
    assert (run_dir / "ingestion_manifest.json").exists()


def test_scan_excludes_run_artifacts_inside_input_tree(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    run_dir = input_dir / "artifacts" / "demo"
    source_file = input_dir / "meeting.wav"
    run_artifact = run_dir / "normalized" / "0000_meeting.wav"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    run_artifact.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_bytes(b"source")
    run_artifact.write_bytes(b"derived")

    scanned = scan_input_files(input_dir, excluded_dirs=[run_dir])

    assert scanned == [source_file]


def test_rerun_normalization_when_checkpoint_is_missing(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    source_file = input_dir / "clip.wav"
    source_file.write_bytes(b"source")

    calls = {"normalize": 0}

    def fake_probe(audio_path: Path) -> AudioProbe:
        del audio_path
        return AudioProbe(
            duration_sec=1.0, sample_rate_hz=16000, channels=1, codec_name="pcm_s16le"
        )

    def fake_normalize(source_path: Path, destination_path: Path) -> None:
        del source_path
        calls["normalize"] += 1
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"normalized")

    processor = IngestionProcessor(probe_fn=fake_probe, normalize_fn=fake_normalize, fail_fast=True)
    run_dir = tmp_path / "runs" / "demo"

    first = processor.process_directory(input_dir=input_dir, run_dir=run_dir)
    assert first.normalized_count == 1
    assert calls["normalize"] == 1

    checkpoint = run_dir / "checkpoints" / "0000_clip.normalize.done"
    checkpoint.unlink()
    second = processor.process_directory(input_dir=input_dir, run_dir=run_dir)

    assert second.normalized_count == 1
    assert calls["normalize"] == 2


def test_file_index_stays_stable_when_early_file_fails_then_succeeds(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    early_file = input_dir / "a.wav"
    later_file = input_dir / "b.wav"
    early_file.write_bytes(b"a")
    later_file.write_bytes(b"b")

    durations = {early_file.resolve(): 1.0, later_file.resolve(): 2.0}
    failed_once = {"value": False}
    normalize_calls = {"a": 0, "b": 0}

    def fake_probe(audio_path: Path) -> AudioProbe:
        return AudioProbe(
            duration_sec=durations[audio_path.resolve()],
            sample_rate_hz=16000,
            channels=1,
            codec_name="pcm_s16le",
        )

    def fake_normalize(source_path: Path, destination_path: Path) -> None:
        name = source_path.name
        if name == "a.wav" and not failed_once["value"]:
            failed_once["value"] = True
            raise RuntimeError("transient failure")
        normalize_calls["a" if name == "a.wav" else "b"] += 1
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"normalized")

    processor = IngestionProcessor(
        probe_fn=fake_probe, normalize_fn=fake_normalize, fail_fast=False
    )
    run_dir = tmp_path / "runs" / "stable-index"

    first = processor.process_directory(input_dir=input_dir, run_dir=run_dir)
    assert first.normalized_count == 1
    assert [item.normalized_name for item in first.files] == ["0001_b.wav"]

    second = processor.process_directory(input_dir=input_dir, run_dir=run_dir)
    assert [item.normalized_name for item in second.files] == ["0000_a.wav", "0001_b.wav"]
    assert second.skipped_count == 1
    assert normalize_calls["a"] == 1
    assert normalize_calls["b"] == 1


def test_opus_input_is_converted_to_wav_artifact(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    source_file = input_dir / "meeting.opus"
    source_file.write_bytes(b"opus-bytes")

    def fake_probe(audio_path: Path) -> AudioProbe:
        del audio_path
        return AudioProbe(duration_sec=3.0, sample_rate_hz=48000, channels=1, codec_name="opus")

    def fake_normalize(source_path: Path, destination_path: Path) -> None:
        del source_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(b"wav-bytes")

    processor = IngestionProcessor(probe_fn=fake_probe, normalize_fn=fake_normalize, fail_fast=True)
    run_dir = tmp_path / "runs" / "opus-demo"

    result = processor.process_directory(input_dir=input_dir, run_dir=run_dir)

    assert result.normalized_count == 1
    assert result.errors == []
    assert len(result.files) == 1
    assert result.files[0].source_name == "meeting.opus"
    assert result.files[0].normalized_name == "0000_meeting.wav"
    assert Path(result.files[0].normalized_path).name == "0000_meeting.wav"
