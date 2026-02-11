from __future__ import annotations

from pathlib import Path

from rec_pipeline.ingestion import AudioProbe, IngestionProcessor, compute_offsets, sort_audio_paths


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
