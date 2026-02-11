from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rec_pipeline.evaluation import (
    EvaluationError,
    compute_cer,
    compute_der_proxy,
    compute_summary_traceability_coverage,
    compute_wer,
    default_thresholds,
    evaluate_dataset,
    validate_evaluation_report_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_metric_calculations() -> None:
    wer = compute_wer("bom dia equipe", "bom equipe")
    cer = compute_cer("abc", "adc")
    der_proxy = compute_der_proxy(["A", "B", "A"], ["A", "A", "A"])
    coverage = compute_summary_traceability_coverage(
        {
            "overview": {"text": "ok", "citations": ["00:00:01.000"]},
            "key_points": [{"text": "ok", "citations": ["00:00:02.000"]}],
            "decisions": [{"text": "ok", "citations": ["00:00:03.000"]}],
            "action_items": [{"text": "ok", "citations": ["00:00:04.000"]}],
            "open_questions": [{"text": "ok", "citations": ["00:00:05.000"]}],
        }
    )

    assert round(wer, 3) == 0.333
    assert round(cer, 3) == 0.333
    assert round(der_proxy, 3) == 0.333
    assert coverage == 1.0

    # Label IDs are arbitrary; perfect swapped labels should still score as perfect diarization.
    swapped_der_proxy = compute_der_proxy(
        ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01"],
        ["SPEAKER_01", "SPEAKER_01", "SPEAKER_00", "SPEAKER_00"],
    )
    assert swapped_der_proxy == 0.0


def test_evaluation_report_schema_and_artifacts(tmp_path: Path) -> None:
    dataset_path = PROJECT_ROOT / "tests" / "fixtures" / "eval_dataset.json"
    output_dir = tmp_path / "reports"

    report = evaluate_dataset(
        dataset_path=dataset_path,
        output_dir=output_dir,
        thresholds=default_thresholds(blocking_gates=False),
    )

    assert report.scenario_count == 1
    assert report.long_multifile_scenario_count == 0
    assert report.gate_status in {"passed", "warning"}

    payload = json.loads((output_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    validate_evaluation_report_payload(payload)
    assert (output_dir / "evaluation_report.md").exists()


def test_evaluate_command_integration(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-report"
    dataset_path = PROJECT_ROOT / "tests" / "fixtures" / "eval_dataset.json"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "rec_pipeline.cli",
            "evaluate",
            "--dataset",
            str(dataset_path),
            "--output",
            str(output_dir),
        ],  # noqa: S607
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert (output_dir / "evaluation_report.json").exists()
    assert (output_dir / "evaluation_report.md").exists()


def test_evaluate_dataset_reports_malformed_json(tmp_path: Path) -> None:
    dataset_path = tmp_path / "broken.json"
    dataset_path.write_text("{", encoding="utf-8")

    with pytest.raises(EvaluationError):
        evaluate_dataset(
            dataset_path=dataset_path,
            output_dir=tmp_path / "reports",
            thresholds=default_thresholds(blocking_gates=False),
        )


def test_evaluate_command_handles_malformed_dataset_without_traceback(tmp_path: Path) -> None:
    output_dir = tmp_path / "cli-broken-report"
    dataset_path = tmp_path / "broken.json"
    dataset_path.write_text("{", encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "rec_pipeline.cli",
            "evaluate",
            "--dataset",
            str(dataset_path),
            "--output",
            str(output_dir),
        ],  # noqa: S607
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "Evaluation failed:" in completed.stdout
    assert "Traceback" not in completed.stderr
