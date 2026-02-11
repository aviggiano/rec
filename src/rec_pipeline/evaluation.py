from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path


class EvaluationError(RuntimeError):
    """Raised when evaluation fails."""


@dataclass(frozen=True)
class EvaluationThresholds:
    max_wer: float
    max_cer: float
    max_der_proxy: float
    min_traceability_coverage: float
    blocking_gates: bool


@dataclass(frozen=True)
class ScenarioMetrics:
    scenario_id: str
    multi_file: bool
    wer: float
    cer: float
    der_proxy: float
    traceability_coverage: float


@dataclass(frozen=True)
class EvaluationReport:
    dataset_path: str
    scenario_count: int
    long_multifile_scenario_count: int
    thresholds: EvaluationThresholds
    metrics: list[ScenarioMetrics]
    aggregate: dict[str, float]
    violations: list[str]
    gate_status: str


def default_thresholds(*, blocking_gates: bool) -> EvaluationThresholds:
    return EvaluationThresholds(
        max_wer=0.35,
        max_cer=0.25,
        max_der_proxy=0.40,
        min_traceability_coverage=0.80,
        blocking_gates=blocking_gates,
    )


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _levenshtein_distance(reference: list[str], hypothesis: list[str]) -> int:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    matrix = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        matrix[i][0] = i
    for j in range(cols):
        matrix[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            substitution_cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + substitution_cost,
            )

    return matrix[-1][-1]


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_tokens = _normalize_text(reference).split()
    hyp_tokens = _normalize_text(hypothesis).split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    distance = _levenshtein_distance(ref_tokens, hyp_tokens)
    return distance / len(ref_tokens)


def compute_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(_normalize_text(reference).replace(" ", ""))
    hyp_chars = list(_normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    distance = _levenshtein_distance(ref_chars, hyp_chars)
    return distance / len(ref_chars)


def compute_der_proxy(reference_speakers: list[str], predicted_speakers: list[str]) -> float:
    if not reference_speakers and not predicted_speakers:
        return 0.0

    total = max(len(reference_speakers), len(predicted_speakers))
    aligned_reference = [
        reference_speakers[index] if index < len(reference_speakers) else "<MISSING>"
        for index in range(total)
    ]
    aligned_predicted = [
        predicted_speakers[index] if index < len(predicted_speakers) else "<MISSING>"
        for index in range(total)
    ]

    predicted_labels = sorted(set(aligned_predicted))
    reference_labels = sorted(set(aligned_reference))
    if len(reference_labels) < len(predicted_labels):
        additional_count = len(predicted_labels) - len(reference_labels)
        reference_labels = reference_labels + [
            f"<UNMAPPED_{index}>" for index in range(additional_count)
        ]

    best_mismatches = total
    for mapping_target in permutations(reference_labels, len(predicted_labels)):
        mapping = dict(zip(predicted_labels, mapping_target, strict=True))
        mismatches = 0
        for reference, predicted in zip(aligned_reference, aligned_predicted, strict=True):
            if reference != mapping[predicted]:
                mismatches += 1
        if mismatches < best_mismatches:
            best_mismatches = mismatches

    return best_mismatches / total


def compute_summary_traceability_coverage(summary_payload: dict[str, object]) -> float:
    section_names = ["overview", "key_points", "decisions", "action_items", "open_questions"]
    if not section_names:
        return 1.0

    covered = 0
    for section_name in section_names:
        section_value = summary_payload.get(section_name)
        if section_name == "overview":
            if _section_has_citation(section_value):
                covered += 1
            continue

        if isinstance(section_value, list) and any(
            _section_has_citation(item) for item in section_value
        ):
            covered += 1

    return covered / len(section_names)


def _section_has_citation(section_value: object) -> bool:
    if not isinstance(section_value, dict):
        return False
    citations = section_value.get("citations")
    return isinstance(citations, list) and any(isinstance(item, str) and item for item in citations)


def evaluate_dataset(
    *,
    dataset_path: Path,
    output_dir: Path,
    thresholds: EvaluationThresholds,
) -> EvaluationReport:
    if not dataset_path.exists():
        raise EvaluationError(f"Evaluation dataset not found: {dataset_path}")

    try:
        dataset_content = dataset_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationError(f"Failed to read evaluation dataset '{dataset_path}': {exc}") from exc
    try:
        payload = json.loads(dataset_content)
    except json.JSONDecodeError as exc:
        raise EvaluationError(
            f"Malformed evaluation dataset JSON in '{dataset_path}': {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EvaluationError("Malformed evaluation dataset: top-level JSON must be an object")
    scenarios_payload = payload.get("scenarios", [])
    if not isinstance(scenarios_payload, list):
        raise EvaluationError("Malformed dataset: 'scenarios' must be a list")

    scenario_metrics: list[ScenarioMetrics] = []
    long_multifile_count = 0

    for scenario in scenarios_payload:
        if not isinstance(scenario, dict):
            continue

        scenario_id = str(scenario.get("id", f"scenario-{len(scenario_metrics)}"))
        multi_file = bool(scenario.get("multi_file", False))
        source_files = scenario.get("source_files", [])
        if multi_file and isinstance(source_files, list) and len(source_files) > 1:
            long_multifile_count += 1

        reference_transcript = str(scenario.get("reference_transcript", ""))
        predicted_transcript = str(scenario.get("predicted_transcript", ""))

        reference_speakers = scenario.get("reference_speakers", [])
        predicted_speakers = scenario.get("predicted_speakers", [])
        reference_speaker_labels = (
            [str(item) for item in reference_speakers]
            if isinstance(reference_speakers, list)
            else []
        )
        predicted_speaker_labels = (
            [str(item) for item in predicted_speakers]
            if isinstance(predicted_speakers, list)
            else []
        )

        summary_payload = scenario.get("summary", {})
        if not isinstance(summary_payload, dict):
            summary_payload = {}

        scenario_metrics.append(
            ScenarioMetrics(
                scenario_id=scenario_id,
                multi_file=multi_file,
                wer=compute_wer(reference_transcript, predicted_transcript),
                cer=compute_cer(reference_transcript, predicted_transcript),
                der_proxy=compute_der_proxy(reference_speaker_labels, predicted_speaker_labels),
                traceability_coverage=compute_summary_traceability_coverage(summary_payload),
            )
        )

    if not scenario_metrics:
        raise EvaluationError("Evaluation dataset has no valid scenarios")

    aggregate = {
        "wer": sum(item.wer for item in scenario_metrics) / len(scenario_metrics),
        "cer": sum(item.cer for item in scenario_metrics) / len(scenario_metrics),
        "der_proxy": sum(item.der_proxy for item in scenario_metrics) / len(scenario_metrics),
        "traceability_coverage": sum(item.traceability_coverage for item in scenario_metrics)
        / len(scenario_metrics),
    }

    violations: list[str] = []
    if aggregate["wer"] > thresholds.max_wer:
        violations.append(
            f"WER {aggregate['wer']:.3f} exceeded max threshold {thresholds.max_wer:.3f}"
        )
    if aggregate["cer"] > thresholds.max_cer:
        violations.append(
            f"CER {aggregate['cer']:.3f} exceeded max threshold {thresholds.max_cer:.3f}"
        )
    if aggregate["der_proxy"] > thresholds.max_der_proxy:
        violations.append(
            "DER proxy "
            f"{aggregate['der_proxy']:.3f} exceeded max threshold {thresholds.max_der_proxy:.3f}"
        )
    if aggregate["traceability_coverage"] < thresholds.min_traceability_coverage:
        violations.append(
            "Traceability coverage "
            f"{aggregate['traceability_coverage']:.3f} is below minimum threshold "
            f"{thresholds.min_traceability_coverage:.3f}"
        )

    gate_status = "passed"
    if violations and thresholds.blocking_gates:
        gate_status = "failed"
    elif violations:
        gate_status = "warning"

    report = EvaluationReport(
        dataset_path=str(dataset_path),
        scenario_count=len(scenario_metrics),
        long_multifile_scenario_count=long_multifile_count,
        thresholds=thresholds,
        metrics=scenario_metrics,
        aggregate=aggregate,
        violations=violations,
        gate_status=gate_status,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = output_dir / "evaluation_report.json"
    report_markdown_path = output_dir / "evaluation_report.md"

    report_payload = {
        "dataset_path": report.dataset_path,
        "scenario_count": report.scenario_count,
        "long_multifile_scenario_count": report.long_multifile_scenario_count,
        "thresholds": asdict(report.thresholds),
        "metrics": [asdict(item) for item in report.metrics],
        "aggregate": report.aggregate,
        "violations": report.violations,
        "gate_status": report.gate_status,
    }
    validate_evaluation_report_payload(report_payload)
    report_json_path.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_markdown_path.write_text(render_evaluation_report_markdown(report), encoding="utf-8")

    return report


def validate_evaluation_report_payload(payload: dict[str, object]) -> None:
    required = {
        "dataset_path",
        "scenario_count",
        "long_multifile_scenario_count",
        "thresholds",
        "metrics",
        "aggregate",
        "violations",
        "gate_status",
    }
    missing = [key for key in required if key not in payload]
    if missing:
        raise EvaluationError(f"Evaluation report missing keys: {', '.join(sorted(missing))}")

    if payload["gate_status"] not in {"passed", "warning", "failed"}:
        raise EvaluationError("Evaluation report has invalid gate status")


def render_evaluation_report_markdown(report: EvaluationReport) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"- Dataset: `{report.dataset_path}`",
        f"- Scenarios: {report.scenario_count}",
        f"- Long multi-file scenarios: {report.long_multifile_scenario_count}",
        f"- Gate status: **{report.gate_status}**",
        "",
        "## Aggregate Metrics",
        f"- WER: {report.aggregate['wer']:.3f}",
        f"- CER: {report.aggregate['cer']:.3f}",
        f"- DER proxy: {report.aggregate['der_proxy']:.3f}",
        f"- Summary traceability coverage: {report.aggregate['traceability_coverage']:.3f}",
        "",
        "## Quality Gates",
    ]

    if report.violations:
        lines.extend(f"- {violation}" for violation in report.violations)
    else:
        lines.append("- No threshold violations")

    lines.extend(["", "## Scenario Metrics"])
    for metric in report.metrics:
        lines.extend(
            [
                f"- `{metric.scenario_id}`",
                f"  - multi_file: {metric.multi_file}",
                f"  - wer: {metric.wer:.3f}",
                f"  - cer: {metric.cer:.3f}",
                f"  - der_proxy: {metric.der_proxy:.3f}",
                f"  - traceability_coverage: {metric.traceability_coverage:.3f}",
            ]
        )

    return "\n".join(lines).strip() + "\n"
