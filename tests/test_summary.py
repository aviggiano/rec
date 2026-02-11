from __future__ import annotations

import json
from pathlib import Path

from rec_pipeline.summary import (
    ChunkSummary,
    SummaryItem,
    SummaryPipeline,
    TranscriptSegment,
    build_summary_payload,
    chunk_transcript,
    synthesize_global_summary,
    validate_summary_payload,
)


def test_chunking_respects_time_and_token_boundaries() -> None:
    segments = [
        TranscriptSegment("um dois tres", 0.0, 1.0),
        TranscriptSegment("quatro cinco seis", 1.0, 2.0),
        TranscriptSegment("sete oito nove", 2.0, 3.0),
    ]

    chunks = chunk_transcript(segments, max_chunk_tokens=5, max_chunk_seconds=1.5)

    assert len(chunks) == 3
    assert [chunk.start_sec for chunk in chunks] == [0.0, 1.0, 2.0]
    assert all(chunk.estimated_tokens <= 3 for chunk in chunks)


def test_merge_logic_populates_all_sections_with_citations() -> None:
    chunk_summaries = [
        ChunkSummary(
            chunk_id=0,
            start_sec=0.0,
            end_sec=10.0,
            summary="Equipe decidiu priorizar",
            citations=["00:00:00.000"],
        ),
        ChunkSummary(
            chunk_id=1,
            start_sec=10.0,
            end_sec=20.0,
            summary="Acao proximo passo e revisar",
            citations=["00:00:10.000"],
        ),
    ]

    document = synthesize_global_summary(chunk_summaries, language="pt")

    for section in [
        document.overview,
        *document.key_points,
        *document.decisions,
        *document.action_items,
        *document.open_questions,
    ]:
        assert section.citations


def test_summary_pipeline_generates_summary_from_transcript_fixture(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "summary"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)

    transcript_payload = {
        "schema_version": "1.0",
        "language": "pt",
        "segment_count": 3,
        "segments": [
            {
                "id": 0,
                "source_name": "a.wav",
                "normalized_name": "0000_a.wav",
                "text": "A equipe decidiu aprovar a pauta.",
                "timing": {
                    "relative_start_sec": 0.0,
                    "relative_end_sec": 1.0,
                    "absolute_start_sec": 0.0,
                    "absolute_end_sec": 1.0,
                },
                "metrics": {"avg_logprob": -0.1, "confidence": 0.9, "no_speech_prob": 0.01},
            },
            {
                "id": 1,
                "source_name": "a.wav",
                "normalized_name": "0000_a.wav",
                "text": "Proximo passo e enviar o resumo.",
                "timing": {
                    "relative_start_sec": 1.0,
                    "relative_end_sec": 2.0,
                    "absolute_start_sec": 1.0,
                    "absolute_end_sec": 2.0,
                },
                "metrics": {"avg_logprob": -0.1, "confidence": 0.9, "no_speech_prob": 0.01},
            },
            {
                "id": 2,
                "source_name": "a.wav",
                "normalized_name": "0000_a.wav",
                "text": "Qual o prazo final?",
                "timing": {
                    "relative_start_sec": 2.0,
                    "relative_end_sec": 3.0,
                    "absolute_start_sec": 2.0,
                    "absolute_end_sec": 3.0,
                },
                "metrics": {"avg_logprob": -0.1, "confidence": 0.9, "no_speech_prob": 0.01},
            },
        ],
    }
    (artifacts_dir / "transcript.json").write_text(
        json.dumps(transcript_payload),
        encoding="utf-8",
    )

    pipeline = SummaryPipeline(
        model_backend="heuristic",
        model_name="unused",
        ollama_base_url="http://localhost:11434",
        llamacpp_server_url="http://localhost:8080",
        max_chunk_tokens=20,
        max_chunk_seconds=60,
        fail_fast=True,
    )

    result = pipeline.build(run_dir=run_dir, language="pt")

    assert result.markdown_path.exists()
    assert result.json_path.exists()

    summary_payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    validate_summary_payload(summary_payload)

    assert summary_payload["overview"]["citations"]
    assert summary_payload["key_points"]
    assert summary_payload["decisions"]
    assert summary_payload["action_items"]
    assert summary_payload["open_questions"]

    built = build_summary_payload(synthesize_global_summary([], language="pt"))
    assert built["language"] == "pt"


def test_empty_transcript_still_produces_non_empty_sections() -> None:
    document = synthesize_global_summary([], language="pt")

    assert isinstance(document.overview, SummaryItem)
    assert document.key_points
    assert document.decisions
    assert document.action_items
    assert document.open_questions


def test_summary_pipeline_skips_model_inference_when_checkpoints_are_current(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "skip"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    transcript_payload = {
        "schema_version": "1.0",
        "language": "pt",
        "segment_count": 1,
        "segments": [
            {
                "id": 0,
                "source_name": "a.wav",
                "normalized_name": "0000_a.wav",
                "text": "texto",
                "timing": {
                    "relative_start_sec": 0.0,
                    "relative_end_sec": 1.0,
                    "absolute_start_sec": 0.0,
                    "absolute_end_sec": 1.0,
                },
                "metrics": {"avg_logprob": -0.1, "confidence": 0.9, "no_speech_prob": 0.01},
            }
        ],
    }
    (artifacts_dir / "transcript.json").write_text(json.dumps(transcript_payload), encoding="utf-8")

    initial_pipeline = SummaryPipeline(
        model_backend="heuristic",
        model_name="unused",
        ollama_base_url="http://localhost:11434",
        llamacpp_server_url="http://localhost:8080",
        max_chunk_tokens=20,
        max_chunk_seconds=60,
        fail_fast=True,
    )
    first_result = initial_pipeline.build(run_dir=run_dir, language="pt")
    assert set(first_result.generated_files) == {"summary.md", "summary.json"}

    class FailingModel:
        def generate(self, prompt: str) -> str:
            del prompt
            raise RuntimeError("should not be called")

    second_pipeline = SummaryPipeline(
        model_backend="heuristic",
        model_name="unused",
        ollama_base_url="http://localhost:11434",
        llamacpp_server_url="http://localhost:8080",
        max_chunk_tokens=20,
        max_chunk_seconds=60,
        fail_fast=True,
        model_override=FailingModel(),
    )
    second_result = second_pipeline.build(run_dir=run_dir, language="pt")

    assert second_result.generated_files == []
    assert set(second_result.skipped_files) == {"summary.md", "summary.json"}
