from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


class SummaryError(RuntimeError):
    """Raised when summary generation fails."""


class SummarySchemaError(SummaryError):
    """Raised when summary schema validation fails."""


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    absolute_start_sec: float
    absolute_end_sec: float


@dataclass(frozen=True)
class TranscriptChunk:
    chunk_id: int
    segments: list[TranscriptSegment]
    start_sec: float
    end_sec: float
    estimated_tokens: int


@dataclass(frozen=True)
class ChunkSummary:
    chunk_id: int
    start_sec: float
    end_sec: float
    summary: str
    citations: list[str]


@dataclass(frozen=True)
class SummaryItem:
    text: str
    citations: list[str]


@dataclass(frozen=True)
class SummaryDocument:
    language: str
    chunk_summaries: list[ChunkSummary]
    overview: SummaryItem
    key_points: list[SummaryItem]
    decisions: list[SummaryItem]
    action_items: list[SummaryItem]
    open_questions: list[SummaryItem]


@dataclass(frozen=True)
class SummaryArtifactResult:
    markdown_path: Path
    json_path: Path
    manifest_path: Path
    generated_files: list[str]
    skipped_files: list[str]


class SummaryModel(Protocol):
    def generate(self, prompt: str) -> str: ...


class HeuristicSummaryModel:
    def generate(self, prompt: str) -> str:
        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        if not lines:
            return "Sem conteudo relevante."
        joined = " ".join(lines)
        sentences = re.split(r"(?<=[.!?])\s+", joined)
        return " ".join(sentences[:2]).strip() or joined[:240]


class OllamaSummaryModel:
    def __init__(self, *, base_url: str, model_name: str, timeout_sec: int = 120) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout_sec = timeout_sec

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
        }
        response = _post_json(
            f"{self._base_url}/api/generate", payload, timeout_sec=self._timeout_sec
        )
        result = response.get("response")
        if not isinstance(result, str) or not result.strip():
            raise SummaryError("Ollama response did not include a valid summary string")
        return result.strip()


class LlamaCppSummaryModel:
    def __init__(self, *, base_url: str, timeout_sec: int = 120) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = timeout_sec

    def generate(self, prompt: str) -> str:
        payload = {
            "prompt": prompt,
            "n_predict": 280,
            "temperature": 0.1,
        }
        response = _post_json(
            f"{self._base_url}/completion", payload, timeout_sec=self._timeout_sec
        )
        candidates = [response.get("content"), response.get("text")]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        raise SummaryError("llama.cpp response did not include a valid summary string")


def _post_json(url: str, payload: dict[str, object], *, timeout_sec: int) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # noqa: S310
            content = response.read().decode("utf-8")
            parsed = json.loads(content)
    except urllib.error.URLError as exc:
        raise SummaryError(f"Local summary backend request failed: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SummaryError("Local summary backend returned malformed JSON")
    return parsed


def load_transcript_segments(run_dir: Path) -> list[TranscriptSegment]:
    transcript_path = run_dir / "artifacts" / "transcript.json"
    if not transcript_path.exists():
        raise SummaryError(f"Missing transcript artifact: {transcript_path}")

    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments_payload = payload.get("segments", [])
    if not isinstance(segments_payload, list):
        raise SummaryError("Malformed transcript payload: 'segments' must be a list")

    segments: list[TranscriptSegment] = []
    for segment_payload in segments_payload:
        timing = segment_payload.get("timing", {})
        if not isinstance(timing, dict):
            continue
        segments.append(
            TranscriptSegment(
                text=str(segment_payload.get("text", "")).strip(),
                absolute_start_sec=float(timing.get("absolute_start_sec", 0.0)),
                absolute_end_sec=float(timing.get("absolute_end_sec", 0.0)),
            )
        )
    return segments


def chunk_transcript(
    segments: Iterable[TranscriptSegment],
    *,
    max_chunk_tokens: int,
    max_chunk_seconds: float,
) -> list[TranscriptChunk]:
    result: list[TranscriptChunk] = []
    current: list[TranscriptSegment] = []
    current_tokens = 0
    chunk_start = 0.0

    for segment in segments:
        text_tokens = max(1, len(segment.text.split()))
        candidate_tokens = current_tokens + text_tokens
        candidate_duration = (
            segment.absolute_end_sec - chunk_start
            if current
            else segment.absolute_end_sec - segment.absolute_start_sec
        )

        should_close = bool(
            current
            and (candidate_tokens > max_chunk_tokens or candidate_duration > max_chunk_seconds)
        )
        if should_close:
            result.append(
                TranscriptChunk(
                    chunk_id=len(result),
                    segments=current,
                    start_sec=chunk_start,
                    end_sec=current[-1].absolute_end_sec,
                    estimated_tokens=current_tokens,
                )
            )
            current = []
            current_tokens = 0

        if not current:
            chunk_start = segment.absolute_start_sec

        current.append(segment)
        current_tokens += text_tokens

    if current:
        result.append(
            TranscriptChunk(
                chunk_id=len(result),
                segments=current,
                start_sec=chunk_start,
                end_sec=current[-1].absolute_end_sec,
                estimated_tokens=current_tokens,
            )
        )

    return result


def summarize_chunk(chunk: TranscriptChunk, *, language: str, model: SummaryModel) -> ChunkSummary:
    chunk_text = "\n".join(segment.text for segment in chunk.segments if segment.text)
    prompt = (
        "Resuma o trecho abaixo em 2 frases objetivas com fatos concretos."
        if language.lower().startswith("pt")
        else "Summarize the chunk below in two concrete factual sentences."
    )
    raw_output = model.generate(f"{prompt}\n\n{chunk_text}")
    summary = raw_output.strip() or (
        "Sem conteudo." if language.startswith("pt") else "No content."
    )
    citations = [format_citation(chunk.start_sec)]
    return ChunkSummary(
        chunk_id=chunk.chunk_id,
        start_sec=chunk.start_sec,
        end_sec=chunk.end_sec,
        summary=summary,
        citations=citations,
    )


def synthesize_global_summary(
    chunk_summaries: list[ChunkSummary], *, language: str
) -> SummaryDocument:
    if not chunk_summaries:
        empty_item = SummaryItem(
            text="Sem dados suficientes." if language.startswith("pt") else "No data.", citations=[]
        )
        return SummaryDocument(
            language=language,
            chunk_summaries=[],
            overview=empty_item,
            key_points=[],
            decisions=[],
            action_items=[],
            open_questions=[],
        )

    citations = [item.citations[0] for item in chunk_summaries if item.citations]
    first_citation = citations[0] if citations else "00:00:00.000"

    overview_text = " ".join(item.summary for item in chunk_summaries[:3]).strip()
    if not overview_text:
        overview_text = (
            "Resumo indisponivel." if language.startswith("pt") else "Summary unavailable."
        )

    key_points = [
        SummaryItem(text=item.summary, citations=item.citations or [first_citation])
        for item in chunk_summaries[:5]
    ]

    decisions = _pick_items_by_keywords(
        chunk_summaries,
        keywords=["decid", "acord", "resolve", "aprov"],
        fallback=(
            "Nenhuma decisao explicita." if language.startswith("pt") else "No explicit decisions."
        ),
        default_citation=first_citation,
    )

    action_items = _pick_items_by_keywords(
        chunk_summaries,
        keywords=["acao", "fazer", "deve", "proximo", "next", "todo"],
        fallback=(
            "Sem acoes explicitas." if language.startswith("pt") else "No explicit action items."
        ),
        default_citation=first_citation,
    )

    open_questions = _pick_items_by_keywords(
        chunk_summaries,
        keywords=["?", "duvida", "pergunta", "question", "uncertain"],
        fallback=(
            "Sem perguntas abertas identificadas."
            if language.startswith("pt")
            else "No open questions identified."
        ),
        default_citation=first_citation,
    )

    return SummaryDocument(
        language=language,
        chunk_summaries=chunk_summaries,
        overview=SummaryItem(text=overview_text, citations=[first_citation]),
        key_points=key_points,
        decisions=decisions,
        action_items=action_items,
        open_questions=open_questions,
    )


def _pick_items_by_keywords(
    chunk_summaries: list[ChunkSummary],
    *,
    keywords: list[str],
    fallback: str,
    default_citation: str,
) -> list[SummaryItem]:
    selected: list[SummaryItem] = []
    for chunk in chunk_summaries:
        content = chunk.summary.lower()
        if any(keyword in content for keyword in keywords):
            selected.append(
                SummaryItem(text=chunk.summary, citations=chunk.citations or [default_citation])
            )
    if selected:
        return selected[:5]
    return [SummaryItem(text=fallback, citations=[default_citation])]


def build_summary_payload(document: SummaryDocument) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "language": document.language,
        "chunk_summaries": [asdict(item) for item in document.chunk_summaries],
        "overview": asdict(document.overview),
        "key_points": [asdict(item) for item in document.key_points],
        "decisions": [asdict(item) for item in document.decisions],
        "action_items": [asdict(item) for item in document.action_items],
        "open_questions": [asdict(item) for item in document.open_questions],
    }


def validate_summary_payload(payload: dict[str, object]) -> None:
    required = {
        "schema_version",
        "language",
        "chunk_summaries",
        "overview",
        "key_points",
        "decisions",
        "action_items",
        "open_questions",
    }
    missing = [key for key in required if key not in payload]
    if missing:
        raise SummarySchemaError(f"Missing required summary keys: {', '.join(sorted(missing))}")

    sections = ["overview", "key_points", "decisions", "action_items", "open_questions"]
    for section in sections:
        value = payload[section]
        if section == "overview":
            _validate_summary_item(value, section_name=section)
        else:
            if not isinstance(value, list):
                raise SummarySchemaError(f"Section '{section}' must be a list")
            if not value:
                raise SummarySchemaError(f"Section '{section}' must have at least one item")
            for item in value:
                _validate_summary_item(item, section_name=section)


def _validate_summary_item(item: object, *, section_name: str) -> None:
    if not isinstance(item, dict):
        raise SummarySchemaError(f"Section '{section_name}' item must be an object")
    text = item.get("text")
    citations = item.get("citations")
    if not isinstance(text, str) or not text.strip():
        raise SummarySchemaError(f"Section '{section_name}' item text must be a non-empty string")
    if not isinstance(citations, list):
        raise SummarySchemaError(f"Section '{section_name}' item citations must be a list")


def render_summary_markdown(document: SummaryDocument) -> str:
    lines = ["# Summary", ""]
    lines.extend(
        ["## Overview", document.overview.text, _citations_line(document.overview.citations), ""]
    )
    lines.extend(_render_section("Key Points", document.key_points))
    lines.extend(_render_section("Decisions", document.decisions))
    lines.extend(_render_section("Action Items", document.action_items))
    lines.extend(_render_section("Open Questions", document.open_questions))
    return "\n".join(lines).strip() + "\n"


def _render_section(title: str, items: list[SummaryItem]) -> list[str]:
    lines = [f"## {title}"]
    for item in items:
        lines.append(f"- {item.text}")
        lines.append(f"  - citations: {', '.join(item.citations) if item.citations else 'n/a'}")
    lines.append("")
    return lines


def _citations_line(citations: list[str]) -> str:
    if not citations:
        return "citations: n/a"
    return f"citations: {', '.join(citations)}"


def format_citation(seconds: float) -> str:
    total_millis = int(round(seconds * 1000))
    hours = total_millis // 3_600_000
    total_millis -= hours * 3_600_000
    minutes = total_millis // 60_000
    total_millis -= minutes * 60_000
    secs = total_millis // 1_000
    millis = total_millis - secs * 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class SummaryPipeline:
    def __init__(
        self,
        *,
        model_backend: str,
        model_name: str,
        ollama_base_url: str,
        llamacpp_server_url: str,
        max_chunk_tokens: int,
        max_chunk_seconds: int,
        fail_fast: bool,
        model_override: SummaryModel | None = None,
    ) -> None:
        self._model_backend = model_backend
        self._model_name = model_name
        self._ollama_base_url = ollama_base_url
        self._llamacpp_server_url = llamacpp_server_url
        self._max_chunk_tokens = max_chunk_tokens
        self._max_chunk_seconds = max_chunk_seconds
        self._fail_fast = fail_fast
        self._model_override = model_override

    def build(self, *, run_dir: Path, language: str) -> SummaryArtifactResult:
        artifacts_dir = run_dir / "artifacts"
        checkpoints_dir = run_dir / "checkpoints"
        logs_dir = run_dir / "logs"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "summary.jsonl"

        markdown_path = artifacts_dir / "summary.md"
        json_path = artifacts_dir / "summary.json"
        manifest_path = artifacts_dir / "summary_manifest.json"

        jobs = [
            ("summary.md", markdown_path, checkpoints_dir / "summary.md.done"),
            ("summary.json", json_path, checkpoints_dir / "summary.json.done"),
        ]

        segments = load_transcript_segments(run_dir)
        chunks = chunk_transcript(
            segments,
            max_chunk_tokens=self._max_chunk_tokens,
            max_chunk_seconds=float(self._max_chunk_seconds),
        )
        model = self._model_override or self._build_model()

        chunk_summaries: list[ChunkSummary] = []
        for chunk in chunks:
            try:
                chunk_summaries.append(summarize_chunk(chunk, language=language, model=model))
            except Exception as exc:  # noqa: BLE001
                if self._fail_fast:
                    raise SummaryError(
                        f"Chunk summarization failed for chunk {chunk.chunk_id}: {exc}"
                    ) from exc
                fallback = HeuristicSummaryModel().generate(
                    "\n".join(segment.text for segment in chunk.segments)
                )
                chunk_summaries.append(
                    ChunkSummary(
                        chunk_id=chunk.chunk_id,
                        start_sec=chunk.start_sec,
                        end_sec=chunk.end_sec,
                        summary=fallback,
                        citations=[format_citation(chunk.start_sec)],
                    )
                )

        document = synthesize_global_summary(chunk_summaries, language=language)
        payload = build_summary_payload(document)
        validate_summary_payload(payload)
        markdown = render_summary_markdown(document)

        generated_files: list[str] = []
        skipped_files: list[str] = []

        for label, output_path, checkpoint_path in jobs:
            if checkpoint_path.exists() and output_path.exists():
                skipped_files.append(label)
                self._log(log_path, "summary_skip_checkpoint", file=label)
                continue

            if label.endswith(".md"):
                output_path.write_text(markdown, encoding="utf-8")
            else:
                output_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )

            checkpoint_path.write_text("ok\n", encoding="utf-8")
            generated_files.append(label)
            self._log(log_path, "summary_generated", file=label)

        manifest_payload = {
            "language": language,
            "model_backend": self._model_backend,
            "model_name": self._model_name,
            "chunk_count": len(chunks),
            "generated_files": generated_files,
            "skipped_files": skipped_files,
            "artifacts": {
                "summary_md": str(markdown_path),
                "summary_json": str(json_path),
            },
            "summary": payload,
        }
        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        return SummaryArtifactResult(
            markdown_path=markdown_path,
            json_path=json_path,
            manifest_path=manifest_path,
            generated_files=generated_files,
            skipped_files=skipped_files,
        )

    def _build_model(self) -> SummaryModel:
        backend = self._model_backend.lower()
        if backend == "ollama":
            return OllamaSummaryModel(base_url=self._ollama_base_url, model_name=self._model_name)
        if backend == "llamacpp":
            return LlamaCppSummaryModel(base_url=self._llamacpp_server_url)
        if backend == "heuristic":
            return HeuristicSummaryModel()
        raise SummaryError(f"Unsupported local summary backend: {self._model_backend}")

    def _log(self, log_path: Path, event: str, **payload: object) -> None:
        entry = {"event": event, "payload": payload}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
