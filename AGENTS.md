# AGENTS.md

Guidance for coding agents working in `rec`.

## Project intent

- `rec` is a local-first long-form audio transcription and summarization pipeline.
- Baseline operation must work without external API keys.
- External providers are optional and opt-in only.

## Core architecture and contracts

- CLI entrypoint: `rec` (`src/rec_pipeline/cli.py`).
- Ordered pipeline stages:
  1. Ingestion/normalization
  2. ASR
  3. Optional diarization
  4. Transcript assembly/writers
  5. Summarization
  6. Optional evaluation (`rec evaluate`)
- Run outputs are run-scoped under `artifacts/<run-name>/`.
- Stage outputs are treated as contracts for downstream stages. Preserve compatibility when changing payloads.

## Resumability and determinism

- Long runs are resume-safe via checkpoint files in `checkpoints/`.
- Stage reruns should skip completed artifacts when checkpoint + output exist.
- File ordering and generated naming must remain deterministic.
- Avoid introducing nondeterministic output ordering.

## Local-first and providers

- Keep `local` as default for ASR and summary providers.
- External provider selection is explicit (`openai`, `deepgram`, `groq`).
- Missing required keys for selected providers must fail fast with actionable messages.
- Preserve `REC_EXTERNAL_FALLBACK_TO_LOCAL` behavior.
- Never log raw secrets; redact keys/tokens in errors and logs.

## Transcript and summary quality expectations

- ASR segment timestamps must be monotonic and valid (`start < end`).
- Transcript artifacts must remain valid/deterministic:
  - `transcript.txt`
  - `transcript.srt`
  - `transcript.json`
- Summary outputs must include these sections:
  - overview
  - key points
  - decisions
  - action items
  - open questions
- Keep timestamp citations for major summary sections when source timing exists.

## Diarization behavior

- Diarization is optional and should degrade gracefully.
- If diarization fails and fail-fast is disabled, continue with `UNKNOWN` speaker labels.
- Preserve speaker-aware outputs and optional per-speaker exports.
- If diarization is enabled, document/token requirements must remain explicit.

## Evaluation and gates

- Evaluation command: `rec evaluate`.
- Maintain metrics and report schema:
  - WER/CER
  - DER proxy
  - summary traceability coverage
- Preserve warning-first gate mode and blocking-gates option.
- Keep at least one long multi-file scenario in evaluation datasets.

## Engineering workflow

- Python 3.11+.
- Preferred checks before submitting changes:
  - `make lint`
  - `make format-check`
  - `make typecheck`
  - `make test`
- Keep mypy strictness intact unless there is a strong reason to relax it.
- Add or update tests for behavioral changes.

## CI/CD and release guardrails

- Keep workflows healthy:
  - `.github/workflows/ci.yml`
  - `.github/workflows/nightly-evaluation.yml`
  - `.github/workflows/release.yml`
- Direct merges to `main` are allowed when local validation is completed.
- Prioritize delivery speed; use PR gates only when they add clear coordination value.

## PR and branching best practices

- Use focused branches and small diffs.
- Direct-to-`main` commits are acceptable for fast iteration.
- If using stacked work, set PR base to the previous branch intentionally.
- Document validation steps in PR descriptions.
- Include migration/compatibility notes when changing artifact schemas.

## Common pitfalls

- Breaking downstream stage by changing JSON payload keys silently.
- Removing checkpoint semantics and causing full reprocessing.
- Making external providers effectively required.
- Emitting logs that include raw API keys/tokens.

## Definition of done for changes

- Behavior is implemented and documented.
- Backward compatibility is addressed or explicitly documented.
- Local quality gates pass.
- Relevant CI/evaluation paths are considered.
