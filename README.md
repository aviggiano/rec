# rec

`rec` is a local-first pipeline for long-form audio transcription and summarization.

## What it does

- Ingests and normalizes recording files for deterministic processing.
- Transcribes recordings with local ASR by default.
- Assembles transcript artifacts (`txt`, `json`, `srt`).
- Generates structured summaries with local-first defaults.
- Supports resumable long runs with per-stage checkpoints and manifests.
- Includes a local `faster-whisper` ASR stage with configurable model/runtime options.

## Local-first policy

The baseline path is designed to run without any external API key.
External providers are optional and only enabled by explicit configuration.

## Quick start (macOS and Linux)

### 1. Prerequisites

- Python 3.11+
- `ffmpeg` / `ffprobe`

### 2. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
# for local ASR with faster-whisper
pip install -e ".[asr]"
cp .env.example .env
```

### 3. Validate developer checks

```bash
make lint
make format-check
make typecheck
make test
```

### 4. CLI smoke test

```bash
rec --help
```

### 5. Run ingestion (resume-safe)

```bash
rec run --input ./recordings --output ./artifacts --run-name session-001
# Running the same command again reuses checkpoints and skips completed normalization.
```

### ASR options

```bash
rec run \
  --input ./recordings \
  --lang pt \
  --asr-model-size small \
  --asr-device auto \
  --asr-compute-type int8 \
  --asr-beam-size 5 \
  --asr-vad-filter \
  --summary-local-backend ollama \
  --summary-model llama3.2
```

Run artifacts are created under `artifacts/<run-name>/` and include:
- `asr/raw/*.segments.json`
- `asr/raw_transcript_segments.json`
- `artifacts/transcript.txt`
- `artifacts/transcript.srt`
- `artifacts/transcript.json`
- `artifacts/summary.md`
- `artifacts/summary.json`

## Configuration

- Defaults come from environment variables and `.env`.
- `.env.example` documents all supported variables.
- External API keys are optional unless you explicitly choose an external provider.

## Development workflow

```bash
pre-commit install
pre-commit run --all-files
```

## License

MIT
