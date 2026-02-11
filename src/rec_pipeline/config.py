from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class RecSettings:
    """Application configuration loaded from environment variables."""

    asr_provider: str
    summary_provider: str
    default_language: str
    output_language: str
    continue_on_error: bool
    openai_api_key: str | None
    deepgram_api_key: str | None
    groq_api_key: str | None
    ollama_base_url: str
    llamacpp_server_url: str


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def load_settings(env_file: str | Path | None = None) -> RecSettings:
    """Load settings from `.env` and environment variables."""

    if env_file is not None:
        load_dotenv(dotenv_path=Path(env_file), override=False)
    else:
        load_dotenv(override=False)

    return RecSettings(
        asr_provider=os.getenv("REC_ASR_PROVIDER", "local"),
        summary_provider=os.getenv("REC_SUMMARY_PROVIDER", "local"),
        default_language=os.getenv("REC_DEFAULT_LANG", "pt"),
        output_language=os.getenv("REC_OUTPUT_LANG", "pt"),
        continue_on_error=_env_bool("REC_CONTINUE_ON_ERROR", True),
        openai_api_key=_env_optional("OPENAI_API_KEY"),
        deepgram_api_key=_env_optional("DEEPGRAM_API_KEY"),
        groq_api_key=_env_optional("GROQ_API_KEY"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        llamacpp_server_url=os.getenv("LLAMACPP_SERVER_URL", "http://localhost:8080"),
    )
