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
    asr_model_size: str
    asr_device: str
    asr_compute_type: str
    asr_beam_size: int
    asr_vad_filter: bool
    asr_max_retries: int
    summary_local_backend: str
    summary_model_name: str
    summary_max_chunk_tokens: int
    summary_max_chunk_seconds: int
    diarization_enabled: bool
    diarization_model_name: str
    diarization_export_speakers: bool
    huggingface_token: str | None
    external_fallback_to_local: bool
    provider_timeout_sec: int
    provider_max_retries: int
    provider_retry_base_delay_sec: float
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


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


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
        asr_model_size=os.getenv("REC_ASR_MODEL_SIZE", "small"),
        asr_device=os.getenv("REC_ASR_DEVICE", "auto"),
        asr_compute_type=os.getenv("REC_ASR_COMPUTE_TYPE", "int8"),
        asr_beam_size=_env_int("REC_ASR_BEAM_SIZE", 5),
        asr_vad_filter=_env_bool("REC_ASR_VAD_FILTER", True),
        asr_max_retries=_env_int("REC_ASR_MAX_RETRIES", 3),
        summary_local_backend=os.getenv("REC_SUMMARY_LOCAL_BACKEND", "ollama"),
        summary_model_name=os.getenv("REC_SUMMARY_MODEL", "llama3.2"),
        summary_max_chunk_tokens=_env_int("REC_SUMMARY_MAX_CHUNK_TOKENS", 1200),
        summary_max_chunk_seconds=_env_int("REC_SUMMARY_MAX_CHUNK_SECONDS", 900),
        diarization_enabled=_env_bool("REC_DIARIZATION_ENABLED", False),
        diarization_model_name=os.getenv(
            "REC_DIARIZATION_MODEL_NAME",
            "pyannote/speaker-diarization-3.1",
        ),
        diarization_export_speakers=_env_bool("REC_DIARIZATION_EXPORT_SPEAKERS", True),
        huggingface_token=_env_optional("HUGGINGFACE_TOKEN"),
        external_fallback_to_local=_env_bool("REC_EXTERNAL_FALLBACK_TO_LOCAL", True),
        provider_timeout_sec=_env_int("REC_PROVIDER_TIMEOUT_SEC", 120),
        provider_max_retries=_env_int("REC_PROVIDER_MAX_RETRIES", 3),
        provider_retry_base_delay_sec=_env_float("REC_PROVIDER_RETRY_BASE_DELAY_SEC", 0.5),
        openai_api_key=_env_optional("OPENAI_API_KEY"),
        deepgram_api_key=_env_optional("DEEPGRAM_API_KEY"),
        groq_api_key=_env_optional("GROQ_API_KEY"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        llamacpp_server_url=os.getenv("LLAMACPP_SERVER_URL", "http://localhost:8080"),
    )
