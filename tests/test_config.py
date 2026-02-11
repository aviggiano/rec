from __future__ import annotations

from pathlib import Path

from rec_pipeline.config import load_settings


def test_config_loading_defaults_without_external_keys(monkeypatch: object, tmp_path: Path) -> None:
    keys = [
        "REC_ASR_PROVIDER",
        "REC_SUMMARY_PROVIDER",
        "REC_DEFAULT_LANG",
        "REC_OUTPUT_LANG",
        "REC_CONTINUE_ON_ERROR",
        "REC_ASR_MODEL_SIZE",
        "REC_ASR_DEVICE",
        "REC_ASR_COMPUTE_TYPE",
        "REC_ASR_BEAM_SIZE",
        "REC_ASR_VAD_FILTER",
        "REC_ASR_MAX_RETRIES",
        "REC_SUMMARY_LOCAL_BACKEND",
        "REC_SUMMARY_MODEL",
        "REC_SUMMARY_MAX_CHUNK_TOKENS",
        "REC_SUMMARY_MAX_CHUNK_SECONDS",
        "REC_DIARIZATION_ENABLED",
        "REC_DIARIZATION_MODEL_NAME",
        "REC_DIARIZATION_EXPORT_SPEAKERS",
        "REC_EXTERNAL_FALLBACK_TO_LOCAL",
        "REC_PROVIDER_TIMEOUT_SEC",
        "REC_PROVIDER_MAX_RETRIES",
        "REC_PROVIDER_RETRY_BASE_DELAY_SEC",
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY",
        "GROQ_API_KEY",
        "HUGGINGFACE_TOKEN",
        "OLLAMA_BASE_URL",
        "LLAMACPP_SERVER_URL",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)  # type: ignore[attr-defined]

    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")

    settings = load_settings(env_file)

    assert settings.asr_provider == "local"
    assert settings.summary_provider == "local"
    assert settings.default_language == "pt"
    assert settings.output_language == "pt"
    assert settings.continue_on_error is True
    assert settings.asr_model_size == "small"
    assert settings.asr_device == "auto"
    assert settings.asr_compute_type == "int8"
    assert settings.asr_beam_size == 5
    assert settings.asr_vad_filter is True
    assert settings.asr_max_retries == 3
    assert settings.summary_local_backend == "ollama"
    assert settings.summary_model_name == "llama3.2"
    assert settings.summary_max_chunk_tokens == 1200
    assert settings.summary_max_chunk_seconds == 900
    assert settings.diarization_enabled is False
    assert settings.diarization_model_name == "pyannote/speaker-diarization-3.1"
    assert settings.diarization_export_speakers is True
    assert settings.huggingface_token is None
    assert settings.external_fallback_to_local is True
    assert settings.provider_timeout_sec == 120
    assert settings.provider_max_retries == 3
    assert settings.provider_retry_base_delay_sec == 0.5
    assert settings.openai_api_key is None
    assert settings.deepgram_api_key is None
    assert settings.groq_api_key is None


def test_config_default_env_file_is_loaded_from_current_working_directory(
    monkeypatch: object, tmp_path: Path
) -> None:
    monkeypatch.delenv("REC_DEFAULT_LANG", raising=False)  # type: ignore[attr-defined]
    env_file = tmp_path / ".env"
    env_file.write_text("REC_DEFAULT_LANG=en\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]

    settings = load_settings()

    assert settings.default_language == "en"
