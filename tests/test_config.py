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
        "OPENAI_API_KEY",
        "DEEPGRAM_API_KEY",
        "GROQ_API_KEY",
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
    assert settings.openai_api_key is None
    assert settings.deepgram_api_key is None
    assert settings.groq_api_key is None
