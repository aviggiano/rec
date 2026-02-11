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
    assert settings.openai_api_key is None
    assert settings.deepgram_api_key is None
    assert settings.groq_api_key is None
