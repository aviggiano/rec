from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from rec_pipeline.asr import TranscriptSegment


class ProviderConfigError(RuntimeError):
    """Raised when provider configuration is invalid."""


class ProviderRequestError(RuntimeError):
    """Raised when provider HTTP/API calls fail."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


_TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_EXTERNAL_PROVIDERS = {"openai", "deepgram", "groq"}
_ASR_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/audio/transcriptions",
    "deepgram": "https://api.deepgram.com/v1/listen",
    "groq": "https://api.groq.com/openai/v1/audio/transcriptions",
}
_SUMMARY_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "deepgram": "https://api.deepgram.com/v1/read",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
}


def redact_secret(secret: str) -> str:
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}...{secret[-4:]}"


def redact_secrets(text: str, secrets: list[str]) -> str:
    sanitized = text
    for secret in secrets:
        if secret:
            sanitized = sanitized.replace(secret, redact_secret(secret))
    return sanitized


def is_external_provider(provider_name: str) -> bool:
    return provider_name in _EXTERNAL_PROVIDERS


def resolve_provider_key(
    *,
    provider_name: str,
    openai_api_key: str | None,
    deepgram_api_key: str | None,
    groq_api_key: str | None,
) -> str | None:
    if provider_name == "openai":
        return openai_api_key
    if provider_name == "deepgram":
        return deepgram_api_key
    if provider_name == "groq":
        return groq_api_key
    return None


def validate_provider_configuration(
    *,
    asr_provider: str,
    summary_provider: str,
    openai_api_key: str | None,
    deepgram_api_key: str | None,
    groq_api_key: str | None,
) -> None:
    allowed = {"local", "openai", "deepgram", "groq"}
    if asr_provider not in allowed:
        raise ProviderConfigError(f"Unsupported ASR provider: {asr_provider}")
    if summary_provider not in allowed:
        raise ProviderConfigError(f"Unsupported summary provider: {summary_provider}")

    for role, provider in [("ASR", asr_provider), ("summary", summary_provider)]:
        if not is_external_provider(provider):
            continue
        key = resolve_provider_key(
            provider_name=provider,
            openai_api_key=openai_api_key,
            deepgram_api_key=deepgram_api_key,
            groq_api_key=groq_api_key,
        )
        if not key:
            if provider == "openai":
                expected = "OPENAI_API_KEY"
            elif provider == "deepgram":
                expected = "DEEPGRAM_API_KEY"
            else:
                expected = "GROQ_API_KEY"
            raise ProviderConfigError(
                f"{role} provider '{provider}' requires {expected} but it is missing. "
                "Either set the key or switch provider to 'local'."
            )


T = TypeVar("T")


def call_with_retry(
    operation: Callable[[], T],
    *,
    max_retries: int,
    base_delay_sec: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    attempts = 0
    while True:
        attempts += 1
        try:
            return operation()
        except ProviderRequestError as exc:
            is_transient = exc.status_code in _TRANSIENT_STATUS_CODES or exc.status_code is None
            if attempts > max_retries or not is_transient:
                raise
            sleep_fn(base_delay_sec * (2 ** (attempts - 1)))


RequestFn = Callable[[str, dict[str, object], str, int], dict[str, object]]


def http_json_request(
    url: str,
    payload: dict[str, object],
    api_key: str,
    timeout_sec: int,
) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # noqa: S310
            body = response.read().decode("utf-8")
            parsed = json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        safe_body = redact_secrets(body, [api_key])
        raise ProviderRequestError(
            f"Provider request failed with status {exc.code}: {safe_body}",
            status_code=exc.code,
        ) from exc
    except urllib.error.URLError as exc:
        raise ProviderRequestError(
            f"Provider request failed: {exc}",
            status_code=None,
        ) from exc

    if not isinstance(parsed, dict):
        raise ProviderRequestError("Provider response is not a JSON object", status_code=None)

    return parsed


@dataclass
class ExternalASRTranscriber:
    provider_name: str
    api_key: str
    model_name: str
    timeout_sec: int
    max_retries: int
    retry_base_delay_sec: float
    request_fn: RequestFn = http_json_request

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        vad_filter: bool,
        beam_size: int,
    ) -> list[TranscriptSegment]:
        del vad_filter, beam_size
        endpoint = _ASR_ENDPOINTS.get(self.provider_name)
        if endpoint is None:
            raise ProviderConfigError(
                f"No ASR endpoint configured for provider '{self.provider_name}'"
            )

        payload: dict[str, object] = {
            "model": self.model_name,
            "language": language,
            "audio_path": str(audio_path),
        }
        response = call_with_retry(
            lambda: self.request_fn(endpoint, payload, self.api_key, self.timeout_sec),
            max_retries=self.max_retries,
            base_delay_sec=self.retry_base_delay_sec,
        )

        segments_payload = response.get("segments")
        if isinstance(segments_payload, list) and segments_payload:
            segments: list[TranscriptSegment] = []
            for index, item in enumerate(segments_payload):
                if not isinstance(item, dict):
                    continue
                start = _coerce_float(item.get("start", item.get("start_sec", 0.0)))
                end = _coerce_float(item.get("end", item.get("end_sec", start + 1.0)))
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                if end <= start:
                    end = start + 0.01
                segments.append(
                    TranscriptSegment(
                        segment_id=index,
                        start_sec=start,
                        end_sec=end,
                        text=text,
                        avg_logprob=_coerce_optional_float(
                            item.get("avg_logprob", item.get("logprob"))
                        ),
                        confidence=_coerce_optional_float(item.get("confidence")),
                        no_speech_prob=_coerce_optional_float(item.get("no_speech_prob")),
                    )
                )
            if segments:
                return segments

        text_value = _extract_response_text(response)
        if text_value:
            return [
                TranscriptSegment(
                    segment_id=0,
                    start_sec=0.0,
                    end_sec=1.0,
                    text=text_value,
                    avg_logprob=None,
                    confidence=None,
                    no_speech_prob=None,
                )
            ]

        raise ProviderRequestError(
            f"External ASR provider '{self.provider_name}' returned no usable transcript"
        )


@dataclass
class ExternalSummaryModel:
    provider_name: str
    api_key: str
    model_name: str
    timeout_sec: int
    max_retries: int
    retry_base_delay_sec: float
    request_fn: RequestFn = http_json_request

    def generate(self, prompt: str) -> str:
        endpoint = _SUMMARY_ENDPOINTS.get(self.provider_name)
        if endpoint is None:
            raise ProviderConfigError(
                f"No summary endpoint configured for provider '{self.provider_name}'"
            )

        payload: dict[str, object] = {
            "model": self.model_name,
            "input": prompt,
        }
        response = call_with_retry(
            lambda: self.request_fn(endpoint, payload, self.api_key, self.timeout_sec),
            max_retries=self.max_retries,
            base_delay_sec=self.retry_base_delay_sec,
        )

        text = _extract_response_text(response)
        if text:
            return text
        raise ProviderRequestError(
            f"External summary provider '{self.provider_name}' returned no usable text"
        )


def _extract_response_text(payload: dict[str, object]) -> str | None:
    for key in ["text", "summary", "response", "content"]:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
            text = choice.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()

    return None


def _coerce_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _coerce_optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
