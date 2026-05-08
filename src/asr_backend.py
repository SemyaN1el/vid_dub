import logging
import os
from dataclasses import dataclass
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import config as cfg

logger = logging.getLogger(__name__)


def _response_to_dict(response: Any) -> dict:
    if isinstance(response, dict):
        return response
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    dict_method = getattr(response, "dict", None)
    if callable(dict_method):
        return dict_method()
    raise TypeError(f"Неподдерживаемый тип ответа ASR API: {type(response)!r}")


def _normalize_verbose_transcript(payload: dict) -> dict:
    """Приводит API-ответ к формату, совместимому с local Whisper."""
    segments = payload.get("segments")
    words = payload.get("words")

    if words and isinstance(words, list):
        has_nested_words = bool(segments) and any(
            isinstance(segment, dict) and segment.get("words")
            for segment in segments
            if isinstance(segment, dict)
        )
        if not has_nested_words:
            payload["segments"] = [{
                "id": 0,
                "start": words[0].get("start", 0.0),
                "end": words[-1].get("end", words[0].get("start", 0.0)),
                "text": payload.get("text", ""),
                "words": words,
            }]

    return payload


@dataclass
class GroqASRBackend:
    model_name: str
    client: Any
    timeout_sec: int

    def transcribe(
        self,
        audio_path: str,
        *,
        word_timestamps: bool = False,
        task: str = "transcribe",
        language: str | None = None,
        fp16: bool | None = None,
    ) -> dict:
        del task, fp16
        timestamp_granularities = ["segment"]
        if word_timestamps:
            timestamp_granularities.append("word")

        with open(audio_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), audio_file.read()),
                model=self.model_name,
                language=language,
                temperature=0.0,
                response_format="verbose_json",
                timestamp_granularities=timestamp_granularities,
            )
        return _normalize_verbose_transcript(_response_to_dict(response))


def _load_asr_model(
    *,
    provider: str,
    local_model_name: str,
    api_model_name: str,
    api_key_env: str,
    base_url: str,
    timeout_sec: int,
    device: str,
    purpose: str,
):
    provider = (provider or "local").strip().lower()

    if provider == "local":
        import whisper

        logger.info(
            "Загружаем локальный Whisper %s на %s для %s...",
            local_model_name,
            device,
            purpose,
        )
        return whisper.load_model(local_model_name).to(device)

    if provider == "groq":
        if OpenAI is None:
            raise ImportError(
                "Для ASR через Groq требуется пакет openai."
            )

        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise EnvironmentError(
                "Для ASR через Groq установите переменную окружения "
                f"{api_key_env}."
            )

        logger.info(
            "Используем Groq ASR %s через %s для %s...",
            api_model_name,
            base_url,
            purpose,
        )
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_sec,
        )
        return GroqASRBackend(
            model_name=api_model_name,
            client=client,
            timeout_sec=timeout_sec,
        )

    raise ValueError(f"Неподдерживаемый ASR provider для {purpose}: {provider}")


def load_main_asr_model():
    return _load_asr_model(
        provider=cfg.ASR_PROVIDER,
        local_model_name=cfg.WHISPER_MODEL_NAME,
        api_model_name=cfg.ASR_API_MODEL,
        api_key_env=cfg.ASR_API_KEY_ENV,
        base_url=cfg.ASR_BASE_URL,
        timeout_sec=cfg.ASR_TIMEOUT_SEC,
        device=cfg.DEVICE,
        purpose="основного ASR",
    )


def load_metrics_asr_model():
    return _load_asr_model(
        provider=cfg.METRICS_ASR_PROVIDER,
        local_model_name=cfg.METRICS_WHISPER_MODEL_NAME,
        api_model_name=cfg.METRICS_ASR_API_MODEL,
        api_key_env=cfg.METRICS_ASR_API_KEY_ENV,
        base_url=cfg.METRICS_ASR_BASE_URL,
        timeout_sec=cfg.METRICS_ASR_TIMEOUT_SEC,
        device=cfg.DEVICE,
        purpose="judge для WER/CER",
    )
