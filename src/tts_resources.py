"""Загрузка и кэширование вспомогательных ресурсов TTS-этапа: модели для guard/ASR-retry и backend локального переписывания SmartSync."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List

from src.translation import load_smart_sync_backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TTSAuxModels:
    babble_guard_model: Any | None = None
    asr_eval_model: Any | None = None
    tail_guard_asr_model: Any | None = None


def existing_reference_paths(speaker_wav: Any) -> List[str]:
    if isinstance(speaker_wav, list):
        return [path for path in speaker_wav if path and os.path.exists(path)]
    return [speaker_wav] if speaker_wav and os.path.exists(speaker_wav) else []


def load_tts_aux_models(
    tail_guard_config: Any,
    smart_sync_config: Any,
) -> TTSAuxModels:
    whisper = None
    babble_guard_model = None
    asr_eval_model = None

    if (
        tail_guard_config.enable_babble_guard
        or tail_guard_config.enable_asr_retry
        or smart_sync_config.enabled
    ):
        import whisper as whisper_module

        whisper = whisper_module

    if tail_guard_config.enable_babble_guard and whisper is not None:
        logger.info(
            "Загружаем Whisper %s для TTS babble guard на %s...",
            tail_guard_config.babble_guard_model_name,
            tail_guard_config.babble_guard_device
        )
        babble_guard_model = whisper.load_model(
            tail_guard_config.babble_guard_model_name
        ).to(tail_guard_config.babble_guard_device)

    if (
        tail_guard_config.enable_asr_retry or smart_sync_config.enabled
    ) and whisper is not None:
        if (
            babble_guard_model is not None
            and tail_guard_config.asr_retry_model_name == tail_guard_config.babble_guard_model_name
            and tail_guard_config.asr_retry_device == tail_guard_config.babble_guard_device
        ):
            asr_eval_model = babble_guard_model
            logger.info(
                "TTS ASR-eval будет использовать уже загруженный Whisper %s на %s.",
                tail_guard_config.asr_retry_model_name,
                tail_guard_config.asr_retry_device
            )
        else:
            logger.info(
                "Загружаем Whisper %s для TTS ASR-eval на %s...",
                tail_guard_config.asr_retry_model_name,
                tail_guard_config.asr_retry_device
            )
            asr_eval_model = whisper.load_model(
                tail_guard_config.asr_retry_model_name
            ).to(tail_guard_config.asr_retry_device)

    tail_guard_asr_model = babble_guard_model or asr_eval_model
    if tail_guard_asr_model is not None and babble_guard_model is None and asr_eval_model is not None:
        logger.info(
            "TTS tail guard будет использовать уже загруженный Whisper %s без отдельной модели.",
            tail_guard_config.asr_retry_model_name
        )

    return TTSAuxModels(
        babble_guard_model=babble_guard_model,
        asr_eval_model=asr_eval_model,
        tail_guard_asr_model=tail_guard_asr_model,
    )


def load_smart_sync_rewrite_backend(
    smart_sync_config: Any,
) -> Any | None:
    if not smart_sync_config.enabled or smart_sync_config.max_rewrites <= 0:
        return None

    try:
        smart_sync_backend = load_smart_sync_backend(device=smart_sync_config.device)
        if smart_sync_backend is not None:
            logger.info(
                "SmartSync rewrite активирован через %s",
                getattr(smart_sync_backend, "model_name", "smart-sync")
            )
        return smart_sync_backend
    except Exception as error:
        logger.warning("SmartSync rewrite отключён по fallback: %s", error)
        return None
