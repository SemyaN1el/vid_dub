"""Text cleanup and post-processing for machine translation output."""

import json
import os
import re
from typing import Any, Dict, List

import config as cfg
from src.translation_common import _normalize_spaces
from src.translation_profiles import (
    _parse_source_target_pairs,
    _profile_list,
    _split_config_list,
)


def _text_replacement_pairs() -> List[Dict[str, str]]:
    """Доменные замены `источник -> замена` из активного профиля и env."""
    raw_items = _profile_list("text_replacements") + _split_config_list(
        os.getenv("MT_TEXT_REPLACEMENTS", getattr(cfg, "MT_TEXT_REPLACEMENTS", ""))
    )
    return _parse_source_target_pairs(raw_items)


def normalize_translated_text(text: str, original_text: str = "") -> str:
    """Подчищает машинный перевод; доменные замены берутся из профиля/env."""
    normalized = _normalize_spaces(text)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"([(\[{])\s+", r"\1", normalized)
    normalized = re.sub(r"\s+([)\]}])", r"\1", normalized)

    original_lower = _normalize_spaces(original_text).lower()
    normalized_lower = normalized.lower()

    if original_lower in {"you're welcome.", "you are welcome."} and normalized_lower in {
        "ни за что.",
        "не за что."
    }:
        return "Пожалуйста."

    for pair in _text_replacement_pairs():
        normalized = re.sub(
            rf"\b{re.escape(pair['source'])}\b",
            pair["target"],
            normalized,
            flags=re.IGNORECASE,
        )

    return normalized.strip()


def normalize_translated_segments(
    translated_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Применяет постобработку ко всем переведённым сегментам."""
    normalized_segments: List[Dict[str, Any]] = []
    for seg in translated_segments:
        item = dict(seg)
        item["text"] = normalize_translated_text(
            seg.get("text", ""),
            seg.get("original_text", "")
        )
        normalized_segments.append(item)
    return normalized_segments


def _cleanup_causal_translation(text: str, tgt_lang: str = "") -> str:
    """Подчищает ответ chat-модели до чистого перевода."""
    cleaned = text.strip().strip("\"'`")
    cleaned = cleaned.replace("<|im_end|>", " ").replace("<|endoftext|>", " ")
    cleaned = _normalize_spaces(cleaned)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            for key in ("translation", "text", "result"):
                value = parsed.get(key)
                if isinstance(value, str):
                    cleaned = value
                    break
            else:
                translations = parsed.get("translations")
                if isinstance(translations, list) and translations and isinstance(translations[0], str):
                    cleaned = translations[0]
        elif isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
            cleaned = parsed[0]
    except json.JSONDecodeError:
        pass

    prefixes = (
        "translation:",
        "translated text:",
        "russian translation:",
        "russian:",
        "перевод:",
        "ответ:",
    )
    lowered = cleaned.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip(" :-\n\t")
            lowered = cleaned.lower()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) > 1:
        cleaned = " ".join(lines)

    if tgt_lang.startswith("rus"):
        cyrillic_start = re.search(r"[А-Яа-яЁё]", cleaned)
        if cyrillic_start:
            cleaned = cleaned[cyrillic_start.start():]

    cleaned = re.sub(r"^(assistant|ассистент)\s*[:,-]?\s*", "", cleaned, flags=re.IGNORECASE)
    return _normalize_spaces(cleaned)
