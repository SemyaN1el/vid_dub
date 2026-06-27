"""Shared low-level helpers for translation modules."""

import re


def _normalize_spaces(text: str) -> str:
    """Чистит лишние пробелы, не меняя смысл текста."""
    return re.sub(r"\s+", " ", text).strip()


def _language_name(lang_code: str) -> str:
    """Преобразует код языка в читаемое имя для prompt-based моделей."""
    prefix = (lang_code or "").split("_", 1)[0]
    mapping = {
        "eng": "English",
        "rus": "Russian",
        "deu": "German",
        "ger": "German",
        "fra": "French",
        "fre": "French",
        "spa": "Spanish",
        "zho": "Chinese",
        "chi": "Chinese",
        "jpn": "Japanese",
        "kor": "Korean",
        "ara": "Arabic",
        "arb": "Arabic",
        "por": "Portuguese",
        "ita": "Italian",
        "nld": "Dutch",
        "dut": "Dutch",
        "pol": "Polish",
        "ukr": "Ukrainian",
        "tur": "Turkish",
        "vie": "Vietnamese",
        "tha": "Thai",
        "ind": "Indonesian",
        "hin": "Hindi",
        "ben": "Bengali",
    }
    return mapping.get(prefix, lang_code or "the target language")


def _is_gemini_model(model_name: str) -> bool:
    return (model_name or "").strip().lower().startswith("gemini-")
