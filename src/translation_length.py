"""Length-aware metadata and prompt guidance for dubbing translation."""

import json
import os
from typing import Any, Dict

import config as cfg
from src.translation_common import _normalize_spaces


def _length_aware_enabled() -> bool:
    raw = os.getenv(
        "MT_LENGTH_AWARE_ENABLED",
        str(getattr(cfg, "MT_LENGTH_AWARE_ENABLED", "1")),
    )
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _soft_target_chars_for_duration(duration_sec: float) -> int:
    cps = float(os.getenv(
        "MT_TARGET_CHARS_PER_SEC",
        str(getattr(cfg, "MT_TARGET_CHARS_PER_SEC", 14.0)),
    ))
    grace = int(os.getenv(
        "MT_LENGTH_GRACE_CHARS",
        str(getattr(cfg, "MT_LENGTH_GRACE_CHARS", 18)),
    ))
    minimum = int(os.getenv(
        "MT_MIN_TARGET_CHARS",
        str(getattr(cfg, "MT_MIN_TARGET_CHARS", 24)),
    ))
    return max(minimum, int(round(max(0.0, duration_sec) * cps)) + grace)


def _segment_duration(segment: Dict[str, Any]) -> float:
    duration = segment.get("source_duration_sec")
    if duration is None:
        try:
            duration = float(segment.get("end", 0.0)) - float(segment.get("start", 0.0))
        except (TypeError, ValueError):
            duration = 0.0
    try:
        return max(0.0, float(duration))
    except (TypeError, ValueError):
        return 0.0


def _translation_segment_metadata(segment: Dict[str, Any]) -> Dict[str, Any]:
    text = _normalize_spaces(str(segment.get("text", "")))
    duration = _segment_duration(segment)
    source_word_count = int(segment.get("source_word_count") or len(text.split()) or 0)
    metadata = {
        "source_duration_sec": round(duration, 2),
        "source_word_count": source_word_count,
        "source_chars": len(text),
    }
    if _length_aware_enabled() and duration > 0:
        metadata["soft_target_max_chars"] = _soft_target_chars_for_duration(duration)
    return metadata


def _format_segment_for_prompt(index: int, text: str, metadata: Dict[str, Any] | None) -> str:
    if not metadata:
        return f"[{index}]: {json.dumps(text, ensure_ascii=False)}"
    payload = {"text": text, **metadata}
    return f"[{index}]: {json.dumps(payload, ensure_ascii=False)}"


def _length_aware_prompt_block() -> str:
    if not _length_aware_enabled():
        return ""
    return (
        "\nDubbing length guidance:\n"
        "- Some segments include source_duration_sec and soft_target_max_chars.\n"
        "- Treat soft_target_max_chars as a strong preference, not a reason to drop meaning-critical information.\n"
        "- Prefer compact natural Russian when the target is tight: remove filler, use shorter synonyms, and avoid inflated written style.\n"
        "- Keep numbers, names, terms, negations, and causal relationships intact.\n"
    )
