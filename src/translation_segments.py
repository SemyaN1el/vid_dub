"""Segment preparation helpers for machine translation."""

import logging
import re
from copy import deepcopy
from typing import Any, Dict, List

from src.translation_common import _normalize_spaces

logger = logging.getLogger(__name__)


def _build_words_with_silence_from_dicts(words: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    normalized_words = []
    for word in words:
        text = _normalize_spaces(str(word.get("text", "")))
        if not text:
            continue
        normalized_words.append({
            "text": text,
            "start": float(word.get("start", 0.0)),
            "end": float(word.get("end", 0.0)),
        })

    for idx, word in enumerate(normalized_words):
        parts.append(word["text"])
        if idx < len(normalized_words) - 1:
            gap_sec = max(0.0, normalized_words[idx + 1]["start"] - word["end"])
            if gap_sec >= 0.04:
                parts.append(f"<{gap_sec:.2f}s>")
    return " ".join(parts)


def _copy_segment_metadata(source_seg: Dict[str, Any], target_seg: Dict[str, Any]) -> Dict[str, Any]:
    for key in (
        "speaker_id",
        "words_with_silence",
        "source_duration_sec",
        "source_word_count",
        "pause_before_sec",
        "pause_after_sec",
    ):
        value = source_seg.get(key)
        if value is not None:
            target_seg[key] = deepcopy(value)

    words = source_seg.get("words")
    if isinstance(words, list):
        target_seg["words"] = [
            {
                "text": _normalize_spaces(str(word.get("text", ""))),
                "start": round(float(word.get("start", 0.0)), 3),
                "end": round(float(word.get("end", 0.0)), 3),
            }
            for word in words
            if _normalize_spaces(str(word.get("text", "")))
        ]
    return target_seg


def _split_words_for_chunks(
    source_words: List[Dict[str, Any]],
    chunks: List[str]
) -> List[List[Dict[str, Any]]]:
    if not source_words or not chunks:
        return [[] for _ in chunks]

    normalized_words = [
        {
            "text": _normalize_spaces(str(word.get("text", ""))),
            "start": round(float(word.get("start", 0.0)), 3),
            "end": round(float(word.get("end", 0.0)), 3),
        }
        for word in source_words
        if _normalize_spaces(str(word.get("text", "")))
    ]
    if not normalized_words:
        return [[] for _ in chunks]

    chunk_word_targets = [max(1, len(chunk.split())) for chunk in chunks]
    total_target_words = sum(chunk_word_targets) or len(chunks)
    cursor = 0
    grouped_words: List[List[Dict[str, Any]]] = []

    for idx, target_words in enumerate(chunk_word_targets):
        remaining_words = len(normalized_words) - cursor
        remaining_chunks = len(chunk_word_targets) - idx
        if idx == len(chunk_word_targets) - 1:
            take = remaining_words
        else:
            proportional_take = max(
                1,
                round(len(normalized_words) * (target_words / total_target_words))
            )
            take = min(proportional_take, max(1, remaining_words - (remaining_chunks - 1)))
        grouped_words.append(normalized_words[cursor:cursor + take])
        cursor += take

    return grouped_words


def _split_text_by_words(text: str, max_chars: int) -> List[str]:
    """Разбивает длинный текст на чанки по словам."""
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = word

    chunks.append(current)
    return chunks


def _split_text_to_chunks(text: str, max_chars: int) -> List[str]:
    """
    Старается резать текст по границам предложений и клауз,
    а к разбиению по словам переходит только как к запасному варианту.
    """
    text = _normalize_spaces(text)
    if not text or len(text) <= max_chars:
        return [text] if text else []

    sentence_parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text)
        if part.strip()
    ]

    units: List[str] = []
    for sentence in sentence_parts:
        if len(sentence) <= max_chars:
            units.append(sentence)
            continue

        clause_parts = [
            part.strip()
            for part in re.split(r"(?<=[,;:])\s+", sentence)
            if part.strip()
        ]
        if len(clause_parts) == 1:
            units.extend(_split_text_by_words(sentence, max_chars))
            continue

        for clause in clause_parts:
            if len(clause) <= max_chars:
                units.append(clause)
            else:
                units.extend(_split_text_by_words(clause, max_chars))

    chunks: List[str] = []
    current = ""

    for unit in units:
        candidate = f"{current} {unit}".strip()
        if not current or len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = unit

    if current:
        chunks.append(current)

    return chunks


def split_segments_for_translation(
    segments: List[Dict[str, Any]],
    max_chars: int
) -> List[Dict[str, Any]]:
    """
    Разбивает слишком длинные сегменты до перевода, чтобы:
    1. не терять хвосты при MT/TTS;
    2. сохранить исходные временные окна пропорционально длине текста.
    """
    if max_chars <= 0:
        return segments

    prepared: List[Dict[str, Any]] = []
    split_count = 0

    for seg in segments:
        text = _normalize_spaces(seg.get("text", ""))
        if not text:
            continue

        chunks = _split_text_to_chunks(text, max_chars)
        source_words = seg.get("words", []) if isinstance(seg.get("words"), list) else []
        split_words = _split_words_for_chunks(source_words, chunks)
        if len(chunks) <= 1:
            copied = dict(seg)
            copied["text"] = text
            prepared.append(copied)
            continue

        split_count += 1
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        duration = max(0.0, end - start)
        weights = [max(len(chunk), 1) for chunk in chunks]
        total_weight = sum(weights) or len(chunks)
        cursor = start

        for idx, chunk in enumerate(chunks):
            if idx == len(chunks) - 1:
                chunk_start = cursor
                chunk_end = end
            else:
                chunk_duration = duration * (weights[idx] / total_weight)
                chunk_start = cursor
                chunk_end = min(end, cursor + chunk_duration)
                cursor = chunk_end

            item = dict(seg)
            item["text"] = chunk
            chunk_words = split_words[idx] if idx < len(split_words) else []
            if chunk_words:
                item["words"] = deepcopy(chunk_words)
                item["words_with_silence"] = _build_words_with_silence_from_dicts(chunk_words)
                item["start"] = round(float(chunk_words[0]["start"]), 3)
                item["end"] = round(float(chunk_words[-1]["end"]), 3)
                item["source_duration_sec"] = round(
                    max(0.0, float(chunk_words[-1]["end"]) - float(chunk_words[0]["start"])),
                    3
                )
                item["source_word_count"] = len(chunk_words)
            else:
                item["start"] = round(chunk_start, 3)
                item["end"] = round(chunk_end, 3)
                item["source_duration_sec"] = round(max(0.0, chunk_end - chunk_start), 3)
                item["source_word_count"] = max(1, len(chunk.split()))

            if idx == 0:
                item["pause_before_sec"] = seg.get("pause_before_sec", 0.0)
            else:
                item["pause_before_sec"] = 0.0
            if idx == len(chunks) - 1:
                item["pause_after_sec"] = seg.get("pause_after_sec", 0.0)
            else:
                item["pause_after_sec"] = 0.0
            prepared.append(item)

    if split_count:
        logger.info(
            "Разбиты длинные сегменты: %s -> %s (split=%s, max_chars=%s)",
            len(segments),
            len(prepared),
            split_count,
            max_chars
        )

    return prepared
