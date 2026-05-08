import json
import logging
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib import error, request

try:
    import torch
except ImportError:
    torch = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import config as cfg

logger = logging.getLogger(__name__)


@dataclass
class GeminiTranslationBackend:
    model_name: str
    api_key: str
    temperature: float
    max_output_tokens: int
    thinking_budget: int
    timeout_sec: int
    min_interval_sec: float
    max_retries: int
    backend: str = "gemini"
    last_request_ts: float = 0.0


@dataclass
class OpenAICompatibleTranslationBackend:
    model_name: str
    api_key: str
    base_url: str
    temperature: float
    max_output_tokens: int
    timeout_sec: int
    min_interval_sec: float
    max_retries: int
    backend: str = "openai_compatible"
    last_request_ts: float = 0.0
    client: Any | None = None


SmartSyncBackend = GeminiTranslationBackend | OpenAICompatibleTranslationBackend


def _single_speaker_id(segments: List[Dict[str, Any]]) -> str | None:
    """Возвращает speaker_id, если у всех сегментов он одинаковый."""
    speaker_ids = {
        seg.get("speaker_id")
        for seg in segments
        if seg.get("speaker_id")
    }
    if len(speaker_ids) == 1:
        return next(iter(speaker_ids))
    return None


def _normalize_spaces(text: str) -> str:
    """Чистит лишние пробелы, не меняя смысл текста."""
    return re.sub(r"\s+", " ", text).strip()


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


def _merge_segment_metadata(source_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if not source_segments:
        return merged

    all_words: List[Dict[str, Any]] = []
    words_with_silence_parts: List[str] = []
    for idx, segment in enumerate(source_segments):
        segment_words = [
            {
                "text": _normalize_spaces(str(word.get("text", ""))),
                "start": round(float(word.get("start", 0.0)), 3),
                "end": round(float(word.get("end", 0.0)), 3),
            }
            for word in segment.get("words", [])
            if _normalize_spaces(str(word.get("text", "")))
        ]
        if segment_words:
            all_words.extend(segment_words)

        pause_profile = _normalize_spaces(str(segment.get("words_with_silence", "")))
        if pause_profile:
            if words_with_silence_parts and idx > 0:
                gap_sec = max(
                    0.0,
                    float(segment.get("start", 0.0)) - float(source_segments[idx - 1].get("end", 0.0))
                )
                if gap_sec >= 0.04:
                    words_with_silence_parts.append(f"<{gap_sec:.2f}s>")
            words_with_silence_parts.append(pause_profile)

    if all_words:
        merged["words"] = all_words
        merged["source_word_count"] = len(all_words)
        merged["source_duration_sec"] = round(
            max(0.0, float(all_words[-1]["end"]) - float(all_words[0]["start"])),
            3
        )
    else:
        merged["source_duration_sec"] = round(
            max(0.0, float(source_segments[-1].get("end", 0.0)) - float(source_segments[0].get("start", 0.0))),
            3
        )
        merged["source_word_count"] = sum(int(seg.get("source_word_count") or 0) for seg in source_segments)

    if words_with_silence_parts:
        merged["words_with_silence"] = " ".join(words_with_silence_parts)

    first = source_segments[0]
    last = source_segments[-1]
    if first.get("pause_before_sec") is not None:
        merged["pause_before_sec"] = deepcopy(first["pause_before_sec"])
    if last.get("pause_after_sec") is not None:
        merged["pause_after_sec"] = deepcopy(last["pause_after_sec"])
    return merged


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


def normalize_translated_text(text: str, original_text: str = "") -> str:
    """Подчищает машинный перевод и исправляет частые доменные промахи."""
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

    replacements = {
        r"\bвремя высадки\b": "время выезда",
        r"\bвысадки\b": "выезда",
        r"\bдля целей регистрации\b": "для регистрации",
        r"\bработник ресепшн\b": "сотрудник стойки регистрации",
        r"\bна ресепшн\b": "на стойке регистрации",
    }

    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

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


def _language_name(lang_code: str) -> str:
    """Преобразует код языка в читаемое имя для prompt-based моделей."""
    prefix = (lang_code or "").split("_", 1)[0]
    mapping = {
        "eng": "English",
        "rus": "Russian",
    }
    return mapping.get(prefix, lang_code or "the target language")


def _cleanup_causal_translation(text: str, tgt_lang: str = "") -> str:
    """Подчищает ответ chat-модели до чистого перевода."""
    cleaned = text.strip().strip("\"'`")
    cleaned = cleaned.replace("<|im_end|>", " ").replace("<|endoftext|>", " ")
    cleaned = _normalize_spaces(cleaned)

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


def _is_gemini_model(model_name: str) -> bool:
    return (model_name or "").strip().lower().startswith("gemini-")


def _build_timing_rewrite_prompts(
    *,
    source_text: str,
    translated_text: str,
    src_lang: str,
    tgt_lang: str,
    target_duration_sec: float,
    available_duration_sec: float,
    current_duration_sec: float,
    words_with_silence: str,
    rewrite_mode: str,
    previous_text: str = "",
    next_text: str = "",
) -> tuple[str, str]:
    src_name = _language_name(src_lang)
    tgt_name = _language_name(tgt_lang)
    timing_delta_sec = max(0.0, current_duration_sec - target_duration_sec)
    if rewrite_mode != "shorter":
        timing_delta_sec = max(0.0, target_duration_sec - current_duration_sec)

    pause_values = [
        float(match)
        for match in re.findall(r"<\s*([0-9]*\.?[0-9]+)\s*s?\s*>", words_with_silence or "")
    ]
    max_pause_sec = max(pause_values) if pause_values else 0.0
    no_pause_budget = max_pause_sec < 0.08

    system_prompt = (
        "You are a dialogue adaptation assistant for a video dubbing pipeline. "
        f"Rewrite an already translated {tgt_name} line so it better matches the timing of the original speech. "
        "Preserve meaning, tone, intent, and named entities. "
        "Return only the rewritten line in the target language."
    )
    context_blocks = []
    if previous_text.strip():
        context_blocks.append(f"Previous dialogue context:\n{previous_text.strip()}")
    context_blocks.append(
        f"Original source ({src_name}):\n{source_text.strip() or '-'}\n\n"
        f"Current translated line ({tgt_name}):\n{translated_text.strip()}"
    )
    if next_text.strip():
        context_blocks.append(f"Next dialogue context:\n{next_text.strip()}")

    if rewrite_mode == "shorter":
        max_words_to_remove = min(
            10,
            max(0, int((timing_delta_sec * 1.8) + 0.999))
        )
        deletion_budget_rule = (
            "Do NOT delete any content words; use only micro-edits, shorter synonyms, punctuation tightening, or contractions if the language allows them."
            if max_words_to_remove == 0
            else (
                f"You may delete at most {max_words_to_remove} content word(s). "
                "Prefer micro-edits first; delete only if still too long."
            )
        )
        user_prompt = (
            f"{chr(10).join(context_blocks)}\n\n"
            f"Timing constraints:\n"
            f"- Original speech duration: {target_duration_sec:.2f}s\n"
            f"- Maximum safe duration in the current timeline: {available_duration_sec:.2f}s\n"
            f"- Current synthesized duration: {current_duration_sec:.2f}s\n"
            f"- Overage to reduce: {timing_delta_sec:.2f}s\n"
            f"- Original speech timing profile: {words_with_silence or '-'}\n\n"
            "Task:\n"
            "- Make the line shorter and denser so it can be spoken naturally inside the safe duration.\n"
            "- It is acceptable to remain slightly longer than the target; ending up too short is worse than being slightly over.\n"
            "- Apply compression in this order: micro-edits first, then remove discourse fillers or redundant intensifiers, then rephrase clauses to use fewer syllables.\n"
            f"- {deletion_budget_rule}\n"
            "- Preserve meaning-critical tokens: names, numbers, dates, technical terms, and negations.\n"
            "- Keep natural spoken rhythm close to the original pauses and clause boundaries.\n"
            "- Keep the result natural for dubbing, not bookish, and consistent with the neighbouring dialogue.\n"
            "- Return one rewritten line only."
        )
    else:
        min_gap_sec = 0.12
        allowed_break_count = 0 if no_pause_budget else (1 if timing_delta_sec <= 0.6 else 2 if timing_delta_sec <= 1.2 else 3)
        hard_cap_added_words = max(2, int((timing_delta_sec * 4.0) + 0.999))
        hard_cap_added_chars = max(18, int((timing_delta_sec * 28.0) + 0.5))
        punctuation_only = timing_delta_sec <= 0.28
        user_prompt = (
            f"{chr(10).join(context_blocks)}\n\n"
            f"Timing constraints:\n"
            f"- Original speech duration: {target_duration_sec:.2f}s\n"
            f"- Current synthesized duration: {current_duration_sec:.2f}s\n"
            f"- Missing time to add: {timing_delta_sec:.2f}s\n"
            f"- Original speech timing profile: {words_with_silence or '-'}\n"
            f"- Largest original inter-word pause: {max_pause_sec:.2f}s\n\n"
            "Task:\n"
            "- Make the line slightly fuller and more natural for speech so it does not end too abruptly.\n"
            "- Prefer the smallest natural change: punctuation first for tiny gaps, then light lexical expansion, and only then a slightly fuller rephrasing.\n"
            f"- Add at most {hard_cap_added_words} new words and at most about {hard_cap_added_chars} additional characters.\n"
            "- Add only light natural phrasing; do not invent new facts, motivations, or side clauses.\n"
            "- Keep names, numbers, and technical terms intact.\n"
            "- Prefer slight underfill over bloated phrasing if the line would otherwise become unnatural.\n"
            f"- {'For this small timing gap, prefer punctuation only and avoid adding new words unless absolutely necessary.' if punctuation_only else 'If you add words, use only short natural expansions such as aspectual or clarifying phrasing.'}\n"
            f"- {'Avoid adding extra pause-like structure because the original line has almost no pause budget.' if no_pause_budget else f'Keep any extra pause-like punctuation aligned with natural clause boundaries; the original pause budget is modest, with at most about {allowed_break_count} natural pause point(s) above {min_gap_sec:.2f}s.'}\n"
            "- Keep the result natural for dubbing and consistent with the neighbouring dialogue.\n"
            "- Return one rewritten line only."
        )
    return system_prompt, user_prompt


def _build_causal_translation_prompt(
    text: str,
    src_lang: str,
    tgt_lang: str,
    tokenizer
) -> str:
    """Строит строгий prompt для instruct/chat-модели."""
    src_name = _language_name(src_lang)
    tgt_name = _language_name(tgt_lang)
    system_prompt = (
        "You are a professional translator. "
        f"Translate from {src_name} to {tgt_name}. "
        "Return only the translation with no explanations, notes, or quotes."
    )
    user_prompt = f"Text:\n{text}"

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True
        )

    return (
        f"{system_prompt}\n\n"
        f"{user_prompt}\n\n"
        f"{tgt_name} translation:"
    )


def _build_gemini_translation_request(
    text: str,
    src_lang: str,
    tgt_lang: str,
    backend: GeminiTranslationBackend,
    max_length: int
) -> Dict[str, Any]:
    src_name = _language_name(src_lang)
    tgt_name = _language_name(tgt_lang)
    system_prompt = (
        "You are a professional translator for a video dubbing pipeline. "
        f"Translate from {src_name} to {tgt_name}. "
        "Preserve meaning, tone, and named entities. "
        "Return only the translation with no explanations, notes, or quotes."
    )
    generation_config: Dict[str, Any] = {
        "temperature": backend.temperature,
        "candidateCount": 1,
        "maxOutputTokens": max(64, min(max_length, backend.max_output_tokens)),
        "responseMimeType": "text/plain",
    }
    if backend.thinking_budget >= 0:
        generation_config["thinkingConfig"] = {
            "thinkingBudget": backend.thinking_budget
        }

    return {
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"Text:\n{text}"}]
            }
        ],
        "generationConfig": generation_config,
    }


def _build_gemini_batch_translation_request(
    texts: List[str],
    src_lang: str,
    tgt_lang: str,
    backend: GeminiTranslationBackend,
    max_length: int
) -> Dict[str, Any]:
    src_name = _language_name(src_lang)
    tgt_name = _language_name(tgt_lang)
    system_prompt = (
        "You are a professional translator for a video dubbing pipeline. "
        f"Translate from {src_name} to {tgt_name}. "
        "Return a JSON array of translated strings in the same order and with the same length "
        "as the input JSON array. Return only valid JSON."
    )
    generation_config: Dict[str, Any] = {
        "temperature": backend.temperature,
        "candidateCount": 1,
        "maxOutputTokens": max(128, min(max_length, backend.max_output_tokens)),
        "responseMimeType": "application/json",
    }
    if backend.thinking_budget >= 0:
        generation_config["thinkingConfig"] = {
            "thinkingBudget": backend.thinking_budget
        }

    return {
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{
                    "text": (
                        "Input JSON array:\n"
                        f"{json.dumps(texts, ensure_ascii=False)}"
                    )
                }]
            }
        ],
        "generationConfig": generation_config,
    }


def _extract_gemini_text(response_json: Dict[str, Any]) -> str:
    candidates = response_json.get("candidates") or []
    if not candidates:
        prompt_feedback = response_json.get("promptFeedback")
        raise RuntimeError(
            f"Gemini не вернул кандидатов. promptFeedback={prompt_feedback}"
        )

    parts = candidates[0].get("content", {}).get("parts", [])
    text = " ".join(
        part.get("text", "")
        for part in parts
        if part.get("text")
    ).strip()
    if not text:
        finish_reason = candidates[0].get("finishReason")
        raise RuntimeError(
            f"Gemini вернул пустой ответ. finishReason={finish_reason}"
        )
    return text


def _parse_gemini_batch_response(
    text: str,
    expected_count: int,
    tgt_lang: str
) -> List[str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini вернул невалидный JSON для batch-перевода: {text}") from exc

    if not isinstance(data, list):
        raise RuntimeError(f"Gemini batch-ответ должен быть JSON-массивом, получено: {type(data).__name__}")
    if len(data) != expected_count:
        raise RuntimeError(
            f"Gemini batch-ответ имеет длину {len(data)}, ожидалось {expected_count}."
        )

    result: List[str] = []
    for item in data:
        if not isinstance(item, str):
            raise RuntimeError("Gemini batch-ответ должен содержать только строки.")
        result.append(_cleanup_causal_translation(item, tgt_lang=tgt_lang))
    return result


def _gemini_generate_translation(
    text: str,
    src_lang: str,
    tgt_lang: str,
    backend: GeminiTranslationBackend,
    max_length: int
) -> str:
    payload = _build_gemini_translation_request(
        text=text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        backend=backend,
        max_length=max_length,
    )
    wait_sec = backend.min_interval_sec - (time.monotonic() - backend.last_request_ts)
    if wait_sec > 0:
        time.sleep(wait_sec)

    req = request.Request(
        url=(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{backend.model_name}:generateContent?key={backend.api_key}"
        ),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error = None
    for attempt in range(backend.max_retries + 1):
        try:
            with request.urlopen(req, timeout=backend.timeout_sec) as resp:
                response_json = json.loads(resp.read().decode("utf-8"))
                backend.last_request_ts = time.monotonic()
                return _cleanup_causal_translation(
                    _extract_gemini_text(response_json),
                    tgt_lang=tgt_lang
                )
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"Gemini API HTTP {exc.code}: {details}")
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= backend.max_retries:
                raise last_error from exc

            retry_after = exc.headers.get("Retry-After")
            if retry_after:
                delay = max(float(retry_after), backend.min_interval_sec)
            else:
                delay = max(backend.min_interval_sec, min(60.0, 2 ** attempt))
            logger.warning(
                "Gemini retry после HTTP %s через %.1f сек (attempt=%s/%s)",
                exc.code,
                delay,
                attempt + 1,
                backend.max_retries + 1
            )
            time.sleep(delay)
        except error.URLError as exc:
            last_error = RuntimeError(f"Не удалось подключиться к Gemini API: {exc}")
            if attempt >= backend.max_retries:
                raise last_error from exc
            delay = max(backend.min_interval_sec, min(60.0, 2 ** attempt))
            logger.warning(
                "Gemini retry после сетевой ошибки через %.1f сек (attempt=%s/%s)",
                delay,
                attempt + 1,
                backend.max_retries + 1
            )
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini translation failed with unknown error.")


def _gemini_generate_batch_translations(
    texts: List[str],
    src_lang: str,
    tgt_lang: str,
    backend: GeminiTranslationBackend,
    max_length: int
) -> List[str]:
    payload = _build_gemini_batch_translation_request(
        texts=texts,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        backend=backend,
        max_length=max_length,
    )

    wait_sec = backend.min_interval_sec - (time.monotonic() - backend.last_request_ts)
    if wait_sec > 0:
        time.sleep(wait_sec)

    req = request.Request(
        url=(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{backend.model_name}:generateContent?key={backend.api_key}"
        ),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error = None
    for attempt in range(backend.max_retries + 1):
        try:
            with request.urlopen(req, timeout=backend.timeout_sec) as resp:
                response_json = json.loads(resp.read().decode("utf-8"))
                backend.last_request_ts = time.monotonic()
                return _parse_gemini_batch_response(
                    _extract_gemini_text(response_json),
                    expected_count=len(texts),
                    tgt_lang=tgt_lang
                )
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"Gemini API HTTP {exc.code}: {details}")
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= backend.max_retries:
                raise last_error from exc

            retry_after = exc.headers.get("Retry-After")
            if retry_after:
                delay = max(float(retry_after), backend.min_interval_sec)
            else:
                delay = max(backend.min_interval_sec, min(60.0, 2 ** attempt))
            logger.warning(
                "Gemini batch retry после HTTP %s через %.1f сек (attempt=%s/%s)",
                exc.code,
                delay,
                attempt + 1,
                backend.max_retries + 1
            )
            time.sleep(delay)
        except error.URLError as exc:
            last_error = RuntimeError(f"Не удалось подключиться к Gemini API: {exc}")
            if attempt >= backend.max_retries:
                raise last_error from exc
            delay = max(backend.min_interval_sec, min(60.0, 2 ** attempt))
            logger.warning(
                "Gemini batch retry после сетевой ошибки через %.1f сек (attempt=%s/%s)",
                delay,
                attempt + 1,
                backend.max_retries + 1
            )
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini batch translation failed with unknown error.")


def _build_gemini_timing_rewrite_request(
    *,
    source_text: str,
    translated_text: str,
    src_lang: str,
    tgt_lang: str,
    backend: GeminiTranslationBackend,
    target_duration_sec: float,
    available_duration_sec: float,
    current_duration_sec: float,
    words_with_silence: str,
    rewrite_mode: str,
    previous_text: str = "",
    next_text: str = "",
) -> Dict[str, Any]:
    system_prompt, user_prompt = _build_timing_rewrite_prompts(
        source_text=source_text,
        translated_text=translated_text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        target_duration_sec=target_duration_sec,
        available_duration_sec=available_duration_sec,
        current_duration_sec=current_duration_sec,
        words_with_silence=words_with_silence,
        rewrite_mode=rewrite_mode,
        previous_text=previous_text,
        next_text=next_text,
    )
    generation_config: Dict[str, Any] = {
        "temperature": min(0.35, max(0.0, backend.temperature)),
        "candidateCount": 1,
        "maxOutputTokens": max(96, min(backend.max_output_tokens, 256)),
        "responseMimeType": "text/plain",
    }
    if backend.thinking_budget >= 0:
        generation_config["thinkingConfig"] = {
            "thinkingBudget": backend.thinking_budget
        }

    return {
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            }
        ],
        "generationConfig": generation_config,
    }


def _extract_openai_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    return ""


def _openai_compatible_rewrite_text_for_timing(
    *,
    source_text: str,
    translated_text: str,
    src_lang: str,
    tgt_lang: str,
    backend: OpenAICompatibleTranslationBackend,
    target_duration_sec: float,
    available_duration_sec: float,
    current_duration_sec: float,
    words_with_silence: str,
    rewrite_mode: str,
    previous_text: str = "",
    next_text: str = "",
) -> str:
    if backend.client is None:
        raise RuntimeError("OpenAI-compatible SmartSync backend is not initialized.")

    system_prompt, user_prompt = _build_timing_rewrite_prompts(
        source_text=source_text,
        translated_text=translated_text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        target_duration_sec=target_duration_sec,
        available_duration_sec=available_duration_sec,
        current_duration_sec=current_duration_sec,
        words_with_silence=words_with_silence,
        rewrite_mode=rewrite_mode,
        previous_text=previous_text,
        next_text=next_text,
    )

    wait_sec = backend.min_interval_sec - (time.monotonic() - backend.last_request_ts)
    if wait_sec > 0:
        time.sleep(wait_sec)

    last_error = None
    for attempt in range(backend.max_retries + 1):
        try:
            response = backend.client.chat.completions.create(
                model=backend.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=min(0.35, max(0.0, backend.temperature)),
                max_tokens=max(96, min(backend.max_output_tokens, 256)),
            )
            backend.last_request_ts = time.monotonic()
            choice = response.choices[0] if getattr(response, "choices", None) else None
            message = getattr(choice, "message", None) if choice is not None else None
            content = getattr(message, "content", "") if message is not None else ""
            text = _extract_openai_message_text(content)
            if not text:
                raise RuntimeError(
                    f"{backend.backend} SmartSync вернул пустой ответ."
                )
            return _cleanup_causal_translation(text, tgt_lang=tgt_lang)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code is None:
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)

            message = str(exc)
            last_error = RuntimeError(
                f"{backend.backend} SmartSync error"
                + (f" HTTP {status_code}" if status_code else "")
                + f": {message}"
            )
            retriable = status_code in {408, 409, 429, 500, 502, 503, 504} or status_code is None
            if not retriable or attempt >= backend.max_retries:
                raise last_error from exc
            delay = max(backend.min_interval_sec, min(30.0, 2 ** attempt))
            logger.warning(
                "%s SmartSync retry через %.1f сек (attempt=%s/%s): %s",
                backend.backend,
                delay,
                attempt + 1,
                backend.max_retries + 1,
                message,
            )
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{backend.backend} SmartSync rewrite failed with unknown error.")


def _gemini_rewrite_text_for_timing(
    *,
    source_text: str,
    translated_text: str,
    src_lang: str,
    tgt_lang: str,
    backend: GeminiTranslationBackend,
    target_duration_sec: float,
    available_duration_sec: float,
    current_duration_sec: float,
    words_with_silence: str,
    rewrite_mode: str,
    previous_text: str = "",
    next_text: str = "",
) -> str:
    payload = _build_gemini_timing_rewrite_request(
        source_text=source_text,
        translated_text=translated_text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        backend=backend,
        target_duration_sec=target_duration_sec,
        available_duration_sec=available_duration_sec,
        current_duration_sec=current_duration_sec,
        words_with_silence=words_with_silence,
        rewrite_mode=rewrite_mode,
        previous_text=previous_text,
        next_text=next_text,
    )
    wait_sec = backend.min_interval_sec - (time.monotonic() - backend.last_request_ts)
    if wait_sec > 0:
        time.sleep(wait_sec)

    req = request.Request(
        url=(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{backend.model_name}:generateContent?key={backend.api_key}"
        ),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error = None
    for attempt in range(backend.max_retries + 1):
        try:
            with request.urlopen(req, timeout=backend.timeout_sec) as resp:
                response_json = json.loads(resp.read().decode("utf-8"))
                backend.last_request_ts = time.monotonic()
                return _cleanup_causal_translation(
                    _extract_gemini_text(response_json),
                    tgt_lang=tgt_lang
                )
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"Gemini SmartSync HTTP {exc.code}: {details}")
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= backend.max_retries:
                raise last_error from exc
            retry_after = exc.headers.get("Retry-After")
            if retry_after:
                delay = max(float(retry_after), backend.min_interval_sec)
            else:
                delay = max(backend.min_interval_sec, min(60.0, 2 ** attempt))
            logger.warning(
                "Gemini SmartSync retry после HTTP %s через %.1f сек (attempt=%s/%s)",
                exc.code,
                delay,
                attempt + 1,
                backend.max_retries + 1
            )
            time.sleep(delay)
        except error.URLError as exc:
            last_error = RuntimeError(f"Не удалось подключиться к Gemini SmartSync API: {exc}")
            if attempt >= backend.max_retries:
                raise last_error from exc
            delay = max(backend.min_interval_sec, min(60.0, 2 ** attempt))
            logger.warning(
                "Gemini SmartSync retry после сетевой ошибки через %.1f сек (attempt=%s/%s)",
                delay,
                attempt + 1,
                backend.max_retries + 1
            )
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini SmartSync rewrite failed with unknown error.")


def _default_openai_compatible_base_url(provider: str) -> str:
    provider = (provider or "").strip().lower()
    if provider == "cerebras":
        return "https://api.cerebras.ai/v1"
    if provider == "groq":
        return "https://api.groq.com/openai/v1"
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    return ""


def load_smart_sync_backend(device: str = "cpu") -> SmartSyncBackend | None:
    del device
    if not cfg.SMART_SYNC_ENABLED:
        return None
    provider = getattr(cfg, "SMART_SYNC_PROVIDER", "gemini").strip().lower()
    api_key = os.getenv(cfg.SMART_SYNC_API_KEY_ENV, "").strip()
    if not api_key:
        raise EnvironmentError(
            "Для SmartSync установите переменную окружения "
            f"{cfg.SMART_SYNC_API_KEY_ENV}."
        )

    if provider == "gemini":
        if not _is_gemini_model(cfg.SMART_SYNC_MODEL_NAME):
            logger.info(
                "SmartSync rewrite пропущен: SMART_SYNC_MODEL_NAME=%s не является Gemini-моделью.",
                cfg.SMART_SYNC_MODEL_NAME,
            )
            return None
        return GeminiTranslationBackend(
            model_name=cfg.SMART_SYNC_MODEL_NAME,
            api_key=api_key,
            temperature=min(cfg.SMART_SYNC_TEMPERATURE, 0.3),
            max_output_tokens=min(cfg.SMART_SYNC_MAX_OUTPUT_TOKENS, 256),
            thinking_budget=cfg.MT_GEMINI_THINKING_BUDGET,
            timeout_sec=cfg.SMART_SYNC_TIMEOUT_SEC,
            min_interval_sec=cfg.SMART_SYNC_MIN_INTERVAL_SEC,
            max_retries=cfg.SMART_SYNC_MAX_RETRIES,
        )

    if provider in {"cerebras", "groq", "openrouter", "openai_compatible"}:
        if OpenAI is None:
            raise ImportError(
                "Для OpenAI-compatible SmartSync требуется пакет openai."
            )
        base_url = getattr(cfg, "SMART_SYNC_BASE_URL", "").strip() or _default_openai_compatible_base_url(provider)
        if not base_url:
            raise EnvironmentError(
                "Для OpenAI-compatible SmartSync установите SMART_SYNC_BASE_URL."
            )
        return OpenAICompatibleTranslationBackend(
            model_name=cfg.SMART_SYNC_MODEL_NAME,
            api_key=api_key,
            base_url=base_url,
            temperature=min(cfg.SMART_SYNC_TEMPERATURE, 0.3),
            max_output_tokens=min(cfg.SMART_SYNC_MAX_OUTPUT_TOKENS, 256),
            timeout_sec=cfg.SMART_SYNC_TIMEOUT_SEC,
            min_interval_sec=cfg.SMART_SYNC_MIN_INTERVAL_SEC,
            max_retries=cfg.SMART_SYNC_MAX_RETRIES,
            backend=provider,
            client=OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=cfg.SMART_SYNC_TIMEOUT_SEC,
            ),
        )

    logger.info(
        "SmartSync rewrite пропущен: SMART_SYNC_PROVIDER=%s не поддерживается.",
        provider,
    )
    return None


def smart_sync_rewrite_segment_text(
    *,
    segment: Dict[str, Any],
    backend: SmartSyncBackend | None,
    src_lang: str,
    tgt_lang: str,
    current_duration_sec: float,
    target_duration_sec: float,
    available_duration_sec: float,
    rewrite_mode: str,
    previous_text: str = "",
    next_text: str = "",
) -> tuple[str | None, Dict[str, Any] | None]:
    if backend is None:
        return None, None

    translated_text = _normalize_spaces(str(segment.get("text", "")))
    source_text = _normalize_spaces(str(segment.get("original_text", "")))
    if len(translated_text.split()) < 2:
        return None, None

    if backend.backend == "gemini":
        rewritten = _gemini_rewrite_text_for_timing(
            source_text=source_text,
            translated_text=translated_text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            backend=backend,
            target_duration_sec=target_duration_sec,
            available_duration_sec=available_duration_sec,
            current_duration_sec=current_duration_sec,
            words_with_silence=_normalize_spaces(str(segment.get("words_with_silence", ""))),
            rewrite_mode=rewrite_mode,
            previous_text=_normalize_spaces(previous_text),
            next_text=_normalize_spaces(next_text),
        )
    else:
        rewritten = _openai_compatible_rewrite_text_for_timing(
            source_text=source_text,
            translated_text=translated_text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            backend=backend,
            target_duration_sec=target_duration_sec,
            available_duration_sec=available_duration_sec,
            current_duration_sec=current_duration_sec,
            words_with_silence=_normalize_spaces(str(segment.get("words_with_silence", ""))),
            rewrite_mode=rewrite_mode,
            previous_text=_normalize_spaces(previous_text),
            next_text=_normalize_spaces(next_text),
        )
    rewritten = normalize_translated_text(rewritten, source_text)
    if not rewritten:
        return None, None
    if _normalize_spaces(rewritten).lower() == translated_text.lower():
        return None, None

    return rewritten, {
        "backend": backend.backend,
        "model_name": backend.model_name,
        "mode": rewrite_mode,
        "source_text": source_text,
        "translated_text": translated_text,
        "rewritten_text": rewritten,
        "target_duration_sec": round(target_duration_sec, 3),
        "available_duration_sec": round(available_duration_sec, 3),
        "current_duration_sec": round(current_duration_sec, 3),
        "words_with_silence": segment.get("words_with_silence"),
        "previous_text": _normalize_spaces(previous_text),
        "next_text": _normalize_spaces(next_text),
    }


def load_translation_model(model_name: str, device: str):
    """Загружает seq2seq- или causal-модель перевода через единый интерфейс."""
    if _is_gemini_model(model_name):
        api_key = os.getenv(cfg.MT_GEMINI_API_KEY_ENV, "").strip()
        if not api_key:
            raise EnvironmentError(
                "Для Gemini установите переменную окружения "
                f"{cfg.MT_GEMINI_API_KEY_ENV}."
            )

        backend = GeminiTranslationBackend(
            model_name=model_name,
            api_key=api_key,
            temperature=cfg.MT_GEMINI_TEMPERATURE,
            max_output_tokens=cfg.MT_GEMINI_MAX_OUTPUT_TOKENS,
            thinking_budget=cfg.MT_GEMINI_THINKING_BUDGET,
            timeout_sec=cfg.MT_GEMINI_TIMEOUT_SEC,
            min_interval_sec=cfg.MT_GEMINI_MIN_INTERVAL_SEC,
            max_retries=cfg.MT_GEMINI_MAX_RETRIES,
        )
        logger.info(
            "Модель перевода загружена: %s (backend=%s)",
            model_name,
            backend.backend
        )
        return backend, None

    if torch is None:
        raise ImportError(
            "Для HuggingFace-моделей перевода требуется torch. "
            "Для облачного перевода используйте MT_MODEL_NAME=gemini-*."
        )

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    model_config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    common_kwargs = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "device_map": "auto" if device == "cuda" else None,
    }

    if getattr(model_config, "is_encoder_decoder", False):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **common_kwargs)
        backend = "seq2seq"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
        backend = "causal"
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    logger.info("Модель перевода загружена: %s (backend=%s)", model_name, backend)
    return model, tokenizer


def _load_model(model_name: str, device: str):
    """Совместимость со старым API загрузки модели."""
    return load_translation_model(model_name, device)


def _translate_batch_seq2seq(
    texts: List[str],
    model,
    tokenizer,
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    max_length: int
) -> List[str]:
    """Переводит список текстов seq2seq-моделью."""
    tokenizer.src_lang = src_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)

    return results


def _translate_batch_causal(
    texts: List[str],
    model,
    tokenizer,
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    max_length: int
) -> List[str]:
    """Переводит список текстов prompt-based causal/chat-моделью."""
    results: List[str] = []
    max_new_tokens = max(64, min(max_length, 256))

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        prompts = [
            _build_causal_translation_prompt(text, src_lang, tgt_lang, tokenizer)
            for text in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(model.device)

        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for output_ids, prompt_len in zip(outputs, prompt_lengths):
            generated_ids = output_ids[int(prompt_len):]
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(_cleanup_causal_translation(decoded, tgt_lang=tgt_lang))

    return results


def _translate_batch(
    texts: List[str],
    model,
    tokenizer,
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    max_length: int
) -> List[str]:
    """Переводит список текстов батчами через подходящий backend модели."""
    if getattr(model, "backend", None) == "gemini":
        results: List[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            if len(batch) == 1:
                results.append(
                    _gemini_generate_translation(
                        text=batch[0],
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        backend=model,
                        max_length=max_length
                    )
                )
            else:
                results.extend(
                    _gemini_generate_batch_translations(
                        texts=batch,
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        backend=model,
                        max_length=max_length
                    )
                )
        return results

    if getattr(model.config, "is_encoder_decoder", False):
        return _translate_batch_seq2seq(
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            batch_size=batch_size,
            max_length=max_length
        )

    return _translate_batch_causal(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        batch_size=batch_size,
        max_length=max_length
    )


def translate_segments_as_sentences(
    model,
    tokenizer,
    segments: List[Dict[str, Any]],
    src_lang: str = "eng_Latn",
    tgt_lang: str = "rus_Cyrl",
    max_pause_merge: float = 0.5,
    batch_size: int = 8,
    max_length: int = 1024
) -> List[Dict[str, Any]]:
    """
    Sentence-level перевод: склеивает сегменты в полные предложения,
    переводит целиком, возвращает один сегмент на предложение.

    Склейка происходит по двум критериям:
        1. Пауза между сегментами < max_pause_merge секунд
        2. Предыдущий сегмент не заканчивается на знак конца предложения (. ! ?)

    Результирующий сегмент имеет:
        start = start первого исходного сегмента
        end   = end   последнего исходного сегмента
        text  = полный перевод предложения

    TTS получает осмысленную фразу целиком и синтезирует её
    в объединённый временной слот.

    Параметры:
        model: загруженная модель NLLB
        tokenizer: токенизатор NLLB
        segments: список сегментов [{text, start, end}, ...]
        src_lang: исходный язык
        tgt_lang: целевой язык
        max_pause_merge: максимальная пауза для склейки сегментов (сек)
        batch_size: размер батча
        max_length: максимальная длина перевода
    """
    import re

    valid_segs = [s for s in segments if s.get("text", "").strip()]
    if not valid_segs:
        logger.error("Нет сегментов для перевода.")
        return []

    SENTENCE_END = re.compile(r"[.!?]\s*$")

    # ── Шаг 1: склеиваем сегменты в предложения ──────────────────────────────
    sentences: List[Dict[str, Any]] = []   # {text, start, end, source_segments}
    current_group = [valid_segs[0]]

    for i in range(1, len(valid_segs)):
        prev = valid_segs[i - 1]
        curr = valid_segs[i]
        pause = curr["start"] - prev["end"]

        # Заканчивается ли предыдущий сегмент на конец предложения?
        ends_sentence = bool(SENTENCE_END.search(prev["text"].strip()))

        if ends_sentence or pause >= max_pause_merge:
            # Сохраняем текущую группу как предложение
            sentences.append({
                "text":            " ".join(s["text"].strip() for s in current_group),
                "start":           current_group[0]["start"],
                "end":             current_group[-1]["end"],
                "source_segments": current_group.copy()
            })
            current_group = [curr]
        else:
            current_group.append(curr)

    # Последняя группа
    sentences.append({
        "text":            " ".join(s["text"].strip() for s in current_group),
        "start":           current_group[0]["start"],
        "end":             current_group[-1]["end"],
        "source_segments": current_group.copy()
    })

    logger.info(f"Sentence-level: {len(valid_segs)} сегментов → {len(sentences)} предложений")

    # ── Шаг 2: переводим предложения батчами ─────────────────────────────────
    texts = [s["text"] for s in sentences]
    translated_texts = _translate_batch(
        texts, model, tokenizer, src_lang, tgt_lang, batch_size, max_length
    )

    # ── Шаг 3: формируем результат ───────────────────────────────────────────
    result: List[Dict[str, Any]] = []
    for sent, trans_text in zip(sentences, translated_texts):
        item = {
            "text":            trans_text,
            "original_text":   sent["text"],
            "start":           sent["start"],
            "end":             sent["end"],
            "merged_count":    len(sent["source_segments"])
        }
        speaker_id = _single_speaker_id(sent["source_segments"])
        if speaker_id:
            item["speaker_id"] = speaker_id
        item.update(_merge_segment_metadata(sent["source_segments"]))
        result.append(item)

    logger.info(f"Sentence-level перевод завершён: {len(result)} предложений")
    return result


def translate_segments(
    model,
    tokenizer,
    segments: List[Dict[str, Any]],
    src_lang: str = "eng_Latn",
    tgt_lang: str = "rus_Cyrl",
    batch_size: int = 8,
    max_length: int = 1024
) -> List[Dict[str, Any]]:
    """
    Базовый перевод: каждый сегмент переводится независимо.
    Быстро, но теряет контекст между фразами.
    """
    valid = [(i, s) for i, s in enumerate(segments) if s.get("text", "").strip()]
    if not valid:
        logger.error("Нет текста для перевода.")
        return []

    indices, segs = zip(*valid)
    texts = [s["text"].strip() for s in segs]

    logger.info(f"Per-segment: переводим {len(texts)} сегментов...")
    translated_texts = _translate_batch(
        texts, model, tokenizer, src_lang, tgt_lang, batch_size, max_length
    )

    result = []
    for i, (orig_idx, seg) in enumerate(zip(indices, segs)):
        item = {
            "text":          translated_texts[i],
            "original_text": seg["text"].strip(),
            "start":         seg["start"],
            "end":           seg["end"]
        }
        item = _copy_segment_metadata(seg, item)
        result.append(item)

    logger.info(f"Переведено сегментов: {len(result)}")
    return result


def translate_segments_sliding_window(
    model,
    tokenizer,
    segments: List[Dict[str, Any]],
    src_lang: str = "eng_Latn",
    tgt_lang: str = "rus_Cyrl",
    window_size: int = 2,
    batch_size: int = 8,
    max_length: int = 1024
) -> List[Dict[str, Any]]:
    """
    Sliding window перевод: каждый сегмент переводится с учётом соседних.

    Для сегмента N модель видит:
        [N-window_size] ... [N-1] | [N] | [N+1] ... [N+window_size]
    Переводится весь контекст, но берётся только перевод центрального сегмента.

    Преимущество над per-segment: модель видит контекст → лучше связность.
    Преимущество над context-aware: временные метки не ломаются.

    Параметры:
        model: загруженная модель NLLB
        tokenizer: токенизатор NLLB
        segments: список сегментов [{text, start, end}, ...]
        src_lang: исходный язык
        tgt_lang: целевой язык
        window_size: количество соседних сегментов с каждой стороны (1-3)
        batch_size: размер батча
        max_length: максимальная длина перевода
    """
    valid_segs = [s for s in segments if s.get("text", "").strip()]
    if not valid_segs:
        logger.error("Нет сегментов для перевода.")
        return []

    # Разделитель между сегментами в контексте
    SEP = " "

    # Строим тексты с контекстом для каждого сегмента
    context_texts = []
    for i in range(len(valid_segs)):
        left  = valid_segs[max(0, i - window_size):i]
        right = valid_segs[i + 1:i + 1 + window_size]
        center = valid_segs[i]

        # Собираем контекст: левые + центр + правые
        context = (
            SEP.join(s["text"].strip() for s in left)
            + (" " if left else "")
            + center["text"].strip()
            + (" " if right else "")
            + SEP.join(s["text"].strip() for s in right)
        ).strip()

        context_texts.append(context)

    logger.info(f"Sliding window (window={window_size}): переводим {len(context_texts)} сегментов...")

    translated_contexts = _translate_batch(
        context_texts, model, tokenizer, src_lang, tgt_lang, batch_size, max_length
    )

    # Извлекаем только перевод центрального сегмента
    # Логика: берём пропорциональную часть из середины перевода
    translated: List[Dict[str, Any]] = []

    for i, (seg, full_translation) in enumerate(zip(valid_segs, translated_contexts)):
        left_count  = min(i, window_size)
        right_count = min(len(valid_segs) - i - 1, window_size)
        total_segs_in_context = left_count + 1 + right_count

        words = full_translation.split()

        if total_segs_in_context == 1:
            # Нет соседей — берём всё
            center_words = words
        else:
            # Оцениваем долю центрального сегмента по длине символов
            left_chars  = sum(len(valid_segs[max(0, i - window_size + j)]["text"]) 
                              for j in range(left_count))
            center_chars = len(seg["text"])
            right_chars = sum(len(valid_segs[i + 1 + j]["text"]) 
                              for j in range(right_count))
            total_chars = left_chars + center_chars + right_chars

            if total_chars == 0:
                center_words = words
            else:
                ratio        = center_chars / total_chars
                n_words      = max(1, round(ratio * len(words)))
                left_words   = max(0, round((left_chars / total_chars) * len(words)))
                center_words = words[left_words:left_words + n_words]

        item = {
            "text":          " ".join(center_words) if center_words else full_translation,
            "original_text": seg["text"].strip(),
            "start":         seg["start"],
            "end":           seg["end"]
        }
        item = _copy_segment_metadata(seg, item)
        translated.append(item)

    logger.info(f"Sliding window перевод завершён: {len(translated)} сегментов")
    return translated


def translate_segments_with_context(
    model,
    tokenizer,
    segments: List[Dict[str, Any]],
    src_lang: str = "eng_Latn",
    tgt_lang: str = "rus_Cyrl",
    max_chunk_chars: int = 800,
    batch_size: int = 8,
    max_length: int = 1024
) -> List[Dict[str, Any]]:
    """
    Контекстный перевод: сегменты объединяются в смысловые чанки,
    переводятся целиком и пропорционально разбиваются обратно.

    Преимущество: переводчик видит соседние фразы — лучше связность и точность.
    """
    valid_segs = [s for s in segments if s.get("text", "").strip()]
    if not valid_segs:
        logger.error("Нет сегментов для перевода.")
        return []

    # Группируем в чанки не разрывая границы сегментов
    chunks: List[str] = []
    chunk_indices: List[List[int]] = []
    current_chunk = ""
    current_idx: List[int] = []

    for i, seg in enumerate(valid_segs):
        text = seg["text"].strip()
        if len(current_chunk) + len(text) + 1 > max_chunk_chars and current_chunk:
            chunks.append(current_chunk.strip())
            chunk_indices.append(current_idx)
            current_chunk = text
            current_idx = [i]
        else:
            current_chunk += (" " if current_chunk else "") + text
            current_idx.append(i)

    if current_chunk:
        chunks.append(current_chunk.strip())
        chunk_indices.append(current_idx)

    logger.info(f"Context-aware: {len(chunks)} чанков из {len(valid_segs)} сегментов...")
    translated_chunks = _translate_batch(
        chunks, model, tokenizer, src_lang, tgt_lang, batch_size, max_length
    )

    # Пропорционально разбиваем переводы обратно на сегменты
    translated: List[Dict[str, Any]] = []

    for trans_text, seg_indices in zip(translated_chunks, chunk_indices):
        words       = trans_text.split()
        chunk_segs  = [valid_segs[i] for i in seg_indices]
        total_chars = sum(len(s["text"].strip()) for s in chunk_segs)
        word_ptr    = 0

        for j, seg in enumerate(chunk_segs):
            is_last = (j == len(chunk_segs) - 1)
            if is_last:
                seg_words = words[word_ptr:]
            else:
                ratio     = len(seg["text"].strip()) / total_chars if total_chars else 0
                n_words   = max(1, round(ratio * len(words)))
                seg_words = words[word_ptr:word_ptr + n_words]
                word_ptr += n_words

            item = {
                "text":          " ".join(seg_words),
                "original_text": seg["text"].strip(),
                "start":         seg["start"],
                "end":           seg["end"]
            }
            item = _copy_segment_metadata(seg, item)
            translated.append(item)

    logger.info(f"Контекстный перевод завершён: {len(translated)} сегментов")
    return translated
