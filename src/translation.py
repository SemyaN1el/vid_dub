"""Машинный перевод сегментов EN->RU: провайдеры (локальный HF/NLLB, OpenAI-совместимые, Gemini), стратегии (per-segment, sentence-boundary-aware), length-aware перевод и техническая QA."""

import json
import logging
import os
import re
import time
from typing import Any, Callable, Dict, List
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
from src.translation_backends import (
    GeminiTranslationBackend,
    OpenAICompatibleTranslationBackend,
    SmartSyncBackend,
)
from src.translation_common import _is_gemini_model, _language_name, _normalize_spaces
from src.translation_length import (
    _format_segment_for_prompt,
    _length_aware_prompt_block,
    _translation_segment_metadata,
)
from src.translation_normalize import (
    _cleanup_causal_translation,
    normalize_translated_segments,
    normalize_translated_text,
)
from src.translation_profiles import (
    _asr_corrections_block,
    _load_translation_profiles,
    _parse_source_target_pairs,
    _profile_system_directives,
    _profile_user_directives_block,
    _split_config_list,
    _terminology_block,
    _translation_style_directives,
)
from src.translation_qa import analyze_translation_quality
from src.translation_segments import _copy_segment_metadata, split_segments_for_translation

logger = logging.getLogger(__name__)


def _no_code_switching_rules(tgt_name: str) -> str:
    return (
        f"The output must be fully in {tgt_name}. Translate ordinary source-language words, "
        "including fillers, adverbs, adjectives, verbs, and common nouns. Keep only product "
        "names, model names, library/framework names, standard acronyms, personal names, "
        "place names without a common target-language form, and exact code-like tokens."
    )


def _build_translation_system_prompt(
    src_lang: str,
    tgt_lang: str,
    *,
    batch: bool,
) -> str:
    src_name = _language_name(src_lang)
    tgt_name = _language_name(tgt_lang)
    cardinality_rule = (
        "Return a JSON object with a translations array in the same order and with exactly "
        "the same number of strings as the input segments. Preserve segment boundaries."
        if batch
        else "Return a JSON object with one translation string."
    )
    return (
        "You are a professional translator for a video dubbing pipeline. "
        f"Translate from {src_name} to {tgt_name}. "
        f"{_translation_style_directives()} "
        f"{_profile_system_directives()} "
        f"{_no_code_switching_rules(tgt_name)} "
        f"{_length_aware_prompt_block()} "
        f"{_terminology_block()} "
        f"{cardinality_rule} Return only valid JSON."
    )


def _build_translation_user_prompt(
    texts: List[str],
    *,
    src_lang: str,
    tgt_lang: str,
    batch: bool,
    segment_metadata: List[Dict[str, Any] | None] | None = None,
) -> str:
    del src_lang, tgt_lang
    correction_block = _asr_corrections_block()
    profile_block = _profile_user_directives_block()
    if batch:
        numbered = "\n".join(
            _format_segment_for_prompt(
                idx,
                text,
                segment_metadata[idx] if segment_metadata and idx < len(segment_metadata) else None,
            )
            for idx, text in enumerate(texts)
        )
        return (
            f"{correction_block}"
            f"{profile_block}"
            "Translate each numbered segment. Sentence fragments are fine; translate them as-is. "
            "Each input segment must map to exactly one output string. "
            "If segment metadata is present, use it only to control length and timing fit.\n\n"
            f"Segments ({len(texts)} total):\n{numbered}"
        )
    metadata_suffix = ""
    if segment_metadata and segment_metadata[0]:
        metadata_suffix = "\n\nSegment metadata:\n" + json.dumps(segment_metadata[0], ensure_ascii=False)
    return (
        f"{correction_block}"
        f"{profile_block}"
        "Translate this text. Sentence fragments are fine; translate them as-is.\n\n"
        f"Text: {json.dumps(texts[0] if texts else '', ensure_ascii=False)}"
        f"{metadata_suffix}"
    )


def _normalize_translation_provider(provider: str) -> str:
    provider = (provider or "").strip().lower()
    aliases = {
        "chatgpt": "openai",
        "openai-compatible": "openai_compatible",
        "openai_compatible": "openai_compatible",
    }
    return aliases.get(provider, provider)


def _infer_translation_provider(model_name: str) -> str:
    explicit = _normalize_translation_provider(os.getenv("MT_PROVIDER", getattr(cfg, "MT_PROVIDER", "")))
    if explicit:
        return explicit

    lowered = (model_name or "").strip().lower()
    if _is_gemini_model(model_name):
        return "gemini"
    if lowered.startswith(("gpt-", "o1", "o3", "o4", "chat-latest")):
        return "openai"
    return "hf"


def _translation_api_key_env(provider: str) -> str:
    configured = os.getenv("MT_OPENAI_API_KEY_ENV", getattr(cfg, "MT_OPENAI_API_KEY_ENV", "")).strip()
    if configured:
        return configured
    defaults = {
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "groq": "GROQ_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "openai_compatible": "OPENAI_API_KEY",
    }
    return defaults.get(provider, "OPENAI_API_KEY")


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
    candidate_count: int = 1,
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

    candidate_count = max(1, int(candidate_count or 1))
    if candidate_count > 1:
        output_rule = (
            f"Return valid JSON only: {{\"candidates\": [\"candidate 1\", \"candidate 2\", ...]}} "
            f"with exactly {candidate_count} target-language candidate strings and no extra keys."
        )
    else:
        output_rule = (
            "Return only the rewritten line in the target language, with no quotes, labels, or alternatives."
        )
    system_prompt = (
        "You are a dialogue adaptation assistant for a video dubbing pipeline. "
        f"Rewrite an already translated {tgt_name} line so it better matches the timing of the original speech. "
        "Preserve meaning, tone, intent, and named entities. "
        "Prefer the least invasive edit that fixes timing. "
        f"{output_rule}"
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
            f"- Final timeline window: {available_duration_sec:.2f}s\n"
            f"- Target synthesized duration before final speed fitting: {target_duration_sec:.2f}s\n"
            f"- Current synthesized duration: {current_duration_sec:.2f}s\n"
            f"- Overage to reduce: {timing_delta_sec:.2f}s\n"
            f"- Original speech timing profile: {words_with_silence or '-'}\n\n"
            "Task:\n"
            "- Make the line only as short as needed for the target synthesized duration; final audio may still be mildly sped up after this rewrite.\n"
            "- Do not over-compress: ending too short or losing content is worse than remaining slightly over the target.\n"
            "- Apply compression in this order: micro-edits first, shorter synonyms, remove filler/adverbs/redundant intensifiers, then rephrase clauses to use fewer syllables.\n"
            f"- {deletion_budget_rule}\n"
            "- Preserve meaning-critical tokens: names, numbers, dates, technical terms, and negations.\n"
            "- Preserve the final semantic unit of the line; do not drop or blur the ending.\n"
            "- Keep at least about three quarters of the content words unless the current line contains obvious filler.\n"
            "- Do not add a new trailing phrase after the final idea, and do not leave the line ending as an unfinished comma fragment.\n"
            "- Keep natural spoken rhythm close to the original pauses and clause boundaries.\n"
            "- Keep the result natural for dubbing, not bookish, and consistent with the neighbouring dialogue.\n"
            + (
                f"- Return exactly {candidate_count} alternatives ordered as: conservative, balanced, strongest safe compression.\n"
                if candidate_count > 1
                else "- Return one rewritten line only."
            )
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
            + (
                f"- Return exactly {candidate_count} alternatives ordered as: conservative, balanced, fullest safe expansion.\n"
                if candidate_count > 1
                else "- Return one rewritten line only."
            )
        )
    return system_prompt, user_prompt


def _build_causal_translation_prompt(
    text: str,
    src_lang: str,
    tgt_lang: str,
    tokenizer,
    segment_metadata: Dict[str, Any] | None = None,
) -> str:
    """Строит строгий prompt для instruct/chat-модели."""
    tgt_name = _language_name(tgt_lang)
    system_prompt = _build_translation_system_prompt(src_lang, tgt_lang, batch=False)
    user_prompt = _build_translation_user_prompt(
        [text],
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        batch=False,
        segment_metadata=[segment_metadata] if segment_metadata else None,
    )

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
        f"JSON response with {tgt_name} translation:"
    )


def _build_gemini_translation_request(
    text: str,
    src_lang: str,
    tgt_lang: str,
    backend: GeminiTranslationBackend,
    max_length: int,
    segment_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    system_prompt = _build_translation_system_prompt(src_lang, tgt_lang, batch=False)
    generation_config: Dict[str, Any] = {
        "temperature": backend.temperature,
        "candidateCount": 1,
        "maxOutputTokens": max(64, min(max_length, backend.max_output_tokens)),
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
                    "text": _build_translation_user_prompt(
                        [text],
                        src_lang=src_lang,
                        tgt_lang=tgt_lang,
                        batch=False,
                        segment_metadata=[segment_metadata] if segment_metadata else None,
                    )
                }]
            }
        ],
        "generationConfig": generation_config,
    }


def _build_gemini_batch_translation_request(
    texts: List[str],
    src_lang: str,
    tgt_lang: str,
    backend: GeminiTranslationBackend,
    max_length: int,
    segment_metadata: List[Dict[str, Any] | None] | None = None,
) -> Dict[str, Any]:
    system_prompt = _build_translation_system_prompt(src_lang, tgt_lang, batch=True)
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
                        _build_translation_user_prompt(
                            texts,
                            src_lang=src_lang,
                            tgt_lang=tgt_lang,
                            batch=True,
                            segment_metadata=segment_metadata,
                        )
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

    if isinstance(data, dict) and isinstance(data.get("translations"), list):
        data = data["translations"]
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
    max_length: int,
    segment_metadata: Dict[str, Any] | None = None,
) -> str:
    payload = _build_gemini_translation_request(
        text=text,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        backend=backend,
        max_length=max_length,
        segment_metadata=segment_metadata,
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
    max_length: int,
    segment_metadata: List[Dict[str, Any] | None] | None = None,
) -> List[str]:
    payload = _build_gemini_batch_translation_request(
        texts=texts,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        backend=backend,
        max_length=max_length,
        segment_metadata=segment_metadata,
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


_BATCH_TRANSLATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["translations"],
    "additionalProperties": False,
}


_SINGLE_TRANSLATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "translation": {"type": "string"},
    },
    "required": ["translation"],
    "additionalProperties": False,
}


_BOUNDARY_AWARE_TRANSLATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "integer"},
                    "translations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["group_id", "translations"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["groups"],
    "additionalProperties": False,
}


def _openai_response_formats(schema_name: str, schema: Dict[str, Any]) -> List[Dict[str, Any] | None]:
    configured = os.getenv(
        "MT_OPENAI_RESPONSE_FORMAT",
        getattr(cfg, "MT_OPENAI_RESPONSE_FORMAT", "json_schema"),
    ).strip().lower()
    if configured in {"none", "off", "disabled"}:
        return [None]
    if configured in {"json_object", "json"}:
        return [{"type": "json_object"}, None]
    return [
        {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
        {"type": "json_object"},
        None,
    ]


def _extract_openai_translation_payload(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


def _parse_openai_single_response(text: str, tgt_lang: str) -> str:
    try:
        data = _extract_openai_translation_payload(text)
    except (json.JSONDecodeError, ValueError):
        return _cleanup_causal_translation(text, tgt_lang=tgt_lang)
    if isinstance(data, dict):
        value = data.get("translation") or data.get("text") or data.get("result")
        if isinstance(value, str):
            return _cleanup_causal_translation(value, tgt_lang=tgt_lang)
    if isinstance(data, list) and data and isinstance(data[0], str):
        return _cleanup_causal_translation(data[0], tgt_lang=tgt_lang)
    raise RuntimeError(f"OpenAI-compatible single response has invalid shape: {text}")


def _parse_openai_batch_response(
    text: str,
    expected_count: int,
    tgt_lang: str,
) -> List[str]:
    data = _extract_openai_translation_payload(text)
    if isinstance(data, dict):
        if isinstance(data.get("translations"), list):
            data = data["translations"]
        elif isinstance(data.get("items"), list):
            data = data["items"]
    if not isinstance(data, list):
        raise RuntimeError(
            f"OpenAI-compatible batch response must be a list or translations object, got {type(data).__name__}."
        )
    if len(data) != expected_count:
        raise RuntimeError(
            f"OpenAI-compatible batch response length is {len(data)}, expected {expected_count}."
        )

    result: List[str] = []
    for item in data:
        if isinstance(item, str):
            result.append(_cleanup_causal_translation(item, tgt_lang=tgt_lang))
            continue
        if isinstance(item, dict):
            value = item.get("translation") or item.get("text")
            if isinstance(value, str):
                result.append(_cleanup_causal_translation(value, tgt_lang=tgt_lang))
                continue
        raise RuntimeError("OpenAI-compatible batch response items must be strings or translation objects.")
    return result


def _openai_completion_create(
    backend: OpenAICompatibleTranslationBackend,
    *,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    response_formats: List[Dict[str, Any] | None],
) -> Any:
    if backend.client is None:
        raise RuntimeError("OpenAI-compatible translation backend is not initialized.")

    wait_sec = backend.min_interval_sec - (time.monotonic() - backend.last_request_ts)
    if wait_sec > 0:
        time.sleep(wait_sec)

    last_error = None
    for response_format in response_formats:
        for attempt in range(backend.max_retries + 1):
            kwargs: Dict[str, Any] = {
                "model": backend.model_name,
                "messages": messages,
                "temperature": backend.temperature,
                "max_tokens": max_tokens,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            try:
                response = backend.client.chat.completions.create(**kwargs)
                backend.last_request_ts = time.monotonic()
                return response
            except Exception as exc:
                message = str(exc)
                status_code = getattr(exc, "status_code", None)
                if status_code is None:
                    response_obj = getattr(exc, "response", None)
                    status_code = getattr(response_obj, "status_code", None)

                unsupported_response_format = (
                    response_format is not None
                    and status_code in {400, 404, 422}
                    and (
                        "response_format" in message
                        or "json_validate_failed" in message
                        or "Failed to validate JSON" in message
                        or "Failed to generate JSON" in message
                    )
                )
                unsupported_max_tokens = status_code in {400, 422} and "max_tokens" in message
                unsupported_temperature = status_code in {400, 422} and "temperature" in message

                if unsupported_max_tokens or unsupported_temperature:
                    retry_kwargs = dict(kwargs)
                    if unsupported_max_tokens:
                        retry_kwargs["max_completion_tokens"] = retry_kwargs.pop("max_tokens")
                    if unsupported_temperature:
                        retry_kwargs.pop("temperature", None)
                    try:
                        response = backend.client.chat.completions.create(**retry_kwargs)
                        backend.last_request_ts = time.monotonic()
                        return response
                    except Exception as retry_exc:
                        exc = retry_exc
                        message = str(retry_exc)
                        status_code = getattr(retry_exc, "status_code", status_code)

                last_error = RuntimeError(
                    f"{backend.backend} translation error"
                    + (f" HTTP {status_code}" if status_code else "")
                    + f": {message}"
                )
                if unsupported_response_format:
                    logger.warning(
                        "%s не принял response_format=%s; пробую следующий формат.",
                        backend.backend,
                        response_format.get("type") if isinstance(response_format, dict) else response_format,
                    )
                    break

                retriable = status_code in {408, 409, 429, 500, 502, 503, 504} or status_code is None
                if not retriable or attempt >= backend.max_retries:
                    raise last_error from exc
                delay = max(backend.min_interval_sec, min(30.0, 2 ** attempt))
                logger.warning(
                    "%s translation retry через %.1f сек (attempt=%s/%s): %s",
                    backend.backend,
                    delay,
                    attempt + 1,
                    backend.max_retries + 1,
                    message,
                )
                time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{backend.backend} translation failed with unknown error.")


def _openai_generate_translation(
    text: str,
    src_lang: str,
    tgt_lang: str,
    backend: OpenAICompatibleTranslationBackend,
    max_length: int,
    segment_metadata: Dict[str, Any] | None = None,
) -> str:
    response = _openai_completion_create(
        backend,
        messages=[
            {"role": "system", "content": _build_translation_system_prompt(src_lang, tgt_lang, batch=False)},
            {
                "role": "user",
                "content": _build_translation_user_prompt(
                    [text],
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    batch=False,
                    segment_metadata=[segment_metadata] if segment_metadata else None,
                ),
            },
        ],
        max_tokens=max(64, min(max_length, backend.max_output_tokens)),
        response_formats=_openai_response_formats("single_translation", _SINGLE_TRANSLATION_SCHEMA),
    )
    choice = response.choices[0] if getattr(response, "choices", None) else None
    message = getattr(choice, "message", None) if choice is not None else None
    content = getattr(message, "content", "") if message is not None else ""
    text_out = _extract_openai_message_text(content)
    if not text_out:
        raise RuntimeError(f"{backend.backend} translation returned an empty response.")
    return _parse_openai_single_response(text_out, tgt_lang=tgt_lang)


def _openai_generate_batch_translations(
    texts: List[str],
    src_lang: str,
    tgt_lang: str,
    backend: OpenAICompatibleTranslationBackend,
    max_length: int,
    segment_metadata: List[Dict[str, Any] | None] | None = None,
) -> List[str]:
    response = _openai_completion_create(
        backend,
        messages=[
            {"role": "system", "content": _build_translation_system_prompt(src_lang, tgt_lang, batch=True)},
            {
                "role": "user",
                "content": _build_translation_user_prompt(
                    texts,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    batch=True,
                    segment_metadata=segment_metadata,
                ),
            },
        ],
        max_tokens=max(128, min(max_length, backend.max_output_tokens)),
        response_formats=_openai_response_formats("translation_response", _BATCH_TRANSLATION_SCHEMA),
    )
    choice = response.choices[0] if getattr(response, "choices", None) else None
    message = getattr(choice, "message", None) if choice is not None else None
    content = getattr(message, "content", "") if message is not None else ""
    text_out = _extract_openai_message_text(content)
    if not text_out:
        raise RuntimeError(f"{backend.backend} batch translation returned an empty response.")
    return _parse_openai_batch_response(text_out, expected_count=len(texts), tgt_lang=tgt_lang)


def _translate_api_chunk_with_fallback(
    *,
    texts: List[str],
    segment_metadata: List[Dict[str, Any] | None] | None = None,
    src_lang: str,
    tgt_lang: str,
    backend: GeminiTranslationBackend | OpenAICompatibleTranslationBackend,
    max_length: int,
    batch_fn: Callable[..., List[str]],
    single_fn: Callable[..., str],
) -> List[str]:
    if not texts:
        return []
    if len(texts) == 1:
        single_metadata = segment_metadata[0] if segment_metadata else None
        return [single_fn(texts[0], src_lang, tgt_lang, backend, max_length, segment_metadata=single_metadata)]

    try:
        return batch_fn(texts, src_lang, tgt_lang, backend, max_length, segment_metadata=segment_metadata)
    except Exception as exc:
        mid = len(texts) // 2
        logger.warning(
            "%s batch-перевод на %s сегм. не прошёл (%s); делю на %s + %s.",
            backend.backend,
            len(texts),
            exc,
            mid,
            len(texts) - mid,
        )
        return (
            _translate_api_chunk_with_fallback(
                texts=texts[:mid],
                segment_metadata=segment_metadata[:mid] if segment_metadata else None,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                backend=backend,
                max_length=max_length,
                batch_fn=batch_fn,
                single_fn=single_fn,
            )
            + _translate_api_chunk_with_fallback(
                texts=texts[mid:],
                segment_metadata=segment_metadata[mid:] if segment_metadata else None,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                backend=backend,
                max_length=max_length,
                batch_fn=batch_fn,
                single_fn=single_fn,
            )
        )


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
    candidate_count: int = 1,
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
        candidate_count=candidate_count,
        previous_text=previous_text,
        next_text=next_text,
    )
    generation_config: Dict[str, Any] = {
        "temperature": min(0.35, max(0.0, backend.temperature)),
        "candidateCount": 1,
        "maxOutputTokens": max(96, min(backend.max_output_tokens, 384 if candidate_count > 1 else 256)),
        "responseMimeType": "application/json" if candidate_count > 1 else "text/plain",
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
    candidate_count: int = 1,
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
        candidate_count=candidate_count,
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
                max_tokens=max(96, min(backend.max_output_tokens, 384 if candidate_count > 1 else 256)),
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
            return text
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
    candidate_count: int = 1,
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
        candidate_count=candidate_count,
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
                return _extract_gemini_text(response_json)
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
    if provider == "openai":
        return ""
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

    if provider in {"openai", "cerebras", "groq", "openrouter", "openai_compatible"}:
        if OpenAI is None:
            raise ImportError(
                "Для OpenAI-compatible SmartSync требуется пакет openai."
            )
        base_url = getattr(cfg, "SMART_SYNC_BASE_URL", "").strip() or _default_openai_compatible_base_url(provider)
        if provider != "openai" and not base_url:
            raise EnvironmentError(
                "Для OpenAI-compatible SmartSync установите SMART_SYNC_BASE_URL."
            )
        if provider == "openai" and base_url:
            logger.warning(
                "SmartSync provider=openai, но SMART_SYNC_BASE_URL=%s: запросы пойдут на этот endpoint.",
                base_url,
            )
        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "timeout": cfg.SMART_SYNC_TIMEOUT_SEC,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
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
            client=OpenAI(**client_kwargs),
        )

    logger.info(
        "SmartSync rewrite пропущен: SMART_SYNC_PROVIDER=%s не поддерживается.",
        provider,
    )
    return None


def _extract_timing_rewrite_candidates(raw_text: str, tgt_lang: str = "") -> List[str]:
    """Extracts one or more SmartSync candidates from plain text, JSON, or bullets."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    raw_candidates: List[str] = []
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            for key in ("candidates", "alternatives", "translations"):
                value = parsed.get(key)
                if isinstance(value, list):
                    raw_candidates.extend(str(item) for item in value if str(item).strip())
                    break
            if not raw_candidates:
                for key in ("translation", "text", "result"):
                    value = parsed.get(key)
                    if isinstance(value, str) and value.strip():
                        raw_candidates.append(value)
                        break
        elif isinstance(parsed, list):
            raw_candidates.extend(str(item) for item in parsed if str(item).strip())
        elif isinstance(parsed, str):
            raw_candidates.append(parsed)
    except json.JSONDecodeError:
        pass

    if not raw_candidates:
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        bullet_candidates = []
        for line in lines:
            candidate = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if candidate:
                bullet_candidates.append(candidate)
        raw_candidates = bullet_candidates if len(bullet_candidates) > 1 else [cleaned]

    candidates: List[str] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        normalized = normalize_translated_text(
            _cleanup_causal_translation(candidate, tgt_lang=tgt_lang),
            "",
        )
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            candidates.append(normalized)
    return candidates


def _smart_sync_candidate_tokens(text: str) -> List[str]:
    return re.findall(r"[\wЁёА-Яа-я]+", text.lower(), flags=re.UNICODE)


def _timing_candidate_preflight_metrics(
    *,
    original_text: str,
    candidate_text: str,
    current_duration_sec: float,
    target_duration_sec: float,
    rewrite_mode: str,
) -> Dict[str, Any]:
    original_chars = max(1, len(original_text))
    candidate_chars = len(candidate_text)
    target_char_ratio = target_duration_sec / max(current_duration_sec, 0.001)
    if rewrite_mode == "shorter":
        target_char_ratio = max(0.52, min(0.98, target_char_ratio))
    else:
        target_char_ratio = max(1.0, min(1.28, target_char_ratio))
    char_ratio = candidate_chars / original_chars

    original_tokens = _smart_sync_candidate_tokens(original_text)
    candidate_tokens = _smart_sync_candidate_tokens(candidate_text)
    original_token_set = set(original_tokens)
    candidate_token_set = set(candidate_tokens)
    overlap = len(original_token_set & candidate_token_set)
    token_precision = overlap / max(1, len(candidate_token_set))
    token_recall = overlap / max(1, len(original_token_set))
    word_ratio = len(candidate_tokens) / max(1, len(original_tokens))

    timing_error = abs(char_ratio - target_char_ratio)
    if rewrite_mode == "shorter":
        underfill_penalty = max(0.0, 0.62 - char_ratio) * 2.5
        overfill_penalty = max(0.0, char_ratio - 1.0) * 2.0
    else:
        underfill_penalty = max(0.0, 1.0 - char_ratio) * 2.0
        overfill_penalty = max(0.0, char_ratio - 1.3) * 2.5
    overlap_penalty = max(0.0, 0.72 - token_precision) * 1.6
    word_penalty = max(0.0, 0.66 - word_ratio) * 1.6
    score = timing_error + underfill_penalty + overfill_penalty + overlap_penalty + word_penalty

    return {
        "char_ratio": round(char_ratio, 4),
        "target_char_ratio": round(target_char_ratio, 4),
        "word_ratio": round(word_ratio, 4),
        "token_precision": round(token_precision, 4),
        "token_recall": round(token_recall, 4),
        "score": round(score, 4),
    }


def _select_timing_rewrite_candidate(
    *,
    original_text: str,
    candidates: List[str],
    current_duration_sec: float,
    target_duration_sec: float,
    rewrite_mode: str,
) -> tuple[str | None, Dict[str, Any]]:
    ranked: List[tuple[float, str, Dict[str, Any]]] = []
    for candidate in candidates:
        if _normalize_spaces(candidate).lower() == _normalize_spaces(original_text).lower():
            continue
        metrics = _timing_candidate_preflight_metrics(
            original_text=original_text,
            candidate_text=candidate,
            current_duration_sec=current_duration_sec,
            target_duration_sec=target_duration_sec,
            rewrite_mode=rewrite_mode,
        )
        ranked.append((float(metrics["score"]), candidate, metrics))

    ranked.sort(key=lambda item: item[0])
    if not ranked:
        return None, {"candidate_count": len(candidates), "ranked": []}

    chosen = ranked[0]
    return chosen[1], {
        "candidate_count": len(candidates),
        "chosen_index": candidates.index(chosen[1]),
        "chosen_metrics": chosen[2],
        "ranked": [
            {
                "candidate": candidate,
                "metrics": metrics,
            }
            for _, candidate, metrics in ranked[:5]
        ],
    }


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
    candidate_count: int = 1,
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
            candidate_count=candidate_count,
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
            candidate_count=candidate_count,
            previous_text=_normalize_spaces(previous_text),
            next_text=_normalize_spaces(next_text),
        )
    candidates = _extract_timing_rewrite_candidates(rewritten, tgt_lang=tgt_lang)
    rewritten, preflight_info = _select_timing_rewrite_candidate(
        original_text=translated_text,
        candidates=candidates,
        current_duration_sec=current_duration_sec,
        target_duration_sec=target_duration_sec,
        rewrite_mode=rewrite_mode,
    )
    if not rewritten:
        return None, None
    rewritten = normalize_translated_text(rewritten, source_text)
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
        "candidate_count": len(candidates),
        "preflight": preflight_info,
        "previous_text": _normalize_spaces(previous_text),
        "next_text": _normalize_spaces(next_text),
    }


def load_translation_model(model_name: str, device: str):
    """Загружает seq2seq- или causal-модель перевода через единый интерфейс."""
    provider = _infer_translation_provider(model_name)
    if provider == "gemini":
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

    if provider in {"openai", "openrouter", "groq", "cerebras", "openai_compatible"}:
        if OpenAI is None:
            raise ImportError(
                "Для OpenAI-compatible перевода требуется пакет openai."
            )
        api_key_env = _translation_api_key_env(provider)
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise EnvironmentError(
                "Для OpenAI-compatible перевода установите переменную окружения "
                f"{api_key_env}."
            )
        base_url = os.getenv(
            "MT_OPENAI_BASE_URL",
            getattr(cfg, "MT_OPENAI_BASE_URL", ""),
        ).strip() or _default_openai_compatible_base_url(provider)
        if provider != "openai" and not base_url:
            raise EnvironmentError(
                "Для OpenAI-compatible перевода установите MT_OPENAI_BASE_URL."
            )
        if provider == "openai" and base_url:
            logger.warning(
                "MT provider=openai, но MT_OPENAI_BASE_URL=%s: запросы пойдут на этот endpoint.",
                base_url,
            )
        timeout_sec = int(os.getenv(
            "MT_OPENAI_TIMEOUT_SEC",
            str(getattr(cfg, "MT_OPENAI_TIMEOUT_SEC", cfg.MT_GEMINI_TIMEOUT_SEC)),
        ))
        temperature = float(os.getenv(
            "MT_OPENAI_TEMPERATURE",
            str(getattr(cfg, "MT_OPENAI_TEMPERATURE", cfg.MT_GEMINI_TEMPERATURE)),
        ))
        max_output_tokens = int(os.getenv(
            "MT_OPENAI_MAX_OUTPUT_TOKENS",
            str(getattr(cfg, "MT_OPENAI_MAX_OUTPUT_TOKENS", cfg.MT_GEMINI_MAX_OUTPUT_TOKENS)),
        ))
        min_interval_sec = float(os.getenv(
            "MT_OPENAI_MIN_INTERVAL_SEC",
            str(getattr(cfg, "MT_OPENAI_MIN_INTERVAL_SEC", 0.0)),
        ))
        max_retries = int(os.getenv(
            "MT_OPENAI_MAX_RETRIES",
            str(getattr(cfg, "MT_OPENAI_MAX_RETRIES", cfg.MT_GEMINI_MAX_RETRIES)),
        ))
        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout_sec,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        backend = OpenAICompatibleTranslationBackend(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_sec=timeout_sec,
            min_interval_sec=min_interval_sec,
            max_retries=max_retries,
            backend=provider,
            client=OpenAI(**client_kwargs),
        )
        logger.info(
            "Модель перевода загружена: %s (backend=%s, base_url=%s)",
            model_name,
            backend.backend,
            backend.base_url or "openai-default",
        )
        return backend, None

    if provider not in {"", "hf"}:
        raise ValueError(
            f"Неподдерживаемый MT_PROVIDER={provider!r}. "
            "Используйте hf, gemini, openai, openrouter, groq, cerebras или openai_compatible."
        )

    if torch is None:
        raise ImportError(
            "Для HuggingFace-моделей перевода требуется torch. "
            "Для облачного перевода используйте MT_MODEL_NAME=gemini-* "
            "или MT_PROVIDER=openai/openai_compatible."
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
    max_length: int,
    segment_metadata: List[Dict[str, Any] | None] | None = None,
) -> List[str]:
    """Переводит список текстов prompt-based causal/chat-моделью."""
    results: List[str] = []
    max_new_tokens = max(64, min(max_length, 256))

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_metadata = segment_metadata[i:i + batch_size] if segment_metadata else [None] * len(batch)
        prompts = [
            _build_causal_translation_prompt(text, src_lang, tgt_lang, tokenizer, meta)
            for text, meta in zip(batch, batch_metadata)
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
    max_length: int,
    segment_metadata: List[Dict[str, Any] | None] | None = None,
) -> List[str]:
    """Переводит список текстов батчами через подходящий backend модели."""
    if getattr(model, "backend", None) == "gemini":
        results: List[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_metadata = segment_metadata[i:i + batch_size] if segment_metadata else None
            results.extend(
                _translate_api_chunk_with_fallback(
                    texts=batch,
                    segment_metadata=batch_metadata,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    backend=model,
                    max_length=max_length,
                    batch_fn=_gemini_generate_batch_translations,
                    single_fn=_gemini_generate_translation,
                )
            )
        return results

    if isinstance(model, OpenAICompatibleTranslationBackend):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_metadata = segment_metadata[i:i + batch_size] if segment_metadata else None
            results.extend(
                _translate_api_chunk_with_fallback(
                    texts=batch,
                    segment_metadata=batch_metadata,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    backend=model,
                    max_length=max_length,
                    batch_fn=_openai_generate_batch_translations,
                    single_fn=_openai_generate_translation,
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
        max_length=max_length,
        segment_metadata=segment_metadata,
    )


def _group_segments_into_sentences(
    segments: List[Dict[str, Any]],
    *,
    max_pause_merge: float = 0.5,
) -> List[Dict[str, Any]]:
    valid_segs = [s for s in segments if s.get("text", "").strip()]
    if not valid_segs:
        return []

    sentence_end = re.compile(r"[.!?]\s*$")
    groups: List[Dict[str, Any]] = []
    current_group = [valid_segs[0]]

    for i in range(1, len(valid_segs)):
        prev = valid_segs[i - 1]
        curr = valid_segs[i]
        pause = float(curr.get("start", 0.0)) - float(prev.get("end", 0.0))
        ends_sentence = bool(sentence_end.search(str(prev.get("text", "")).strip()))

        if ends_sentence or pause >= max_pause_merge:
            groups.append({
                "text": " ".join(str(s.get("text", "")).strip() for s in current_group),
                "start": current_group[0].get("start", 0.0),
                "end": current_group[-1].get("end", 0.0),
                "source_segments": current_group.copy(),
            })
            current_group = [curr]
        else:
            current_group.append(curr)

    groups.append({
        "text": " ".join(str(s.get("text", "")).strip() for s in current_group),
        "start": current_group[0].get("start", 0.0),
        "end": current_group[-1].get("end", 0.0),
        "source_segments": current_group.copy(),
    })
    return groups


def _parse_boundary_aware_response(
    text: str,
    *,
    expected_counts: List[int],
) -> List[List[str]]:
    data = _extract_openai_translation_payload(text)
    if not isinstance(data, dict) or not isinstance(data.get("groups"), list):
        raise RuntimeError("Boundary-aware response must be a JSON object with a groups array.")

    groups = data["groups"]
    if len(groups) != len(expected_counts):
        raise RuntimeError(
            f"Boundary-aware response group count is {len(groups)}, expected {len(expected_counts)}."
        )

    result: List[List[str]] = []
    by_id: Dict[int, Any] = {}
    for fallback_idx, group in enumerate(groups):
        if not isinstance(group, dict):
            raise RuntimeError("Boundary-aware response group items must be objects.")
        group_id = group.get("group_id", fallback_idx)
        if not isinstance(group_id, int):
            raise RuntimeError("Boundary-aware group_id must be an integer.")
        by_id[group_id] = group

    for group_id, expected_count in enumerate(expected_counts):
        group = by_id.get(group_id)
        if not isinstance(group, dict):
            raise RuntimeError(f"Boundary-aware response is missing group_id={group_id}.")
        translations = group.get("translations")
        if not isinstance(translations, list):
            raise RuntimeError(f"Boundary-aware group_id={group_id} has no translations array.")
        if len(translations) != expected_count:
            raise RuntimeError(
                f"Boundary-aware group_id={group_id} has {len(translations)} translations, "
                f"expected {expected_count}."
            )

        row: List[str] = []
        for item in translations:
            if not isinstance(item, str):
                raise RuntimeError("Boundary-aware translations must be strings.")
            row.append(_normalize_spaces(item))
        result.append(row)

    return result


def _openai_boundary_aware_translate_groups(
    *,
    backend: OpenAICompatibleTranslationBackend,
    sentence_groups: List[Dict[str, Any]],
    src_lang: str,
    tgt_lang: str,
    max_length: int,
) -> List[List[str]]:
    src_name = _language_name(src_lang)
    tgt_name = _language_name(tgt_lang)
    groups_payload = []
    for group_id, group in enumerate(sentence_groups):
        source_segments = group["source_segments"]
        segment_payload = []
        for idx, segment in enumerate(source_segments):
            segment_payload.append({
                "index": idx,
                "source_text": _normalize_spaces(str(segment.get("text", ""))),
                **_translation_segment_metadata(segment),
            })
        groups_payload.append({
            "group_id": group_id,
            "source_sentence": group["text"],
            "segments": segment_payload,
        })

    system_prompt = (
        "You are a professional translator for a video dubbing pipeline. "
        f"Translate from {src_name} to {tgt_name}. "
        "Each group is one sentence or coherent sentence fragment split into original timed source segments. "
        "Translate the full group with sentence-level context, but return exactly one target-language string "
        "per input segment. Preserve segment boundaries, order, numbers, names, terms, negations, references, "
        "and causal relationships. Fragments are allowed when the source segment is a fragment. "
        "Use compact natural spoken Russian when timing metadata is tight. "
        f"{_translation_style_directives()} "
        f"{_profile_system_directives()} "
        f"{_no_code_switching_rules(tgt_name)} "
        f"{_length_aware_prompt_block()} "
        f"{_terminology_block()} "
        "Return only valid JSON."
    )
    user_prompt = (
        f"{_asr_corrections_block()}"
        f"{_profile_user_directives_block()}"
        "Translate these boundary-aware groups. For each group, return the same group_id and a translations "
        "array with exactly the same number of strings as group.segments. Do not merge or drop segments.\n\n"
        f"{json.dumps({'groups': groups_payload}, ensure_ascii=False, indent=2)}"
    )
    response = _openai_completion_create(
        backend,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max(256, min(max_length * 2, backend.max_output_tokens)),
        response_formats=_openai_response_formats(
            "boundary_aware_translation",
            _BOUNDARY_AWARE_TRANSLATION_SCHEMA,
        ),
    )
    choice = response.choices[0] if getattr(response, "choices", None) else None
    message = getattr(choice, "message", None) if choice is not None else None
    content = getattr(message, "content", "") if message is not None else ""
    text_out = _extract_openai_message_text(content)
    if not text_out:
        raise RuntimeError(f"{backend.backend} boundary-aware translation returned an empty response.")
    return _parse_boundary_aware_response(
        text_out,
        expected_counts=[len(group["source_segments"]) for group in sentence_groups],
    )


def translate_segments_sentence_boundary_aware(
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
    Boundary-aware sentence translation:
    модель видит целое предложение, но сразу возвращает перевод по исходным
    timed-сегментам. Это сохраняет сетку для TTS без отдельного back-projection.
    """
    sentence_groups = _group_segments_into_sentences(
        segments,
        max_pause_merge=max_pause_merge,
    )
    if not sentence_groups:
        logger.error("Нет сегментов для перевода.")
        return []

    if not isinstance(model, OpenAICompatibleTranslationBackend):
        logger.warning(
            "Boundary-aware translation is implemented for OpenAI-compatible backends; "
            "falling back to per-segment."
        )
        return translate_segments(
            model=model,
            tokenizer=tokenizer,
            segments=segments,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            batch_size=batch_size,
            max_length=max_length,
        )

    logger.info(
        "Boundary-aware sentence translation: %s сегментов -> %s предложений",
        sum(len(group["source_segments"]) for group in sentence_groups),
        len(sentence_groups),
    )

    result: List[Dict[str, Any]] = []
    api_batches = 0
    for start_idx in range(0, len(sentence_groups), batch_size):
        batch_groups = sentence_groups[start_idx:start_idx + batch_size]
        projected_batch = _openai_boundary_aware_translate_groups(
            backend=model,
            sentence_groups=batch_groups,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_length=max_length,
        )
        api_batches += 1

        for group, projected in zip(batch_groups, projected_batch):
            source_segments = group["source_segments"]
            for idx, (source_seg, projected_text) in enumerate(zip(source_segments, projected)):
                item = {
                    "text": projected_text,
                    "original_text": _normalize_spaces(str(source_seg.get("text", ""))),
                    "start": source_seg["start"],
                    "end": source_seg["end"],
                    "sentence_boundary_aware": {
                        "group_size": len(source_segments),
                        "group_index": idx,
                        "source_sentence": group["text"],
                    },
                }
                item = _copy_segment_metadata(source_seg, item)
                result.append(item)

    logger.info(
        "Boundary-aware sentence translation завершён: %s сегментов (api_batches=%s)",
        len(result),
        api_batches,
    )
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
    metadata = [_translation_segment_metadata(s) for s in segs]

    logger.info(f"Per-segment: переводим {len(texts)} сегментов...")
    translated_texts = _translate_batch(
        texts, model, tokenizer, src_lang, tgt_lang, batch_size, max_length,
        segment_metadata=metadata,
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


