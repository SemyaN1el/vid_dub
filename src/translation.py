import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List
from urllib import error, request

try:
    import torch
except ImportError:
    torch = None

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
            item["start"] = round(chunk_start, 3)
            item["end"] = round(chunk_end, 3)
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
        if seg.get("speaker_id"):
            item["speaker_id"] = seg["speaker_id"]
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
        if seg.get("speaker_id"):
            item["speaker_id"] = seg["speaker_id"]
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
            if seg.get("speaker_id"):
                item["speaker_id"] = seg["speaker_id"]
            translated.append(item)

    logger.info(f"Контекстный перевод завершён: {len(translated)} сегментов")
    return translated
