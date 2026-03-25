import logging
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


def _load_model(model_name: str, device: str):
    """Загружает модель и токенизатор NLLB."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    # Не вызываем .to("cuda") — device_map="auto" уже всё расставил
    model.eval()
    return model, tokenizer


def _translate_batch(
    texts: List[str],
    model,
    tokenizer,
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    max_length: int
) -> List[str]:
    """Переводит список текстов батчами через model.generate."""
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
        result.append({
            "text":            trans_text,
            "original_text":   sent["text"],
            "start":           sent["start"],
            "end":             sent["end"],
            "merged_count":    len(sent["source_segments"])
        })

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
        result.append({
            "text":          translated_texts[i],
            "original_text": seg["text"].strip(),
            "start":         seg["start"],
            "end":           seg["end"]
        })

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

    tokenizer.src_lang = src_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

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

    # Переводим батчами
    translated_contexts = []
    for i in range(0, len(context_texts), batch_size):
        batch = context_texts[i:i + batch_size]
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
        translated_contexts.extend(decoded)

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

        translated.append({
            "text":          " ".join(center_words) if center_words else full_translation,
            "original_text": seg["text"].strip(),
            "start":         seg["start"],
            "end":           seg["end"]
        })

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

            translated.append({
                "text":          " ".join(seg_words),
                "original_text": seg["text"].strip(),
                "start":         seg["start"],
                "end":           seg["end"]
            })

    logger.info(f"Контекстный перевод завершён: {len(translated)} сегментов")
    return translated
