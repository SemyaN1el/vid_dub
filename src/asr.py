import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class WordSegment:
    text:  str
    start: float
    end:   float


def transcribe_and_segment(
    model_asr,
    audio_path: str,
    max_pause_between_sentences: float = 0.3,
    max_audio_length_for_ref: float = 15.0,
    output_ref_path: str = "speaker_ref.wav"
) -> List[Dict[str, Any]]:
    """
    Транскрибирует аудио и разбивает на сегменты по паузам.

    Параметры:
        model_asr: загруженная модель Whisper
        audio_path: путь к WAV-файлу
        max_pause_between_sentences: порог паузы для разделения сегментов (сек)
        max_audio_length_for_ref: максимальная длина референсного сегмента (сек)
        output_ref_path: путь для сохранения референсного аудио спикера

    Возвращает:
        List[Dict]: список сегментов [{text, start, end}, ...]
    """
    audio_data, sample_rate = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    result = model_asr.transcribe(audio_path, word_timestamps=True, task="transcribe")

    # Собираем все слова с временными метками
    words: List[WordSegment] = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words.append(WordSegment(text=w["word"], start=w["start"], end=w["end"]))

    if not words:
        logger.error("Слова с временными метками не найдены.")
        return []

    # Разбиваем на сегменты по паузам
    segments = []
    current = {"text": words[0].text, "start": words[0].start, "end": words[0].end}

    for i in range(1, len(words)):
        gap = words[i].start - words[i - 1].end
        if gap > max_pause_between_sentences:
            current["text"] = current["text"].strip()
            segments.append(current)
            current = {"text": words[i].text, "start": words[i].start, "end": words[i].end}
        else:
            current["text"] += " " + words[i].text
            current["end"]   = words[i].end

    current["text"] = current["text"].strip()
    segments.append(current)
    logger.info(f"Сегментов получено: {len(segments)}")

    # Сохраняем референсный сегмент (самый длинный в пределах лимита)
    valid = [s for s in segments if (s["end"] - s["start"]) <= max_audio_length_for_ref]
    if valid:
        best = max(valid, key=lambda s: s["end"] - s["start"])
        start_s = int(best["start"] * sample_rate)
        end_s   = int(best["end"]   * sample_rate)
        sf.write(output_ref_path, audio_data[start_s:end_s], sample_rate)
        logger.info(f"Референс сохранён: {output_ref_path} ({best['end'] - best['start']:.2f} сек)")
    else:
        logger.warning("Подходящий сегмент для референса не найден.")

    return segments
