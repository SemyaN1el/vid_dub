import os
import re
import logging
from copy import deepcopy
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Tuple

import soundfile as sf
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range, speedup
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    """Удаляет спецсимволы и эмодзи перед подачей в TTS."""
    text = text.lower()
    text = re.sub(r"[^\w\s]",            "", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    text = re.sub(r"\s+",                " ", text).strip()
    return text


def generate_audio_segment(
    model_tts,
    text: str,
    output_path: str,
    speaker_wav: str,
    language: str,
    gpt_cond_latent=None,
    speaker_embedding=None
) -> Tuple[str, float]:
    """
    Синтезирует аудио для одного текстового сегмента.

    Параметры:
        model_tts: загруженная модель XTTS-v2
        text: текст для синтеза
        output_path: путь для сохранения сегмента
        speaker_wav: путь к референсному аудио спикера
        language: язык синтеза ('ru', 'en', ...)
        gpt_cond_latent: предвычисленный латент (опционально, для ускорения)
        speaker_embedding: предвычисленный эмбеддинг (опционально)

    Возвращает:
        Tuple[str, float]: (путь к файлу, длительность в секундах)
    """
    if gpt_cond_latent is None or speaker_embedding is None:
        gpt_cond_latent, speaker_embedding = model_tts.get_conditioning_latents(
            audio_path=speaker_wav
        )

    with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    output = model_tts.inference(
        text=text,
        language=language,
        speaker_embedding=speaker_embedding,
        gpt_cond_latent=gpt_cond_latent
    )
    sf.write(tmp_path, output["wav"], 24000)

    audio = AudioSegment.from_wav(tmp_path)
    audio.export(output_path, format="wav")
    os.remove(tmp_path)

    duration = len(audio) / 1000.0
    return output_path, duration


def synthesize_segments_with_timing(
    model_tts,
    segments: List[Dict[str, Any]],
    output_audio_path: str,
    speaker_wav: str,
    language: str,
    segments_dir: str = "./data/output/temp/audio_segments",
    max_speedup_factor: float = 1.2,
    min_pause_between_segments: float = 0.2,
    fade_in_out_ms: int = 50,
    crossfade_ms: int = 30,
    max_shift_left_seconds: float = 0.5,
    threshold_compression: float = -15.0,
    ratio_compression: float = 2.0,
    attack_compression: int = 25,
    release_compression: int = 50,
    target_dBFS: float = -15.0
) -> None:
    """
    Синтезирует дубляж с временно́й синхронизацией сегментов.

    Алгоритм:
        1. Предвычисляет conditioning latents один раз для всего пайплайна.
        2. Для каждого сегмента генерирует аудио и проверяет вписывается ли
           оно в отведённое время с учётом паузы до следующего сегмента.
        3. При необходимости ускоряет сегмент (не более max_speedup_factor).
        4. Применяет fade-in/out и кроссфейды между соседними сегментами.
        5. Нормализует итоговое аудио.
    """
    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

    # Предвычисляем латенты один раз
    logger.info("Вычисляем conditioning latents...")
    gpt_cond_latent, speaker_embedding = model_tts.get_conditioning_latents(
        audio_path=speaker_wav
    )

    full_duration_ms = int((max(s["end"] for s in segments) + 5) * 1000)
    full_audio = AudioSegment.silent(duration=full_duration_ms)

    # Сохраняем оригинальные временные метки
    for seg in segments:
        if "original_start" not in seg:
            seg["original_start"] = seg.get("start", 0.0)

    prev_end_sec = 0.0

    for i, seg in tqdm(enumerate(segments), total=len(segments), desc="Синтез"):
        orig_start  = seg["original_start"]
        cur_start   = seg["start"]
        cur_start_ms = int(cur_start * 1000)

        # Сдвиг первого сегмента влево при наличии свободного времени
        if i == 0:
            avail_shift = min(max(0.0, orig_start - prev_end_sec), max_shift_left_seconds)
            if avail_shift > 0.05:
                seg["start"] -= avail_shift
                seg["start"]  = max(orig_start - max_shift_left_seconds, seg["start"])
                cur_start     = seg["start"]
                cur_start_ms  = int(cur_start * 1000)

        # Защита от слишком раннего сдвига
        if cur_start - orig_start < -max_shift_left_seconds:
            seg["start"] = orig_start
            cur_start    = orig_start
            cur_start_ms = int(orig_start * 1000)

        # Очистка текста
        clean = _clean_text(seg["text"])
        seg["cleaned_text"] = clean
        if not clean:
            logger.warning(f"[{i}] Пустой сегмент после очистки — пропуск.")
            continue

        # Генерация аудио
        seg_path = os.path.join(segments_dir, f"seg_{int(seg['start'] * 1000)}.wav")
        seg_path, _ = generate_audio_segment(
            model_tts=model_tts,
            text=clean,
            output_path=seg_path,
            speaker_wav=speaker_wav,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding
        )

        seg_audio       = AudioSegment.from_wav(seg_path)
        generated_ms    = len(seg_audio)

        orig_start_val = seg["start"]
        orig_end_val   = seg.get("end", seg["start"] + 1.0)
        if orig_start_val > orig_end_val:
            orig_start_val, orig_end_val = orig_end_val, orig_start_val

        original_ms = int((orig_end_val - orig_start_val) * 1000)

        # Доступное время с учётом паузы до следующего сегмента
        if i < len(segments) - 1:
            next_start_ms   = int(segments[i + 1]["start"] * 1000)
            before_next_ms  = max(0, next_start_ms - int(orig_end_val * 1000))
        else:
            before_next_ms = 0

        extra_ms      = max(0, before_next_ms - int(min_pause_between_segments * 1000))
        available_ms  = max(100, original_ms + extra_ms)

        # Ускорение при необходимости
        corrected = seg_audio
        if generated_ms > available_ms > 0:
            factor = min(generated_ms / available_ms, max_speedup_factor)
            corrected = speedup(seg_audio, playback_speed=factor)
            logger.info(f"[{i}] Ускорение {factor:.2f}x")

        corrected_ms = len(corrected)
        corrected    = corrected.fade_in(fade_in_out_ms).fade_out(fade_in_out_ms)
        seg["corrected_audio"] = corrected

        # Вставляем в итоговое аудио
        full_audio = (
            full_audio[:cur_start_ms]
            + corrected
            + full_audio[cur_start_ms + corrected_ms:]
        )

        actual_end = cur_start + corrected_ms / 1000.0
        seg["corrected_duration_sec"] = corrected_ms / 1000.0
        prev_end_sec = actual_end

        # Обновляем начало следующего сегмента (только вправо)
        if i < len(segments) - 1:
            nxt = segments[i + 1]
            nxt.setdefault("original_start", nxt["start"])
            nxt["start"] = max(actual_end, nxt["start"])

    # Кроссфейды между соседними сегментами
    for i in range(len(segments) - 1):
        curr, nxt = segments[i], segments[i + 1]
        if "corrected_audio" not in curr or "corrected_audio" not in nxt:
            continue
        curr_ms   = int(curr["start"] * 1000)
        nxt_ms    = int(nxt["start"]  * 1000)
        overlap   = curr_ms + len(curr["corrected_audio"]) - nxt_ms
        if 0 < overlap < crossfade_ms:
            merged = curr["corrected_audio"].append(nxt["corrected_audio"], crossfade=overlap)
            full_audio = full_audio[:curr_ms] + merged + full_audio[curr_ms + len(merged):]

    # Финальная нормализация
    full_audio = compress_dynamic_range(
        full_audio,
        threshold=threshold_compression,
        ratio=ratio_compression,
        attack=attack_compression,
        release=release_compression
    )
    gain = target_dBFS - full_audio.dBFS
    full_audio = full_audio.apply_gain(gain)

    full_audio.export(output_audio_path, format="wav")
    logger.info(f"Финальное аудио сохранено: {output_audio_path}")
