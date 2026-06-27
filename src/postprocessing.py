"""Постобработка: микширование синтезированного голоса с фоном и приглушённым оригиналом, сборка финального видео средствами FFmpeg."""

import os
import subprocess
import logging
from typing import Optional

from pydub import AudioSegment

logger = logging.getLogger(__name__)


def mix_audio_tracks(
    voice_over_path: str,
    background_path: str,
    output_path: str,
    original_audio_path: Optional[str] = None,
    voice_gain: float = -3.0,
    background_gain: float = -5.0,
    original_gain: float = -10.0
) -> str:
    """
    Микширует дубляж, фоновую музыку и (опционально) оригинальный звук.

    Параметры:
        voice_over_path: сгенерированная речь на целевом языке
        background_path: фоновая дорожка (music/SFX от Demucs)
        output_path: путь для сохранения результата
        original_audio_path: оригинальный звук (опционально, как тихий фон)
        voice_gain: усиление голоса (дБ)
        background_gain: усиление фона (дБ)
        original_gain: усиление оригинала (дБ)

    Возвращает:
        str: путь к смикшированному файлу
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    voice_audio = AudioSegment.from_wav(voice_over_path).apply_gain(voice_gain)
    bg_audio    = AudioSegment.from_wav(background_path).apply_gain(background_gain)

    # Зацикливаем фон до длины голосовой дорожки
    max_dur = len(voice_audio)
    if len(bg_audio) == 0:
        logger.warning("Фоновая дорожка пустая, микширую дубляж поверх тишины.")
        bg_audio = AudioSegment.silent(duration=max_dur, frame_rate=voice_audio.frame_rate)
    else:
        bg_audio = (bg_audio * (max_dur // len(bg_audio) + 2))[:max_dur]

    mixed = bg_audio.overlay(voice_audio)

    if original_audio_path and os.path.exists(original_audio_path):
        orig = AudioSegment.from_wav(original_audio_path).apply_gain(original_gain)
        mixed = mixed.overlay(orig)

    mixed.export(output_path, format="wav")
    logger.info(f"Микс сохранён: {output_path}")
    return output_path


def add_audio_to_video(
    video_path: str,
    audio_path: str,
    output_video_path: str
) -> None:
    """
    Интегрирует аудиодорожку в видео с помощью FFmpeg.

    Параметры:
        video_path: путь к исходному видео
        audio_path: путь к новой аудиодорожке
        output_video_path: путь для сохранения финального видео
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видео не найдено: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Аудио не найдено: {audio_path}")

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-y",
        output_video_path
    ]

    logger.info(f"Сборка видео: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FFmpeg STDERR: {result.stderr}")
        raise RuntimeError(f"Не удалось собрать видео: {result.stderr}")

    logger.info(f"Финальное видео сохранено: {output_video_path}")
