"""Предобработка аудио: извлечение дорожки (FFmpeg), разделение на речь и фон (HTDemucs) и нормализация вокала с умеренным шумоподавлением."""

import os
import subprocess
import logging
from typing import Tuple

import numpy as np
import noisereduce as nr
from scipy.io.wavfile import read, write
from pydub import AudioSegment

from utils.helpers import create_path, normalize_path
from config import DEVICE, SUFFIX

logger = logging.getLogger(__name__)


def extract_audio_from_video(video_path: str, audio_path: str) -> None:
    """
    Извлекает аудиодорожку из видеофайла с помощью FFmpeg.

    Параметры:
        video_path: путь к исходному видео
        audio_path: путь для сохранения аудио (.wav)
    """
    if os.path.exists(audio_path):
        logger.info(f"Аудио уже существует, пропуск: {audio_path}")
        return

    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    cmd = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", "-y", audio_path]

    logger.info(f"Извлечение аудио: {video_path} → {audio_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FFmpeg STDERR: {result.stderr}")
        raise RuntimeError(f"Не удалось извлечь аудио: {result.stderr}")

    logger.info("Аудио успешно извлечено.")


def separate_audio_sources(
    input_audio_path: str,
    temp_dir: str,
    model_name: str = "htdemucs",
    output_format: str = "wav",
    device: str = DEVICE,
    suffix: str = SUFFIX
) -> Tuple[str, str]:
    """
    Разделяет аудио на вокал и фон с помощью Demucs.

    Параметры:
        input_audio_path: путь к исходному аудио
        temp_dir: директория для временных файлов
        model_name: модель Demucs ('htdemucs', 'htdemucs_ft', 'mdx_extra')
        output_format: формат выходных файлов
        device: устройство ('cuda' или 'cpu')
        suffix: суффикс для именования выходных файлов

    Возвращает:
        Tuple[str, str]: (путь к вокалу, путь к фону drums+bass+other)
    """
    if not os.path.exists(input_audio_path):
        raise FileNotFoundError(f"Входной файл не найден: {input_audio_path}")

    input_audio_path = normalize_path(input_audio_path)
    temp_dir         = normalize_path(temp_dir)
    output_dir       = normalize_path(os.path.dirname(input_audio_path))

    # Строим команду: опции ДО входного файла.
    # --two-stems=vocals даёт фон со всеми не-вокальными стемами;
    # 4-стемный режим терял drums и bass в финальном миксе.
    cmd = ["demucs", "-n", model_name, "--two-stems", "vocals"]
    if device == "cuda":
        cmd.extend(["-d", "cuda"])
    cmd.extend(["--out", output_dir])
    if output_format == "mp3":
        cmd.append("--mp3")
    elif output_format == "flac":
        cmd.append("--flac")
    cmd.append(input_audio_path)

    logger.info(f"Запуск Demucs: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Demucs STDERR:\n{result.stderr}")
        raise RuntimeError(f"Demucs завершился с ошибкой:\n{result.stderr}")

    base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
    stems_dir = normalize_path(os.path.join(output_dir, model_name, base_name))
    logger.info(f"Стемы в: {stems_dir}")

    voice_source      = os.path.join(stems_dir, "vocals.wav")
    background_source = os.path.join(stems_dir, "no_vocals.wav")
    if not os.path.exists(background_source):
        legacy_background = os.path.join(stems_dir, "other.wav")
        if os.path.exists(legacy_background):
            logger.warning(
                "Demucs вернул 4-стемный выход; фон без drums/bass: %s",
                legacy_background
            )
            background_source = legacy_background

    if not os.path.exists(voice_source) or not os.path.exists(background_source):
        raise FileNotFoundError(f"Demucs не создал ожидаемые файлы в {stems_dir}")

    output_voice_path      = os.path.join(temp_dir, f"vocals_{suffix}.wav")
    output_background_path = os.path.join(temp_dir, f"background_{suffix}.wav")

    # Force/resume may rerun preprocessing in the same job directory.  On
    # Windows os.rename() refuses to overwrite the stems left by the previous
    # run, while os.replace() atomically replaces an existing destination.
    os.replace(voice_source, output_voice_path)
    os.replace(background_source, output_background_path)

    logger.info(f"Вокал сохранён: {output_voice_path}")
    logger.info(f"Фон сохранён:   {output_background_path}")

    return output_voice_path, output_background_path


def preprocess_audio_for_asr(
    input_path: str,
    output_path: str,
    target_sample_rate: int = 16000,
    noise_reduce: bool = True,
    gain_increase: float = 0.0,
    prop_decrease: float = 0.3,
    n_fft: int = 512,
    hop_length: int = 256
) -> str:
    """
    Предобрабатывает аудио для ASR: ресемплинг, моно, денойзинг, усиление.

    Параметры:
        input_path: путь к исходному аудио
        output_path: путь для сохранения обработанного аудио
        target_sample_rate: целевая частота дискретизации (16000 для ASR)
        noise_reduce: применять ли шумоподавление
        gain_increase: усиление громкости в дБ
        prop_decrease: степень подавления шума (0.0 – 1.0)
        n_fft: размер FFT для спектрального анализа
        hop_length: шаг между фреймами

    Возвращает:
        str: путь к обработанному файлу
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_wav = input_path + "_tmp.wav"

    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.apply_gain(gain_increase)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(target_sample_rate)
        audio.export(temp_wav, format="wav")

        if noise_reduce:
            rate, data = read(temp_wav)
            reduced = nr.reduce_noise(
                y=data, sr=rate,
                prop_decrease=prop_decrease,
                n_fft=n_fft,
                hop_length=hop_length
            )
            write(output_path, rate, reduced.astype(np.int16))
        else:
            audio.export(output_path, format="wav")

        logger.info(f"Аудио обработано и сохранено: {output_path}")
        return output_path

    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
