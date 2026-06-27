"""Анализ активной речи: выделение речевых участков и измерение их уровня (dBFS) для нормализации громкости и метрик."""

import math
from typing import Dict, Iterable, List

import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def _dbfs_from_square_sum(
    square_sum: float,
    sample_count: int,
    max_possible_amplitude: float,
) -> float | None:
    if sample_count <= 0 or square_sum <= 0.0 or max_possible_amplitude <= 0:
        return None

    rms = math.sqrt(square_sum / sample_count)
    if rms <= 0.0:
        return None
    return 20.0 * math.log10(rms / max_possible_amplitude)


def _ranges_dbfs(audio: AudioSegment, ranges: Iterable[List[int]]) -> float | None:
    if len(audio) == 0 or audio.rms == 0:
        return None

    samples = np.asarray(audio.get_array_of_samples(), dtype=np.float64)
    if samples.size == 0:
        return None

    channels = max(1, int(audio.channels or 1))
    frame_count = samples.size // channels
    square_sum = 0.0
    sample_count = 0

    for start_ms, end_ms in ranges:
        if end_ms <= start_ms:
            continue

        start_frame = max(0, min(frame_count, int(start_ms * audio.frame_rate / 1000)))
        end_frame = max(0, min(frame_count, int(end_ms * audio.frame_rate / 1000)))
        start_idx = start_frame * channels
        end_idx = end_frame * channels
        if end_idx <= start_idx:
            continue

        chunk = samples[start_idx:end_idx]
        square_sum += float(np.dot(chunk, chunk))
        sample_count += int(chunk.size)

    return _dbfs_from_square_sum(
        square_sum=square_sum,
        sample_count=sample_count,
        max_possible_amplitude=float(audio.max_possible_amplitude),
    )


def active_speech_ranges(
    audio: AudioSegment,
    min_silence_len: int = 120,
    silence_margin_db: float = 16.0,
    seek_step: int = 5,
) -> List[List[int]]:
    if len(audio) == 0:
        return []

    silence_threshold = max(-50.0, audio.dBFS - silence_margin_db)
    return detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_threshold,
        seek_step=seek_step,
    )


def active_speech_dbfs(
    audio: AudioSegment,
    min_silence_len: int = 120,
    silence_margin_db: float = 16.0,
    seek_step: int = 5,
) -> float | None:
    if len(audio) == 0:
        return None

    ranges = active_speech_ranges(
        audio,
        min_silence_len=min_silence_len,
        silence_margin_db=silence_margin_db,
        seek_step=seek_step,
    )
    if not ranges:
        return audio.dBFS if audio.rms else None

    return _ranges_dbfs(audio, ranges)


def active_speech_stats(
    audio: AudioSegment,
    min_silence_len: int = 120,
    silence_margin_db: float = 16.0,
    seek_step: int = 5,
    fallback_active_ratio: float | None = None,
) -> Dict[str, float | None]:
    if len(audio) == 0:
        return {
            "active_dbfs": None,
            "active_ratio": 0.0,
            "peak_dbfs": None,
            "full_dbfs": None,
        }

    ranges = active_speech_ranges(
        audio,
        min_silence_len=min_silence_len,
        silence_margin_db=silence_margin_db,
        seek_step=seek_step,
    )
    if not ranges:
        active_ratio = fallback_active_ratio if fallback_active_ratio is not None else 1.0
        return {
            "active_dbfs": audio.dBFS if audio.rms else None,
            "active_ratio": active_ratio if audio.rms else 0.0,
            "peak_dbfs": audio.max_dBFS if audio.rms else None,
            "full_dbfs": audio.dBFS if audio.rms else None,
        }

    active_ms = sum(
        max(0, int(end_ms) - int(start_ms))
        for start_ms, end_ms in ranges
        if end_ms > start_ms
    )
    active_dbfs = _ranges_dbfs(audio, ranges)

    return {
        "active_dbfs": active_dbfs,
        "active_ratio": active_ms / len(audio) if len(audio) else 0.0,
        "peak_dbfs": audio.max_dBFS if audio.rms else None,
        "full_dbfs": audio.dBFS if audio.rms else None,
    }
