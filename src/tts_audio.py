from dataclasses import dataclass
from typing import Dict

from pydub import AudioSegment
from pydub.effects import compress_dynamic_range
from pydub.silence import detect_nonsilent


@dataclass(frozen=True)
class AudioLevelConfig:
    threshold_compression: float = -15.0
    ratio_compression: float = 2.0
    attack_compression: int = 25
    release_compression: int = 50
    target_dbfs: float = -15.0
    reference_gain_offset_db: float = 3.0
    max_segment_boost_db: float = 6.0
    max_segment_cut_db: float = 12.0
    peak_ceiling_dbfs: float = -2.0
    enable_final_compression: bool = False
    enable_segment_matching: bool = False
    segment_match_padding_ms: int = 120
    segment_match_strength: float = 0.7
    segment_match_max_delta_db: float = 4.0
    segment_match_min_active_ratio: float = 0.35


def _active_speech_dbfs(
    audio: AudioSegment,
    min_silence_len: int = 120,
    silence_margin_db: float = 16.0
) -> float | None:
    """Оценивает громкость только активной речи, без длинных пауз."""
    if len(audio) == 0:
        return None

    silence_threshold = max(-50.0, audio.dBFS - silence_margin_db)
    ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_threshold,
        seek_step=5
    )
    if not ranges:
        return audio.dBFS if audio.rms else None

    active_audio = audio[0:0]
    for start_ms, end_ms in ranges:
        if end_ms > start_ms:
            active_audio += audio[start_ms:end_ms]

    return active_audio.dBFS if active_audio.rms else None


def _active_speech_stats(
    audio: AudioSegment,
    min_silence_len: int = 120,
    silence_margin_db: float = 16.0
) -> Dict[str, float | None]:
    """Оценивает локальные параметры речи внутри сегмента."""
    if len(audio) == 0:
        return {
            "active_dbfs": None,
            "active_ratio": 0.0,
            "peak_dbfs": None,
            "full_dbfs": None,
        }

    silence_threshold = max(-50.0, audio.dBFS - silence_margin_db)
    ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_threshold,
        seek_step=5
    )
    if not ranges:
        return {
            "active_dbfs": audio.dBFS if audio.rms else None,
            "active_ratio": 1.0 if audio.rms else 0.0,
            "peak_dbfs": audio.max_dBFS if audio.rms else None,
            "full_dbfs": audio.dBFS if audio.rms else None,
        }

    active_audio = audio[0:0]
    active_ms = 0
    for start_ms, end_ms in ranges:
        if end_ms > start_ms:
            active_audio += audio[start_ms:end_ms]
            active_ms += end_ms - start_ms

    return {
        "active_dbfs": active_audio.dBFS if len(active_audio) and active_audio.rms else None,
        "active_ratio": active_ms / len(audio) if len(audio) else 0.0,
        "peak_dbfs": audio.max_dBFS if audio.rms else None,
        "full_dbfs": audio.dBFS if audio.rms else None,
    }


def _match_segment_level(
    audio: AudioSegment,
    target_active_dbfs: float | None,
    max_boost_db: float,
    max_cut_db: float
) -> AudioSegment:
    """Подгоняет громкость сегмента к целевому уровню активной речи."""
    if target_active_dbfs is None or len(audio) == 0:
        return audio

    current_active_dbfs = _active_speech_dbfs(audio)
    if current_active_dbfs is None:
        return audio

    gain_delta = target_active_dbfs - current_active_dbfs
    gain_delta = max(-max_cut_db, min(max_boost_db, gain_delta))
    return audio.apply_gain(gain_delta)


def _compute_segment_target_level(
    source_audio: AudioSegment | None,
    segment_start_sec: float,
    segment_end_sec: float,
    default_target_active_dbfs: float | None,
    reference_gain_offset_db: float,
    strength: float,
    max_delta_db: float,
    padding_ms: int,
    min_active_ratio: float
) -> tuple[float | None, Dict[str, float | None]]:
    """
    Смещает целевой уровень сегмента к локальному уровню исходного вокала.
    Делает это мягко, чтобы не раскачать громкость на шумных или слабых кусках.
    """
    empty_stats = {
        "active_dbfs": None,
        "active_ratio": 0.0,
        "peak_dbfs": None,
        "full_dbfs": None,
    }
    if source_audio is None:
        return default_target_active_dbfs, empty_stats

    total_ms = len(source_audio)
    start_ms = max(0, int(segment_start_sec * 1000) - padding_ms)
    end_ms = min(total_ms, int(segment_end_sec * 1000) + padding_ms)
    if end_ms <= start_ms:
        return default_target_active_dbfs, empty_stats

    source_segment = source_audio[start_ms:end_ms]
    stats = _active_speech_stats(source_segment)
    source_active_dbfs = stats["active_dbfs"]
    active_ratio = float(stats["active_ratio"] or 0.0)

    if source_active_dbfs is None or active_ratio < min_active_ratio:
        return default_target_active_dbfs, stats

    local_target = source_active_dbfs + reference_gain_offset_db
    if default_target_active_dbfs is None:
        return local_target, stats

    delta = local_target - default_target_active_dbfs
    delta = max(-max_delta_db, min(max_delta_db, delta))
    adjusted_target = default_target_active_dbfs + delta * strength
    return adjusted_target, stats


def _apply_peak_ceiling(audio: AudioSegment, peak_ceiling_dbfs: float) -> AudioSegment:
    """Не даёт итоговому аудио доходить до клиппинга."""
    if len(audio) == 0 or audio.max_dBFS == float("-inf"):
        return audio
    if audio.max_dBFS <= peak_ceiling_dbfs:
        return audio
    return audio.apply_gain(peak_ceiling_dbfs - audio.max_dBFS)


def apply_final_audio_processing(
    audio: AudioSegment,
    config: AudioLevelConfig
) -> AudioSegment:
    """Применяет финальную компрессию и peak ceiling к полной дорожке."""
    if config.enable_final_compression:
        audio = compress_dynamic_range(
            audio,
            threshold=config.threshold_compression,
            ratio=config.ratio_compression,
            attack=config.attack_compression,
            release=config.release_compression
        )
    return _apply_peak_ceiling(audio, config.peak_ceiling_dbfs)
