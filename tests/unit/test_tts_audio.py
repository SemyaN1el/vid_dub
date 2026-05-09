import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from src.tts_audio import (
    AudioLevelConfig,
    _active_speech_dbfs,
    _apply_peak_ceiling,
    _compute_segment_target_level,
    _match_segment_level,
    apply_final_audio_processing,
)


def _silence(duration_ms: int) -> AudioSegment:
    return AudioSegment.silent(duration=duration_ms, frame_rate=44100)


def _tone(target_dbfs: float, duration_ms: int = 300) -> AudioSegment:
    audio = Sine(440).to_audio_segment(duration=duration_ms)
    return audio.apply_gain(target_dbfs - audio.dBFS)


def test_active_speech_dbfs_ignores_surrounding_silence() -> None:
    tone = _tone(-18.0)
    audio = _silence(300) + tone + _silence(300)

    active_dbfs = _active_speech_dbfs(audio)

    assert active_dbfs is not None
    assert active_dbfs > audio.dBFS + 3.0
    assert active_dbfs == pytest.approx(tone.dBFS, abs=1.0)


def test_match_segment_level_clamps_gain_boost() -> None:
    audio = _tone(-24.0)

    matched = _match_segment_level(
        audio,
        target_active_dbfs=-14.0,
        max_boost_db=4.0,
        max_cut_db=12.0,
    )

    assert matched.dBFS == pytest.approx(audio.dBFS + 4.0, abs=0.3)


def test_compute_segment_target_level_blends_local_source_level() -> None:
    source_audio = _silence(200) + _tone(-20.0, duration_ms=400) + _silence(200)

    target_dbfs, stats = _compute_segment_target_level(
        source_audio=source_audio,
        segment_start_sec=0.2,
        segment_end_sec=0.6,
        default_target_active_dbfs=-15.0,
        reference_gain_offset_db=3.0,
        strength=0.5,
        max_delta_db=4.0,
        padding_ms=0,
        min_active_ratio=0.35,
    )

    assert target_dbfs == pytest.approx(-16.0, abs=0.8)
    assert stats["active_ratio"] == pytest.approx(1.0, abs=0.05)


def test_apply_peak_ceiling_limits_loud_audio() -> None:
    audio = _tone(-3.0)

    limited = _apply_peak_ceiling(audio, peak_ceiling_dbfs=-8.0)

    assert limited.max_dBFS <= -8.0 + 0.01
    assert len(limited) == len(audio)


def test_final_audio_processing_applies_peak_ceiling_without_compression() -> None:
    audio = _tone(-4.0)
    config = AudioLevelConfig(
        enable_final_compression=False,
        peak_ceiling_dbfs=-9.0,
    )

    processed = apply_final_audio_processing(audio, config)

    assert processed.max_dBFS <= -9.0 + 0.01
    assert len(processed) == len(audio)
