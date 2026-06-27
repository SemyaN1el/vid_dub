from pydub import AudioSegment
from pydub.generators import Sine

from src.tts import (
    AudioLevelConfig,
    SegmentRoutingConfig,
    SmartSyncConfig,
    TailGuardConfig,
    TTSRuntimeConfig,
    _assemble_timeline,
    _fit_audio_to_timing_window,
    _serialize_tts_segments,
)


def test_serialize_tts_segments_drops_runtime_audio_objects() -> None:
    segments = _serialize_tts_segments([
        {
            "text": "Привет.",
            "corrected_duration_sec": 1.2,
            "corrected_audio": AudioSegment.silent(duration=100),
        }
    ])

    assert segments == [{"text": "Привет.", "corrected_duration_sec": 1.2}]


def test_assemble_timeline_preserves_overlapped_tail() -> None:
    tone = Sine(440).to_audio_segment(duration=500).apply_gain(-12)
    silence = AudioSegment.silent(duration=500, frame_rate=tone.frame_rate)
    left_only = AudioSegment.from_mono_audiosegments(tone, silence)
    right_only = AudioSegment.from_mono_audiosegments(silence, tone)

    timeline = _assemble_timeline(
        placements=[(0, left_only), (250, right_only)],
        replacements=[],
        min_duration_ms=1000,
    )

    overlap_left, overlap_right = timeline[300:450].split_to_mono()

    assert len(timeline) >= 1000
    assert overlap_left.rms > 0
    assert overlap_right.rms > 0


def test_assemble_timeline_places_segment_at_position() -> None:
    tone = Sine(440).to_audio_segment(duration=300).apply_gain(-12)

    timeline = _assemble_timeline(
        placements=[(500, tone)],
        replacements=[],
        min_duration_ms=1000,
    )

    assert timeline[:450].rms == 0
    assert timeline[520:750].rms > 0
    assert timeline[850:].rms == 0


def test_assemble_timeline_replacement_overwrites_region() -> None:
    tone = Sine(440).to_audio_segment(duration=400).apply_gain(-12)
    quiet = AudioSegment.silent(duration=400, frame_rate=tone.frame_rate)

    timeline = _assemble_timeline(
        placements=[(100, tone)],
        replacements=[(100, quiet)],
        min_duration_ms=800,
    )

    assert timeline[120:480].rms == 0
    assert len(timeline) == 800


def test_assemble_timeline_without_parts_returns_silence() -> None:
    timeline = _assemble_timeline(
        placements=[],
        replacements=[],
        min_duration_ms=600,
    )

    assert len(timeline) == 600
    assert timeline.rms == 0


def test_fit_audio_to_timing_window_trims_surplus_edge_silence() -> None:
    tone = Sine(440).to_audio_segment(duration=500).apply_gain(-12)
    audio = AudioSegment.silent(duration=300) + tone + AudioSegment.silent(duration=350)

    fitted, info = _fit_audio_to_timing_window(
        audio,
        available_ms=2000,
        max_speedup_factor=1.35,
        speedup_tail_padding_ms=120,
        trim_edge_silence=True,
        trim_keep_edge_ms=90,
        trim_min_edge_ms=180,
    )

    assert len(fitted) < len(audio)
    assert info["pre_trim_leading_ms"] > 0
    assert info["pre_trim_trailing_ms"] > 0
    assert info["applied_speedup_factor"] == 1.0
    assert info["speedup_capped"] is False


def test_tts_config_objects_group_runtime_options() -> None:
    runtime = TTSRuntimeConfig(
        max_speedup_factor=1.35,
        speedup_trim_keep_edge_ms=80,
        grouping_max_chars=180,
    )
    smart_sync = SmartSyncConfig(enabled=True, max_rewrites=2, target_speedup_factor=1.25)
    tail_guard = TailGuardConfig(enable_asr_retry=True, asr_retry_attempts=3)
    routing = SegmentRoutingConfig(enabled=True, max_refs_per_segment=3)
    audio = AudioLevelConfig(target_dbfs=-16.0, enable_segment_matching=True)

    assert runtime.max_speedup_factor == 1.35
    assert runtime.speedup_trim_keep_edge_ms == 80
    assert runtime.grouping_max_chars == 180
    assert smart_sync.enabled is True
    assert smart_sync.max_rewrites == 2
    assert smart_sync.target_speedup_factor == 1.25
    assert tail_guard.enable_asr_retry is True
    assert tail_guard.asr_retry_attempts == 3
    assert routing.enabled is True
    assert routing.max_refs_per_segment == 3
    assert audio.target_dbfs == -16.0
    assert audio.enable_segment_matching is True
