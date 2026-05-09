from pydub import AudioSegment

from src.tts import (
    AudioLevelConfig,
    SegmentRoutingConfig,
    SmartSyncConfig,
    TailGuardConfig,
    TTSRuntimeConfig,
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


def test_tts_config_objects_group_runtime_options() -> None:
    runtime = TTSRuntimeConfig(max_speedup_factor=1.15, grouping_max_chars=180)
    smart_sync = SmartSyncConfig(enabled=True, max_rewrites=2)
    tail_guard = TailGuardConfig(enable_asr_retry=True, asr_retry_attempts=3)
    routing = SegmentRoutingConfig(enabled=True, max_refs_per_segment=3)
    audio = AudioLevelConfig(target_dbfs=-16.0, enable_segment_matching=True)

    assert runtime.max_speedup_factor == 1.15
    assert runtime.grouping_max_chars == 180
    assert smart_sync.enabled is True
    assert smart_sync.max_rewrites == 2
    assert tail_guard.enable_asr_retry is True
    assert tail_guard.asr_retry_attempts == 3
    assert routing.enabled is True
    assert routing.max_refs_per_segment == 3
    assert audio.target_dbfs == -16.0
    assert audio.enable_segment_matching is True
