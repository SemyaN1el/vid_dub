from types import SimpleNamespace

from src.tts_config import (
    build_audio_level_config,
    build_segment_routing_config,
    build_smart_sync_config,
    build_tail_guard_config,
    build_tts_runtime_config,
)


def test_tts_config_factories_read_project_config_values() -> None:
    config = SimpleNamespace(
        MAX_SPEEDUP_FACTOR=1.08,
        TTS_GROUPING_ENABLED=True,
        TTS_GROUPING_MAX_CHARS=160,
        SMART_SYNC_ENABLED=True,
        SMART_SYNC_MAX_REWRITES=3,
        ENABLE_TTS_ASR_RETRY=True,
        TTS_ASR_RETRY_ATTEMPTS=5,
        SEGMENT_ROUTING_ENABLED=True,
        SEGMENT_ROUTING_MAX_REFS=4,
        TARGET_DBFS=-17.0,
        SEGMENT_MATCHING_ENABLED=True,
    )

    runtime = build_tts_runtime_config(config)
    smart_sync = build_smart_sync_config(config)
    tail_guard = build_tail_guard_config(config)
    routing = build_segment_routing_config(config)
    audio = build_audio_level_config(config)

    assert runtime.max_speedup_factor == 1.08
    assert runtime.enable_grouping is True
    assert runtime.grouping_max_chars == 160
    assert smart_sync.enabled is True
    assert smart_sync.max_rewrites == 3
    assert tail_guard.enable_asr_retry is True
    assert tail_guard.asr_retry_attempts == 5
    assert routing.enabled is True
    assert routing.max_refs_per_segment == 4
    assert audio.target_dbfs == -17.0
    assert audio.enable_segment_matching is True


def test_tts_config_factories_accept_cli_overrides() -> None:
    config = SimpleNamespace(
        TTS_GROUPING_ENABLED=True,
        SMART_SYNC_ENABLED=False,
        ENABLE_TTS_BABBLE_GUARD=False,
        ENABLE_TTS_ASR_RETRY=False,
        SEGMENT_ROUTING_ENABLED=True,
    )

    runtime = build_tts_runtime_config(config, enable_grouping=False)
    smart_sync = build_smart_sync_config(config, enabled=True)
    tail_guard = build_tail_guard_config(
        config,
        enable_babble_guard=True,
        enable_asr_retry=True,
    )
    routing = build_segment_routing_config(config, enabled=False)

    assert runtime.enable_grouping is False
    assert smart_sync.enabled is True
    assert tail_guard.enable_babble_guard is True
    assert tail_guard.enable_asr_retry is True
    assert routing.enabled is False
