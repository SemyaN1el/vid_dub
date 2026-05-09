from types import SimpleNamespace

from src.config_snapshot import build_tts_config_snapshot


def test_build_tts_config_snapshot_groups_key_tts_settings() -> None:
    config = SimpleNamespace(
        XTTS_TEMPERATURE=0.55,
        XTTS_TOP_P=0.82,
        XTTS_REPETITION_PENALTY=2.35,
        SMART_SYNC_ENABLED=True,
        SMART_SYNC_PROVIDER="groq",
        SMART_SYNC_MAX_REWRITES=1,
        TTS_GROUPING_ENABLED=True,
        SEGMENT_ROUTING_ENABLED=True,
        ENABLE_TTS_BABBLE_GUARD=False,
        ENABLE_TTS_ASR_RETRY=True,
        SEGMENT_MATCHING_ENABLED=False,
        TARGET_DBFS=-15.0,
    )

    snapshot = build_tts_config_snapshot(config)

    assert snapshot["xtts_generation"]["temperature"] == 0.55
    assert snapshot["xtts_generation"]["top_p"] == 0.82
    assert snapshot["xtts_generation"]["repetition_penalty"] == 2.35
    assert snapshot["smart_sync"]["enabled"] is True
    assert snapshot["smart_sync"]["provider"] == "groq"
    assert snapshot["runtime"]["grouping_enabled"] is True
    assert snapshot["segment_routing"]["enabled"] is True
    assert snapshot["tail_guards"]["babble_guard_enabled"] is False
    assert snapshot["tail_guards"]["asr_retry_enabled"] is True
    assert snapshot["audio_level"]["segment_matching_enabled"] is False
    assert snapshot["audio_level"]["target_dbfs"] == -15.0
