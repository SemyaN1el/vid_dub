from types import SimpleNamespace

from src.config_snapshot import build_pipeline_config_snapshot, build_tts_config_snapshot


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


def test_build_pipeline_config_snapshot_exposes_runtime_sections_without_secrets() -> None:
    config = SimpleNamespace(
        SEED=42,
        DEVICE="cpu",
        INPUT_PATH="./data/input",
        OUTPUT_PATH="./data/output",
        TEST_OUTPUT_PATH="./data/test",
        MODEL_TTS_DIR="./original_tts_model",
        INPUT_VIDEO_EXTENSIONS=(".mp4", ".mov"),
        ASR_PROVIDER="groq",
        WHISPER_MODEL_NAME="small",
        ASR_API_MODEL="whisper-large-v3",
        ASR_API_KEY_ENV="GROQ_API_KEY",
        MT_MODEL_NAME="gemini-2.5-flash",
        MT_STRATEGY="per-segment",
        MT_GEMINI_API_KEY_ENV="GEMINI_API_KEY",
        METRICS_ASR_PROVIDER="local",
        METRICS_WHISPER_MODEL_NAME="small",
        SUBTITLE_MODE="soft",
        SUBTITLE_USE_ORIGINAL=False,
        SUBTITLE_ASS_FONT="Arial",
        SUBTITLE_ASS_FONT_SIZE=24,
        XTTS_TEMPERATURE=0.55,
        SMART_SYNC_ENABLED=True,
        SEGMENT_MATCHING_ENABLED=False,
    )

    snapshot = build_pipeline_config_snapshot(config)

    assert snapshot["runtime"]["seed"] == 42
    assert snapshot["paths"]["input_video_extensions"] == (".mp4", ".mov")
    assert snapshot["asr"]["api_key_env"] == "GROQ_API_KEY"
    assert "api_key" not in snapshot["asr"]
    assert snapshot["translation"]["model_name"] == "gemini-2.5-flash"
    assert snapshot["subtitles"]["mode"] == "soft"
    assert snapshot["tts"]["xtts_generation"]["temperature"] == 0.55
    assert snapshot["tts"]["smart_sync"]["enabled"] is True
