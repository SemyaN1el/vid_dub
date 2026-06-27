"""Снимки фактической конфигурации запуска (пайплайн, TTS, SmartSync, routing) для воспроизводимости и отчётов."""

from __future__ import annotations

from typing import Any


ConfigSource = Any


def _pick(config: ConfigSource, fields: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for output_name, config_name in fields:
        if hasattr(config, config_name):
            snapshot[output_name] = getattr(config, config_name)
    return snapshot


def build_tts_config_snapshot(config: ConfigSource) -> dict[str, Any]:
    snapshot = {
        "provider": getattr(config, "TTS_PROVIDER", "xtts"),
        "xtts_generation": _pick(
            config,
            (
                ("temperature", "XTTS_TEMPERATURE"),
                ("top_p", "XTTS_TOP_P"),
                ("top_k", "XTTS_TOP_K"),
                ("length_penalty", "XTTS_LENGTH_PENALTY"),
                ("repetition_penalty", "XTTS_REPETITION_PENALTY"),
            ),
        ),
        "elevenlabs": _pick(
            config,
            (
                ("api_key_env", "ELEVENLABS_API_KEY_ENV"),
                ("base_url", "ELEVENLABS_BASE_URL"),
                ("voice_id_configured", "ELEVENLABS_VOICE_ID"),
                ("voice_name", "ELEVENLABS_VOICE_NAME"),
                ("clone_voice", "ELEVENLABS_CLONE_VOICE"),
                ("remove_background_noise", "ELEVENLABS_REMOVE_BACKGROUND_NOISE"),
                ("model_id", "ELEVENLABS_MODEL_ID"),
                ("output_format", "ELEVENLABS_OUTPUT_FORMAT"),
                ("language_code", "ELEVENLABS_LANGUAGE_CODE"),
                ("apply_text_normalization", "ELEVENLABS_APPLY_TEXT_NORMALIZATION"),
                ("enable_logging", "ELEVENLABS_ENABLE_LOGGING"),
                ("timeout_sec", "ELEVENLABS_TIMEOUT_SEC"),
                ("min_interval_sec", "ELEVENLABS_MIN_INTERVAL_SEC"),
                ("max_retries", "ELEVENLABS_MAX_RETRIES"),
                ("stability", "ELEVENLABS_STABILITY"),
                ("similarity_boost", "ELEVENLABS_SIMILARITY_BOOST"),
                ("style", "ELEVENLABS_STYLE"),
                ("use_speaker_boost", "ELEVENLABS_USE_SPEAKER_BOOST"),
                ("speed", "ELEVENLABS_SPEED"),
            ),
        ),
        "runtime": _pick(
            config,
            (
                ("language", "LANGUAGE"),
                ("max_speedup_factor", "MAX_SPEEDUP_FACTOR"),
                ("max_next_start_shift_sec", "MAX_NEXT_START_SHIFT_SEC"),
                ("speedup_tail_padding_ms", "SPEEDUP_TAIL_PADDING_MS"),
                ("speedup_trim_edge_silence", "SPEEDUP_TRIM_EDGE_SILENCE"),
                ("speedup_trim_keep_edge_ms", "SPEEDUP_TRIM_KEEP_EDGE_MS"),
                ("speedup_trim_min_edge_ms", "SPEEDUP_TRIM_MIN_EDGE_MS"),
                ("min_pause_between_segments", "MIN_PAUSE_SEGMENTS"),
                ("fade_in_out_ms", "FADE_IN_OUT_MS"),
                ("crossfade_ms", "CROSSFADE_MS"),
                ("max_shift_left_seconds", "MAX_SHIFT_LEFT_SEC"),
                ("grouping_enabled", "TTS_GROUPING_ENABLED"),
                ("grouping_max_gap_sec", "TTS_GROUPING_MAX_GAP_SEC"),
                ("grouping_max_segments", "TTS_GROUPING_MAX_SEGMENTS"),
                ("grouping_max_chars", "TTS_GROUPING_MAX_CHARS"),
                ("grouping_max_duration_sec", "TTS_GROUPING_MAX_DURATION_SEC"),
            ),
        ),
        "smart_sync": _pick(
            config,
            (
                ("enabled", "SMART_SYNC_ENABLED"),
                ("provider", "SMART_SYNC_PROVIDER"),
                ("model_name", "SMART_SYNC_MODEL_NAME"),
                ("max_rewrites", "SMART_SYNC_MAX_REWRITES"),
                ("candidate_count", "SMART_SYNC_CANDIDATES"),
                ("target_speedup_factor", "SMART_SYNC_TARGET_SPEEDUP_FACTOR"),
                ("trigger_speed_factor", "SMART_SYNC_TRIGGER_SPEED_FACTOR"),
                ("min_fill_ratio", "SMART_SYNC_MIN_FILL_RATIO"),
                ("min_improvement_ms", "SMART_SYNC_MIN_IMPROVEMENT_MS"),
                ("allow_lengthen", "SMART_SYNC_ALLOW_LENGTHEN"),
                ("accept_min_fill_ratio", "SMART_SYNC_ACCEPT_MIN_FILL_RATIO"),
                ("accept_min_text_similarity", "SMART_SYNC_ACCEPT_MIN_TEXT_SIMILARITY"),
                ("accept_min_word_ratio", "SMART_SYNC_ACCEPT_MIN_WORD_RATIO"),
                ("accept_min_token_precision", "SMART_SYNC_ACCEPT_MIN_TOKEN_PRECISION"),
                ("accept_min_asr_score", "SMART_SYNC_ACCEPT_MIN_ASR_SCORE"),
                ("accept_max_asr_drop", "SMART_SYNC_ACCEPT_MAX_ASR_DROP"),
            ),
        ),
        "segment_routing": _pick(
            config,
            (
                ("enabled", "SEGMENT_ROUTING_ENABLED"),
                ("short_segment_sec", "SEGMENT_ROUTING_SHORT_SEC"),
                ("max_refs_per_segment", "SEGMENT_ROUTING_MAX_REFS"),
                ("min_segment_sec", "SEGMENT_ROUTING_MIN_SEC"),
                ("min_segment_words", "SEGMENT_ROUTING_MIN_WORDS"),
                ("confidence_margin", "SEGMENT_ROUTING_CONFIDENCE_MARGIN"),
            ),
        ),
        "tail_guards": _pick(
            config,
            (
                ("cheap_tail_guard_enabled", "ENABLE_TTS_CHEAP_TAIL_GUARD"),
                (
                    "cheap_tail_guard_max_segment_sec",
                    "TTS_CHEAP_TAIL_GUARD_MAX_SEGMENT_SEC",
                ),
                (
                    "cheap_tail_guard_min_overhang_ms",
                    "TTS_CHEAP_TAIL_GUARD_MIN_OVERHANG_MS",
                ),
                ("cheap_tail_guard_max_trim_ms", "TTS_CHEAP_TAIL_GUARD_MAX_TRIM_MS"),
                ("babble_guard_enabled", "ENABLE_TTS_BABBLE_GUARD"),
                ("babble_guard_model_name", "TTS_BABBLE_GUARD_MODEL_NAME"),
                ("babble_guard_device", "TTS_BABBLE_GUARD_DEVICE"),
                ("babble_guard_max_segment_sec", "TTS_BABBLE_GUARD_MAX_SEGMENT_SEC"),
                ("babble_guard_max_trim_ms", "TTS_BABBLE_GUARD_MAX_TRIM_MS"),
                ("asr_retry_enabled", "ENABLE_TTS_ASR_RETRY"),
                ("asr_retry_model_name", "TTS_ASR_RETRY_MODEL_NAME"),
                ("asr_retry_attempts", "TTS_ASR_RETRY_ATTEMPTS"),
                ("asr_retry_min_score", "TTS_ASR_RETRY_MIN_SCORE"),
                ("short_segment_tail_trim_enabled", "SHORT_SEGMENT_TAIL_TRIM_ENABLED"),
                ("short_segment_tail_trim_max_ms", "SHORT_SEGMENT_TAIL_TRIM_MAX_MS"),
            ),
        ),
        "audio_level": _pick(
            config,
            (
                ("target_dbfs", "TARGET_DBFS"),
                ("reference_gain_offset_db", "REFERENCE_GAIN_OFFSET_DB"),
                ("max_segment_boost_db", "MAX_SEGMENT_BOOST_DB"),
                ("max_segment_cut_db", "MAX_SEGMENT_CUT_DB"),
                ("peak_ceiling_dbfs", "PEAK_CEILING_DBFS"),
                ("final_compression_enabled", "ENABLE_FINAL_COMPRESSION"),
                ("segment_matching_enabled", "SEGMENT_MATCHING_ENABLED"),
                ("segment_match_padding_ms", "SEGMENT_MATCH_PADDING_MS"),
                ("segment_match_strength", "SEGMENT_MATCH_STRENGTH"),
                ("segment_match_max_delta_db", "SEGMENT_MATCH_MAX_DELTA_DB"),
                ("segment_match_min_active_ratio", "SEGMENT_MATCH_MIN_ACTIVE_RATIO"),
            ),
        ),
    }
    if snapshot.get("provider") == "elevenlabs":
        snapshot.setdefault("segment_routing", {})["enabled"] = False
    return snapshot


def build_pipeline_config_snapshot(config: ConfigSource) -> dict[str, Any]:
    return {
        "runtime": _pick(
            config,
            (
                ("seed", "SEED"),
                ("device", "DEVICE"),
                ("default_job_name", "DEFAULT_JOB_NAME"),
                ("default_speaker_id", "DEFAULT_SPEAKER_ID"),
            ),
        ),
        "paths": _pick(
            config,
            (
                ("input_path", "INPUT_PATH"),
                ("output_path", "OUTPUT_PATH"),
                ("test_output_path", "TEST_OUTPUT_PATH"),
                ("model_tts_dir", "MODEL_TTS_DIR"),
                ("finetuned_tts_dir", "FINETUNED_TTS_DIR"),
                ("input_video_extensions", "INPUT_VIDEO_EXTENSIONS"),
            ),
        ),
        "asr": _pick(
            config,
            (
                ("provider", "ASR_PROVIDER"),
                ("whisper_model", "WHISPER_MODEL_NAME"),
                ("language", "ASR_LANGUAGE"),
                ("api_model", "ASR_API_MODEL"),
                ("api_key_env", "ASR_API_KEY_ENV"),
                ("timeout_sec", "ASR_TIMEOUT_SEC"),
            ),
        ),
        "translation": _pick(
            config,
            (
                ("provider", "MT_PROVIDER"),
                ("model_name", "MT_MODEL_NAME"),
                ("strategy", "MT_STRATEGY"),
                ("profile", "MT_PROFILE"),
                ("profiles_path", "MT_PROFILES_PATH"),
                ("style", "MT_STYLE"),
                ("batch_size", "MT_BATCH_SIZE"),
                ("max_length", "MT_MAX_LENGTH"),
                ("max_segment_chars", "MT_MAX_SEGMENT_CHARS"),
                ("length_aware_enabled", "MT_LENGTH_AWARE_ENABLED"),
                ("target_chars_per_sec", "MT_TARGET_CHARS_PER_SEC"),
                ("length_grace_chars", "MT_LENGTH_GRACE_CHARS"),
                ("min_target_chars", "MT_MIN_TARGET_CHARS"),
                ("qa_max_char_ratio", "MT_QA_MAX_CHAR_RATIO"),
                ("qa_max_target_ratio", "MT_QA_MAX_TARGET_RATIO"),
                ("gemini_api_key_env", "MT_GEMINI_API_KEY_ENV"),
                ("gemini_temperature", "MT_GEMINI_TEMPERATURE"),
                ("gemini_timeout_sec", "MT_GEMINI_TIMEOUT_SEC"),
                ("openai_api_key_env", "MT_OPENAI_API_KEY_ENV"),
                ("openai_base_url", "MT_OPENAI_BASE_URL"),
                ("openai_response_format", "MT_OPENAI_RESPONSE_FORMAT"),
                ("openai_temperature", "MT_OPENAI_TEMPERATURE"),
                ("openai_timeout_sec", "MT_OPENAI_TIMEOUT_SEC"),
            ),
        ),
        "metrics": _pick(
            config,
            (
                ("asr_provider", "METRICS_ASR_PROVIDER"),
                ("whisper_model", "METRICS_WHISPER_MODEL_NAME"),
                ("asr_api_model", "METRICS_ASR_API_MODEL"),
                ("asr_api_key_env", "METRICS_ASR_API_KEY_ENV"),
            ),
        ),
        "subtitles": _pick(
            config,
            (
                ("mode", "SUBTITLE_MODE"),
                ("use_original", "SUBTITLE_USE_ORIGINAL"),
                ("ass_font", "SUBTITLE_ASS_FONT"),
                ("ass_font_size", "SUBTITLE_ASS_FONT_SIZE"),
            ),
        ),
        "tts": build_tts_config_snapshot(config),
    }
