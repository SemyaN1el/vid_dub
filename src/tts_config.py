"""Сборка конфигураций TTS-этапа из настроек проекта: runtime, SmartSync, routing сегментов, tail-guards и аудио-уровни."""

from __future__ import annotations

from typing import Any

from src.tts import (
    SegmentRoutingConfig,
    SmartSyncConfig,
    TailGuardConfig,
    TTSRuntimeConfig,
)
from src.tts_audio import AudioLevelConfig


def _value(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def build_tts_runtime_config(
    config: Any,
    *,
    enable_grouping: bool | None = None,
) -> TTSRuntimeConfig:
    return TTSRuntimeConfig(
        max_speedup_factor=_value(config, "MAX_SPEEDUP_FACTOR", 1.35),
        max_next_start_shift_sec=_value(config, "MAX_NEXT_START_SHIFT_SEC", 0.25),
        speedup_tail_padding_ms=_value(config, "SPEEDUP_TAIL_PADDING_MS", 120),
        speedup_trim_edge_silence=_value(config, "SPEEDUP_TRIM_EDGE_SILENCE", True),
        speedup_trim_keep_edge_ms=_value(config, "SPEEDUP_TRIM_KEEP_EDGE_MS", 90),
        speedup_trim_min_edge_ms=_value(config, "SPEEDUP_TRIM_MIN_EDGE_MS", 180),
        min_pause_between_segments=_value(config, "MIN_PAUSE_SEGMENTS", 0.2),
        fade_in_out_ms=_value(config, "FADE_IN_OUT_MS", 50),
        crossfade_ms=_value(config, "CROSSFADE_MS", 30),
        max_shift_left_seconds=_value(config, "MAX_SHIFT_LEFT_SEC", 0.5),
        enable_grouping=(
            _value(config, "TTS_GROUPING_ENABLED", True)
            if enable_grouping is None
            else enable_grouping
        ),
        grouping_max_gap_sec=_value(config, "TTS_GROUPING_MAX_GAP_SEC", 0.6),
        grouping_max_segments=_value(config, "TTS_GROUPING_MAX_SEGMENTS", 2),
        grouping_max_chars=_value(config, "TTS_GROUPING_MAX_CHARS", 180),
        grouping_max_duration_sec=_value(config, "TTS_GROUPING_MAX_DURATION_SEC", 8.5),
    )


def build_smart_sync_config(
    config: Any,
    *,
    enabled: bool | None = None,
) -> SmartSyncConfig:
    return SmartSyncConfig(
        enabled=_value(config, "SMART_SYNC_ENABLED", False) if enabled is None else enabled,
        device=_value(config, "DEVICE", "cpu"),
        src_lang=_value(config, "MT_SRC_LANG", "eng_Latn"),
        tgt_lang=_value(config, "MT_TGT_LANG", "rus_Cyrl"),
        max_rewrites=_value(config, "SMART_SYNC_MAX_REWRITES", 1),
        candidate_count=_value(config, "SMART_SYNC_CANDIDATES", 3),
        target_speedup_factor=_value(config, "SMART_SYNC_TARGET_SPEEDUP_FACTOR", 1.25),
        trigger_speed_factor=_value(config, "SMART_SYNC_TRIGGER_SPEED_FACTOR", 1.08),
        min_fill_ratio=_value(config, "SMART_SYNC_MIN_FILL_RATIO", 0.82),
        min_improvement_ms=_value(config, "SMART_SYNC_MIN_IMPROVEMENT_MS", 180),
        allow_lengthen=_value(config, "SMART_SYNC_ALLOW_LENGTHEN", True),
        accept_min_fill_ratio=_value(config, "SMART_SYNC_ACCEPT_MIN_FILL_RATIO", 0.8),
        accept_min_text_similarity=_value(config, "SMART_SYNC_ACCEPT_MIN_TEXT_SIMILARITY", 0.55),
        accept_min_word_ratio=_value(config, "SMART_SYNC_ACCEPT_MIN_WORD_RATIO", 0.66),
        accept_min_token_precision=_value(config, "SMART_SYNC_ACCEPT_MIN_TOKEN_PRECISION", 0.72),
        accept_min_asr_score=_value(config, "SMART_SYNC_ACCEPT_MIN_ASR_SCORE", 0.9),
        accept_max_asr_drop=_value(config, "SMART_SYNC_ACCEPT_MAX_ASR_DROP", 0.03),
    )


def build_tail_guard_config(
    config: Any,
    *,
    enable_babble_guard: bool | None = None,
    enable_asr_retry: bool | None = None,
) -> TailGuardConfig:
    return TailGuardConfig(
        enable_cheap_tail_guard=_value(config, "ENABLE_TTS_CHEAP_TAIL_GUARD", True),
        cheap_tail_guard_max_segment_sec=_value(config, "TTS_CHEAP_TAIL_GUARD_MAX_SEGMENT_SEC", 3.2),
        cheap_tail_guard_min_overhang_ms=_value(config, "TTS_CHEAP_TAIL_GUARD_MIN_OVERHANG_MS", 180),
        cheap_tail_guard_min_gap_ms=_value(config, "TTS_CHEAP_TAIL_GUARD_MIN_GAP_MS", 80),
        cheap_tail_guard_min_island_ms=_value(config, "TTS_CHEAP_TAIL_GUARD_MIN_ISLAND_MS", 80),
        cheap_tail_guard_max_island_ms=_value(config, "TTS_CHEAP_TAIL_GUARD_MAX_ISLAND_MS", 450),
        cheap_tail_guard_search_window_ms=_value(config, "TTS_CHEAP_TAIL_GUARD_SEARCH_WINDOW_MS", 900),
        cheap_tail_guard_max_trim_ms=_value(config, "TTS_CHEAP_TAIL_GUARD_MAX_TRIM_MS", 700),
        enable_babble_guard=(
            _value(config, "ENABLE_TTS_BABBLE_GUARD", False)
            if enable_babble_guard is None
            else enable_babble_guard
        ),
        babble_guard_model_name=_value(config, "TTS_BABBLE_GUARD_MODEL_NAME", "base"),
        babble_guard_device=_value(config, "TTS_BABBLE_GUARD_DEVICE", "cpu"),
        babble_guard_max_segment_sec=_value(config, "TTS_BABBLE_GUARD_MAX_SEGMENT_SEC", 4.0),
        babble_guard_min_gap_ms=_value(config, "TTS_BABBLE_GUARD_MIN_GAP_MS", 80),
        babble_guard_min_island_ms=_value(config, "TTS_BABBLE_GUARD_MIN_ISLAND_MS", 80),
        babble_guard_max_island_ms=_value(config, "TTS_BABBLE_GUARD_MAX_ISLAND_MS", 450),
        babble_guard_search_window_ms=_value(config, "TTS_BABBLE_GUARD_SEARCH_WINDOW_MS", 900),
        babble_guard_max_trim_ms=_value(config, "TTS_BABBLE_GUARD_MAX_TRIM_MS", 700),
        babble_guard_anchor_words=_value(config, "TTS_BABBLE_GUARD_ANCHOR_WORDS", 2),
        babble_guard_min_score_gain=_value(config, "TTS_BABBLE_GUARD_MIN_SCORE_GAIN", 0.08),
        enable_asr_retry=(
            _value(config, "ENABLE_TTS_ASR_RETRY", False)
            if enable_asr_retry is None
            else enable_asr_retry
        ),
        asr_retry_model_name=_value(config, "TTS_ASR_RETRY_MODEL_NAME", "base"),
        asr_retry_device=_value(config, "TTS_ASR_RETRY_DEVICE", "cpu"),
        asr_retry_max_segment_sec=_value(config, "TTS_ASR_RETRY_MAX_SEGMENT_SEC", 2.5),
        asr_retry_attempts=_value(config, "TTS_ASR_RETRY_ATTEMPTS", 2),
        asr_retry_min_score=_value(config, "TTS_ASR_RETRY_MIN_SCORE", 0.94),
        enable_short_segment_tail_trim=_value(config, "SHORT_SEGMENT_TAIL_TRIM_ENABLED", False),
        short_segment_tail_trim_min_overhang_ms=_value(config, "SHORT_SEGMENT_TAIL_TRIM_MIN_OVERHANG_MS", 280),
        short_segment_tail_trim_max_ms=_value(config, "SHORT_SEGMENT_TAIL_TRIM_MAX_MS", 500),
        short_segment_tail_trim_max_ratio=_value(config, "SHORT_SEGMENT_TAIL_TRIM_MAX_RATIO", 0.22),
    )


def build_segment_routing_config(
    config: Any,
    *,
    enabled: bool | None = None,
) -> SegmentRoutingConfig:
    return SegmentRoutingConfig(
        enabled=_value(config, "SEGMENT_ROUTING_ENABLED", True) if enabled is None else enabled,
        short_segment_sec=_value(config, "SEGMENT_ROUTING_SHORT_SEC", 2.2),
        max_refs_per_segment=_value(config, "SEGMENT_ROUTING_MAX_REFS", 2),
        min_segment_sec=_value(config, "SEGMENT_ROUTING_MIN_SEC", 0.9),
        min_segment_words=_value(config, "SEGMENT_ROUTING_MIN_WORDS", 3),
        confidence_margin=_value(config, "SEGMENT_ROUTING_CONFIDENCE_MARGIN", 0.45),
    )


def build_audio_level_config(config: Any) -> AudioLevelConfig:
    return AudioLevelConfig(
        threshold_compression=_value(config, "THRESHOLD_COMPRESSION", -15.0),
        ratio_compression=_value(config, "RATIO_COMPRESSION", 2.0),
        attack_compression=_value(config, "ATTACK_COMPRESSION", 25),
        release_compression=_value(config, "RELEASE_COMPRESSION", 50),
        target_dbfs=_value(config, "TARGET_DBFS", -15.0),
        reference_gain_offset_db=_value(config, "REFERENCE_GAIN_OFFSET_DB", 3.0),
        max_segment_boost_db=_value(config, "MAX_SEGMENT_BOOST_DB", 6.0),
        max_segment_cut_db=_value(config, "MAX_SEGMENT_CUT_DB", 12.0),
        peak_ceiling_dbfs=_value(config, "PEAK_CEILING_DBFS", -2.0),
        enable_final_compression=_value(config, "ENABLE_FINAL_COMPRESSION", False),
        enable_segment_matching=_value(config, "SEGMENT_MATCHING_ENABLED", False),
        segment_match_padding_ms=_value(config, "SEGMENT_MATCH_PADDING_MS", 120),
        segment_match_strength=_value(config, "SEGMENT_MATCH_STRENGTH", 0.7),
        segment_match_max_delta_db=_value(config, "SEGMENT_MATCH_MAX_DELTA_DB", 4.0),
        segment_match_min_active_ratio=_value(config, "SEGMENT_MATCH_MIN_ACTIVE_RATIO", 0.35),
    )
