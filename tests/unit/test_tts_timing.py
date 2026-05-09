from pydub import AudioSegment

from src.tts_timing import (
    _allocate_timing_extension_ms,
    _timing_speech_stats,
    build_segment_timing_window,
    update_next_segment_start,
)


def test_allocate_timing_extension_prefers_symmetric_borrow() -> None:
    assert _allocate_timing_extension_ms(
        needed_extension_ms=800,
        available_before_ms=400,
        available_after_ms=1000,
    ) == (400, 400)


def test_timing_speech_stats_handles_silent_audio() -> None:
    stats = _timing_speech_stats(AudioSegment.silent(duration=300))

    assert stats == {
        "effective_duration_ms": 0,
        "leading_silence_ms": 300,
        "trailing_silence_ms": 300,
    }


def test_build_segment_timing_window_borrows_from_neighbors() -> None:
    segments = [
        {"start": 1.0, "end": 2.0},
        {"start": 3.2, "end": 4.0},
    ]

    window = build_segment_timing_window(
        segments=segments,
        index=0,
        original_start_sec=1.0,
        prev_end_sec=0.4,
        timing_duration_ms=1800,
        min_pause_between_segments=0.2,
        max_shift_left_seconds=0.5,
        max_next_start_shift_sec=None,
    )

    assert window.original_ms == 1000
    assert window.borrowed_before_ms == 399
    assert window.borrowed_after_ms == 401
    assert window.cur_start_sec == 0.601
    assert window.cur_start_ms == 601
    assert window.available_ms == 1800
    assert window.timing_window_end_sec == 2.401


def test_update_next_segment_start_respects_shift_cap() -> None:
    segments = [
        {"start": 0.0, "end": 1.0},
        {"start": 2.0, "end": 3.0},
    ]

    update_next_segment_start(
        segments=segments,
        index=0,
        actual_end_sec=3.0,
        max_next_start_shift_sec=0.5,
    )

    assert segments[1]["original_start"] == 2.0
    assert segments[1]["start"] == 2.5
