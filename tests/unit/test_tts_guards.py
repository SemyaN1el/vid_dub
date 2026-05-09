from pydub import AudioSegment
from pydub.generators import Sine

from src.tts_guards import (
    _compute_safe_tail_trim_ms,
    _is_better_recognition_eval,
    _segment_recognition_score,
    _trim_trailing_speech_island_fast,
)


def test_compute_safe_tail_trim_uses_only_trailing_silence_budget() -> None:
    assert _compute_safe_tail_trim_ms(
        original_ms=1000,
        corrected_ms=1500,
        trailing_silence_ms=260,
        short_segment_sec=2.0,
        min_overhang_ms=180,
        max_trim_ms=700,
        max_trim_ratio=0.5,
    ) == 160


def test_segment_recognition_score_penalizes_extra_tail() -> None:
    score = _segment_recognition_score(
        expected_text="hello world",
        recognized_text="hello world extra words",
        anchor_words=2,
    )

    assert score["has_suffix"] is True
    assert score["has_extra_tail"] is True
    assert score["score"] < score["similarity"]


def test_is_better_recognition_eval_prefers_no_extra_tail() -> None:
    candidate = {
        "has_extra_tail": False,
        "has_suffix": True,
        "score": 0.82,
        "similarity": 0.82,
    }
    incumbent = {
        "has_extra_tail": True,
        "has_suffix": True,
        "score": 0.95,
        "similarity": 0.95,
    }

    assert _is_better_recognition_eval(candidate, incumbent) is True


def test_trim_trailing_speech_island_fast_removes_small_tail_island() -> None:
    tone = Sine(440).to_audio_segment(duration=500).apply_gain(-6)
    tail = Sine(660).to_audio_segment(duration=180).apply_gain(-6)
    audio = tone + AudioSegment.silent(duration=120) + tail

    trimmed, info = _trim_trailing_speech_island_fast(
        audio=audio,
        expected_text="hello.",
        original_ms=500,
        segment_duration_sec=0.5,
        tts_group_size=1,
        max_segment_sec=2.0,
        min_overhang_ms=180,
        min_gap_ms=80,
        min_island_ms=80,
        max_island_ms=250,
        search_window_ms=500,
        max_trim_ms=400,
    )

    assert info is not None
    assert info["trim_ms"] == 180
    assert len(trimmed) == 620
