from src.tts_text import (
    _build_tts_retry_text_variants,
    _clean_text,
    _prepare_tts_segments,
)


def test_clean_text_removes_emoji_and_normalizes_spaces() -> None:
    assert _clean_text("  Hello   world! 🎙️  ") == "Hello world!"


def test_build_tts_retry_text_variants_for_xtts_boundary() -> None:
    variants = _build_tts_retry_text_variants(
        "Привет",
        original_text="Hello.",
        tts_backend_name="xtts",
    )

    assert variants[0] == "Привет"
    assert "Привет!" in variants
    assert "Привет?" in variants


def test_prepare_tts_segments_groups_continuation_segments() -> None:
    segments = [
        {
            "text": "Это начало,",
            "original_text": "This starts,",
            "start": 0.0,
            "end": 1.0,
            "speaker_id": "spk_0",
        },
        {
            "text": "продолжение",
            "original_text": "continues",
            "start": 1.1,
            "end": 2.0,
            "speaker_id": "spk_0",
        },
    ]

    grouped = _prepare_tts_segments(
        segments,
        enable_grouping=True,
        max_gap_sec=0.6,
        max_group_segments=2,
        max_group_chars=120,
        max_group_duration_sec=8.5,
    )

    assert len(grouped) == 1
    assert grouped[0]["text"] == "Это начало, продолжение"
    assert grouped[0]["original_text"] == "This starts, continues"
    assert grouped[0]["tts_group_indices"] == [0, 1]
    assert grouped[0]["tts_group_size"] == 2
