from src.asr import WordSegment, _segment_from_words


def test_segment_from_words_preserves_word_metadata() -> None:
    segment = _segment_from_words(
        [
            WordSegment(" Hello ", 1.0, 1.2),
            WordSegment("world", 1.35, 1.8),
        ],
        speaker_id="spk_0",
        pause_before_sec=0.4,
        pause_after_sec=0.7,
    )

    assert segment["text"] == "Hello world"
    assert segment["start"] == 1.0
    assert segment["end"] == 1.8
    assert segment["source_word_count"] == 2
    assert segment["source_duration_sec"] == 0.8
    assert segment["pause_before_sec"] == 0.4
    assert segment["pause_after_sec"] == 0.7
    assert segment["words_with_silence"] == "Hello <0.15s> world"
    assert segment["words"] == [
        {"text": "Hello", "start": 1.0, "end": 1.2},
        {"text": "world", "start": 1.35, "end": 1.8},
    ]
