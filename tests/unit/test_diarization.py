from copy import deepcopy

from src.diarization import (
    apply_diarization_to_segments,
    normalize_diarization_turns,
    summarize_diarization,
)


def test_normalize_diarization_turns_maps_labels_by_first_appearance() -> None:
    turns = normalize_diarization_turns([
        {"start": 2.0, "end": 3.0, "speaker_label": "SPEAKER_B"},
        {"start": 0.0, "end": 1.0, "speaker_label": "SPEAKER_A"},
        {"start": 1.0, "end": 2.0, "speaker_label": "SPEAKER_B"},
    ])

    assert turns == [
        {"start": 0.0, "end": 1.0, "speaker_id": "spk_0", "speaker_label": "SPEAKER_A"},
        {"start": 1.0, "end": 2.0, "speaker_id": "spk_1", "speaker_label": "SPEAKER_B"},
        {"start": 2.0, "end": 3.0, "speaker_id": "spk_1", "speaker_label": "SPEAKER_B"},
    ]


def test_apply_diarization_splits_segment_when_speaker_changes() -> None:
    segments = [{
        "text": "Hello there yes now",
        "start": 0.0,
        "end": 1.8,
        "speaker_id": "spk_0",
        "pause_before_sec": 0.2,
        "pause_after_sec": 0.5,
        "words": [
            {"text": "Hello", "start": 0.0, "end": 0.4},
            {"text": "there", "start": 0.45, "end": 0.8},
            {"text": "yes", "start": 1.0, "end": 1.3},
            {"text": "now", "start": 1.35, "end": 1.8},
        ],
    }]

    original = deepcopy(segments)
    turns = [
        {"start": 0.0, "end": 0.9, "speaker_label": "A"},
        {"start": 0.95, "end": 2.0, "speaker_label": "B"},
    ]

    diarized = apply_diarization_to_segments(
        segments,
        turns,
        default_speaker_id="spk_0",
        max_pause_between_sentences=0.3,
    )

    assert segments == original
    assert len(diarized) == 2
    assert diarized[0]["text"] == "Hello there"
    assert diarized[0]["speaker_id"] == "spk_0"
    assert diarized[0]["pause_before_sec"] == 0.2
    assert diarized[0]["pause_after_sec"] == 0.2
    assert diarized[0]["diarization_split_count"] == 2
    assert diarized[1]["text"] == "yes now"
    assert diarized[1]["speaker_id"] == "spk_1"
    assert diarized[1]["pause_before_sec"] == 0.2
    assert diarized[1]["pause_after_sec"] == 0.5
    assert diarized[1]["words"][0]["speaker_id"] == "spk_1"


def test_apply_diarization_uses_default_for_uncovered_words() -> None:
    segments = [{
        "text": "Uncovered word",
        "start": 5.0,
        "end": 5.8,
        "words": [
            {"text": "Uncovered", "start": 5.0, "end": 5.4},
            {"text": "word", "start": 5.45, "end": 5.8},
        ],
    }]

    diarized = apply_diarization_to_segments(
        segments,
        [{"start": 0.0, "end": 1.0, "speaker_label": "A"}],
        default_speaker_id="spk_fallback",
    )

    assert len(diarized) == 1
    assert diarized[0]["speaker_id"] == "spk_fallback"
    assert diarized[0]["words_with_silence"] == "Uncovered <0.05s> word"


def test_summarize_diarization_counts_speakers_and_splits() -> None:
    turns = [
        {"start": 0.0, "end": 1.0, "speaker_label": "A"},
        {"start": 1.0, "end": 3.0, "speaker_label": "B"},
    ]
    segments = [
        {"start": 0.0, "end": 1.0, "speaker_id": "spk_0", "diarization_split_count": 2},
        {"start": 1.0, "end": 2.0, "speaker_id": "spk_1", "diarization_split_count": 2},
        {"start": 2.0, "end": 3.0, "speaker_id": "spk_1"},
    ]

    summary = summarize_diarization(turns, segments)

    assert summary["speaker_count"] == 2
    assert summary["segment_count_by_speaker"] == {"spk_0": 1, "spk_1": 2}
    assert summary["turn_seconds_by_speaker"] == {"spk_0": 1.0, "spk_1": 2.0}
    assert summary["split_segment_count"] == 2
