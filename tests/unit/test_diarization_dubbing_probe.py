from pathlib import Path

from pydub import AudioSegment

from scripts.diarization_dubbing_probe import (
    _build_llm_translation_messages,
    _build_llm_timing_constraints,
    _group_segments_by_speaker,
    _mix_audio_layers,
    _parse_llm_translation_response,
    _reference_paths_from_profile,
    _summarize_llm_timing_constraints,
    _translation_constraint_violations,
)


def test_group_segments_by_speaker_copies_segments() -> None:
    segments = [
        {"speaker_id": "spk_1", "text": "one"},
        {"speaker_id": "spk_0", "text": "two"},
        {"text": "fallback"},
    ]

    grouped = _group_segments_by_speaker(segments, default_speaker_id="spk_default")

    assert list(grouped) == ["spk_0", "spk_1", "spk_default"]
    assert grouped["spk_0"][0]["text"] == "two"
    assert grouped["spk_1"][0]["text"] == "one"
    assert grouped["spk_default"][0]["text"] == "fallback"

    grouped["spk_0"][0]["text"] = "changed"
    assert segments[1]["text"] == "two"


def test_reference_paths_from_profile_prefers_existing_clips(tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    merged = tmp_path / "merged.wav"
    clip.write_bytes(b"clip")
    merged.write_bytes(b"merged")

    profile = {
        "merged_reference_path": str(merged),
        "clips": [
            {"path": str(tmp_path / "missing.wav")},
            {"path": str(clip)},
        ],
    }

    assert _reference_paths_from_profile(profile) == [str(clip)]


def test_reference_paths_from_profile_falls_back_to_merged(tmp_path: Path) -> None:
    merged = tmp_path / "merged.wav"
    merged.write_bytes(b"merged")

    assert _reference_paths_from_profile({"merged_reference_path": str(merged), "clips": []}) == [
        str(merged)
    ]


def test_mix_audio_layers_uses_longest_duration(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    output = tmp_path / "mixed.wav"

    AudioSegment.silent(duration=500).export(first, format="wav")
    AudioSegment.silent(duration=900).export(second, format="wav")

    mixed = _mix_audio_layers([first, second], output)

    assert output.is_file()
    assert len(mixed) == 900


def test_parse_llm_translation_response_accepts_fenced_json() -> None:
    response = """
    ```json
    [
      {"id": 0, "translation": "Привет."},
      {"id": 1, "translation": "Как дела?"}
    ]
    ```
    """

    assert _parse_llm_translation_response(response, [0, 1]) == [
        "Привет.",
        "Как дела?",
    ]


def test_parse_llm_translation_response_rejects_id_mismatch() -> None:
    response = '[{"id": 2, "translation": "Не тот id."}]'

    try:
        _parse_llm_translation_response(response, [0])
    except Exception as exc:
        assert "id mismatch" in str(exc)
    else:
        raise AssertionError("Expected id mismatch error")


def test_parse_llm_translation_response_accepts_batch_object() -> None:
    response = """
    {
      "batch": [
        {"id": 0, "translation": "Первый."},
        {"id": 1, "translation": "Второй."}
      ]
    }
    """

    assert _parse_llm_translation_response(response, [0, 1]) == ["Первый.", "Второй."]


def test_parse_llm_translation_response_strips_think_block() -> None:
    response = """
    <think>internal reasoning that must not leak</think>
    [{"id": 0, "translation": "Коротко."}]
    """

    assert _parse_llm_translation_response(response, [0]) == ["Коротко."]


def test_build_llm_translation_messages_includes_context_and_batch() -> None:
    messages = _build_llm_translation_messages(
        batch=[
            {"speaker_id": "spk_0", "start": 1.0, "end": 2.0, "text": "Hello there."},
        ],
        previous_context=[
            {"speaker_id": "spk_1", "start": 0.0, "end": 0.8, "text": "Before."},
        ],
        next_context=[
            {"speaker_id": "spk_1", "start": 2.2, "end": 3.0, "text": "After."},
        ],
        src_lang="eng_Latn",
        tgt_lang="rus_Cyrl",
    )

    assert messages[0]["role"] == "system"
    assert "audiovisual translator" in messages[0]["content"]
    assert "Hello there." in messages[1]["content"]
    assert "Before." in messages[1]["content"]
    assert "After." in messages[1]["content"]


def test_build_llm_timing_constraints_uses_duration_budget() -> None:
    constraints = _build_llm_timing_constraints(
        [
            {"start": 0.0, "end": 2.0},
            {"source_duration_sec": 20.0},
            {"start": 3.0, "end": 3.2},
        ],
        target_chars_per_sec=10.0,
        min_chars=8,
        max_chars=50,
    )

    assert constraints[0]["max_chars"] == 20
    assert constraints[1]["max_chars"] == 50
    assert constraints[2]["max_chars"] == 8


def test_build_llm_translation_messages_includes_timing_contract() -> None:
    messages = _build_llm_translation_messages(
        batch=[
            {"speaker_id": "spk_0", "start": 1.0, "end": 2.0, "text": "Hello there."},
        ],
        previous_context=[],
        next_context=[],
        src_lang="eng_Latn",
        tgt_lang="rus_Cyrl",
        timing_constraints=[
            {
                "id": 0,
                "duration_sec": 1.0,
                "max_chars": 16,
                "target_chars_per_sec": 16.0,
            }
        ],
    )

    assert '"max_chars": 16' in messages[1]["content"]
    assert "timing_contract" in messages[1]["content"]


def test_translation_constraint_violations_respects_tolerance() -> None:
    constraints = [
        {"id": 0, "max_chars": 10},
        {"id": 1, "max_chars": 10},
    ]

    violations = _translation_constraint_violations(
        ["1234567890", "123456789012"],
        constraints,
        tolerance_ratio=1.1,
    )

    assert violations == [
        {
            "id": 1,
            "max_chars": 10,
            "tolerated_chars": 11,
            "actual_chars": 12,
            "translation": "123456789012",
        }
    ]


def test_summarize_llm_timing_constraints_counts_over_limit() -> None:
    summary = _summarize_llm_timing_constraints(
        [
            {
                "llm_timing_constraint": {
                    "ratio": 0.8,
                    "within_limit": True,
                    "within_tolerance": True,
                }
            },
            {
                "llm_timing_constraint": {
                    "ratio": 1.2,
                    "within_limit": False,
                    "within_tolerance": False,
                }
            },
            {"text": "no constraint"},
        ]
    )

    assert summary == {
        "enabled": True,
        "segment_count": 2,
        "over_limit_count": 1,
        "over_tolerance_count": 1,
        "avg_ratio": 1.0,
        "max_ratio": 1.2,
    }
