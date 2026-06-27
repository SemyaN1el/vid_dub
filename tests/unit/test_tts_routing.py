from src.tts_routing import _reference_route_score, _select_segment_references


def test_reference_route_score_prefers_matching_duration_and_question() -> None:
    segment = {
        "text": "Are you ready?",
        "start": 0.0,
        "end": 1.2,
    }
    matching_clip = {
        "path": "matching.wav",
        "text": "Ready?",
        "duration_sec": 1.1,
        "duration_bucket": "short",
    }
    mismatched_clip = {
        "path": "mismatched.wav",
        "text": "This is a long calm sentence.",
        "duration_sec": 8.0,
        "duration_bucket": "long",
    }

    assert _reference_route_score(segment, matching_clip) > _reference_route_score(segment, mismatched_clip)


def test_select_segment_references_returns_fallback_without_profile(tmp_path) -> None:
    fallback = tmp_path / "fallback.wav"
    fallback.write_bytes(b"")

    selected = _select_segment_references(
        segment={"text": "Hi there", "start": 0.0, "end": 1.0},
        speaker_wav=str(fallback),
        speaker_profile=None,
        max_refs_per_segment=1,
        short_segment_sec=2.2,
        min_segment_sec=0.5,
        min_segment_words=1,
        confidence_margin=0.45,
    )

    assert selected == [str(fallback)]


def test_select_segment_references_uses_routing_clip_when_it_scores_better(tmp_path) -> None:
    fallback = tmp_path / "fallback.wav"
    routing = tmp_path / "routing.wav"
    fallback.write_bytes(b"")
    routing.write_bytes(b"")

    segment = {
        "text": "Are you ready?",
        "start": 0.0,
        "end": 1.2,
    }
    speaker_profile = {
        "clips": [
            {
                "path": str(fallback),
                "text": "This is a long calm sentence.",
                "duration_sec": 8.0,
                "duration_bucket": "long",
            }
        ],
        "routing_clips": [
            {
                "path": str(routing),
                "text": "Ready?",
                "duration_sec": 1.1,
                "duration_bucket": "short",
                "selection_score": 6.0,
            }
        ],
    }

    selected = _select_segment_references(
        segment=segment,
        speaker_wav=str(fallback),
        speaker_profile=speaker_profile,
        max_refs_per_segment=1,
        short_segment_sec=2.2,
        min_segment_sec=0.5,
        min_segment_words=1,
        confidence_margin=0.0,
    )

    assert selected == [str(routing)]
