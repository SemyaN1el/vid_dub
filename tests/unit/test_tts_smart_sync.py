from typing import Any

from src.tts import (
    _smart_sync_acceptance_gate,
    _smart_sync_candidate_options,
    _smart_sync_distance_ms,
)


def _eval(
    score: float,
    *,
    has_extra_tail: bool = False,
    has_suffix: bool = True,
) -> dict[str, Any]:
    return {
        "score": score,
        "has_extra_tail": has_extra_tail,
        "has_suffix": has_suffix,
    }


def _gate(
    *,
    source_text: str = "добрый день коллеги начинаем короткий обзор",
    rewritten_text: str = "добрый день коллеги короткий обзор",
    rewrite_mode: str = "shorter",
    baseline_duration_ms: int = 3400,
    rewritten_duration_ms: int = 2200,
    target_duration_ms: int = 2600,
    baseline_eval: dict[str, Any] | None = None,
    rewritten_eval: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    return _smart_sync_acceptance_gate(
        source_text=source_text,
        rewritten_text=rewritten_text,
        rewrite_mode=rewrite_mode,
        baseline_duration_ms=baseline_duration_ms,
        rewritten_duration_ms=rewritten_duration_ms,
        target_duration_ms=target_duration_ms,
        baseline_eval=baseline_eval,
        rewritten_eval=rewritten_eval,
        min_fill_ratio=0.72,
        min_text_similarity=0.55,
        min_word_ratio=0.58,
        min_token_precision=0.72,
        min_asr_score=0.90,
        max_asr_drop=0.03,
    )


def test_smart_sync_distance_shorter_penalizes_only_over_target() -> None:
    assert _smart_sync_distance_ms(2400, 2600, "shorter") == 0
    assert _smart_sync_distance_ms(2600, 2600, "shorter") == 0
    assert _smart_sync_distance_ms(2900, 2600, "shorter") == 300
    assert _smart_sync_distance_ms(2400, 2600, "longer") == 200


def test_smart_sync_acceptance_accepts_shorter_rewrite_with_preserved_meaning() -> None:
    accepted, metrics = _gate(
        baseline_eval=_eval(0.91),
        rewritten_eval=_eval(0.93),
    )

    assert accepted is True
    assert metrics["reject_reason"] is None
    assert metrics["fill_ratio"] >= 0.72
    assert metrics["text_similarity"] >= 0.55
    assert metrics["token_precision"] >= 0.72


def test_smart_sync_acceptance_rejects_low_text_similarity() -> None:
    accepted, metrics = _gate(
        rewritten_text="совсем другой смысл здесь пропал",
        baseline_eval=_eval(0.91),
        rewritten_eval=_eval(0.94),
    )

    assert accepted is False
    assert metrics["reject_reason"] == "text_similarity"


def test_smart_sync_acceptance_rejects_low_asr_score() -> None:
    accepted, metrics = _gate(
        baseline_eval=None,
        rewritten_eval=_eval(0.82),
    )

    assert accepted is False
    assert metrics["reject_reason"] == "asr_score"


def test_smart_sync_acceptance_rejects_asr_drop_from_baseline() -> None:
    accepted, metrics = _gate(
        baseline_eval=_eval(0.98),
        rewritten_eval=_eval(0.92),
    )

    assert accepted is False
    assert metrics["reject_reason"] == "asr_drop"


def test_smart_sync_acceptance_allows_small_asr_noise() -> None:
    accepted, metrics = _gate(
        baseline_eval=_eval(0.98),
        rewritten_eval=_eval(0.96),
    )

    assert accepted is True
    assert metrics["reject_reason"] is None
    assert metrics["asr_drop"] == 0.02


def test_smart_sync_acceptance_rejects_extra_tail_and_missing_suffix() -> None:
    accepted_tail, tail_metrics = _gate(
        baseline_eval=_eval(0.91),
        rewritten_eval=_eval(0.95, has_extra_tail=True),
    )
    accepted_suffix, suffix_metrics = _gate(
        baseline_eval=_eval(0.91),
        rewritten_eval=_eval(0.95, has_suffix=False),
    )

    assert accepted_tail is False
    assert tail_metrics["reject_reason"] == "extra_tail"
    assert accepted_suffix is False
    assert suffix_metrics["reject_reason"] == "missing_suffix"


def test_smart_sync_candidate_options_keep_ranked_fallbacks() -> None:
    options = _smart_sync_candidate_options(
        "Короткий основной вариант.",
        {
            "model_name": "test-model",
            "preflight": {
                "ranked": [
                    {"candidate": "Короткий основной вариант.", "metrics": {"score": 0.1}},
                    {"candidate": "Другой безопасный вариант.", "metrics": {"score": 0.2}},
                    {"candidate": "Другой безопасный вариант.", "metrics": {"score": 0.3}},
                ]
            },
        },
    )

    assert [text for text, _ in options] == [
        "Короткий основной вариант.",
        "Другой безопасный вариант.",
    ]
    assert options[1][1]["candidate_rank"] == 2
    assert options[1][1]["candidate_metrics"] == {"score": 0.2}
