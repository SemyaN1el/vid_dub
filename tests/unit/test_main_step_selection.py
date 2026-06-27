from main import ALL_STEPS, resolve_requested_steps


def test_all_steps_include_metrics_by_default() -> None:
    assert resolve_requested_steps("all") == ALL_STEPS


def test_skip_metrics_removes_only_metrics_from_all_plan() -> None:
    assert resolve_requested_steps("all", skip_metrics=True) == [
        "preprocess",
        "asr",
        "translate",
        "tts",
        "postprocess",
        "subtitles",
    ]


def test_explicit_metrics_step_is_not_suppressed() -> None:
    assert resolve_requested_steps("metrics", skip_metrics=True) == ["metrics"]
