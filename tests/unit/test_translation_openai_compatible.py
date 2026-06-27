from types import SimpleNamespace

from src import translation as tr


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=item)
                )
            ]
        )


class _FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


class _FakeOpenAIClient(_FakeClient):
    init_kwargs = []

    def __init__(self, **kwargs):
        self.init_kwargs.append(kwargs)
        super().__init__([])


def _backend(responses):
    return tr.OpenAICompatibleTranslationBackend(
        model_name="gpt-test",
        api_key="test",
        base_url="",
        temperature=0.1,
        max_output_tokens=256,
        timeout_sec=10,
        min_interval_sec=0.0,
        max_retries=0,
        backend="openai",
        client=_FakeClient(responses),
    )


def test_openai_compatible_translation_uses_batch_json(monkeypatch) -> None:
    tr._load_translation_profiles.cache_clear()
    monkeypatch.delenv("MT_PROFILE", raising=False)
    monkeypatch.setattr(tr.cfg, "MT_STYLE", "news", raising=False)
    monkeypatch.setattr(tr.cfg, "MT_PROFILE", "", raising=False)
    monkeypatch.setattr(tr.cfg, "MT_ASR_CORRECTIONS", "03 -> o3|04 -> o4", raising=False)
    monkeypatch.setattr(tr.cfg, "MT_OPENAI_RESPONSE_FORMAT", "json_object", raising=False)
    monkeypatch.setenv("MT_LENGTH_AWARE_ENABLED", "1")

    backend = _backend([
        '{"translations":["Привет.","Мир."]}',
    ])

    translated = tr.translate_segments(
        model=backend,
        tokenizer=None,
        segments=[
            {"text": "Hello.", "start": 0.0, "end": 1.0},
            {"text": "World.", "start": 1.0, "end": 2.0},
        ],
        batch_size=8,
    )

    assert [seg["text"] for seg in translated] == ["Привет.", "Мир."]
    call = backend.client.chat.completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert "broadcast-style Russian" in call["messages"][0]["content"]
    assert "03 -> o3" in call["messages"][1]["content"]
    assert "source_duration_sec" in call["messages"][1]["content"]
    assert "soft_target_max_chars" in call["messages"][1]["content"]


def test_openai_compatible_translation_applies_profile_prompt(monkeypatch) -> None:
    tr._load_translation_profiles.cache_clear()
    monkeypatch.setenv("MT_PROFILE", "movie")
    monkeypatch.setenv("MT_PROFILES_PATH", "translation_profiles.yaml")
    monkeypatch.setattr(tr.cfg, "MT_PROFILE", "movie", raising=False)
    monkeypatch.setattr(tr.cfg, "MT_PROFILES_PATH", "translation_profiles.yaml", raising=False)
    monkeypatch.setattr(tr.cfg, "MT_STYLE", "compact", raising=False)
    monkeypatch.setattr(tr.cfg, "MT_ASR_CORRECTIONS", "", raising=False)
    monkeypatch.setattr(tr.cfg, "MT_OPENAI_RESPONSE_FORMAT", "json_object", raising=False)

    backend = _backend([
        '{"translation":"Не сейчас."}',
    ])

    translated = tr.translate_segments(
        model=backend,
        tokenizer=None,
        segments=[
            {"text": "Not now.", "start": 0.0, "end": 1.0},
        ],
        batch_size=8,
    )

    assert translated[0]["text"] == "Не сейчас."
    call = backend.client.chat.completions.calls[0]
    assert "Genre profile (movie)" in call["messages"][0]["content"]
    assert "subtext" in call["messages"][0]["content"]
    assert "Profile-specific instructions" in call["messages"][1]["content"]
    assert "performable" in call["messages"][1]["content"]


def test_smart_sync_shorter_prompt_targets_pre_speedfit_duration() -> None:
    system_prompt, user_prompt = tr._build_timing_rewrite_prompts(
        source_text="The next step is splitting the data into train and test sets.",
        translated_text="Следующим шагом является разделение данных на набор обучения и набор тестирования.",
        src_lang="eng_Latn",
        tgt_lang="rus_Cyrl",
        target_duration_sec=4.7,
        available_duration_sec=3.8,
        current_duration_sec=6.2,
        words_with_silence="The <0.05s> next <0.04s> step",
        rewrite_mode="shorter",
        candidate_count=3,
    )

    assert "least invasive edit" in system_prompt
    assert "Return valid JSON only" in system_prompt
    assert '"candidates"' in system_prompt
    assert "Final timeline window: 3.80s" in user_prompt
    assert "Target synthesized duration before final speed fitting: 4.70s" in user_prompt
    assert "final audio may still be mildly sped up" in user_prompt
    assert "Preserve the final semantic unit" in user_prompt
    assert "unfinished comma fragment" in user_prompt
    assert "Return exactly 3 alternatives" in user_prompt


def test_smart_sync_candidate_preflight_prefers_safe_timing_match() -> None:
    original = "Следующим шагом является разделение данных на набор обучения и набор тестирования."
    candidates = [
        "Разделяем данные.",
        "Следующий шаг - разделить данные на обучающий и тестовый наборы.",
        "Следующим шагом является разделение данных на набор обучения и набор тестирования с дополнительной проверкой.",
    ]

    selected, info = tr._select_timing_rewrite_candidate(
        original_text=original,
        candidates=candidates,
        current_duration_sec=6.2,
        target_duration_sec=4.7,
        rewrite_mode="shorter",
    )

    assert selected == "Следующий шаг - разделить данные на обучающий и тестовый наборы."
    assert info["candidate_count"] == 3
    assert info["chosen_metrics"]["word_ratio"] >= 0.66


def test_smart_sync_candidate_parser_reads_json_candidates() -> None:
    candidates = tr._extract_timing_rewrite_candidates(
        '{"candidates":["Первый вариант.","Второй вариант."]}',
        tgt_lang="rus_Cyrl",
    )

    assert candidates == ["Первый вариант.", "Второй вариант."]


def test_openai_compatible_batch_falls_back_by_splitting(monkeypatch) -> None:
    tr._load_translation_profiles.cache_clear()
    monkeypatch.delenv("MT_PROFILE", raising=False)
    monkeypatch.setattr(tr.cfg, "MT_STYLE", "standard", raising=False)
    monkeypatch.setattr(tr.cfg, "MT_PROFILE", "", raising=False)
    monkeypatch.setattr(tr.cfg, "MT_ASR_CORRECTIONS", "", raising=False)
    monkeypatch.setattr(tr.cfg, "MT_OPENAI_RESPONSE_FORMAT", "json_object", raising=False)

    backend = _backend([
        '{"translations":["Недостаточно."]}',
        '{"translation":"Первый."}',
        '{"translations":["Второй.","Третий."]}',
    ])

    translated = tr.translate_segments(
        model=backend,
        tokenizer=None,
        segments=[
            {"text": "One.", "start": 0.0, "end": 1.0},
            {"text": "Two.", "start": 1.0, "end": 2.0},
            {"text": "Three.", "start": 2.0, "end": 3.0},
        ],
        batch_size=8,
    )

    assert [seg["text"] for seg in translated] == ["Первый.", "Второй.", "Третий."]
    assert len(backend.client.chat.completions.calls) == 3


def test_translation_quality_flags_operational_risks(monkeypatch) -> None:
    monkeypatch.setenv("MT_LENGTH_AWARE_ENABLED", "1")
    monkeypatch.setenv("MT_TARGET_CHARS_PER_SEC", "10.0")
    monkeypatch.setenv("MT_LENGTH_GRACE_CHARS", "0")
    monkeypatch.setenv("MT_MIN_TARGET_CHARS", "10")
    monkeypatch.setenv("MT_QA_MAX_TARGET_RATIO", "1.1")
    monkeypatch.setenv("MT_QA_MAX_CHAR_RATIO", "1.2")
    monkeypatch.setenv("MT_PRESERVE_TERMS", "API")

    report = tr.analyze_translation_quality([
        {
            "original_text": "Version 3 works.",
            "text": "Очень длинный translation без числа.",
            "start": 0.0,
            "end": 1.0,
        }
    ])

    assert report["issue_count"] == 1
    issues = set(report["segments"][0]["issues"])
    assert "missing_number" in issues
    assert "latin_word_in_translation" in issues
    assert "over_timing_soft_limit" in issues
