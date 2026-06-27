from pydub import AudioSegment

from src.tts import _synthesize_best_retry_take


def test_selects_candidate_with_best_raw_recognition() -> None:
    audio_bad = AudioSegment.silent(duration=100)
    audio_good = AudioSegment.silent(duration=120)
    synthesized = {"Привет мир!": audio_bad, "Привет мир?": audio_good}
    recognized = {id(audio_bad): "привет", id(audio_good): "привет мир"}
    synth_calls: list[str] = []

    def synthesize_fn(text: str) -> AudioSegment:
        synth_calls.append(text)
        return synthesized[text]

    best_audio, best_text, best_eval = _synthesize_best_retry_take(
        retry_text_variants=["Привет мир!", "Привет мир?"],
        fallback_text="Привет мир!",
        attempts=2,
        synthesize_fn=synthesize_fn,
        transcribe_fn=lambda audio: recognized[id(audio)],
        expected_text="Привет мир!",
        anchor_words=2,
    )

    assert synth_calls == ["Привет мир!", "Привет мир?"]
    assert best_audio is audio_good
    assert best_text == "Привет мир?"
    assert best_eval is not None and best_eval["score"] >= 0.995


def test_stops_synthesizing_after_perfect_take() -> None:
    synth_calls: list[str] = []

    def synthesize_fn(text: str) -> AudioSegment:
        synth_calls.append(text)
        return AudioSegment.silent(duration=100)

    best_audio, _, _ = _synthesize_best_retry_take(
        retry_text_variants=["Привет мир!"],
        fallback_text="Привет мир!",
        attempts=4,
        synthesize_fn=synthesize_fn,
        transcribe_fn=lambda audio: "привет мир",
        expected_text="Привет мир!",
        anchor_words=2,
    )

    assert len(synth_calls) == 1
    assert best_audio is not None


def test_uses_fallback_text_when_variants_empty() -> None:
    synth_calls: list[str] = []
    produced: list[AudioSegment] = []

    def synthesize_fn(text: str) -> AudioSegment:
        synth_calls.append(text)
        audio = AudioSegment.silent(duration=100)
        produced.append(audio)
        return audio

    best_audio, best_text, _ = _synthesize_best_retry_take(
        retry_text_variants=[],
        fallback_text="Привет мир!",
        attempts=2,
        synthesize_fn=synthesize_fn,
        transcribe_fn=lambda audio: "привет",
        expected_text="Привет мир!",
        anchor_words=2,
    )

    assert synth_calls == ["Привет мир!", "Привет мир!"]
    assert best_audio is produced[0]
    assert best_text == "Привет мир!"


def test_zero_attempts_returns_nothing() -> None:
    best_audio, best_text, best_eval = _synthesize_best_retry_take(
        retry_text_variants=["Привет мир!"],
        fallback_text="Привет мир!",
        attempts=0,
        synthesize_fn=lambda text: AudioSegment.silent(duration=100),
        transcribe_fn=lambda audio: "привет мир",
        expected_text="Привет мир!",
        anchor_words=2,
    )

    assert best_audio is None
    assert best_text is None
    assert best_eval is None
