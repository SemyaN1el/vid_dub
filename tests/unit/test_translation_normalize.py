import json
from pathlib import Path

import config as cfg
from src.translation import (
    _load_translation_profiles,
    _parse_source_target_pairs,
    normalize_translated_text,
)


def _clear_replacement_sources(monkeypatch) -> None:
    monkeypatch.delenv("MT_PROFILE", raising=False)
    monkeypatch.delenv("MT_TEXT_REPLACEMENTS", raising=False)
    monkeypatch.setattr(cfg, "MT_PROFILE", "", raising=False)
    monkeypatch.setattr(cfg, "MT_TEXT_REPLACEMENTS", "", raising=False)


def test_generic_cleanup_keeps_domain_words_intact(monkeypatch) -> None:
    _clear_replacement_sources(monkeypatch)

    assert normalize_translated_text("Привет ,  мир !") == "Привет, мир!"
    assert (
        normalize_translated_text("Время высадки на Луну.")
        == "Время высадки на Луну."
    )


def test_you_are_welcome_special_case(monkeypatch) -> None:
    _clear_replacement_sources(monkeypatch)

    assert normalize_translated_text("Не за что.", "You're welcome.") == "Пожалуйста."


def test_profile_text_replacements_applied(tmp_path: Path, monkeypatch) -> None:
    _clear_replacement_sources(monkeypatch)
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        json.dumps({
            "profiles": {
                "hotel": {
                    "text_replacements": [
                        "высадки -> выезда",
                        "на ресепшн -> на стойке регистрации",
                    ],
                },
            },
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setenv("MT_PROFILES_PATH", str(profile_path))
    monkeypatch.setenv("MT_PROFILE", "hotel")

    assert (
        normalize_translated_text("Время высадки уточните на ресепшн.")
        == "Время выезда уточните на стойке регистрации."
    )


def test_env_text_replacements_applied_without_profile(monkeypatch) -> None:
    _clear_replacement_sources(monkeypatch)
    monkeypatch.setenv("MT_TEXT_REPLACEMENTS", "высадки -> выезда")

    assert normalize_translated_text("Время высадки.") == "Время выезда."


def test_repo_hotel_profile_provides_replacements() -> None:
    profiles_path = Path(__file__).resolve().parents[2] / "translation_profiles.yaml"
    profiles = _load_translation_profiles(str(profiles_path))

    hotel = profiles.get("hotel")
    assert isinstance(hotel, dict)
    pairs = _parse_source_target_pairs([
        str(item) for item in hotel.get("text_replacements", [])
    ])
    assert {"source": "высадки", "target": "выезда"} in pairs
