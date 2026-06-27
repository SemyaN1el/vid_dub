"""Translation profiles, styles, terminology, and prompt config helpers."""

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import config as cfg
from src.translation_common import _normalize_spaces

logger = logging.getLogger(__name__)


_TRANSLATION_STYLE_DIRECTIVES: Dict[str, str] = {
    "standard": (
        "Use natural spoken Russian suitable for dubbing. Preserve meaning, tone, "
        "named entities, numbers, and technical terms. Avoid bookish phrasing."
    ),
    "academic": (
        "Use a precise academic register. Preserve technical terminology and "
        "formal structure, but keep the line pronounceable for voice-over."
    ),
    "casual": (
        "Use natural conversational Russian, as if explaining aloud. Prefer "
        "short clear phrasing over literal syntax."
    ),
    "news": (
        "Use concise broadcast-style Russian. Cut filler and redundancy while "
        "preserving facts, numbers, names, negations, and causality."
    ),
    "compact": (
        "Use compact spoken Russian for tight dubbing timing. Preserve all "
        "meaning-critical details, but choose shorter natural phrasing."
    ),
}


def _translation_profiles_path() -> str:
    return os.getenv(
        "MT_PROFILES_PATH",
        getattr(cfg, "MT_PROFILES_PATH", "./translation_profiles.yaml"),
    ).strip()


def _translation_profile_name() -> str:
    return os.getenv("MT_PROFILE", getattr(cfg, "MT_PROFILE", "")).strip().lower()


@lru_cache(maxsize=8)
def _load_translation_profiles(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}

    profile_path = Path(path)
    if not profile_path.is_absolute():
        profile_path = Path.cwd() / profile_path
    if not profile_path.exists():
        logger.warning("MT profile file not found: %s", profile_path)
        return {}

    try:
        if profile_path.suffix.lower() == ".json":
            data = json.loads(profile_path.read_text(encoding="utf-8"))
        else:
            try:
                import yaml
            except ImportError:
                logger.warning(
                    "PyYAML is not installed, MT_PROFILE will be ignored for %s.",
                    profile_path,
                )
                return {}
            data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load MT profile file %s: %s", profile_path, exc)
        return {}

    if not isinstance(data, dict):
        logger.warning("MT profile file %s must contain a mapping.", profile_path)
        return {}
    profiles = data.get("profiles", data)
    if not isinstance(profiles, dict):
        logger.warning("MT profile file %s has invalid profiles section.", profile_path)
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}
    for name, profile in profiles.items():
        if isinstance(profile, dict):
            normalized[str(name).strip().lower()] = profile
    return normalized


def _active_translation_profile() -> Dict[str, Any]:
    name = _translation_profile_name()
    if not name:
        return {}
    profiles = _load_translation_profiles(_translation_profiles_path())
    profile = profiles.get(name)
    if not isinstance(profile, dict):
        logger.warning(
            "Unknown MT_PROFILE=%s; available profiles: %s",
            name,
            ", ".join(sorted(profiles)) or "none",
        )
        return {}
    return profile


def _split_config_list(raw: Any) -> List[str]:
    if isinstance(raw, (list, tuple)):
        return [_normalize_spaces(str(item)) for item in raw if _normalize_spaces(str(item))]
    return [
        _normalize_spaces(part)
        for part in re.split(r"[|\n;]+", str(raw or ""))
        if _normalize_spaces(part)
    ]


def _profile_text(key: str) -> str:
    value = _active_translation_profile().get(key, "")
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value if str(item).strip())
    return _normalize_spaces(str(value))


def _profile_list(key: str) -> List[str]:
    value = _active_translation_profile().get(key, [])
    if value is None:
        return []
    return _split_config_list(value)


def _profile_system_directives() -> str:
    profile_name = _translation_profile_name()
    directives = _profile_text("system_directives")
    if not profile_name or not directives:
        return ""
    return f"Genre profile ({profile_name}): {directives}"


def _profile_user_directives_block() -> str:
    directives = _profile_text("user_directives")
    if not directives:
        return ""
    return f"\nProfile-specific instructions:\n{directives}\n"


def _translation_style_directives() -> str:
    style = os.getenv("MT_STYLE", getattr(cfg, "MT_STYLE", "standard")).strip().lower() or "standard"
    if style not in _TRANSLATION_STYLE_DIRECTIVES:
        logger.warning(
            "Неизвестный MT_STYLE=%s; используется standard. Доступно: %s",
            style,
            ", ".join(sorted(_TRANSLATION_STYLE_DIRECTIVES)),
        )
        style = "standard"
    return _TRANSLATION_STYLE_DIRECTIVES[style]


def _asr_corrections() -> List[str]:
    raw = os.getenv("MT_ASR_CORRECTIONS", getattr(cfg, "MT_ASR_CORRECTIONS", ""))
    corrections = _profile_list("asr_corrections") + _split_config_list(raw)

    seen: set[str] = set()
    deduped: List[str] = []
    for item in corrections:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _asr_corrections_block() -> str:
    corrections = _asr_corrections()
    if not corrections:
        return ""
    lines = "\n".join(f"- {item}" for item in corrections)
    return (
        "\nBefore translating, apply these source-text corrections when they fit the context. "
        "Do not mention the incorrect form in the output:\n"
        f"{lines}\n"
    )


def _glossary_entries() -> List[str]:
    return _profile_list("glossary") + _split_config_list(
        os.getenv("MT_GLOSSARY", getattr(cfg, "MT_GLOSSARY", ""))
    )


def _preserve_terms() -> List[str]:
    return _profile_list("preserve_terms") + _split_config_list(
        os.getenv("MT_PRESERVE_TERMS", getattr(cfg, "MT_PRESERVE_TERMS", ""))
    )


def _terminology_block() -> str:
    glossary = _glossary_entries()
    preserve_terms = _preserve_terms()
    parts: List[str] = []
    if glossary:
        parts.append(
            "Terminology glossary. Use these translations when the source term appears and the context fits:\n"
            + "\n".join(f"- {item}" for item in glossary)
        )
    if preserve_terms:
        parts.append(
            "Preserve these exact tokens unchanged unless grammar absolutely requires inflection:\n"
            + "\n".join(f"- {item}" for item in preserve_terms)
        )
    if not parts:
        return ""
    return "\n" + "\n\n".join(parts) + "\n"


def _parse_source_target_pairs(items: List[str]) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    for item in items:
        if "->" not in item:
            continue
        source, target = item.split("->", 1)
        source = _normalize_spaces(source)
        target = _normalize_spaces(target)
        if source and target:
            pairs.append({"source": source, "target": target})
    return pairs
