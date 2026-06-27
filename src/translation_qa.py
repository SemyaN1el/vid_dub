"""Deterministic operational QA for translated dubbing segments."""

import os
import re
from typing import Any, Dict, List

import config as cfg
from src.translation_common import _normalize_spaces
from src.translation_length import _translation_segment_metadata
from src.translation_profiles import _preserve_terms


def _numbers_in_text(text: str) -> set[str]:
    return {
        match.group(0).replace(",", ".")
        for match in re.finditer(r"\b\d+(?:[.,]\d+)?%?\b", text or "")
    }


def _latin_tokens(text: str) -> List[str]:
    return re.findall(r"\b[A-Za-z][A-Za-z0-9_.+-]*\b", text or "")


def _qa_max_char_ratio() -> float:
    return float(os.getenv(
        "MT_QA_MAX_CHAR_RATIO",
        str(getattr(cfg, "MT_QA_MAX_CHAR_RATIO", 1.65)),
    ))


def _qa_max_target_ratio() -> float:
    return float(os.getenv(
        "MT_QA_MAX_TARGET_RATIO",
        str(getattr(cfg, "MT_QA_MAX_TARGET_RATIO", 1.25)),
    ))


def analyze_translation_quality(
    translated_segments: List[Dict[str, Any]],
    *,
    tgt_lang: str = "rus_Cyrl",
) -> Dict[str, Any]:
    """Cheap deterministic QA for MT output before TTS.

    This does not judge translation quality. It flags operational risks that
    usually hurt dubbing: empty text, missing numbers, leftover Latin words,
    and translations that are likely too long for the source timing window.
    """
    issues_by_segment: List[Dict[str, Any]] = []
    char_ratios: List[float] = []
    target_ratios: List[float] = []
    preserve_terms = {term.lower() for term in _preserve_terms()}

    for idx, segment in enumerate(translated_segments):
        text = _normalize_spaces(str(segment.get("text", "")))
        source_text = _normalize_spaces(str(segment.get("original_text", "")))
        issues: List[str] = []

        if not text:
            issues.append("empty_translation")

        source_chars = max(1, len(source_text))
        char_ratio = len(text) / source_chars
        char_ratios.append(char_ratio)
        if len(source_text) >= 12 and char_ratio > _qa_max_char_ratio():
            issues.append("high_char_ratio")

        metadata = _translation_segment_metadata({
            "text": source_text,
            "start": segment.get("start", 0.0),
            "end": segment.get("end", 0.0),
            "source_duration_sec": segment.get("source_duration_sec"),
            "source_word_count": segment.get("source_word_count"),
        })
        soft_target = metadata.get("soft_target_max_chars")
        target_ratio = None
        if soft_target:
            target_ratio = len(text) / max(1, int(soft_target))
            target_ratios.append(target_ratio)
            if target_ratio > _qa_max_target_ratio():
                issues.append("over_timing_soft_limit")

        source_numbers = _numbers_in_text(source_text)
        translated_numbers = _numbers_in_text(text)
        missing_numbers = sorted(source_numbers - translated_numbers)
        if missing_numbers:
            issues.append("missing_number")

        latin_leaks: List[str] = []
        if tgt_lang.startswith("rus"):
            for token in _latin_tokens(text):
                normalized = token.lower()
                if normalized in preserve_terms:
                    continue
                if token.isupper() and len(token) <= 8:
                    continue
                if re.search(r"\d", token):
                    continue
                latin_leaks.append(token)
            if latin_leaks:
                issues.append("latin_word_in_translation")

        if issues:
            issues_by_segment.append({
                "index": idx,
                "start": segment.get("start"),
                "end": segment.get("end"),
                "issues": issues,
                "text": text,
                "original_text": source_text,
                "char_ratio": round(char_ratio, 3),
                "target_ratio": round(target_ratio, 3) if target_ratio is not None else None,
                "soft_target_max_chars": soft_target,
                "missing_numbers": missing_numbers,
                "latin_words": latin_leaks,
            })

    return {
        "translated_count": len(translated_segments),
        "issue_count": len(issues_by_segment),
        "empty_count": sum("empty_translation" in item["issues"] for item in issues_by_segment),
        "high_char_ratio_count": sum("high_char_ratio" in item["issues"] for item in issues_by_segment),
        "over_timing_soft_limit_count": sum("over_timing_soft_limit" in item["issues"] for item in issues_by_segment),
        "missing_number_count": sum("missing_number" in item["issues"] for item in issues_by_segment),
        "latin_word_count": sum("latin_word_in_translation" in item["issues"] for item in issues_by_segment),
        "mean_char_ratio": round(sum(char_ratios) / len(char_ratios), 4) if char_ratios else 0.0,
        "mean_target_ratio": round(sum(target_ratios) / len(target_ratios), 4) if target_ratios else 0.0,
        "segments": issues_by_segment,
    }
