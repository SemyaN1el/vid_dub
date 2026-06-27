"""Защиты синтеза речи: babble-guard, повторный синтез по ASR-проверке (ASR-retry) и обрезка хвостовых артефактов коротких сегментов."""

import os
import re
from difflib import SequenceMatcher
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Tuple

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from src.tts_text import _ends_with_terminal_punctuation
from src.tts_timing import _edge_silence_ms


def _compute_short_segment_tail_trim_ms(
    original_ms: int,
    corrected_ms: int,
    short_segment_sec: float,
    min_overhang_ms: int,
    max_trim_ms: int,
    max_trim_ratio: float
) -> int:
    """Подрезает хвост только у коротких сегментов с явным переизбытком длины."""
    if original_ms <= 0 or corrected_ms <= 0:
        return 0

    if original_ms / 1000.0 > short_segment_sec:
        return 0

    overhang_ms = corrected_ms - original_ms
    if overhang_ms < min_overhang_ms:
        return 0

    trim_ms = min(
        int(overhang_ms),
        int(max_trim_ms),
        int(corrected_ms * max_trim_ratio)
    )
    if trim_ms <= 0:
        return 0

    min_remaining_ms = max(int(original_ms * 0.75), 450)
    if corrected_ms - trim_ms < min_remaining_ms:
        return 0

    return trim_ms


def _compute_safe_tail_trim_ms(
    original_ms: int,
    corrected_ms: int,
    trailing_silence_ms: int,
    short_segment_sec: float,
    min_overhang_ms: int,
    max_trim_ms: int,
    max_trim_ratio: float
) -> int:
    """
    Разрешает tail trim только за счёт тишины в конце.
    Последние фонемы и естественный release речи не трогаем.
    """
    trim_candidate_ms = _compute_short_segment_tail_trim_ms(
        original_ms=original_ms,
        corrected_ms=corrected_ms,
        short_segment_sec=short_segment_sec,
        min_overhang_ms=min_overhang_ms,
        max_trim_ms=max_trim_ms,
        max_trim_ratio=max_trim_ratio
    )
    if trim_candidate_ms <= 0 or trailing_silence_ms <= 0:
        return 0

    keep_tail_silence_ms = max(60, min(140, original_ms // 10))
    trim_budget_ms = max(0, trailing_silence_ms - keep_tail_silence_ms)
    if trim_budget_ms <= 0:
        return 0

    return min(trim_candidate_ms, trim_budget_ms)


def _normalize_word_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _normalized_text_similarity(expected_text: str, recognized_text: str) -> float:
    expected_norm = " ".join(_normalize_word_tokens(expected_text))
    recognized_norm = " ".join(_normalize_word_tokens(recognized_text))
    if not expected_norm or not recognized_norm:
        return 0.0
    return SequenceMatcher(None, expected_norm, recognized_norm).ratio()


def _token_overlap_stats(
    expected_words: List[str],
    recognized_words: List[str],
) -> Tuple[int, float, float]:
    if not expected_words or not recognized_words:
        return 0, 0.0, 0.0

    expected_counts: Dict[str, int] = {}
    recognized_counts: Dict[str, int] = {}
    for word in expected_words:
        expected_counts[word] = expected_counts.get(word, 0) + 1
    for word in recognized_words:
        recognized_counts[word] = recognized_counts.get(word, 0) + 1

    overlap = 0
    for word, expected_count in expected_counts.items():
        overlap += min(expected_count, recognized_counts.get(word, 0))

    recall = overlap / max(1, len(expected_words))
    precision = overlap / max(1, len(recognized_words))
    return overlap, recall, precision


def _contains_expected_suffix(
    expected_text: str,
    recognized_text: str,
    anchor_words: int
) -> bool:
    expected_words = _normalize_word_tokens(expected_text)
    recognized_words = _normalize_word_tokens(recognized_text)
    if not expected_words or not recognized_words:
        return False

    anchor_size = min(anchor_words, len(expected_words))
    anchor = expected_words[-anchor_size:]
    for idx in range(0, len(recognized_words) - anchor_size + 1):
        if recognized_words[idx:idx + anchor_size] == anchor:
            return True
    return False


def _has_extra_trailing_words(
    expected_text: str,
    recognized_text: str,
    anchor_words: int
) -> bool:
    expected_words = _normalize_word_tokens(expected_text)
    recognized_words = _normalize_word_tokens(recognized_text)
    if len(recognized_words) <= len(expected_words):
        return False
    if not _contains_expected_suffix(expected_text, recognized_text, anchor_words):
        return False

    anchor_size = min(anchor_words, len(expected_words))
    anchor = expected_words[-anchor_size:]
    trailing_words = 0
    for idx in range(0, len(recognized_words) - anchor_size + 1):
        if recognized_words[idx:idx + anchor_size] == anchor:
            trailing_words = len(recognized_words) - (idx + anchor_size)
    return trailing_words > 0


def _segment_recognition_score(
    expected_text: str,
    recognized_text: str,
    anchor_words: int
) -> Dict[str, Any]:
    similarity = _normalized_text_similarity(expected_text, recognized_text)
    has_extra_tail = _has_extra_trailing_words(expected_text, recognized_text, anchor_words)
    has_suffix = _contains_expected_suffix(expected_text, recognized_text, anchor_words)

    score = similarity
    if has_extra_tail:
        score -= 0.35
    if not has_suffix:
        score -= 0.15

    return {
        "recognized_text": recognized_text,
        "similarity": similarity,
        "has_extra_tail": has_extra_tail,
        "has_suffix": has_suffix,
        "score": score,
    }


def _recognition_eval_rank(eval_result: Dict[str, Any] | None) -> tuple[int, int, float, float]:
    if not eval_result:
        return (0, 0, 0.0, 0.0)
    return (
        0 if eval_result.get("has_extra_tail") else 1,
        1 if eval_result.get("has_suffix", False) else 0,
        float(eval_result.get("score") or 0.0),
        float(eval_result.get("similarity") or 0.0),
    )


def _is_better_recognition_eval(
    candidate_eval: Dict[str, Any] | None,
    incumbent_eval: Dict[str, Any] | None,
) -> bool:
    if candidate_eval is None:
        return False
    if incumbent_eval is None:
        return True
    return _recognition_eval_rank(candidate_eval) > _recognition_eval_rank(incumbent_eval)


def _find_trailing_speech_island_trim_ms(
    audio: AudioSegment,
    min_gap_ms: int,
    min_island_ms: int,
    max_island_ms: int,
    search_window_ms: int,
    max_trim_ms: int,
    min_silence_len: int = 40,
    silence_margin_db: float = 18.0
) -> int:
    if len(audio) == 0:
        return 0

    silence_threshold = max(-50.0, audio.dBFS - silence_margin_db)
    ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_threshold,
        seek_step=2
    )
    if len(ranges) < 2:
        return 0

    prev_start_ms, prev_end_ms = ranges[-2]
    last_start_ms, last_end_ms = ranges[-1]
    last_island_ms = max(0, last_end_ms - last_start_ms)
    gap_ms = max(0, last_start_ms - prev_end_ms)
    trim_ms = max(0, len(audio) - last_start_ms)

    if last_island_ms < min_island_ms or last_island_ms > max_island_ms:
        return 0
    if gap_ms < min_gap_ms:
        return 0
    if trim_ms > max_trim_ms or trim_ms > search_window_ms:
        return 0

    return trim_ms


def _transcribe_short_audio(
    model_asr,
    audio: AudioSegment,
    language: str
) -> str:
    if len(audio) == 0:
        return ""

    with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        audio.export(tmp_path, format="wav")
        use_fp16 = False
        model_device = getattr(model_asr, "device", None)
        if model_device is not None:
            device_type = getattr(model_device, "type", str(model_device))
            use_fp16 = str(device_type).startswith("cuda")
        result = model_asr.transcribe(tmp_path, language=language, fp16=use_fp16)
        return " ".join(
            segment["text"].strip()
            for segment in result.get("segments", [])
            if segment.get("text")
        ).strip()
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _trim_trailing_speech_island_fast(
    audio: AudioSegment,
    expected_text: str,
    original_ms: int,
    segment_duration_sec: float,
    tts_group_size: int,
    max_segment_sec: float,
    min_overhang_ms: int,
    min_gap_ms: int,
    min_island_ms: int,
    max_island_ms: int,
    search_window_ms: int,
    max_trim_ms: int
) -> tuple[AudioSegment, Dict[str, Any] | None]:
    if len(audio) == 0:
        return audio, None
    if tts_group_size != 1 or segment_duration_sec > max_segment_sec:
        return audio, None
    if not _ends_with_terminal_punctuation(expected_text):
        return audio, None

    overhang_ms = len(audio) - max(0, original_ms)
    if overhang_ms < min_overhang_ms:
        return audio, None

    trim_ms = _find_trailing_speech_island_trim_ms(
        audio=audio,
        min_gap_ms=min_gap_ms,
        min_island_ms=min_island_ms,
        max_island_ms=max_island_ms,
        search_window_ms=search_window_ms,
        max_trim_ms=max_trim_ms
    )
    if trim_ms <= 0:
        return audio, None

    return audio[:-trim_ms], {
        "trim_ms": trim_ms,
        "overhang_ms": overhang_ms,
    }


def _trim_trailing_babble(
    audio: AudioSegment,
    expected_text: str,
    model_asr,
    language: str,
    segment_duration_sec: float,
    tts_group_size: int,
    max_segment_sec: float,
    min_gap_ms: int,
    min_island_ms: int,
    max_island_ms: int,
    search_window_ms: int,
    max_trim_ms: int,
    anchor_words: int,
    min_score_gain: float = 0.08,
) -> tuple[AudioSegment, Dict[str, Any] | None]:
    if model_asr is None or len(audio) == 0:
        return audio, None
    if tts_group_size != 1 or segment_duration_sec > max_segment_sec:
        return audio, None
    if not _ends_with_terminal_punctuation(expected_text):
        return audio, None

    detected_trim_ms = _find_trailing_speech_island_trim_ms(
        audio=audio,
        min_gap_ms=min_gap_ms,
        min_island_ms=min_island_ms,
        max_island_ms=max_island_ms,
        search_window_ms=search_window_ms,
        max_trim_ms=max_trim_ms
    )

    recognized_before = _transcribe_short_audio(model_asr, audio, language)
    score_before = _segment_recognition_score(
        expected_text,
        recognized_before,
        anchor_words
    )

    baseline_suspicious = (
        detected_trim_ms > 0
        or score_before["has_extra_tail"]
        or not score_before["has_suffix"]
    )
    if not baseline_suspicious:
        return audio, None

    if detected_trim_ms <= 0 or detected_trim_ms >= len(audio):
        return audio, None

    trimmed_audio = audio[:-detected_trim_ms]
    recognized_after = _transcribe_short_audio(model_asr, trimmed_audio, language)
    score_after = _segment_recognition_score(
        expected_text,
        recognized_after,
        anchor_words
    )
    if score_after["has_extra_tail"]:
        return audio, None
    if not score_after["has_suffix"]:
        return audio, None

    improved = (
        score_after["score"] >= score_before["score"] + min_score_gain
        and score_after["similarity"] >= score_before["similarity"]
    )
    if score_before["has_extra_tail"]:
        improved = (
            score_after["score"] >= score_before["score"] + (min_score_gain * 0.5)
            and score_after["similarity"] + 0.02 >= score_before["similarity"]
        )
    if not improved:
        return audio, None

    info = {
        "trim_ms": detected_trim_ms,
        "recognized_before": recognized_before,
        "recognized_after": recognized_after,
        "score_before": round(score_before["score"], 4),
        "score_after": round(score_after["score"], 4),
        "reason": "extra_tail" if score_before["has_extra_tail"] else "score_gain",
    }
    return trimmed_audio, info
