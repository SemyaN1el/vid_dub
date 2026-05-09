import logging
import re
from copy import deepcopy
from typing import Any, Dict, List


logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    """Чистит только действительно мешающие символы, сохраняя интонацию."""
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    text = re.sub(r"[^\w\s.,!?;:()\"'%\-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _duration_bucket(duration_sec: float) -> str:
    if duration_sec < 4.0:
        return "short"
    if duration_sec < 7.0:
        return "medium"
    return "long"


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def _is_question(text: str) -> bool:
    return text.strip().endswith("?")


def _strip_trailing_closers(text: str) -> str:
    stripped = text.strip()
    while stripped and stripped[-1] in "\"')]}»”":
        stripped = stripped[:-1].rstrip()
    return stripped


def _ends_with_terminal_punctuation(text: str) -> bool:
    stripped = _strip_trailing_closers(text)
    return bool(stripped) and stripped[-1] in ".!?…"


def _ends_with_continuation_punctuation(text: str) -> bool:
    stripped = _strip_trailing_closers(text)
    return bool(stripped) and stripped[-1] in ",;:-"


def _starts_with_lowercase_letter(text: str) -> bool:
    for char in text.strip():
        if char.isalpha():
            return char.islower()
    return False


def _preferred_terminal_punctuation(text: str) -> str:
    stripped = _strip_trailing_closers(text)
    if stripped.endswith("?"):
        return "?"
    if stripped.endswith("!"):
        return "!"
    return "."


def _replace_terminal_suffix(text: str, suffix: str) -> str:
    base = _clean_text(text).rstrip(" .,!?:;_-")
    if not base:
        return _clean_text(text)
    return f"{base}{suffix}"


def _stabilize_tts_text(
    text: str,
    *,
    original_text: str = "",
    next_text: str = "",
    gap_after_sec: float | None = None,
    pause_after_sec: float | None = None,
    tts_group_size: int = 1,
    force_terminal: bool = False,
) -> str:
    cleaned = _clean_text(text)
    if not cleaned or tts_group_size != 1:
        return cleaned
    if _ends_with_terminal_punctuation(cleaned):
        return cleaned
    if next_text and _starts_with_lowercase_letter(next_text):
        return cleaned

    should_add_terminal = force_terminal
    if not should_add_terminal:
        if original_text and _ends_with_terminal_punctuation(original_text):
            should_add_terminal = True
        elif pause_after_sec is not None and pause_after_sec >= 0.28:
            should_add_terminal = True
        elif gap_after_sec is not None and gap_after_sec >= 0.48:
            should_add_terminal = True
        elif (
            next_text.strip()
            and not _starts_with_lowercase_letter(next_text)
            and len(cleaned) >= 48
            and not _ends_with_continuation_punctuation(cleaned)
        ):
            should_add_terminal = True

    if not should_add_terminal:
        return cleaned

    terminal = _preferred_terminal_punctuation(original_text or cleaned)
    base = cleaned.rstrip(" ,;:-")
    return f"{base}{terminal}" if base else cleaned


def _build_tts_retry_text_variants(
    text: str,
    *,
    original_text: str = "",
    next_text: str = "",
    gap_after_sec: float | None = None,
    pause_after_sec: float | None = None,
    tts_group_size: int = 1,
    tts_backend_name: str = "",
) -> List[str]:
    variants: List[str] = []
    base_clean = _clean_text(text)
    if base_clean:
        variants.append(base_clean)

    backend_name = (tts_backend_name or "").strip().lower()
    if backend_name != "xtts":
        return variants

    stabilized = _stabilize_tts_text(
        text,
        original_text=original_text,
        next_text=next_text,
        gap_after_sec=gap_after_sec,
        pause_after_sec=pause_after_sec,
        tts_group_size=tts_group_size,
        force_terminal=False,
    )
    boundary_detected = (
        stabilized != base_clean
        or _ends_with_terminal_punctuation(base_clean)
        or _ends_with_terminal_punctuation(original_text)
    )
    if not boundary_detected:
        return variants

    preferred = _preferred_terminal_punctuation(original_text or base_clean)
    suffix_candidates: List[str]
    if preferred == "?":
        suffix_candidates = ["?", "!", "_"]
    elif preferred == "!":
        suffix_candidates = ["!", "?", "_"]
    else:
        suffix_candidates = ["!", "?", "_"]

    for suffix in suffix_candidates:
        candidate = _replace_terminal_suffix(base_clean, suffix)
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants


def _join_segment_text(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    separator = "" if left.endswith(("-", "—", "–", "/")) else " "
    return f"{left}{separator}{right}".strip()


def _same_speaker(segment_a: Dict[str, Any], segment_b: Dict[str, Any]) -> bool:
    speaker_a = segment_a.get("speaker_id")
    speaker_b = segment_b.get("speaker_id")
    return not (speaker_a and speaker_b and speaker_a != speaker_b)


def _should_group_for_tts(
    current: Dict[str, Any],
    nxt: Dict[str, Any],
    gap_sec: float,
    max_gap_sec: float,
    max_chars: int,
    max_duration_sec: float
) -> bool:
    if gap_sec < 0 or gap_sec > max_gap_sec:
        return False

    if not _same_speaker(current, nxt):
        return False

    current_start = float(current.get("start", 0.0))
    next_end = float(nxt.get("end", nxt.get("start", 0.0)))
    group_duration_sec = max(0.0, next_end - current_start)
    if group_duration_sec > max_duration_sec:
        return False

    current_text = str(current.get("text") or "").strip()
    current_original_text = str(current.get("original_text") or "").strip()
    next_text = str(nxt.get("text") or "").strip()
    next_original_text = str(nxt.get("original_text") or "").strip()

    combined_chars = len(_join_segment_text(current_text, next_text))
    if combined_chars > max_chars:
        return False

    continuation = (
        _ends_with_continuation_punctuation(current_text)
        or _ends_with_continuation_punctuation(current_original_text)
        or _starts_with_lowercase_letter(next_text)
        or _starts_with_lowercase_letter(next_original_text)
    )
    if not continuation:
        return False

    terminal_current = (
        _ends_with_terminal_punctuation(current_text)
        and _ends_with_terminal_punctuation(current_original_text)
    )
    if terminal_current and not (
        _starts_with_lowercase_letter(next_text)
        or _starts_with_lowercase_letter(next_original_text)
    ):
        return False

    return True


def _prepare_tts_segments(
    segments: List[Dict[str, Any]],
    enable_grouping: bool,
    max_gap_sec: float,
    max_group_segments: int,
    max_group_chars: int,
    max_group_duration_sec: float
) -> List[Dict[str, Any]]:
    working_segments = [deepcopy(segment) for segment in segments]
    if not enable_grouping or len(working_segments) < 2:
        return working_segments

    grouped_segments: List[Dict[str, Any]] = []
    current_group = working_segments[0]
    current_group["tts_group_indices"] = [0]
    current_group["tts_group_size"] = 1
    current_group["tts_group_gap_sec"] = 0.0

    for idx, next_segment in enumerate(working_segments[1:], start=1):
        gap_sec = float(next_segment.get("start", 0.0)) - float(current_group.get("end", 0.0))
        can_group = (
            int(current_group.get("tts_group_size", 1)) < max_group_segments
            and _should_group_for_tts(
                current=current_group,
                nxt=next_segment,
                gap_sec=gap_sec,
                max_gap_sec=max_gap_sec,
                max_chars=max_group_chars,
                max_duration_sec=max_group_duration_sec
            )
        )

        if can_group:
            current_group["text"] = _join_segment_text(
                str(current_group.get("text") or ""),
                str(next_segment.get("text") or "")
            )
            current_group["original_text"] = _join_segment_text(
                str(current_group.get("original_text") or ""),
                str(next_segment.get("original_text") or "")
            )
            current_group["end"] = next_segment.get("end", current_group.get("end"))
            current_group["original_end"] = next_segment.get(
                "original_end",
                next_segment.get("end", current_group.get("end"))
            )
            if not current_group.get("speaker_id") and next_segment.get("speaker_id"):
                current_group["speaker_id"] = next_segment["speaker_id"]
            current_group["tts_group_indices"].append(idx)
            current_group["tts_group_size"] = len(current_group["tts_group_indices"])
            current_group["tts_group_gap_sec"] = float(current_group["tts_group_gap_sec"]) + max(0.0, gap_sec)
            logger.info(
                "TTS grouping: сегменты %s -> %s (gap %.3fs)",
                current_group["tts_group_indices"][:-1],
                current_group["tts_group_indices"],
                gap_sec
            )
            continue

        grouped_segments.append(current_group)
        current_group = next_segment
        current_group["tts_group_indices"] = [idx]
        current_group["tts_group_size"] = 1
        current_group["tts_group_gap_sec"] = 0.0

    grouped_segments.append(current_group)
    return grouped_segments
