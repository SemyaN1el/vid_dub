import os
import re
import logging
import subprocess
from copy import deepcopy
from difflib import SequenceMatcher
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Tuple

import soundfile as sf
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range, speedup
from pydub.silence import detect_nonsilent
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    """Чистит только действительно мешающие символы, сохраняя интонацию."""
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    text = re.sub(r"[^\w\s.,!?;:()\"'%\-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _active_speech_dbfs(
    audio: AudioSegment,
    min_silence_len: int = 120,
    silence_margin_db: float = 16.0
) -> float | None:
    """Оценивает громкость только активной речи, без длинных пауз."""
    if len(audio) == 0:
        return None

    silence_threshold = max(-50.0, audio.dBFS - silence_margin_db)
    ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_threshold,
        seek_step=5
    )
    if not ranges:
        return audio.dBFS if audio.rms else None

    active_audio = audio[0:0]
    for start_ms, end_ms in ranges:
        if end_ms > start_ms:
            active_audio += audio[start_ms:end_ms]

    return active_audio.dBFS if active_audio.rms else None


def _active_speech_stats(
    audio: AudioSegment,
    min_silence_len: int = 120,
    silence_margin_db: float = 16.0
) -> Dict[str, float | None]:
    """Оценивает локальные параметры речи внутри сегмента."""
    if len(audio) == 0:
        return {
            "active_dbfs": None,
            "active_ratio": 0.0,
            "peak_dbfs": None,
            "full_dbfs": None,
        }

    silence_threshold = max(-50.0, audio.dBFS - silence_margin_db)
    ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_threshold,
        seek_step=5
    )
    if not ranges:
        return {
            "active_dbfs": audio.dBFS if audio.rms else None,
            "active_ratio": 1.0 if audio.rms else 0.0,
            "peak_dbfs": audio.max_dBFS if audio.rms else None,
            "full_dbfs": audio.dBFS if audio.rms else None,
        }

    active_audio = audio[0:0]
    active_ms = 0
    for start_ms, end_ms in ranges:
        if end_ms > start_ms:
            active_audio += audio[start_ms:end_ms]
            active_ms += end_ms - start_ms

    return {
        "active_dbfs": active_audio.dBFS if len(active_audio) and active_audio.rms else None,
        "active_ratio": active_ms / len(audio) if len(audio) else 0.0,
        "peak_dbfs": audio.max_dBFS if audio.rms else None,
        "full_dbfs": audio.dBFS if audio.rms else None,
    }


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


def _reference_route_score(segment: Dict[str, Any], clip: Dict[str, Any]) -> float:
    """Оценивает, насколько reference clip подходит для текущего сегмента."""
    segment_text = str(segment.get("original_text") or segment.get("text") or "").strip()
    clip_text = str(clip.get("text") or "").strip()

    segment_duration = max(0.2, float(segment.get("end", 0.0) - segment.get("start", 0.0)))
    clip_duration = max(0.2, float(clip.get("duration_sec") or segment_duration))

    segment_bucket = _duration_bucket(segment_duration)
    clip_bucket = str(clip.get("duration_bucket") or _duration_bucket(clip_duration))

    segment_words = _word_count(segment_text)
    clip_words = _word_count(clip_text)
    segment_question = _is_question(segment_text)
    clip_question = _is_question(clip_text)

    score = 0.0
    score += max(-1.0, 1.8 - abs(clip_duration - segment_duration) / max(segment_duration, 1.2))

    if segment_bucket == clip_bucket:
        score += 0.7
    elif {segment_bucket, clip_bucket} in ({"short", "medium"}, {"medium", "long"}):
        score += 0.25
    else:
        score -= 0.15

    if segment_question and clip_question:
        score += 0.45
    elif segment_question != clip_question:
        score -= 0.2

    if segment_words and clip_words:
        score += max(-0.4, 0.8 - abs(clip_words - segment_words) / max(segment_words, 4))

    if "," in segment_text and "," in clip_text:
        score += 0.15
    if "!" in segment_text and "!" in clip_text:
        score += 0.15

    base_selection_score = float(clip.get("selection_score") or clip.get("score") or 0.0)
    score += min(base_selection_score, 6.0) * 0.08
    return score


def _pick_reference_subset(
    segment: Dict[str, Any],
    clips: List[Dict[str, Any]],
    desired_refs: int
) -> List[Dict[str, Any]]:
    """Выбирает небольшой, но разнообразный поднабор reference-клипов."""
    if not clips:
        return []

    desired_refs = max(1, min(desired_refs, len(clips)))
    ranked_clips = sorted(
        clips,
        key=lambda clip: _reference_route_score(segment, clip),
        reverse=True
    )

    selected: List[Dict[str, Any]] = []
    while ranked_clips and len(selected) < desired_refs:
        best_clip = None
        best_score = float("-inf")

        for clip in ranked_clips:
            if any(item["path"] == clip["path"] for item in selected):
                continue

            score = _reference_route_score(segment, clip)
            selected_buckets = {item.get("duration_bucket") for item in selected}
            if selected and clip.get("duration_bucket") not in selected_buckets:
                score += 0.18

            if score > best_score:
                best_score = score
                best_clip = clip

        if best_clip is None:
            break
        selected.append(best_clip)

    return selected


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


def _edge_silence_ms(
    audio: AudioSegment,
    min_silence_len: int = 30,
    silence_margin_db: float = 18.0
) -> Dict[str, int]:
    """Оценивает тишину по краям сегмента, чтобы не срезать живую речь."""
    if len(audio) == 0:
        return {"leading": 0, "trailing": 0}

    silence_threshold = max(-50.0, audio.dBFS - silence_margin_db)
    ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_threshold,
        seek_step=2
    )
    if not ranges:
        return {"leading": len(audio), "trailing": len(audio)}

    leading_ms = max(0, int(ranges[0][0]))
    trailing_ms = max(0, int(len(audio) - ranges[-1][1]))
    return {"leading": leading_ms, "trailing": trailing_ms}


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
    anchor_words: int
) -> tuple[AudioSegment, Dict[str, Any] | None]:
    if model_asr is None or len(audio) == 0:
        return audio, None
    if tts_group_size != 1 or segment_duration_sec > max_segment_sec:
        return audio, None
    if not _ends_with_terminal_punctuation(expected_text):
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

    recognized_before = _transcribe_short_audio(model_asr, audio, language)
    if not _has_extra_trailing_words(expected_text, recognized_before, anchor_words):
        return audio, None

    trimmed_audio = audio[:-trim_ms]
    recognized_after = _transcribe_short_audio(model_asr, trimmed_audio, language)
    if _has_extra_trailing_words(expected_text, recognized_after, anchor_words):
        return audio, None
    if not _contains_expected_suffix(expected_text, recognized_after, anchor_words):
        return audio, None

    info = {
        "trim_ms": trim_ms,
        "recognized_before": recognized_before,
        "recognized_after": recognized_after,
    }
    return trimmed_audio, info


def _build_atempo_chain(playback_speed: float) -> List[float]:
    """Разбивает коэффициент скорости на допустимую цепочку atempo."""
    if playback_speed <= 0:
        raise ValueError("playback_speed must be positive")

    factors: List[float] = []
    remainder = playback_speed

    while remainder > 2.0:
        factors.append(2.0)
        remainder /= 2.0

    while remainder < 0.5:
        factors.append(0.5)
        remainder /= 0.5

    factors.append(remainder)
    return factors


def _time_stretch_ffmpeg(
    audio: AudioSegment,
    playback_speed: float
) -> AudioSegment:
    """
    Меняет темп через ffmpeg/atempo.
    Это надёжнее для концовок фраз, чем pydub.speedup, который выкидывает чанки.
    """
    if len(audio) == 0 or abs(playback_speed - 1.0) < 0.01:
        return audio

    filter_chain = ",".join(
        f"atempo={factor:.5f}"
        for factor in _build_atempo_chain(playback_speed)
    )

    src_path = dst_path = None
    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as src_tmp:
            src_path = src_tmp.name
        with NamedTemporaryFile(suffix=".wav", delete=False) as dst_tmp:
            dst_path = dst_tmp.name

        audio.export(src_path, format="wav")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            src_path,
            "-filter:a",
            filter_chain,
            dst_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ffmpeg atempo failed")

        return AudioSegment.from_wav(dst_path)
    except Exception as exc:
        logger.warning(
            "FFmpeg atempo не сработал, fallback на pydub.speedup: %s",
            exc
        )
        return speedup(audio, playback_speed=playback_speed)
    finally:
        for path in (src_path, dst_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def _select_segment_references(
    segment: Dict[str, Any],
    speaker_wav,
    speaker_profile: Dict[str, Any] | None,
    max_refs_per_segment: int,
    short_segment_sec: float,
    min_segment_sec: float,
    min_segment_words: int,
    confidence_margin: float
) -> List[str]:
    """Выбирает 1-2 наиболее подходящих референса под конкретную реплику."""
    if isinstance(speaker_wav, list):
        fallback_paths = [path for path in speaker_wav if path and os.path.exists(path)]
    else:
        fallback_paths = [speaker_wav] if speaker_wav and os.path.exists(speaker_wav) else []

    if not fallback_paths:
        return []

    if not speaker_profile:
        return fallback_paths

    stable_clips = [
        clip
        for clip in speaker_profile.get("clips", [])
        if clip.get("path") and os.path.exists(clip["path"])
    ]
    routing_clips = [
        clip
        for clip in (
            speaker_profile.get("routing_clips")
            or speaker_profile.get("clips", [])
        )
        if clip.get("path") and os.path.exists(clip["path"])
    ]
    if not routing_clips:
        return fallback_paths

    segment_text = str(segment.get("original_text") or segment.get("text") or "").strip()
    segment_words = _word_count(segment_text)
    segment_duration = max(0.2, float(segment.get("end", 0.0) - segment.get("start", 0.0)))
    if (
        segment_duration > short_segment_sec
        or segment_duration < min_segment_sec
        or segment_words < min_segment_words
    ):
        return fallback_paths

    desired_refs = max(1, min(max_refs_per_segment, len(routing_clips)))
    stable_selected = _pick_reference_subset(segment, stable_clips, desired_refs)
    stable_selected_paths = [clip["path"] for clip in stable_selected if clip.get("path")]

    stable_best_score = max(
        (_reference_route_score(segment, clip) for clip in stable_clips),
        default=float("-inf")
    )
    ranked_clips = sorted(
        routing_clips,
        key=lambda clip: _reference_route_score(segment, clip),
        reverse=True
    )
    if not ranked_clips:
        return stable_selected_paths or fallback_paths

    best_routing_score = _reference_route_score(segment, ranked_clips[0])
    if best_routing_score < stable_best_score + confidence_margin:
        return stable_selected_paths or fallback_paths

    selected = _pick_reference_subset(segment, ranked_clips, desired_refs)
    selected_paths = [clip["path"] for clip in selected if clip.get("path")]
    return selected_paths or stable_selected_paths or fallback_paths


def _match_segment_level(
    audio: AudioSegment,
    target_active_dbfs: float | None,
    max_boost_db: float,
    max_cut_db: float
) -> AudioSegment:
    """Подгоняет громкость сегмента к целевому уровню активной речи."""
    if target_active_dbfs is None or len(audio) == 0:
        return audio

    current_active_dbfs = _active_speech_dbfs(audio)
    if current_active_dbfs is None:
        return audio

    gain_delta = target_active_dbfs - current_active_dbfs
    gain_delta = max(-max_cut_db, min(max_boost_db, gain_delta))
    return audio.apply_gain(gain_delta)


def _compute_segment_target_level(
    source_audio: AudioSegment | None,
    segment_start_sec: float,
    segment_end_sec: float,
    default_target_active_dbfs: float | None,
    reference_gain_offset_db: float,
    strength: float,
    max_delta_db: float,
    padding_ms: int,
    min_active_ratio: float
) -> tuple[float | None, Dict[str, float | None]]:
    """
    Смещает целевой уровень сегмента к локальному уровню исходного вокала.
    Делает это мягко, чтобы не раскачать громкость на шумных или слабых кусках.
    """
    empty_stats = {
        "active_dbfs": None,
        "active_ratio": 0.0,
        "peak_dbfs": None,
        "full_dbfs": None,
    }
    if source_audio is None:
        return default_target_active_dbfs, empty_stats

    total_ms = len(source_audio)
    start_ms = max(0, int(segment_start_sec * 1000) - padding_ms)
    end_ms = min(total_ms, int(segment_end_sec * 1000) + padding_ms)
    if end_ms <= start_ms:
        return default_target_active_dbfs, empty_stats

    source_segment = source_audio[start_ms:end_ms]
    stats = _active_speech_stats(source_segment)
    source_active_dbfs = stats["active_dbfs"]
    active_ratio = float(stats["active_ratio"] or 0.0)

    if source_active_dbfs is None or active_ratio < min_active_ratio:
        return default_target_active_dbfs, stats

    local_target = source_active_dbfs + reference_gain_offset_db
    if default_target_active_dbfs is None:
        return local_target, stats

    delta = local_target - default_target_active_dbfs
    delta = max(-max_delta_db, min(max_delta_db, delta))
    adjusted_target = default_target_active_dbfs + delta * strength
    return adjusted_target, stats


def _apply_peak_ceiling(audio: AudioSegment, peak_ceiling_dbfs: float) -> AudioSegment:
    """Не даёт итоговому аудио доходить до клиппинга."""
    if len(audio) == 0 or audio.max_dBFS == float("-inf"):
        return audio
    if audio.max_dBFS <= peak_ceiling_dbfs:
        return audio
    return audio.apply_gain(peak_ceiling_dbfs - audio.max_dBFS)


def generate_audio_segment(
    tts_backend,
    text: str,
    output_path: str,
    speaker_wav,
    language: str,
    conditioning=None,
    speaker_profile: Dict[str, Any] | None = None,
) -> Tuple[str, float]:
    """
    Синтезирует аудио для одного текстового сегмента.

    Параметры:
        tts_backend: загруженный backend TTS
        text: текст для синтеза
        output_path: путь для сохранения сегмента
        speaker_wav: путь или список путей к референсам спикера
        language: язык синтеза ('ru', 'en', ...)
        conditioning: предвычисленное conditioning backend-а
        speaker_profile: профиль спикера с текстами референсов (опционально)

    Возвращает:
        Tuple[str, float]: (путь к файлу, длительность в секундах)
    """
    if conditioning is None:
        if isinstance(speaker_wav, list):
            reference_paths = [path for path in speaker_wav if path and os.path.exists(path)]
        else:
            reference_paths = [speaker_wav] if speaker_wav and os.path.exists(speaker_wav) else []
        conditioning = tts_backend.prepare_conditioning(
            reference_paths=reference_paths,
            speaker_profile=speaker_profile
        )

    wav, sample_rate = tts_backend.synthesize(
        text=text,
        language=language,
        conditioning=conditioning
    )
    sf.write(output_path, wav, sample_rate)

    audio = AudioSegment.from_wav(output_path)

    duration = len(audio) / 1000.0
    return output_path, duration


def synthesize_segments_with_timing(
    tts_backend,
    segments: List[Dict[str, Any]],
    output_audio_path: str,
    speaker_wav,
    language: str,
    speaker_profile: Dict[str, Any] | None = None,
    reference_audio_path: str | None = None,
    source_vocals_path: str | None = None,
    segments_dir: str = "./data/output/temp/audio_segments",
    max_speedup_factor: float = 1.2,
    max_next_start_shift_sec: float | None = None,
    speedup_tail_padding_ms: int = 120,
    min_pause_between_segments: float = 0.2,
    fade_in_out_ms: int = 50,
    crossfade_ms: int = 30,
    max_shift_left_seconds: float = 0.5,
    threshold_compression: float = -15.0,
    ratio_compression: float = 2.0,
    attack_compression: int = 25,
    release_compression: int = 50,
    target_dBFS: float = -15.0,
    reference_gain_offset_db: float = 3.0,
    max_segment_boost_db: float = 6.0,
    max_segment_cut_db: float = 12.0,
    peak_ceiling_dbfs: float = -2.0,
    enable_final_compression: bool = False,
    enable_segment_routing: bool = False,
    short_segment_sec: float = 2.2,
    max_refs_per_segment: int = 2,
    min_segment_routing_sec: float = 0.9,
    min_segment_routing_words: int = 3,
    routing_confidence_margin: float = 0.45,
    enable_tts_grouping: bool = True,
    tts_grouping_max_gap_sec: float = 0.6,
    tts_grouping_max_segments: int = 2,
    tts_grouping_max_chars: int = 220,
    tts_grouping_max_duration_sec: float = 8.5,
    enable_tts_cheap_tail_guard: bool = True,
    tts_cheap_tail_guard_max_segment_sec: float = 3.2,
    tts_cheap_tail_guard_min_overhang_ms: int = 180,
    tts_cheap_tail_guard_min_gap_ms: int = 80,
    tts_cheap_tail_guard_min_island_ms: int = 80,
    tts_cheap_tail_guard_max_island_ms: int = 450,
    tts_cheap_tail_guard_search_window_ms: int = 900,
    tts_cheap_tail_guard_max_trim_ms: int = 700,
    enable_tts_babble_guard: bool = False,
    tts_babble_guard_model_name: str = "small",
    tts_babble_guard_device: str = "cpu",
    tts_babble_guard_max_segment_sec: float = 4.0,
    tts_babble_guard_min_gap_ms: int = 80,
    tts_babble_guard_min_island_ms: int = 80,
    tts_babble_guard_max_island_ms: int = 450,
    tts_babble_guard_search_window_ms: int = 900,
    tts_babble_guard_max_trim_ms: int = 700,
    tts_babble_guard_anchor_words: int = 2,
    enable_tts_asr_retry: bool = False,
    tts_asr_retry_max_segment_sec: float = 2.5,
    tts_asr_retry_attempts: int = 4,
    tts_asr_retry_min_score: float = 0.9,
    enable_short_segment_tail_trim: bool = False,
    short_segment_tail_trim_min_overhang_ms: int = 280,
    short_segment_tail_trim_max_ms: int = 500,
    short_segment_tail_trim_max_ratio: float = 0.22,
    enable_segment_matching: bool = False,
    segment_match_padding_ms: int = 120,
    segment_match_strength: float = 0.7,
    segment_match_max_delta_db: float = 4.0,
    segment_match_min_active_ratio: float = 0.35
) -> None:
    """
    Синтезирует дубляж с временно́й синхронизацией сегментов.

    Алгоритм:
        1. Предвычисляет conditioning latents один раз для всего пайплайна.
        2. Для каждого сегмента генерирует аудио и проверяет вписывается ли
           оно в отведённое время с учётом паузы до следующего сегмента.
        3. При необходимости ускоряет сегмент (не более max_speedup_factor).
        4. Применяет fade-in/out и кроссфейды между соседними сегментами.
        5. Нормализует итоговое аудио.
    """
    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

    original_segment_count = len(segments)
    segments = _prepare_tts_segments(
        segments=segments,
        enable_grouping=enable_tts_grouping,
        max_gap_sec=tts_grouping_max_gap_sec,
        max_group_segments=tts_grouping_max_segments,
        max_group_chars=tts_grouping_max_chars,
        max_group_duration_sec=tts_grouping_max_duration_sec
    )
    if len(segments) != original_segment_count:
        logger.info(
            "TTS grouping: %s -> %s сегментов",
            original_segment_count,
            len(segments)
        )

    babble_guard_model = None
    if enable_tts_babble_guard:
        import whisper

        logger.info(
            "Загружаем Whisper %s для TTS babble guard на %s...",
            tts_babble_guard_model_name,
            tts_babble_guard_device
        )
        babble_guard_model = whisper.load_model(tts_babble_guard_model_name).to(
            tts_babble_guard_device
        )

    if isinstance(speaker_wav, list):
        default_reference_paths = [path for path in speaker_wav if path and os.path.exists(path)]
    else:
        default_reference_paths = [speaker_wav] if speaker_wav and os.path.exists(speaker_wav) else []
    if not default_reference_paths:
        raise ValueError("Не найдено ни одного speaker reference для TTS.")

    conditioning_cache: Dict[Tuple[str, ...], Any] = {}
    logged_routes: set[Tuple[str, ...]] = set()

    def get_conditioning(reference_paths: List[str]) -> Any:
        key = tuple(reference_paths)
        cached = conditioning_cache.get(key)
        if cached is not None:
            return cached

        logger.info(
            "Подготавливаем conditioning для %s backend-а на %s reference clip(s)...",
            getattr(tts_backend, "name", "tts"),
            len(key)
        )
        conditioning_cache[key] = tts_backend.prepare_conditioning(
            reference_paths=list(key),
            speaker_profile=speaker_profile
        )
        return conditioning_cache[key]

    reference_path = reference_audio_path
    if reference_path is None:
        reference_path = default_reference_paths[0]
    reference_audio = AudioSegment.from_wav(reference_path)
    reference_active_dbfs = _active_speech_dbfs(reference_audio)
    source_vocals_audio = None
    if source_vocals_path and os.path.exists(source_vocals_path):
        source_vocals_audio = AudioSegment.from_wav(source_vocals_path).set_channels(1)
    target_active_dbfs = target_dBFS
    if reference_active_dbfs is not None:
        target_active_dbfs = reference_active_dbfs + reference_gain_offset_db
        logger.info(
            "Целевой уровень активной речи: %.2f dBFS (ref %.2f + %.2f)",
            target_active_dbfs,
            reference_active_dbfs,
            reference_gain_offset_db
        )
    else:
        logger.warning(
            "Не удалось оценить громкость референса, используем fallback %.2f dBFS",
            target_dBFS
        )

    full_duration_ms = int((max(s["end"] for s in segments) + 5) * 1000)
    full_audio = AudioSegment.silent(duration=full_duration_ms)

    # Сохраняем оригинальные временные метки
    for seg in segments:
        if "original_start" not in seg:
            seg["original_start"] = seg.get("start", 0.0)
        if "original_end" not in seg:
            seg["original_end"] = seg.get("end", seg.get("start", 0.0) + 1.0)

    prev_end_sec = 0.0

    for i, seg in tqdm(enumerate(segments), total=len(segments), desc="Синтез"):
        orig_start  = seg["original_start"]
        orig_end    = seg["original_end"]
        cur_start   = seg["start"]
        cur_start_ms = int(cur_start * 1000)

        # Сдвиг первого сегмента влево при наличии свободного времени
        if i == 0:
            avail_shift = min(max(0.0, orig_start - prev_end_sec), max_shift_left_seconds)
            if avail_shift > 0.05:
                seg["start"] -= avail_shift
                seg["start"]  = max(orig_start - max_shift_left_seconds, seg["start"])
                cur_start     = seg["start"]
                cur_start_ms  = int(cur_start * 1000)

        # Защита от слишком раннего сдвига
        if cur_start - orig_start < -max_shift_left_seconds:
            seg["start"] = orig_start
            cur_start    = orig_start
            cur_start_ms = int(orig_start * 1000)

        # Очистка текста
        clean = _clean_text(seg["text"])
        seg["cleaned_text"] = clean
        if not clean:
            logger.warning(f"[{i}] Пустой сегмент после очистки — пропуск.")
            continue

        selected_references = default_reference_paths
        if enable_segment_routing:
            routed_references = _select_segment_references(
                segment=seg,
                speaker_wav=default_reference_paths,
                speaker_profile=speaker_profile,
                max_refs_per_segment=max_refs_per_segment,
                short_segment_sec=short_segment_sec,
                min_segment_sec=min_segment_routing_sec,
                min_segment_words=min_segment_routing_words,
                confidence_margin=routing_confidence_margin
            )
            if routed_references:
                selected_references = routed_references

        route_key = tuple(selected_references)
        if route_key not in logged_routes:
            logged_routes.add(route_key)
            logger.info(
                "TTS route [%s]: %s",
                clean[:60],
                ", ".join(os.path.basename(path) for path in route_key)
            )

        conditioning = get_conditioning(selected_references)
        seg["tts_reference_paths"] = list(selected_references)
        seg["tts_reference_count"] = len(selected_references)

        # Генерация аудио
        seg_path = os.path.join(segments_dir, f"seg_{int(seg['start'] * 1000)}.wav")
        seg_path, _ = generate_audio_segment(
            tts_backend=tts_backend,
            text=clean,
            output_path=seg_path,
            speaker_wav=selected_references,
            language=language,
            conditioning=conditioning,
            speaker_profile=speaker_profile
        )

        seg_audio       = AudioSegment.from_wav(seg_path)
        generated_ms    = len(seg_audio)

        orig_start_val = seg["start"]
        orig_end_val   = seg.get("end", seg["start"] + 1.0)
        if orig_start_val > orig_end_val:
            orig_start_val, orig_end_val = orig_end_val, orig_start_val

        original_ms = int((orig_end_val - orig_start_val) * 1000)

        # Доступное время с учётом паузы до следующего сегмента
        if i < len(segments) - 1:
            next_start_ms   = int(segments[i + 1]["start"] * 1000)
            before_next_ms  = max(0, next_start_ms - int(orig_end_val * 1000))
        else:
            before_next_ms = 0

        extra_ms      = max(0, before_next_ms - int(min_pause_between_segments * 1000))
        available_ms  = max(100, original_ms + extra_ms)

        segment_target_dbfs = target_active_dbfs
        if enable_segment_matching:
            segment_target_dbfs, source_stats = _compute_segment_target_level(
                source_audio=source_vocals_audio,
                segment_start_sec=orig_start,
                segment_end_sec=orig_end,
                default_target_active_dbfs=target_active_dbfs,
                reference_gain_offset_db=reference_gain_offset_db,
                strength=segment_match_strength,
                max_delta_db=segment_match_max_delta_db,
                padding_ms=segment_match_padding_ms,
                min_active_ratio=segment_match_min_active_ratio
            )
            seg["source_active_dbfs"] = source_stats["active_dbfs"]
            seg["source_active_ratio"] = source_stats["active_ratio"]
            seg["segment_target_dbfs"] = segment_target_dbfs

        retry_candidate = (
            enable_tts_asr_retry
            and babble_guard_model is not None
            and int(seg.get("tts_group_size", 1) or 1) == 1
            and max(0.0, orig_end_val - orig_start_val) <= tts_asr_retry_max_segment_sec
            and _ends_with_terminal_punctuation(clean)
        )

        def finalize_candidate(candidate_audio: AudioSegment) -> tuple[AudioSegment, Dict[str, Any] | None, Dict[str, Any] | None, Dict[str, Any] | None]:
            corrected_candidate = candidate_audio
            if len(candidate_audio) > available_ms > 0:
                factor = min(len(candidate_audio) / available_ms, max_speedup_factor)
                speedup_input = candidate_audio + AudioSegment.silent(duration=max(0, speedup_tail_padding_ms))
                corrected_candidate = _time_stretch_ffmpeg(speedup_input, playback_speed=factor)
            if enable_short_segment_tail_trim:
                edge_silence_candidate = _edge_silence_ms(corrected_candidate)
                tail_trim_ms = _compute_safe_tail_trim_ms(
                    original_ms=original_ms,
                    corrected_ms=len(corrected_candidate),
                    trailing_silence_ms=edge_silence_candidate["trailing"],
                    short_segment_sec=short_segment_sec,
                    min_overhang_ms=short_segment_tail_trim_min_overhang_ms,
                    max_trim_ms=short_segment_tail_trim_max_ms,
                    max_trim_ratio=short_segment_tail_trim_max_ratio
                )
                if tail_trim_ms:
                    corrected_candidate = corrected_candidate[:-tail_trim_ms]

            corrected_candidate = _match_segment_level(
                corrected_candidate,
                target_active_dbfs=segment_target_dbfs,
                max_boost_db=max_segment_boost_db,
                max_cut_db=max_segment_cut_db
            )
            cheap_tail_info = None
            if enable_tts_cheap_tail_guard:
                corrected_candidate, cheap_tail_info = _trim_trailing_speech_island_fast(
                    audio=corrected_candidate,
                    expected_text=clean,
                    original_ms=original_ms,
                    segment_duration_sec=max(0.0, orig_end_val - orig_start_val),
                    tts_group_size=int(seg.get("tts_group_size", 1) or 1),
                    max_segment_sec=tts_cheap_tail_guard_max_segment_sec,
                    min_overhang_ms=tts_cheap_tail_guard_min_overhang_ms,
                    min_gap_ms=tts_cheap_tail_guard_min_gap_ms,
                    min_island_ms=tts_cheap_tail_guard_min_island_ms,
                    max_island_ms=tts_cheap_tail_guard_max_island_ms,
                    search_window_ms=tts_cheap_tail_guard_search_window_ms,
                    max_trim_ms=tts_cheap_tail_guard_max_trim_ms
                )
            corrected_candidate, candidate_babble_info = _trim_trailing_babble(
                audio=corrected_candidate,
                expected_text=clean,
                model_asr=babble_guard_model,
                language=language,
                segment_duration_sec=max(0.0, orig_end_val - orig_start_val),
                tts_group_size=int(seg.get("tts_group_size", 1) or 1),
                max_segment_sec=tts_babble_guard_max_segment_sec,
                min_gap_ms=tts_babble_guard_min_gap_ms,
                min_island_ms=tts_babble_guard_min_island_ms,
                max_island_ms=tts_babble_guard_max_island_ms,
                search_window_ms=tts_babble_guard_search_window_ms,
                max_trim_ms=tts_babble_guard_max_trim_ms,
                anchor_words=tts_babble_guard_anchor_words
            )

            candidate_eval = None
            if retry_candidate:
                recognized_candidate = _transcribe_short_audio(
                    babble_guard_model,
                    corrected_candidate,
                    language
                )
                candidate_eval = _segment_recognition_score(
                    clean,
                    recognized_candidate,
                    tts_babble_guard_anchor_words
                )
            return corrected_candidate, cheap_tail_info, candidate_babble_info, candidate_eval

        corrected, cheap_tail_info, babble_guard_info, asr_eval = finalize_candidate(seg_audio)

        if retry_candidate and asr_eval and asr_eval["score"] < tts_asr_retry_min_score:
            best_audio = corrected
            best_cheap_tail_info = cheap_tail_info
            best_babble_guard_info = babble_guard_info
            best_eval = asr_eval

            for attempt in range(1, tts_asr_retry_attempts + 1):
                with NamedTemporaryFile(suffix=".wav", delete=False) as retry_tmp:
                    retry_path = retry_tmp.name
                try:
                    generate_audio_segment(
                        tts_backend=tts_backend,
                        text=clean,
                        output_path=retry_path,
                        speaker_wav=selected_references,
                        language=language,
                        conditioning=conditioning,
                        speaker_profile=speaker_profile
                    )
                    retry_audio = AudioSegment.from_wav(retry_path)
                finally:
                    if os.path.exists(retry_path):
                        try:
                            os.remove(retry_path)
                        except OSError:
                            pass

                retry_corrected, retry_cheap_tail_info, retry_babble_info, retry_eval = finalize_candidate(retry_audio)
                if retry_eval and retry_eval["score"] > best_eval["score"]:
                    best_audio = retry_corrected
                    best_cheap_tail_info = retry_cheap_tail_info
                    best_babble_guard_info = retry_babble_info
                    best_eval = retry_eval

                if retry_eval and retry_eval["score"] >= 0.995:
                    break

            if best_eval["score"] > asr_eval["score"]:
                corrected = best_audio
                cheap_tail_info = best_cheap_tail_info
                babble_guard_info = best_babble_guard_info
                asr_eval = best_eval
                logger.info(
                    "[%s] TTS retry improved ASR score to %.3f | %s",
                    i,
                    asr_eval["score"],
                    asr_eval["recognized_text"]
                )

        if cheap_tail_info:
            seg["cheap_tail_guard_trim_ms"] = cheap_tail_info["trim_ms"]
            seg["cheap_tail_guard_overhang_ms"] = cheap_tail_info["overhang_ms"]
            logger.info(
                "[%s] Cheap tail guard trim %s мс (overhang %s мс)",
                i,
                cheap_tail_info["trim_ms"],
                cheap_tail_info["overhang_ms"]
            )
        if babble_guard_info:
            seg["babble_guard_trim_ms"] = babble_guard_info["trim_ms"]
            seg["babble_guard_before"] = babble_guard_info["recognized_before"]
            seg["babble_guard_after"] = babble_guard_info["recognized_after"]
            logger.info(
                "[%s] TTS babble guard trim %s мс | до: %s | после: %s",
                i,
                babble_guard_info["trim_ms"],
                babble_guard_info["recognized_before"],
                babble_guard_info["recognized_after"]
            )
        if asr_eval:
            seg["tts_asr_score"] = asr_eval["score"]
            seg["tts_asr_similarity"] = asr_eval["similarity"]
            seg["tts_asr_recognized"] = asr_eval["recognized_text"]
        corrected_ms = len(corrected)
        edge_silence = _edge_silence_ms(corrected)
        fade_in_ms = min(fade_in_out_ms, edge_silence["leading"])
        fade_out_ms = min(fade_in_out_ms, edge_silence["trailing"])
        if fade_in_ms > 0:
            corrected = corrected.fade_in(fade_in_ms)
        if fade_out_ms > 0:
            corrected = corrected.fade_out(fade_out_ms)
        seg["edge_leading_silence_ms"] = edge_silence["leading"]
        seg["edge_trailing_silence_ms"] = edge_silence["trailing"]
        seg["fade_in_ms"] = fade_in_ms
        seg["fade_out_ms"] = fade_out_ms
        seg["corrected_audio"] = corrected

        # Вставляем в итоговое аудио
        full_audio = (
            full_audio[:cur_start_ms]
            + corrected
            + full_audio[cur_start_ms + corrected_ms:]
        )

        actual_end = cur_start + corrected_ms / 1000.0
        seg["corrected_duration_sec"] = corrected_ms / 1000.0
        prev_end_sec = actual_end

        # Обновляем начало следующего сегмента (только вправо)
        if i < len(segments) - 1:
            nxt = segments[i + 1]
            next_original_start = nxt.setdefault("original_start", nxt["start"])
            next_start_floor = nxt["start"]
            if max_next_start_shift_sec is None:
                nxt["start"] = max(actual_end, next_start_floor)
            else:
                max_allowed_start = next_original_start + max_next_start_shift_sec
                bounded_actual_end = min(actual_end, max_allowed_start)
                nxt["start"] = max(next_start_floor, bounded_actual_end)

    # Кроссфейды между соседними сегментами
    for i in range(len(segments) - 1):
        curr, nxt = segments[i], segments[i + 1]
        if "corrected_audio" not in curr or "corrected_audio" not in nxt:
            continue
        curr_ms   = int(curr["start"] * 1000)
        nxt_ms    = int(nxt["start"]  * 1000)
        overlap   = curr_ms + len(curr["corrected_audio"]) - nxt_ms
        if 0 < overlap < crossfade_ms:
            merged = curr["corrected_audio"].append(nxt["corrected_audio"], crossfade=overlap)
            full_audio = full_audio[:curr_ms] + merged + full_audio[curr_ms + len(merged):]

    if enable_final_compression:
        full_audio = compress_dynamic_range(
            full_audio,
            threshold=threshold_compression,
            ratio=ratio_compression,
            attack=attack_compression,
            release=release_compression
        )
    full_audio = _apply_peak_ceiling(full_audio, peak_ceiling_dbfs)

    full_audio.export(output_audio_path, format="wav")
    logger.info(f"Финальное аудио сохранено: {output_audio_path}")
