"""Модель временной укладки сегмента: доступное окно с резервом соседних пауз, обработка кромочной тишины и сдвиг старта следующего сегмента."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from pydub import AudioSegment
from pydub.silence import detect_nonsilent


@dataclass(frozen=True)
class SegmentTimingWindow:
    window_start_sec: float
    window_end_sec: float
    cur_start_sec: float
    cur_start_ms: int
    original_ms: int
    available_ms: int
    borrowed_before_ms: int
    borrowed_after_ms: int

    @property
    def timing_window_start_sec(self) -> float:
        return round(self.cur_start_sec, 3)

    @property
    def timing_window_end_sec(self) -> float:
        return round(self.window_end_sec + self.borrowed_after_ms / 1000.0, 3)


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


def _timing_speech_stats(
    audio: AudioSegment,
    keep_edge_ms: int = 60,
) -> Dict[str, int]:
    """
    Оценивает «реальную» речевую длительность для тайминга, не наказывая
    сегмент за длинные стартовые/хвостовые паузы XTTS.
    """
    if len(audio) == 0:
        return {
            "effective_duration_ms": 0,
            "leading_silence_ms": 0,
            "trailing_silence_ms": 0,
        }

    edge = _edge_silence_ms(audio)
    effective_duration_ms = max(
        0,
        len(audio)
        - max(0, edge["leading"] - keep_edge_ms)
        - max(0, edge["trailing"] - keep_edge_ms)
    )
    return {
        "effective_duration_ms": int(effective_duration_ms),
        "leading_silence_ms": int(edge["leading"]),
        "trailing_silence_ms": int(edge["trailing"]),
    }


def _allocate_timing_extension_ms(
    needed_extension_ms: int,
    available_before_ms: int,
    available_after_ms: int,
) -> Tuple[int, int]:
    """Распределяет дополнительное окно слева/справа максимально симметрично."""
    if needed_extension_ms <= 0:
        return 0, 0

    total_available_ms = max(0, available_before_ms) + max(0, available_after_ms)
    if total_available_ms <= 0:
        return 0, 0

    target_extension_ms = min(needed_extension_ms, total_available_ms)
    half_needed_ms = target_extension_ms / 2.0

    if half_needed_ms <= available_before_ms and half_needed_ms <= available_after_ms:
        return int(round(half_needed_ms)), int(round(half_needed_ms))

    if available_before_ms < half_needed_ms:
        borrow_before_ms = max(0, available_before_ms)
        borrow_after_ms = min(target_extension_ms - borrow_before_ms, max(0, available_after_ms))
        return int(round(borrow_before_ms)), int(round(borrow_after_ms))

    borrow_after_ms = max(0, available_after_ms)
    borrow_before_ms = min(target_extension_ms - borrow_after_ms, max(0, available_before_ms))
    return int(round(borrow_before_ms)), int(round(borrow_after_ms))


def build_segment_timing_window(
    segments: List[Dict[str, Any]],
    index: int,
    original_start_sec: float,
    prev_end_sec: float,
    timing_duration_ms: int,
    min_pause_between_segments: float,
    max_shift_left_seconds: float,
    max_next_start_shift_sec: float | None,
) -> SegmentTimingWindow:
    segment = segments[index]
    cur_start = float(segment.get("start", original_start_sec))
    cur_start_ms = int(cur_start * 1000)

    window_start_sec = float(segment.get("start", original_start_sec))
    window_end_sec = float(segment.get("end", window_start_sec + 1.0))
    if window_start_sec > window_end_sec:
        window_start_sec, window_end_sec = window_end_sec, window_start_sec

    original_ms = int((window_end_sec - window_start_sec) * 1000)

    available_before_ms = max(
        0,
        int(
            (
                window_start_sec
                - (prev_end_sec + min_pause_between_segments)
            ) * 1000
        )
    )
    available_before_ms = min(
        available_before_ms,
        max(0, int(max_shift_left_seconds * 1000))
    )

    available_after_ms = 0
    if index < len(segments) - 1:
        next_segment = segments[index + 1]
        next_current_start = float(next_segment.get("start", next_segment.get("original_start", 0.0)))
        gap_after_ms = max(
            0,
            int(
                (
                    next_current_start
                    - window_end_sec
                    - min_pause_between_segments
                ) * 1000
            )
        )
        available_after_ms = gap_after_ms
        if max_next_start_shift_sec is not None:
            next_original_start = float(
                next_segment.setdefault("original_start", next_segment.get("start", next_current_start))
            )
            max_allowed_next_start = next_original_start + max_next_start_shift_sec
            available_after_ms += max(
                0,
                int((max_allowed_next_start - next_current_start) * 1000)
            )

    borrowed_before_ms = 0
    borrowed_after_ms = 0
    needed_extension_ms = max(0, timing_duration_ms - original_ms)
    total_borrowable_ms = available_before_ms + available_after_ms
    if (
        needed_extension_ms > 0
        and total_borrowable_ms >= max(120, int(needed_extension_ms * 0.5))
    ):
        borrowed_before_ms, borrowed_after_ms = _allocate_timing_extension_ms(
            needed_extension_ms=needed_extension_ms,
            available_before_ms=available_before_ms,
            available_after_ms=available_after_ms,
        )
        if borrowed_before_ms or borrowed_after_ms:
            cur_start = max(
                prev_end_sec + min_pause_between_segments,
                window_start_sec - borrowed_before_ms / 1000.0,
            )
            cur_start_ms = int(round(cur_start * 1000))

    available_ms = max(100, original_ms + borrowed_before_ms + borrowed_after_ms)
    return SegmentTimingWindow(
        window_start_sec=window_start_sec,
        window_end_sec=window_end_sec,
        cur_start_sec=cur_start,
        cur_start_ms=cur_start_ms,
        original_ms=original_ms,
        available_ms=available_ms,
        borrowed_before_ms=borrowed_before_ms,
        borrowed_after_ms=borrowed_after_ms,
    )


def update_next_segment_start(
    segments: List[Dict[str, Any]],
    index: int,
    actual_end_sec: float,
    max_next_start_shift_sec: float | None,
) -> None:
    """Сдвигает старт следующего сегмента только вправо и с учетом лимита."""
    if index >= len(segments) - 1:
        return

    next_segment = segments[index + 1]
    next_original_start = next_segment.setdefault("original_start", next_segment["start"])
    next_start_floor = next_segment["start"]
    if max_next_start_shift_sec is None:
        next_segment["start"] = max(actual_end_sec, next_start_floor)
        return

    max_allowed_start = next_original_start + max_next_start_shift_sec
    bounded_actual_end = min(actual_end_sec, max_allowed_start)
    next_segment["start"] = max(next_start_floor, bounded_actual_end)
