"""Диаризация говорящих (pyannote) — задел под мультиголосый дубляж; основной пайплайн работает в режиме single-speaker."""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def _normalize_word_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _interval_overlap(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def normalize_diarization_turns(
    turns: Iterable[dict[str, Any]],
    *,
    speaker_prefix: str = "spk",
) -> list[dict[str, Any]]:
    """
    Normalizes diarization turns to the project's speaker_id convention.

    Input turns may use either speaker_id or speaker labels such as SPEAKER_00.
    Speaker IDs are assigned in first-appearance order to keep outputs stable
    across pyannote label formats.
    """
    normalized: list[dict[str, Any]] = []
    label_to_id: dict[str, str] = {}

    for raw_turn in sorted(turns, key=lambda item: float(item.get("start", 0.0))):
        start = float(raw_turn.get("start", 0.0))
        end = float(raw_turn.get("end", start))
        if end <= start:
            continue

        raw_label = (
            raw_turn.get("speaker_label")
            or raw_turn.get("label")
            or raw_turn.get("speaker")
            or raw_turn.get("speaker_id")
            or f"SPEAKER_{len(label_to_id):02d}"
        )
        speaker_label = str(raw_label)

        existing_speaker_id = raw_turn.get("speaker_id")
        if isinstance(existing_speaker_id, str) and existing_speaker_id.startswith(f"{speaker_prefix}_"):
            speaker_id = existing_speaker_id
            label_to_id.setdefault(speaker_label, speaker_id)
        else:
            speaker_id = label_to_id.setdefault(
                speaker_label,
                f"{speaker_prefix}_{len(label_to_id)}",
            )

        normalized.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "speaker_id": speaker_id,
            "speaker_label": speaker_label,
        })

    return normalized


def load_diarization_turns(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        turns = payload.get("turns", [])
    else:
        turns = payload
    if not isinstance(turns, list):
        raise ValueError(f"Diarization JSON must contain a list of turns: {path}")
    return normalize_diarization_turns(turns)


def save_diarization_turns(path: str, turns: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"turns": turns}, f, ensure_ascii=False, indent=2)


def _turns_from_pyannote_output(output: Any) -> list[dict[str, Any]]:
    diarization = getattr(output, "speaker_diarization", output)
    turns: list[dict[str, Any]] = []

    itertracks = getattr(diarization, "itertracks", None)
    if callable(itertracks):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker_label": str(speaker),
            })
        return turns

    for item in diarization:
        if len(item) == 2:
            turn, speaker = item
        elif len(item) == 3:
            turn, _, speaker = item
        else:
            raise ValueError(f"Unsupported pyannote diarization item: {item!r}")
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker_label": str(speaker),
        })
    return turns


def run_pyannote_diarization(
    audio_path: str,
    *,
    model_name: str,
    token: str,
    device: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict[str, Any]]:
    """
    Runs pyannote only when explicitly requested by the probe script.

    pyannote.audio remains an optional dependency: importing this module does
    not import pyannote, so the existing project runtime is unaffected.
    """
    try:
        import torch
        from pyannote.audio import Pipeline
        import pyannote.audio.core.model as pyannote_model
    except ImportError as error:
        raise ImportError(
            "Для diarization probe установите optional dependency: "
            "pip install pyannote.audio"
        ) from error

    if not token:
        raise EnvironmentError(
            "Для pyannote diarization нужен Hugging Face token. "
            "Передайте --hf-token или задайте переменную окружения с token."
        )

    pl_load = getattr(pyannote_model, "pl_load", None)
    if callable(pl_load) and "weights_only" not in getattr(pl_load, "__annotations__", {}):
        import inspect

        if "weights_only" not in inspect.signature(pl_load).parameters:
            original_pl_load = pl_load

            def _pl_load_compat(*args, **kwargs):
                weights_only = kwargs.pop("weights_only", False)
                if weights_only is not False:
                    return original_pl_load(*args, **kwargs)

                original_torch_load = torch.load

                def _torch_load_compat(*load_args, **load_kwargs):
                    load_kwargs.setdefault("weights_only", False)
                    return original_torch_load(*load_args, **load_kwargs)

                torch.load = _torch_load_compat
                try:
                    return original_pl_load(*args, **kwargs)
                finally:
                    torch.load = original_torch_load

            pyannote_model.pl_load = _pl_load_compat

    original_torch_load = torch.load

    def _torch_load_trusted_checkpoint(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = _torch_load_trusted_checkpoint
    try:
        pipeline = Pipeline.from_pretrained(model_name, token=token)
    finally:
        torch.load = original_torch_load
    if device:
        pipeline.to(torch.device(device))

    kwargs: dict[str, int] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = int(num_speakers)
    if min_speakers is not None:
        kwargs["min_speakers"] = int(min_speakers)
    if max_speakers is not None:
        kwargs["max_speakers"] = int(max_speakers)

    logger.info("Запускаем pyannote diarization: %s", model_name)
    output = pipeline(audio_path, **kwargs)
    return normalize_diarization_turns(_turns_from_pyannote_output(output))


def _speaker_for_interval(
    start: float,
    end: float,
    turns: list[dict[str, Any]],
    *,
    default_speaker_id: str,
) -> tuple[str, float]:
    overlaps: dict[str, float] = defaultdict(float)
    for turn in turns:
        overlap = _interval_overlap(
            start,
            end,
            float(turn["start"]),
            float(turn["end"]),
        )
        if overlap > 0:
            overlaps[str(turn["speaker_id"])] += overlap

    if not overlaps:
        return default_speaker_id, 0.0

    speaker_id, overlap_sec = max(overlaps.items(), key=lambda item: item[1])
    return speaker_id, round(overlap_sec, 3)


def _build_words_with_silence(words: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    cleaned_words = [word for word in words if _normalize_word_text(word.get("text", ""))]
    for idx, word in enumerate(cleaned_words):
        parts.append(_normalize_word_text(word["text"]))
        if idx < len(cleaned_words) - 1:
            gap_sec = max(0.0, float(cleaned_words[idx + 1]["start"]) - float(word["end"]))
            if gap_sec >= 0.04:
                parts.append(f"<{gap_sec:.2f}s>")
    return " ".join(parts)


def _segment_from_word_dicts(
    words: list[dict[str, Any]],
    *,
    speaker_id: str,
    pause_before_sec: float,
    pause_after_sec: float,
    source_segment_index: int,
    split_index: int,
    split_count: int,
) -> dict[str, Any]:
    cleaned_words = [
        {
            **word,
            "text": _normalize_word_text(word.get("text", "")),
            "start": round(float(word.get("start", 0.0)), 3),
            "end": round(float(word.get("end", 0.0)), 3),
        }
        for word in words
        if _normalize_word_text(word.get("text", ""))
    ]
    if not cleaned_words:
        raise ValueError("Cannot build diarized segment without words.")

    start = float(cleaned_words[0]["start"])
    end = float(cleaned_words[-1]["end"])
    return {
        "text": " ".join(word["text"] for word in cleaned_words),
        "start": round(start, 3),
        "end": round(end, 3),
        "speaker_id": speaker_id,
        "words": [
            {
                "text": word["text"],
                "start": word["start"],
                "end": word["end"],
                "speaker_id": word.get("speaker_id", speaker_id),
                "diarization_overlap_sec": word.get("diarization_overlap_sec", 0.0),
            }
            for word in cleaned_words
        ],
        "words_with_silence": _build_words_with_silence(cleaned_words),
        "source_duration_sec": round(max(0.0, end - start), 3),
        "source_word_count": len(cleaned_words),
        "pause_before_sec": round(max(0.0, pause_before_sec), 3),
        "pause_after_sec": round(max(0.0, pause_after_sec), 3),
        "diarization_source_segment_index": source_segment_index,
        "diarization_split_index": split_index,
        "diarization_split_count": split_count,
    }


def _normalized_segment_words(segment: dict[str, Any]) -> list[dict[str, Any]]:
    words = segment.get("words")
    if not isinstance(words, list):
        return []

    normalized_words: list[dict[str, Any]] = []
    for word in words:
        text = _normalize_word_text(word.get("text", word.get("word", "")))
        if not text:
            continue
        start = float(word.get("start", 0.0))
        end = float(word.get("end", start))
        if end <= start:
            continue
        normalized_words.append({
            "text": text,
            "start": start,
            "end": end,
        })
    return sorted(normalized_words, key=lambda item: item["start"])


def apply_diarization_to_segments(
    segments: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    *,
    default_speaker_id: str = "spk_0",
    max_pause_between_sentences: float = 0.3,
) -> list[dict[str, Any]]:
    """
    Produces a diarized copy of ASR segments without mutating source segments.

    Existing segment boundaries are preserved unless a speaker changes inside
    one ASR segment. In that case the segment is split at the word boundary.
    """
    normalized_turns = normalize_diarization_turns(turns)
    diarized: list[dict[str, Any]] = []

    for source_index, segment in enumerate(segments):
        words = _normalized_segment_words(segment)
        if not words:
            item = deepcopy(segment)
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
            speaker_id, overlap_sec = _speaker_for_interval(
                start,
                end,
                normalized_turns,
                default_speaker_id=default_speaker_id,
            )
            item["speaker_id"] = speaker_id
            item["diarization_overlap_sec"] = overlap_sec
            item["diarization_source_segment_index"] = source_index
            item["diarization_split_index"] = 0
            item["diarization_split_count"] = 1
            diarized.append(item)
            continue

        assigned_words: list[dict[str, Any]] = []
        for word in words:
            speaker_id, overlap_sec = _speaker_for_interval(
                float(word["start"]),
                float(word["end"]),
                normalized_turns,
                default_speaker_id=default_speaker_id,
            )
            assigned_words.append({
                **word,
                "speaker_id": speaker_id,
                "diarization_overlap_sec": overlap_sec,
            })

        groups: list[tuple[str, list[dict[str, Any]], float, float]] = []
        current_speaker = str(assigned_words[0]["speaker_id"])
        current_words = [assigned_words[0]]
        group_pause_before = float(segment.get("pause_before_sec", 0.0) or 0.0)

        for word in assigned_words[1:]:
            previous_word = current_words[-1]
            gap_sec = max(0.0, float(word["start"]) - float(previous_word["end"]))
            should_split = (
                str(word["speaker_id"]) != current_speaker
                or gap_sec > max_pause_between_sentences
            )
            if should_split:
                groups.append((current_speaker, current_words, group_pause_before, gap_sec))
                current_speaker = str(word["speaker_id"])
                current_words = [word]
                group_pause_before = gap_sec
            else:
                current_words.append(word)

        groups.append((
            current_speaker,
            current_words,
            group_pause_before,
            float(segment.get("pause_after_sec", 0.0) or 0.0),
        ))

        split_count = len(groups)
        for split_index, (speaker_id, group_words, pause_before, pause_after) in enumerate(groups):
            diarized.append(_segment_from_word_dicts(
                group_words,
                speaker_id=speaker_id,
                pause_before_sec=pause_before,
                pause_after_sec=pause_after,
                source_segment_index=source_index,
                split_index=split_index,
                split_count=split_count,
            ))

    return diarized


def summarize_diarization(
    turns: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_turns = normalize_diarization_turns(turns)
    turn_seconds: Counter[str] = Counter()
    for turn in normalized_turns:
        turn_seconds[str(turn["speaker_id"])] += max(
            0.0,
            float(turn["end"]) - float(turn["start"]),
        )

    segment_counts: Counter[str] = Counter(
        str(segment.get("speaker_id", ""))
        for segment in segments
        if segment.get("speaker_id")
    )
    segment_seconds: Counter[str] = Counter()
    for segment in segments:
        speaker_id = segment.get("speaker_id")
        if not speaker_id:
            continue
        segment_seconds[str(speaker_id)] += max(
            0.0,
            float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)),
        )

    split_segments = sum(
        1
        for segment in segments
        if int(segment.get("diarization_split_count", 1) or 1) > 1
    )

    return {
        "turn_count": len(normalized_turns),
        "speaker_count": len({turn["speaker_id"] for turn in normalized_turns}),
        "segment_count": len(segments),
        "split_segment_count": split_segments,
        "turn_seconds_by_speaker": {
            key: round(value, 3)
            for key, value in sorted(turn_seconds.items())
        },
        "segment_count_by_speaker": dict(sorted(segment_counts.items())),
        "segment_seconds_by_speaker": {
            key: round(value, 3)
            for key, value in sorted(segment_seconds.items())
        },
    }
