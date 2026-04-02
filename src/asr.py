import logging
import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median
from typing import Any, Dict, List

import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

logger = logging.getLogger(__name__)


@dataclass
class WordSegment:
    text:  str
    start: float
    end:   float


def _normalize_reference_text(text: str) -> str:
    """Нормализует текст для дедупликации похожих референсов."""
    normalized = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _text_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def _duration_bucket(duration_sec: float) -> str:
    if duration_sec < 4.0:
        return "short"
    if duration_sec < 7.0:
        return "medium"
    return "long"


def _timeline_bucket(
    start_sec: float,
    min_start_sec: float,
    max_end_sec: float
) -> str:
    span = max(max_end_sec - min_start_sec, 1.0)
    position = (start_sec - min_start_sec) / span
    if position < 0.33:
        return "early"
    if position < 0.66:
        return "middle"
    return "late"


def _active_speech_stats(audio: AudioSegment) -> Dict[str, float | None]:
    """Оценивает плотность речи и уровень сигнала внутри клипа."""
    if len(audio) == 0 or audio.rms == 0:
        return {
            "active_ratio": 0.0,
            "active_dbfs": None,
            "peak_dbfs": None,
            "full_dbfs": None,
        }

    silence_threshold = max(-50.0, audio.dBFS - 16.0)
    regions = detect_nonsilent(
        audio,
        min_silence_len=120,
        silence_thresh=silence_threshold,
        seek_step=5
    )

    active_audio = audio[0:0]
    active_ms = 0
    for start_ms, end_ms in regions:
        if end_ms > start_ms:
            active_audio += audio[start_ms:end_ms]
            active_ms += end_ms - start_ms

    active_ratio = active_ms / len(audio) if len(audio) else 0.0
    active_dbfs = active_audio.dBFS if len(active_audio) and active_audio.rms else audio.dBFS
    return {
        "active_ratio": active_ratio,
        "active_dbfs": active_dbfs,
        "peak_dbfs": audio.max_dBFS,
        "full_dbfs": audio.dBFS,
    }


def _reference_score(
    segment: Dict[str, Any],
    target_duration: float,
    target_active_dbfs: float | None
) -> float:
    """Скоринг кандидата для speaker profile с учётом качества и разнообразия."""
    duration = float(segment["duration_sec"])
    text = segment.get("text", "").strip()
    active_ratio = float(segment.get("active_ratio") or 0.0)
    active_dbfs = segment.get("active_dbfs")
    peak_dbfs = segment.get("peak_dbfs")

    duration_score = 2.0 - abs(duration - target_duration) / max(target_duration, 1.0)
    word_bonus = min(len(text.split()), 18) * 0.06
    ratio_score = 1.2 - abs(active_ratio - 0.72) * 2.0

    loudness_score = 0.0
    if target_active_dbfs is not None and active_dbfs is not None:
        loudness_score = 1.0 - min(abs(active_dbfs - target_active_dbfs), 12.0) / 6.0

    peak_score = 0.0
    if peak_dbfs is not None:
        if peak_dbfs > -2.0:
            peak_score = -1.0
        elif peak_dbfs > -6.0:
            peak_score = 0.15
        else:
            peak_score = 0.35

    question_bonus = 0.2 if text.rstrip().endswith("?") else 0.0
    comma_bonus = 0.1 if "," in text else 0.0

    return duration_score + word_bonus + ratio_score + loudness_score + peak_score + question_bonus + comma_bonus


def _segments_too_close(
    left: Dict[str, Any],
    right: Dict[str, Any],
    min_gap_sec: float
) -> bool:
    return (
        left["start"] < right["end"] + min_gap_sec
        and right["start"] < left["end"] + min_gap_sec
    )


def export_speaker_profile(
    reference_audio_path: str,
    segments: List[Dict[str, Any]],
    output_ref_path: str,
    output_refs_dir: str | None = None,
    output_profile_path: str | None = None,
    max_reference_clips: int = 5,
    max_routing_clips: int = 12,
    min_reference_sec: float = 2.0,
    max_reference_sec: float = 10.0,
    target_reference_sec: float = 6.0,
    min_reference_text_chars: int = 20,
    padding_ms: int = 120,
    min_gap_sec: float = 1.5,
    speaker_id: str | None = None
) -> Dict[str, Any]:
    """
    Экспортирует multi-reference speaker profile из одного видео.

    Создаёт:
    - набор отдельных reference clips;
    - merged reference wav для метрик и fallback;
    - json-манифест профиля.
    """
    if not os.path.exists(reference_audio_path):
        raise FileNotFoundError(f"Референсное аудио не найдено: {reference_audio_path}")

    audio = AudioSegment.from_wav(reference_audio_path).set_channels(1)
    total_audio_ms = len(audio)
    candidates: List[Dict[str, Any]] = []

    for idx, seg in enumerate(segments):
        if speaker_id and seg.get("speaker_id") != speaker_id:
            continue

        text = seg.get("text", "").strip()
        duration = float(seg["end"] - seg["start"])
        if duration < min_reference_sec or duration > max_reference_sec:
            continue
        if len(text) < min_reference_text_chars:
            continue

        start_ms = max(0, int(seg["start"] * 1000) - padding_ms)
        end_ms = min(total_audio_ms, int(seg["end"] * 1000) + padding_ms)
        clip = audio[start_ms:end_ms]
        if len(clip) == 0 or clip.rms == 0:
            continue

        speech_stats = _active_speech_stats(clip)
        candidate = dict(seg)
        candidate["source_index"] = idx
        candidate["duration_sec"] = round(duration, 3)
        candidate["clip_start_ms"] = start_ms
        candidate["clip_end_ms"] = end_ms
        candidate["text_signature"] = _normalize_reference_text(text)
        candidate["duration_bucket"] = _duration_bucket(duration)
        candidate["active_ratio"] = round(float(speech_stats["active_ratio"] or 0.0), 4)
        candidate["active_dbfs"] = (
            round(float(speech_stats["active_dbfs"]), 3)
            if speech_stats["active_dbfs"] is not None else None
        )
        candidate["peak_dbfs"] = (
            round(float(speech_stats["peak_dbfs"]), 3)
            if speech_stats["peak_dbfs"] is not None else None
        )
        candidate["full_dbfs"] = (
            round(float(speech_stats["full_dbfs"]), 3)
            if speech_stats["full_dbfs"] is not None else None
        )
        candidate["question"] = text.rstrip().endswith("?")
        candidates.append(candidate)

    if not candidates:
        raise ValueError("Не удалось подобрать сегменты для speaker profile.")

    target_active_dbfs = None
    active_levels = [
        item["active_dbfs"]
        for item in candidates
        if item.get("active_dbfs") is not None
    ]
    if active_levels:
        target_active_dbfs = float(median(active_levels))

    for candidate in candidates:
        candidate["score"] = round(
            _reference_score(candidate, target_reference_sec, target_active_dbfs),
            4
        )

    deduped_candidates: List[Dict[str, Any]] = []
    seen_signatures: Dict[str, Dict[str, Any]] = {}
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        signature = candidate["text_signature"]
        existing = seen_signatures.get(signature)
        if existing is None or candidate["score"] > existing["score"]:
            seen_signatures[signature] = candidate
    deduped_candidates.extend(seen_signatures.values())

    ordered = sorted(
        deduped_candidates,
        key=lambda item: (item["score"], item["duration_sec"]),
        reverse=True
    )
    if ordered:
        min_start_sec = min(item["start"] for item in ordered)
        max_end_sec = max(item["end"] for item in ordered)
        for candidate in ordered:
            candidate["timeline_bucket"] = _timeline_bucket(
                float(candidate["start"]),
                min_start_sec,
                max_end_sec
            )

    selected: List[Dict[str, Any]] = []
    while len(selected) < max_reference_clips:
        best_candidate = None
        best_score = float("-inf")

        for candidate in ordered:
            if candidate in selected:
                continue
            if any(_segments_too_close(candidate, existing, min_gap_sec) for existing in selected):
                continue

            similarity_penalty = 0.0
            if selected:
                max_similarity = max(
                    _text_similarity(candidate["text_signature"], existing["text_signature"])
                    for existing in selected
                )
                if max_similarity >= 0.93:
                    continue
                similarity_penalty = max(0.0, max_similarity - 0.55) * 2.2

            diversity_bonus = 0.0
            selected_buckets = {item["duration_bucket"] for item in selected}
            if candidate["duration_bucket"] not in selected_buckets:
                diversity_bonus += 0.35
            if candidate["question"] and not any(item["question"] for item in selected):
                diversity_bonus += 0.35

            final_score = candidate["score"] + diversity_bonus - similarity_penalty
            candidate["selection_score"] = round(final_score, 4)
            if final_score > best_score:
                best_score = final_score
                best_candidate = candidate

        if best_candidate is None:
            break
        selected.append(best_candidate)

    if len(selected) < max_reference_clips:
        used_indices = {item["source_index"] for item in selected}
        for candidate in ordered:
            if candidate["source_index"] in used_indices:
                continue
            if any(_segments_too_close(candidate, existing, min_gap_sec) for existing in selected):
                continue
            selected.append(candidate)
            used_indices.add(candidate["source_index"])
            if len(selected) >= max_reference_clips:
                break

    selected = sorted(selected, key=lambda item: item["start"])

    routing_target = max(max_reference_clips, max_routing_clips)
    routing_selected: List[Dict[str, Any]] = list(selected)
    used_indices = {item["source_index"] for item in routing_selected}
    routing_min_gap_sec = max(0.6, min_gap_sec * 0.5)

    while len(routing_selected) < routing_target:
        best_candidate = None
        best_score = float("-inf")

        for candidate in ordered:
            if candidate["source_index"] in used_indices:
                continue
            if any(
                _segments_too_close(candidate, existing, routing_min_gap_sec)
                for existing in routing_selected
            ):
                continue

            similarity_penalty = 0.0
            if routing_selected:
                max_similarity = max(
                    _text_similarity(candidate["text_signature"], existing["text_signature"])
                    for existing in routing_selected
                )
                if max_similarity >= 0.96:
                    continue
                similarity_penalty = max(0.0, max_similarity - 0.7) * 1.8

            diversity_bonus = 0.0
            selected_durations = {item["duration_bucket"] for item in routing_selected}
            selected_timelines = {item.get("timeline_bucket") for item in routing_selected}
            if candidate["duration_bucket"] not in selected_durations:
                diversity_bonus += 0.2
            if candidate["timeline_bucket"] not in selected_timelines:
                diversity_bonus += 0.25
            if candidate["question"] and not any(item["question"] for item in routing_selected):
                diversity_bonus += 0.2

            final_score = candidate["score"] + diversity_bonus - similarity_penalty
            candidate["routing_selection_score"] = round(final_score, 4)
            if final_score > best_score:
                best_score = final_score
                best_candidate = candidate

        if best_candidate is None:
            break
        routing_selected.append(best_candidate)
        used_indices.add(best_candidate["source_index"])

    routing_selected = sorted(routing_selected, key=lambda item: item["start"])

    clip_paths: List[str] = []
    clip_metadata: List[Dict[str, Any]] = []
    routing_clip_metadata: List[Dict[str, Any]] = []
    if output_refs_dir:
        os.makedirs(output_refs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_ref_path), exist_ok=True)

    merged_audio = AudioSegment.silent(duration=0)
    gap_audio = AudioSegment.silent(duration=150)

    for idx, seg in enumerate(selected):
        clip = audio[seg["clip_start_ms"]:seg["clip_end_ms"]]

        if output_refs_dir:
            clip_path = os.path.join(output_refs_dir, f"ref_{idx:02d}.wav")
            clip.export(clip_path, format="wav")
            clip_paths.append(clip_path)
        else:
            clip_path = output_ref_path if idx == 0 else ""

        merged_audio += clip
        if idx < len(selected) - 1:
            merged_audio += gap_audio

        clip_metadata.append({
            "path": clip_path,
            "text": seg.get("text", "").strip(),
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "duration_sec": round(len(clip) / 1000.0, 3),
            "source_index": seg["source_index"],
            "speaker_id": seg.get("speaker_id"),
            "score": seg.get("score"),
            "selection_score": seg.get("selection_score", seg.get("score")),
            "active_ratio": seg.get("active_ratio"),
            "active_dbfs": seg.get("active_dbfs"),
            "peak_dbfs": seg.get("peak_dbfs"),
            "duration_bucket": seg.get("duration_bucket"),
            "timeline_bucket": seg.get("timeline_bucket"),
        })

    for idx, seg in enumerate(routing_selected):
        clip = audio[seg["clip_start_ms"]:seg["clip_end_ms"]]
        route_clip_path = ""
        if output_refs_dir:
            route_clip_path = os.path.join(output_refs_dir, f"route_ref_{idx:02d}.wav")
            clip.export(route_clip_path, format="wav")

        routing_clip_metadata.append({
            "path": route_clip_path,
            "text": seg.get("text", "").strip(),
            "start": round(float(seg["start"]), 3),
            "end": round(float(seg["end"]), 3),
            "duration_sec": round(len(clip) / 1000.0, 3),
            "source_index": seg["source_index"],
            "speaker_id": seg.get("speaker_id"),
            "score": seg.get("score"),
            "selection_score": seg.get(
                "routing_selection_score",
                seg.get("selection_score", seg.get("score"))
            ),
            "active_ratio": seg.get("active_ratio"),
            "active_dbfs": seg.get("active_dbfs"),
            "peak_dbfs": seg.get("peak_dbfs"),
            "duration_bucket": seg.get("duration_bucket"),
            "timeline_bucket": seg.get("timeline_bucket"),
        })

    if len(merged_audio) == 0:
        raise ValueError("Speaker profile получился пустым.")

    merged_audio.export(output_ref_path, format="wav")

    profile = {
        "merged_reference_path": output_ref_path,
        "reference_audio_path": reference_audio_path,
        "candidate_count": len(candidates),
        "deduped_candidate_count": len(deduped_candidates),
        "target_active_dbfs": target_active_dbfs,
        "clips": clip_metadata,
        "routing_clips": routing_clip_metadata,
    }
    if output_profile_path:
        with open(output_profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

    logger.info(
        "Speaker profile сохранён: %s refs, routing=%s, merged=%s",
        len(clip_metadata),
        len(routing_clip_metadata),
        output_ref_path
    )
    return profile


def transcribe_and_segment(
    model_asr,
    audio_path: str,
    max_pause_between_sentences: float = 0.3,
    max_audio_length_for_ref: float = 15.0,
    output_ref_path: str = "speaker_ref.wav",
    default_speaker_id: str = "spk_0",
    reference_audio_path: str | None = None,
    output_refs_dir: str | None = None,
    output_profile_path: str | None = None,
    max_reference_clips: int = 5,
    max_routing_clips: int = 12,
    min_reference_sec: float = 2.0,
    max_reference_sec: float = 10.0,
    target_reference_sec: float = 6.0,
    min_reference_text_chars: int = 20,
    reference_padding_ms: int = 120,
    min_reference_gap_sec: float = 1.5
) -> List[Dict[str, Any]]:
    """
    Транскрибирует аудио и разбивает на сегменты по паузам.

    Параметры:
        model_asr: загруженная модель Whisper
        audio_path: путь к WAV-файлу
        max_pause_between_sentences: порог паузы для разделения сегментов (сек)
        max_audio_length_for_ref: максимальная длина референсного сегмента (сек)
        output_ref_path: путь для сохранения референсного аудио спикера
        default_speaker_id: speaker_id по умолчанию для single-speaker режима
        reference_audio_path: из какого аудио вырезать референсы для TTS
        output_refs_dir: директория с отдельными reference clips
        output_profile_path: json-манифест speaker profile

    Возвращает:
        List[Dict]: список сегментов [{text, start, end, speaker_id}, ...]
    """
    audio_data, sample_rate = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    result = model_asr.transcribe(audio_path, word_timestamps=True, task="transcribe")

    # Собираем все слова с временными метками
    words: List[WordSegment] = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words.append(WordSegment(text=w["word"], start=w["start"], end=w["end"]))

    if not words:
        logger.error("Слова с временными метками не найдены.")
        return []

    # Разбиваем на сегменты по паузам
    segments = []
    current = {
        "text": words[0].text,
        "start": words[0].start,
        "end": words[0].end,
        "speaker_id": default_speaker_id
    }

    for i in range(1, len(words)):
        gap = words[i].start - words[i - 1].end
        if gap > max_pause_between_sentences:
            current["text"] = current["text"].strip()
            segments.append(current)
            current = {
                "text": words[i].text,
                "start": words[i].start,
                "end": words[i].end,
                "speaker_id": default_speaker_id
            }
        else:
            current["text"] += " " + words[i].text
            current["end"]   = words[i].end

    current["text"] = current["text"].strip()
    segments.append(current)
    logger.info(f"Сегментов получено: {len(segments)}")

    try:
        export_speaker_profile(
            reference_audio_path=reference_audio_path or audio_path,
            segments=segments,
            output_ref_path=output_ref_path,
            output_refs_dir=output_refs_dir,
            output_profile_path=output_profile_path,
            max_reference_clips=max_reference_clips,
            max_routing_clips=max_routing_clips,
            min_reference_sec=min_reference_sec,
            max_reference_sec=min(max_audio_length_for_ref, max_reference_sec),
            target_reference_sec=target_reference_sec,
            min_reference_text_chars=min_reference_text_chars,
            padding_ms=reference_padding_ms,
            min_gap_sec=min_reference_gap_sec,
            speaker_id=default_speaker_id
        )
    except ValueError:
        logger.warning("Не удалось собрать multi-reference profile, сохраняю одиночный референс.")
        valid = [s for s in segments if (s["end"] - s["start"]) <= max_audio_length_for_ref]
        if valid:
            best = max(valid, key=lambda s: s["end"] - s["start"])
            start_s = int(best["start"] * sample_rate)
            end_s = int(best["end"] * sample_rate)
            sf.write(output_ref_path, audio_data[start_s:end_s], sample_rate)
            logger.info(
                "Fallback reference сохранён: %s (%.2f сек)",
                output_ref_path,
                best["end"] - best["start"]
            )
        else:
            logger.warning("Подходящий сегмент для референса не найден.")

    return segments
