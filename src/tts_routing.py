import os
from typing import Any, Dict, List

from src.tts_text import _duration_bucket, _is_question, _word_count


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
