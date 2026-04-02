import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Tuple

from pydub import AudioSegment

logger = logging.getLogger(__name__)


def normalize_dataset_text(text: str) -> str:
    """Готовит текст для metadata-манифестов XTTS."""
    normalized = re.sub(r"\s+", " ", text.replace("|", " ")).strip()
    return normalized


def _validate_segment(
    text: str,
    duration_sec: float,
    min_sec: float,
    max_sec: float,
    min_chars: int,
    max_chars: int
) -> str | None:
    """Возвращает причину пропуска или None, если сегмент подходит."""
    if duration_sec < min_sec:
        return "too_short"
    if duration_sec > max_sec:
        return "too_long"
    if len(text) < min_chars:
        return "text_too_short"
    if len(text) > max_chars:
        return "text_too_long"
    if len(text.split()) < 2:
        return "not_enough_words"
    return None


def _write_manifest(manifest_path: str, records: List[Dict[str, Any]]) -> None:
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(f"{item['audio_file']}|{item['text']}|{item['speaker_name']}\n")


def _dataset_paths(dataset_root: str) -> Dict[str, str]:
    clips_dir = os.path.join(dataset_root, "clips")
    return {
        "root": dataset_root,
        "clips": clips_dir,
        "train_manifest": os.path.join(dataset_root, "metadata_train.csv"),
        "eval_manifest": os.path.join(dataset_root, "metadata_eval.csv"),
        "summary": os.path.join(dataset_root, "dataset_summary.json"),
        "references": os.path.join(dataset_root, "reference_candidates.json"),
    }


def prepare_finetune_dataset(
    audio_path: str,
    segments: List[Dict[str, Any]],
    dataset_root: str,
    speaker_name: str,
    sample_rate: int,
    min_sec: float,
    max_sec: float,
    min_chars: int,
    max_chars: int,
    padding_ms: int,
    eval_ratio: float,
    max_eval_samples: int,
    reference_clips: int,
    seed: int,
    target_speaker_id: str | None = None
) -> Dict[str, Any]:
    """
    Готовит датасет под XTTS fine-tuning из сегментов ASR и WAV-дорожки.

    Результат:
    - clips/*.wav
    - metadata_train.csv
    - metadata_eval.csv
    - dataset_summary.json
    - reference_candidates.json
    """
    paths = _dataset_paths(dataset_root)
    os.makedirs(paths["clips"], exist_ok=True)

    source_audio = AudioSegment.from_wav(audio_path).set_channels(1).set_frame_rate(sample_rate)
    total_audio_ms = len(source_audio)

    kept_records: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}

    for idx, seg in enumerate(segments):
        if target_speaker_id and seg.get("speaker_id") != target_speaker_id:
            skipped["speaker_mismatch"] = skipped.get("speaker_mismatch", 0) + 1
            continue

        text = normalize_dataset_text(seg.get("text", ""))
        if not text:
            skipped["empty_text"] = skipped.get("empty_text", 0) + 1
            continue

        start_sec = float(seg.get("start", 0.0))
        end_sec = float(seg.get("end", start_sec))
        if end_sec <= start_sec:
            skipped["invalid_timing"] = skipped.get("invalid_timing", 0) + 1
            continue

        duration_sec = end_sec - start_sec
        reason = _validate_segment(
            text=text,
            duration_sec=duration_sec,
            min_sec=min_sec,
            max_sec=max_sec,
            min_chars=min_chars,
            max_chars=max_chars
        )
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        start_ms = max(0, int(start_sec * 1000) - padding_ms)
        end_ms = min(total_audio_ms, int(end_sec * 1000) + padding_ms)
        clip = source_audio[start_ms:end_ms]

        if len(clip) <= 0 or clip.rms == 0:
            skipped["silent_clip"] = skipped.get("silent_clip", 0) + 1
            continue

        file_name = f"{len(kept_records):05d}.wav"
        clip_path = os.path.join(paths["clips"], file_name)
        clip.export(clip_path, format="wav")

        kept_records.append({
            "audio_file": f"clips/{file_name}",
            "text": text,
            "speaker_name": speaker_name,
            "speaker_id": seg.get("speaker_id"),
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "duration_sec": round(len(clip) / 1000.0, 3),
            "char_count": len(text),
            "word_count": len(text.split()),
        })

    if len(kept_records) < 2:
        raise ValueError(
            "Недостаточно сегментов для fine-tuning. "
            f"Подготовлено только {len(kept_records)} клипов."
        )

    rng = random.Random(seed)
    shuffled = kept_records.copy()
    rng.shuffle(shuffled)

    eval_count = min(max(1, round(len(shuffled) * eval_ratio)), max_eval_samples, len(shuffled) - 1)
    eval_records = sorted(shuffled[:eval_count], key=lambda item: item["audio_file"])
    train_records = sorted(shuffled[eval_count:], key=lambda item: item["audio_file"])

    _write_manifest(paths["train_manifest"], train_records)
    _write_manifest(paths["eval_manifest"], eval_records)

    references = sorted(
        kept_records,
        key=lambda item: item["duration_sec"],
        reverse=True
    )[:reference_clips]

    summary = {
        "dataset_root": os.path.abspath(paths["root"]),
        "source_audio_path": os.path.abspath(audio_path),
        "speaker_name": speaker_name,
        "target_speaker_id": target_speaker_id,
        "sample_rate": sample_rate,
        "total_segments": len(segments),
        "kept_clips": len(kept_records),
        "train_clips": len(train_records),
        "eval_clips": len(eval_records),
        "total_duration_sec": round(sum(item["duration_sec"] for item in kept_records), 2),
        "avg_clip_duration_sec": round(
            sum(item["duration_sec"] for item in kept_records) / len(kept_records), 2
        ),
        "skipped": skipped,
    }

    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(paths["references"], "w", encoding="utf-8") as f:
        json.dump(references, f, ensure_ascii=False, indent=2)

    logger.info(
        "Fine-tuning датасет готов: %s клипов, %.2f минут, train=%s, eval=%s",
        summary["kept_clips"],
        summary["total_duration_sec"] / 60.0,
        summary["train_clips"],
        summary["eval_clips"]
    )
    logger.info("Манифест train: %s", paths["train_manifest"])
    logger.info("Манифест eval:  %s", paths["eval_manifest"])

    return {
        "paths": paths,
        "summary": summary,
        "references": references,
    }
