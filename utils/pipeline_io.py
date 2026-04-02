import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def sanitize_job_name(name: str) -> str:
    """Нормализует имя задания для имён артефактов."""
    normalized = re.sub(r"[^\w.-]+", "_", name.strip(), flags=re.UNICODE)
    normalized = normalized.strip("._-")
    if not normalized:
        raise ValueError("Имя задания пустое после нормализации.")
    return normalized


def discover_input_videos(input_dir: str, extensions: Iterable[str]) -> List[str]:
    """Ищет видео в директории ввода по расширениям."""
    if not os.path.isdir(input_dir):
        return []

    allowed_exts = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in extensions
    }

    videos: List[str] = []
    for entry in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, entry)
        if os.path.isfile(path) and os.path.splitext(entry)[1].lower() in allowed_exts:
            videos.append(os.path.abspath(path))

    return videos


def resolve_input_video(
    video_path: Optional[str],
    input_dir: str,
    extensions: Iterable[str],
    legacy_suffix: Optional[str] = None
) -> str:
    """
    Разрешает путь к входному видео.

    Приоритет:
        1. Явный --video
        2. Legacy --suffix → video_<suffix>.<ext>
        3. Единственное видео в data/input
    """
    if video_path:
        resolved = os.path.abspath(os.path.expanduser(video_path))
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Видео не найдено: {resolved}")
        return resolved

    normalized_exts = [
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in extensions
    ]

    if legacy_suffix:
        for ext in normalized_exts:
            candidate = os.path.join(input_dir, f"video_{legacy_suffix}{ext}")
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        raise FileNotFoundError(
            f"Не найдено legacy-видео для suffix='{legacy_suffix}' в {input_dir}"
        )

    discovered = discover_input_videos(input_dir, normalized_exts)
    if not discovered:
        raise FileNotFoundError(
            f"В директории {input_dir} не найдено ни одного видео. Укажите --video."
        )
    if len(discovered) > 1:
        joined = ", ".join(Path(path).name for path in discovered)
        raise ValueError(
            "Найдено несколько входных видео. Укажите одно явно через --video. "
            f"Доступно: {joined}"
        )

    return discovered[0]


def derive_job_name(
    video_path: str,
    explicit_job_name: Optional[str] = None,
    legacy_suffix: Optional[str] = None
) -> str:
    """Формирует идентификатор задания для артефактов."""
    if explicit_job_name:
        return sanitize_job_name(explicit_job_name)

    if legacy_suffix:
        return sanitize_job_name(legacy_suffix)

    stem = Path(video_path).stem
    if stem.lower().startswith("video_") and len(stem) > len("video_"):
        stem = stem[len("video_"):]

    return sanitize_job_name(stem or "job")


def build_pipeline_paths(
    video_path: str,
    job_name: str,
    output_root: str,
    test_output_root: str,
    test: bool = False
) -> Dict[str, str]:
    """Строит словарь путей пайплайна из видео и имени задания."""
    base_out = test_output_root if test else output_root
    temp = os.path.join(base_out, "temp")

    return {
        "job_name":            job_name,
        "input":               os.path.dirname(video_path),
        "output":              base_out,
        "temp":                temp,
        "original_video":      os.path.abspath(video_path),
        "original_audio":      os.path.join(temp,     f"original_extracted_audio_{job_name}.wav"),
        "speaker_ref":         os.path.join(temp,     f"speaker_ref_{job_name}.wav"),
        "speaker_refs_dir":    os.path.join(temp,     f"speaker_refs_{job_name}"),
        "speaker_profile":     os.path.join(temp,     f"speaker_profile_{job_name}.json"),
        "vocals":              os.path.join(temp,     f"vocals_{job_name}.wav"),
        "vocals_processed":    os.path.join(temp,     f"vocals_processed_{job_name}.wav"),
        "background":          os.path.join(temp,     f"background_{job_name}.wav"),
        "final_voice":         os.path.join(base_out, f"final_dubbing_{job_name}.wav"),
        "final_mix":           os.path.join(base_out, f"final_mix_{job_name}.wav"),
        "final_video":         os.path.join(base_out, f"final_video_{job_name}.mp4"),
        "metrics_summary":     os.path.join(base_out, f"metrics_{job_name}.json"),
        "segments":            os.path.join(base_out, f"segments_{job_name}.json"),
        "translated_segments": os.path.join(base_out, f"translated_segments_{job_name}.json"),
        "audio_segments_dir":  os.path.join(temp,     "audio_segments"),
    }
