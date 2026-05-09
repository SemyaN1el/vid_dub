import json
import os
from pathlib import Path
from typing import Any


STEP_ARTIFACTS: dict[str, tuple[tuple[str, str], ...]] = {
    "preprocess": (
        ("original_audio", "file"),
        ("vocals", "file"),
        ("background", "file"),
        ("vocals_processed", "file"),
    ),
    "asr": (
        ("segments", "json_list"),
        ("speaker_ref", "file"),
    ),
    "translate": (
        ("translated_segments", "json_list"),
    ),
    "tts": (
        ("final_voice", "file"),
        ("translated_segments", "json_list"),
    ),
    "postprocess": (
        ("final_mix", "file"),
        ("final_video", "file"),
    ),
    "metrics": (
        ("metrics_summary", "json_object"),
        ("run_report", "file"),
    ),
}


def _file_ready(path: str | os.PathLike[str], min_bytes: int = 1) -> bool:
    resolved = Path(path)
    return resolved.is_file() and resolved.stat().st_size >= min_bytes


def _load_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_list_ready(path: str | os.PathLike[str]) -> bool:
    if not _file_ready(path):
        return False
    try:
        data = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, list) and len(data) > 0


def _json_object_ready(path: str | os.PathLike[str]) -> bool:
    if not _file_ready(path):
        return False
    try:
        data = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and len(data) > 0


def _artifact_ready(path: str | os.PathLike[str], kind: str) -> bool:
    if kind == "file":
        return _file_ready(path)
    if kind == "json_list":
        return _json_list_ready(path)
    if kind == "json_object":
        return _json_object_ready(path)
    raise ValueError(f"Неизвестный тип resume artifact: {kind}")


def _manifest_file_path(value: Any, subtitles_dir: str | os.PathLike[str]) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(subtitles_dir) / path
    return path


def _subtitle_manifest_status(
    paths: dict[str, str],
    subtitle_mode: str,
) -> tuple[bool, list[str]]:
    subtitles_dir = paths["subtitles_dir"]
    manifest_path = Path(subtitles_dir) / "subtitles_manifest.json"
    missing: list[str] = []

    if not _file_ready(manifest_path):
        return False, ["subtitles_manifest.json"]

    try:
        manifest = _load_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False, ["subtitles_manifest.json"]
    if not isinstance(manifest, dict):
        return False, ["subtitles_manifest.json"]

    required_keys = ["srt", "vtt", "ass"]
    required_keys.extend({
        "soft": ["video_soft"],
        "hard": ["video_hard"],
        "both": ["video_soft", "video_hard"],
    }[subtitle_mode])

    for key in required_keys:
        path = _manifest_file_path(manifest.get(key), subtitles_dir)
        if path is None or not _file_ready(path):
            missing.append(f"subtitles_manifest.{key}")

    return len(missing) == 0, missing


def step_resume_status(
    paths: dict[str, str],
    step_name: str,
    subtitle_mode: str = "soft",
) -> tuple[bool, list[str]]:
    """
    Проверяет, можно ли пропустить шаг при --resume.

    Возвращает (complete, missing_or_invalid_labels).
    """
    if step_name == "subtitles":
        return _subtitle_manifest_status(paths, subtitle_mode)

    specs = STEP_ARTIFACTS.get(step_name)
    if not specs:
        return False, [f"{step_name}: resume contract is not defined"]

    missing = [
        key
        for key, kind in specs
        if key not in paths or not _artifact_ready(paths[key], kind)
    ]
    return len(missing) == 0, missing
