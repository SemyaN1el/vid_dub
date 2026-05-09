from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from pydub import AudioSegment


MetricMap = dict[str, Any]
SegmentList = list[dict[str, Any]]


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_config_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_duration(value: float | None) -> str:
    if value is None:
        return "n/a"
    minutes, seconds = divmod(max(0.0, value), 60.0)
    if minutes >= 1:
        return f"{int(minutes)}m {seconds:04.1f}s"
    return f"{seconds:.1f}s"


def _relative_or_abs(
    path: str | os.PathLike[str] | None,
    base_dir: str | os.PathLike[str],
) -> str:
    if not path:
        return "n/a"
    resolved = Path(path)
    try:
        return str(resolved.resolve().relative_to(Path(base_dir).resolve())).replace("\\", "/")
    except (OSError, ValueError):
        return str(resolved)


def _file_status(path: str | os.PathLike[str] | None) -> str:
    if not path:
        return "missing"
    resolved = Path(path)
    if not resolved.exists():
        return "missing"
    if not resolved.is_file():
        return "not a file"
    return f"ok ({resolved.stat().st_size} bytes)"


def _audio_duration_sec(path: str | os.PathLike[str] | None) -> float | None:
    if not path or not Path(path).exists():
        return None
    try:
        return len(AudioSegment.from_file(path)) / 1000.0
    except Exception:
        return None


def _video_duration_sec(path: str | os.PathLike[str] | None) -> float | None:
    if not path or not Path(path).exists():
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            text=True,
            timeout=15,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def _count_where(segments: Iterable[dict[str, Any]], key: str) -> int:
    count = 0
    for segment in segments:
        value = _as_float(segment.get(key))
        if value is not None and value > 0:
            count += 1
    return count


def summarize_tts_segments(translated_segments: SegmentList) -> dict[str, Any]:
    grouped = [
        segment
        for segment in translated_segments
        if int(segment.get("tts_group_size", 1) or 1) > 1
    ]
    ratio_values: list[float] = []
    over_window = 0
    for segment in translated_segments:
        corrected_sec = _as_float(segment.get("corrected_duration_sec"))
        window_ms = _as_float(segment.get("timing_window_ms"))
        if corrected_sec is None or window_ms is None or window_ms <= 0:
            continue
        ratio = corrected_sec / (window_ms / 1000.0)
        ratio_values.append(ratio)
        if ratio > 1.02:
            over_window += 1

    return {
        "translated_count": len(translated_segments),
        "grouped_segment_count": len(grouped),
        "grouped_source_segment_count": sum(
            int(segment.get("tts_group_size", 1) or 1)
            for segment in grouped
        ),
        "timing_ratio_avg": (
            sum(ratio_values) / len(ratio_values) if ratio_values else None
        ),
        "timing_ratio_max": max(ratio_values) if ratio_values else None,
        "over_window_count": over_window,
        "cheap_tail_trim_count": _count_where(
            translated_segments,
            "cheap_tail_guard_trim_ms",
        ),
        "babble_guard_trim_count": _count_where(
            translated_segments,
            "babble_guard_trim_ms",
        ),
        "smart_sync_count": sum(
            1 for segment in translated_segments if segment.get("smart_sync")
        ),
        "tts_retry_count": sum(
            1 for segment in translated_segments if segment.get("tts_retry_text_used")
        ),
    }


def _metric_status(metrics: MetricMap) -> str:
    speaker = _as_float(metrics.get("speaker_verification"))
    wer = _as_float(metrics.get("wer"))
    cer = _as_float(metrics.get("cer"))
    labse = _as_float(metrics.get("labse_mean"))

    bad = (
        (speaker is not None and speaker < 0.60)
        or (wer is not None and wer > 0.40)
        or (cer is not None and cer > 0.30)
        or (labse is not None and labse < 0.65)
    )
    warning = (
        (speaker is not None and speaker < 0.75)
        or (wer is not None and wer > 0.20)
        or (cer is not None and cer > 0.15)
        or (labse is not None and labse < 0.80)
    )
    if bad:
        return "BAD"
    if warning:
        return "WARNING"
    return "OK"


def _nested_value(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _tts_config_rows(tts_config: Any) -> list[tuple[str, Any]]:
    if not isinstance(tts_config, dict):
        return []

    rows = [
        (
            "XTTS temperature",
            _nested_value(tts_config, "xtts_generation", "temperature"),
        ),
        ("XTTS top_p", _nested_value(tts_config, "xtts_generation", "top_p")),
        (
            "XTTS repetition_penalty",
            _nested_value(tts_config, "xtts_generation", "repetition_penalty"),
        ),
        ("SmartSync enabled", _nested_value(tts_config, "smart_sync", "enabled")),
        ("SmartSync provider", _nested_value(tts_config, "smart_sync", "provider")),
        (
            "SmartSync max_rewrites",
            _nested_value(tts_config, "smart_sync", "max_rewrites"),
        ),
        ("TTS grouping enabled", _nested_value(tts_config, "runtime", "grouping_enabled")),
        (
            "Segment routing enabled",
            _nested_value(tts_config, "segment_routing", "enabled"),
        ),
        (
            "Segment matching enabled",
            _nested_value(tts_config, "audio_level", "segment_matching_enabled"),
        ),
        (
            "Babble guard enabled",
            _nested_value(tts_config, "tail_guards", "babble_guard_enabled"),
        ),
        (
            "ASR retry enabled",
            _nested_value(tts_config, "tail_guards", "asr_retry_enabled"),
        ),
        ("Target dBFS", _nested_value(tts_config, "audio_level", "target_dbfs")),
    ]
    return [(label, value) for label, value in rows if value is not None]


def build_run_report(
    *,
    paths: dict[str, str],
    segments: SegmentList,
    translated_segments: SegmentList,
    metrics_summary: dict[str, Any],
    mode: str,
) -> str:
    metrics = metrics_summary.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    tts_summary = summarize_tts_segments(translated_segments)
    output_dir = paths.get("output", "")
    created_at = metrics_summary.get("created_at") or datetime.now().isoformat(
        timespec="seconds"
    )

    artifact_paths = {
        "final_video": paths.get("final_video"),
        "final_dubbing": paths.get("final_voice"),
        "final_mix": paths.get("final_mix"),
        "tts_config": paths.get("tts_config_snapshot"),
        "metrics": paths.get("metrics_summary"),
        "translated_segments": paths.get("translated_segments"),
        "subtitles_manifest": os.path.join(
            paths.get("subtitles_dir", ""),
            "subtitles_manifest.json",
        ),
    }

    lines = [
        "# Pipeline Run Report",
        "",
        "## Summary",
        "",
        f"- Job: `{metrics_summary.get('job_name') or paths.get('job_name', 'n/a')}`",
        f"- Mode: `{mode}`",
        f"- Created: `{created_at}`",
        f"- Verdict: **{_metric_status(metrics)}**",
        f"- Output: `{output_dir}`",
        "",
        "## Durations",
        "",
        f"- Source video: {_format_duration(_video_duration_sec(paths.get('original_video')))}",
        f"- Final dubbing: {_format_duration(_audio_duration_sec(paths.get('final_voice')))}",
        f"- Final mix: {_format_duration(_audio_duration_sec(paths.get('final_mix')))}",
        "",
        "## Segments",
        "",
        f"- Source segments: {len(segments)}",
        f"- Translated/TTS segments: {tts_summary['translated_count']}",
        (
            "- Grouped TTS segments: "
            f"{tts_summary['grouped_segment_count']} groups / "
            f"{tts_summary['grouped_source_segment_count']} source segments"
        ),
        f"- Segments over timing window: {tts_summary['over_window_count']}",
        f"- Average corrected/window ratio: {_format_float(tts_summary['timing_ratio_avg'], 3)}",
        f"- Max corrected/window ratio: {_format_float(tts_summary['timing_ratio_max'], 3)}",
        f"- Cheap tail guard trims: {tts_summary['cheap_tail_trim_count']}",
        f"- Babble guard trims: {tts_summary['babble_guard_trim_count']}",
        f"- SmartSync accepted rewrites: {tts_summary['smart_sync_count']}",
        f"- TTS retry text changes: {tts_summary['tts_retry_count']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Speaker Verification | {_format_float(_as_float(metrics.get('speaker_verification')))} |",
        f"| WER | {_format_float(_as_float(metrics.get('wer')))} |",
        f"| CER | {_format_float(_as_float(metrics.get('cer')))} |",
        f"| LaBSE mean | {_format_float(_as_float(metrics.get('labse_mean')))} |",
        f"| LaBSE min | {_format_float(_as_float(metrics.get('labse_min')))} |",
        f"| LaBSE max | {_format_float(_as_float(metrics.get('labse_max')))} |",
        "",
    ]

    tts_config_rows = _tts_config_rows(metrics_summary.get("tts_config"))
    if tts_config_rows:
        lines.extend([
            "## TTS Config",
            "",
            "| Setting | Value |",
            "| --- | ---: |",
        ])
        for label, value in tts_config_rows:
            lines.append(f"| {label} | `{_format_config_value(value)}` |")
        lines.append("")

    lines.extend([
        "## Artifacts",
        "",
        "| Artifact | Path | Status |",
        "| --- | --- | --- |",
    ])
    for label, path in artifact_paths.items():
        lines.append(
            f"| {label} | `{_relative_or_abs(path, output_dir)}` | {_file_status(path)} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_run_report(
    *,
    paths: dict[str, str],
    segments: SegmentList,
    translated_segments: SegmentList,
    metrics_summary: dict[str, Any],
    mode: str,
) -> str:
    report_path = paths["run_report"]
    report = build_run_report(
        paths=paths,
        segments=segments,
        translated_segments=translated_segments,
        metrics_summary=metrics_summary,
        mode=mode,
    )
    Path(report_path).write_text(report, encoding="utf-8")
    return report_path
