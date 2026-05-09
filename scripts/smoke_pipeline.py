from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.pipeline_io import build_pipeline_paths, sanitize_job_name


DEFAULT_VIDEO = PROJECT_ROOT / "data" / "input" / "smoke_20s.mp4"
DEFAULT_JOB_NAME = "smoke_pipeline"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "output"
DEFAULT_TEST_OUTPUT_ROOT = PROJECT_ROOT / "data" / "test"


class SmokeValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SmokeSummary:
    output_dir: Path
    segment_count: int
    translated_count: int
    metrics: dict[str, float]


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _require_file(path: str | Path, label: str, min_bytes: int = 1) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise SmokeValidationError(f"Missing {label}: {resolved}")
    if not resolved.is_file():
        raise SmokeValidationError(f"{label} is not a file: {resolved}")
    size = resolved.stat().st_size
    if size < min_bytes:
        raise SmokeValidationError(
            f"{label} is too small: {resolved} ({size} bytes)"
        )
    return resolved


def _load_json(path: str | Path, label: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise SmokeValidationError(f"Cannot read {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SmokeValidationError(f"Invalid JSON in {label}: {path}") from exc


def _load_non_empty_list(path: str | Path, label: str) -> list[dict[str, Any]]:
    data = _load_json(path, label)
    if not isinstance(data, list) or not data:
        raise SmokeValidationError(f"{label} must be a non-empty JSON list: {path}")
    if not all(isinstance(item, dict) for item in data):
        raise SmokeValidationError(f"{label} must contain JSON objects: {path}")
    return data


def _manifest_path(value: Any, subtitles_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SmokeValidationError(f"Subtitle manifest is missing {label}")
    path = Path(value)
    if not path.is_absolute():
        path = subtitles_dir / path
    return path


def validate_artifacts(
    paths: dict[str, str],
    subtitle_mode: str = "soft",
) -> SmokeSummary:
    output_dir = Path(paths["output"])

    _require_file(paths["final_voice"], "final_dubbing.wav", min_bytes=128)
    _require_file(paths["final_mix"], "final_mix.wav", min_bytes=128)
    _require_file(paths["final_video"], "final_video.mp4", min_bytes=128)
    _require_file(paths["speaker_ref"], "speaker_ref.wav", min_bytes=128)
    _require_file(paths["run_report"], "run_report.md")

    segments_path = _require_file(paths["segments"], "segments.json")
    translated_path = _require_file(
        paths["translated_segments"],
        "translated_segments.json",
    )
    segments = _load_non_empty_list(segments_path, "segments.json")
    translated = _load_non_empty_list(
        translated_path,
        "translated_segments.json",
    )

    metrics_path = _require_file(paths["metrics_summary"], "metrics.json")
    metrics_payload = _load_json(metrics_path, "metrics.json")
    if not isinstance(metrics_payload, dict):
        raise SmokeValidationError("metrics.json must be a JSON object")
    metrics = metrics_payload.get("metrics")
    if not isinstance(metrics, dict):
        raise SmokeValidationError("metrics.json must contain a metrics object")

    required_metric_keys = ("speaker_verification", "wer", "cer", "labse_mean")
    metric_values: dict[str, float] = {}
    for key in required_metric_keys:
        value = metrics.get(key)
        if not isinstance(value, (int, float)):
            raise SmokeValidationError(f"metrics.{key} must be numeric")
        metric_values[key] = float(value)

    subtitles_dir = Path(paths["subtitles_dir"])
    manifest_path = subtitles_dir / "subtitles_manifest.json"
    _require_file(manifest_path, "subtitles_manifest.json")
    manifest = _load_json(manifest_path, "subtitles_manifest.json")
    if not isinstance(manifest, dict):
        raise SmokeValidationError("subtitles_manifest.json must be a JSON object")

    for key in ("srt", "vtt", "ass"):
        _require_file(
            _manifest_path(manifest.get(key), subtitles_dir, key),
            f"subtitle {key}",
        )

    expected_video_keys = {
        "soft": ("video_soft",),
        "hard": ("video_hard",),
        "both": ("video_soft", "video_hard"),
    }[subtitle_mode]
    for key in expected_video_keys:
        _require_file(
            _manifest_path(manifest.get(key), subtitles_dir, key),
            f"subtitle {key}",
        )

    return SmokeSummary(
        output_dir=output_dir,
        segment_count=len(segments),
        translated_count=len(translated),
        metrics=metric_values,
    )


def clean_test_job_dir(job_dir: Path, test_root: Path = DEFAULT_TEST_OUTPUT_ROOT) -> None:
    target = job_dir.resolve()
    allowed_root = test_root.resolve()
    if target == allowed_root or allowed_root not in target.parents:
        raise SmokeValidationError(
            f"Refusing to delete outside test output root: {target}"
        )
    if target.exists():
        shutil.rmtree(target)


def build_main_command(args: argparse.Namespace, video_path: Path, job_name: str) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--video",
        str(video_path),
        "--job-name",
        job_name,
        "--step",
        "all",
        "--test",
        "--subtitle-mode",
        args.subtitle_mode,
    ]
    if args.mt_model:
        cmd.extend(["--mt-model", args.mt_model])
    if args.mt_strategy:
        cmd.extend(["--mt-strategy", args.mt_strategy])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and validate a short test-mode smoke pipeline.",
    )
    parser.add_argument(
        "--video",
        default=str(DEFAULT_VIDEO),
        help="Short input video for the smoke run.",
    )
    parser.add_argument(
        "--job-name",
        default=DEFAULT_JOB_NAME,
        help="Job name under data/test/.",
    )
    parser.add_argument(
        "--subtitle-mode",
        choices=["soft", "hard", "both"],
        default="soft",
        help="Subtitle mode passed to main.py.",
    )
    parser.add_argument(
        "--mt-model",
        default=None,
        help="Optional translation model override passed to main.py.",
    )
    parser.add_argument(
        "--mt-strategy",
        choices=["per-segment", "sentence-level", "sliding-window", "context-aware"],
        default=None,
        help="Optional translation strategy override passed to main.py.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Validate existing artifacts without running main.py.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove the existing data/test/<job-name> directory before run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = _resolve_project_path(args.video)
    if not video_path.exists():
        print(f"Smoke input video not found: {video_path}", file=sys.stderr)
        return 2

    try:
        job_name = sanitize_job_name(args.job_name)
    except ValueError as exc:
        print(f"Invalid smoke job name: {exc}", file=sys.stderr)
        return 2
    paths = build_pipeline_paths(
        video_path=str(video_path),
        job_name=job_name,
        output_root=str(DEFAULT_OUTPUT_ROOT),
        test_output_root=str(DEFAULT_TEST_OUTPUT_ROOT),
        test=True,
    )
    job_dir = Path(paths["output"])

    if not args.skip_run:
        if not args.no_clean:
            try:
                clean_test_job_dir(job_dir)
            except SmokeValidationError as exc:
                print(f"Smoke cleanup failed: {exc}", file=sys.stderr)
                return 1
        cmd = build_main_command(args, video_path, job_name)
        print("Running smoke pipeline:", flush=True)
        print(" ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            return result.returncode

    try:
        summary = validate_artifacts(paths, subtitle_mode=args.subtitle_mode)
    except SmokeValidationError as exc:
        print(f"Smoke validation failed: {exc}", file=sys.stderr)
        return 1

    print("Smoke validation passed", flush=True)
    print(f"Output: {summary.output_dir}", flush=True)
    print(
        "Segments: "
        f"{summary.segment_count} source / {summary.translated_count} translated",
        flush=True,
    )
    print(
        "Metrics: "
        f"SV={summary.metrics['speaker_verification']:.4f}, "
        f"WER={summary.metrics['wer']:.4f}, "
        f"CER={summary.metrics['cer']:.4f}, "
        f"LaBSE={summary.metrics['labse_mean']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
