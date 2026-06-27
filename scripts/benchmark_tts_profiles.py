from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting import summarize_tts_segments
from utils.pipeline_io import build_pipeline_paths, sanitize_job_name


DEFAULT_VIDEO = PROJECT_ROOT / "data" / "input" / "smoke_20s.mp4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "output"
DEFAULT_TEST_OUTPUT_ROOT = PROJECT_ROOT / "data" / "test"
DEFAULT_JOB_PREFIX = "tts_benchmark"

RUN_MAIN_WITH_OVERRIDES = """
import json
import os
import runpy
import sys

import config as cfg

overrides = json.loads(os.environ.get("VDUB_CONFIG_OVERRIDES", "{}"))
for key, value in overrides.items():
    setattr(cfg, key, value)

sys.argv = ["main.py"] + sys.argv[1:]
runpy.run_path(os.environ["VDUB_MAIN_PATH"], run_name="__main__")
"""


@dataclass(frozen=True)
class TTSBenchmarkProfile:
    name: str
    description: str
    overrides: Mapping[str, Any]


DEFAULT_PROFILES: tuple[TTSBenchmarkProfile, ...] = (
    TTSBenchmarkProfile(
        name="baseline",
        description="Current config.py defaults",
        overrides={},
    ),
    TTSBenchmarkProfile(
        name="smart_sync_on",
        description="Enable SmartSync rewrite",
        overrides={"SMART_SYNC_ENABLED": True},
    ),
    TTSBenchmarkProfile(
        name="smart_sync_off",
        description="Disable SmartSync rewrite",
        overrides={"SMART_SYNC_ENABLED": False},
    ),
    TTSBenchmarkProfile(
        name="segment_matching_on",
        description="Enable source-vocals local level matching",
        overrides={"SEGMENT_MATCHING_ENABLED": True},
    ),
    TTSBenchmarkProfile(
        name="segment_matching_off",
        description="Disable source-vocals local level matching",
        overrides={"SEGMENT_MATCHING_ENABLED": False},
    ),
    TTSBenchmarkProfile(
        name="babble_guard_on",
        description="Enable TTS babble guard",
        overrides={"ENABLE_TTS_BABBLE_GUARD": True},
    ),
    TTSBenchmarkProfile(
        name="babble_guard_off",
        description="Disable TTS babble guard",
        overrides={"ENABLE_TTS_BABBLE_GUARD": False},
    ),
    TTSBenchmarkProfile(
        name="xtts_conservative",
        description="Lower temperature/top_p, stronger repetition penalty",
        overrides={
            "XTTS_TEMPERATURE": 0.45,
            "XTTS_TOP_P": 0.75,
            "XTTS_REPETITION_PENALTY": 2.60,
        },
    ),
    TTSBenchmarkProfile(
        name="xtts_expressive",
        description="Higher temperature/top_p, softer repetition penalty",
        overrides={
            "XTTS_TEMPERATURE": 0.65,
            "XTTS_TOP_P": 0.90,
            "XTTS_REPETITION_PENALTY": 2.10,
        },
    ),
)

PROFILE_BY_NAME = {profile.name: profile for profile in DEFAULT_PROFILES}


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _safe_delete_test_job_dir(job_dir: Path, test_root: Path = DEFAULT_TEST_OUTPUT_ROOT) -> None:
    target = job_dir.resolve()
    allowed_root = test_root.resolve()
    if target == allowed_root or allowed_root not in target.parents:
        raise RuntimeError(f"Refusing to delete outside test output root: {target}")
    if target.exists():
        shutil.rmtree(target)


def select_profiles(profile_names: list[str] | None) -> list[TTSBenchmarkProfile]:
    if not profile_names:
        return list(DEFAULT_PROFILES)

    unknown = [name for name in profile_names if name not in PROFILE_BY_NAME]
    if unknown:
        available = ", ".join(PROFILE_BY_NAME)
        raise ValueError(f"Unknown TTS benchmark profile(s): {', '.join(unknown)}. Available: {available}")
    return [PROFILE_BY_NAME[name] for name in profile_names]


def build_main_args(
    *,
    video_path: Path,
    job_name: str,
    step: str,
    subtitle_mode: str,
    mt_model: str | None,
    mt_strategy: str | None,
    resume: bool = False,
    force_step: str | None = None,
) -> list[str]:
    args = [
        "--video",
        str(video_path),
        "--job-name",
        job_name,
        "--step",
        step,
        "--test",
        "--subtitle-mode",
        subtitle_mode,
    ]
    if resume:
        args.append("--resume")
    if force_step:
        args.extend(["--force-step", force_step])
    if mt_model:
        args.extend(["--mt-model", mt_model])
    if mt_strategy:
        args.extend(["--mt-strategy", mt_strategy])
    return args


def run_main(
    *,
    main_args: list[str],
    overrides: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> None:
    cmd = [sys.executable, "-c", RUN_MAIN_WITH_OVERRIDES, *main_args]
    printable = " ".join(["python", "main.py", *main_args])
    if overrides:
        printable = f"VDUB_CONFIG_OVERRIDES={dict(overrides)} {printable}"
    print(printable, flush=True)
    if dry_run:
        return

    env = os.environ.copy()
    env["VDUB_MAIN_PATH"] = str(PROJECT_ROOT / "main.py")
    env["VDUB_CONFIG_OVERRIDES"] = json.dumps(dict(overrides or {}))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {printable}")


def prepare_upstream_artifacts(
    *,
    video_path: Path,
    prep_job_name: str,
    subtitle_mode: str,
    mt_model: str | None,
    mt_strategy: str | None,
    dry_run: bool,
    skip_prep: bool,
) -> dict[str, str]:
    prep_paths = build_pipeline_paths(
        video_path=str(video_path),
        job_name=prep_job_name,
        output_root=str(DEFAULT_OUTPUT_ROOT),
        test_output_root=str(DEFAULT_TEST_OUTPUT_ROOT),
        test=True,
    )
    if skip_prep:
        return prep_paths

    for step in ("preprocess", "asr", "translate"):
        run_main(
            main_args=build_main_args(
                video_path=video_path,
                job_name=prep_job_name,
                step=step,
                subtitle_mode=subtitle_mode,
                mt_model=mt_model,
                mt_strategy=mt_strategy,
                resume=True,
            ),
            dry_run=dry_run,
        )
    return prep_paths


def copy_upstream_artifacts(prep_paths: dict[str, str], profile_paths: dict[str, str]) -> None:
    Path(profile_paths["temp"]).mkdir(parents=True, exist_ok=True)

    for key in ("segments", "translated_segments"):
        shutil.copy2(prep_paths[key], profile_paths[key])

    for key in (
        "original_audio",
        "vocals",
        "vocals_processed",
        "background",
        "speaker_ref",
    ):
        source = Path(prep_paths[key])
        if source.exists():
            target = Path(profile_paths[key])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    source_refs = Path(prep_paths["speaker_refs_dir"])
    target_refs = Path(profile_paths["speaker_refs_dir"])
    if source_refs.exists():
        if target_refs.exists():
            shutil.rmtree(target_refs)
        shutil.copytree(source_refs, target_refs)

    source_profile = Path(prep_paths["speaker_profile"])
    if source_profile.exists():
        profile = _load_json(source_profile)
        if isinstance(profile, dict):
            profile = rewrite_speaker_profile_paths(profile, prep_paths, profile_paths)
        target_profile = Path(profile_paths["speaker_profile"])
        target_profile.parent.mkdir(parents=True, exist_ok=True)
        target_profile.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def rewrite_speaker_profile_paths(
    profile: dict[str, Any],
    prep_paths: dict[str, str],
    profile_paths: dict[str, str],
) -> dict[str, Any]:
    rewritten = dict(profile)
    replacements = {
        str(Path(prep_paths["speaker_ref"]).resolve()): str(Path(profile_paths["speaker_ref"]).resolve()),
        str(Path(prep_paths["vocals"]).resolve()): str(Path(profile_paths["vocals"]).resolve()),
    }
    prep_refs = Path(prep_paths["speaker_refs_dir"]).resolve()
    profile_refs = Path(profile_paths["speaker_refs_dir"]).resolve()

    def rewrite_path(value: Any) -> Any:
        if not isinstance(value, str) or not value:
            return value
        resolved = str(Path(value).resolve())
        if resolved in replacements:
            return replacements[resolved]
        try:
            relative = Path(resolved).relative_to(prep_refs)
        except ValueError:
            return value
        return str(profile_refs / relative)

    rewritten["merged_reference_path"] = rewrite_path(rewritten.get("merged_reference_path"))
    rewritten["reference_audio_path"] = rewrite_path(rewritten.get("reference_audio_path"))
    for list_key in ("clips", "routing_clips"):
        items = rewritten.get(list_key)
        if not isinstance(items, list):
            continue
        new_items = []
        for item in items:
            if not isinstance(item, dict):
                new_items.append(item)
                continue
            new_item = dict(item)
            new_item["path"] = rewrite_path(new_item.get("path"))
            new_items.append(new_item)
        rewritten[list_key] = new_items
    return rewritten


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _format_metric(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def _nested_config_value(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _format_config_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


def _format_config_brief(result: Mapping[str, Any]) -> str:
    parts = [
        f"SS={_format_config_value(result.get('smart_sync_enabled'))}",
        f"SM={_format_config_value(result.get('segment_matching_enabled'))}",
        f"BG={_format_config_value(result.get('babble_guard_enabled'))}",
        f"T={_format_config_value(result.get('xtts_temperature'))}",
        f"P={_format_config_value(result.get('xtts_top_p'))}",
        f"RP={_format_config_value(result.get('xtts_repetition_penalty'))}",
    ]
    return " ".join(part for part in parts if not part.endswith("="))


def collect_profile_result(
    profile: TTSBenchmarkProfile,
    job_name: str,
    paths: dict[str, str],
) -> dict[str, Any]:
    metrics_payload = _load_json(paths["metrics_summary"])
    translated_segments = _load_json(paths["translated_segments"])
    metrics = metrics_payload.get("metrics") if isinstance(metrics_payload, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    tts_config = metrics_payload.get("tts_config") if isinstance(metrics_payload, dict) else {}
    if not isinstance(tts_config, dict):
        tts_config = {}
    if not isinstance(translated_segments, list):
        translated_segments = []
    tts_summary = summarize_tts_segments(translated_segments)

    return {
        "profile": profile.name,
        "job_name": job_name,
        "description": profile.description,
        "overrides": dict(profile.overrides),
        "tts_config": tts_config,
        "smart_sync_enabled": _nested_config_value(tts_config, "smart_sync", "enabled"),
        "segment_matching_enabled": _nested_config_value(
            tts_config,
            "audio_level",
            "segment_matching_enabled",
        ),
        "babble_guard_enabled": _nested_config_value(
            tts_config,
            "tail_guards",
            "babble_guard_enabled",
        ),
        "xtts_temperature": _nested_config_value(
            tts_config,
            "xtts_generation",
            "temperature",
        ),
        "xtts_top_p": _nested_config_value(tts_config, "xtts_generation", "top_p"),
        "xtts_repetition_penalty": _nested_config_value(
            tts_config,
            "xtts_generation",
            "repetition_penalty",
        ),
        "speaker_verification": _metric(metrics, "speaker_verification"),
        "wer": _metric(metrics, "wer"),
        "cer": _metric(metrics, "cer"),
        "labse_mean": _metric(metrics, "labse_mean"),
        "labse_min": _metric(metrics, "labse_min"),
        "labse_max": _metric(metrics, "labse_max"),
        "translated_segments": tts_summary["translated_count"],
        "grouped_segments": tts_summary["grouped_segment_count"],
        "over_window": tts_summary["over_window_count"],
        "cheap_tail_trims": tts_summary["cheap_tail_trim_count"],
        "babble_guard_trims": tts_summary["babble_guard_trim_count"],
        "smart_sync_rewrites": tts_summary["smart_sync_count"],
        "tts_retry_changes": tts_summary["tts_retry_count"],
        "output_dir": paths["output"],
        "final_video": paths["final_video"],
        "final_dubbing": paths["final_voice"],
        "tts_config_json": paths["tts_config_snapshot"],
        "run_report": paths["run_report"],
        "metrics_json": paths["metrics_summary"],
    }


def write_benchmark_summary(
    *,
    results: list[dict[str, Any]],
    summary_dir: Path,
    prep_job_name: str,
    video_path: Path,
) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "tts_benchmark_summary.csv"
    json_path = summary_dir / "tts_benchmark_summary.json"
    md_path = summary_dir / "tts_benchmark_summary.md"

    fieldnames = [
        "profile",
        "job_name",
        "speaker_verification",
        "wer",
        "cer",
        "labse_mean",
        "smart_sync_enabled",
        "segment_matching_enabled",
        "babble_guard_enabled",
        "xtts_temperature",
        "xtts_top_p",
        "xtts_repetition_penalty",
        "translated_segments",
        "grouped_segments",
        "over_window",
        "cheap_tail_trims",
        "babble_guard_trims",
        "smart_sync_rewrites",
        "tts_retry_changes",
        "output_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in fieldnames})

    json_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "video": str(video_path),
                "prep_job_name": prep_job_name,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# TTS Profile Benchmark",
        "",
        f"- Video: `{video_path}`",
        f"- Prep job: `{prep_job_name}`",
        "",
        "| Profile | Config | SV | WER | CER | LaBSE | Over window | SmartSync | Babble trims | Report |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        report_path = Path(str(result["run_report"]))
        lines.append(
            "| "
            f"{result['profile']} | "
            f"`{_format_config_brief(result)}` | "
            f"{_format_metric(result['speaker_verification'])} | "
            f"{_format_metric(result['wer'])} | "
            f"{_format_metric(result['cer'])} | "
            f"{_format_metric(result['labse_mean'])} | "
            f"{result['over_window']} | "
            f"{result['smart_sync_rewrites']} | "
            f"{result['babble_guard_trims']} | "
            f"`{report_path}` |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Benchmark summary written to: {md_path}", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"JSON: {json_path}", flush=True)


def _copy_if_exists(source: str | Path | None, target_dir: Path) -> str | None:
    if not source:
        return None
    source_path = Path(source)
    if not source_path.is_file():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / source_path.name
    shutil.copy2(source_path, target_path)
    return target_path.name


def write_listen_pack(
    *,
    results: list[dict[str, Any]],
    summary_dir: Path,
    prep_job_name: str,
    video_path: Path,
) -> None:
    pack_dir = summary_dir / "listen_pack"
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)

    readme_lines = [
        "# TTS Benchmark Listen Pack",
        "",
        f"- Video: `{video_path}`",
        f"- Prep job: `{prep_job_name}`",
        "",
        "| Profile | Config | SV | WER | CER | LaBSE | Artifacts |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for result in results:
        profile_name = str(result["profile"])
        profile_dir = pack_dir / profile_name
        copied: list[str] = []
        for artifact_key in (
            "final_video",
            "final_dubbing",
            "run_report",
            "metrics_json",
            "tts_config_json",
        ):
            copied_name = _copy_if_exists(result.get(artifact_key), profile_dir)
            if copied_name:
                copied.append(copied_name)

        profile_readme = [
            f"# {profile_name}",
            "",
            f"- Description: {result.get('description', '')}",
            f"- Config: `{_format_config_brief(result)}`",
            f"- Speaker verification: {_format_metric(result.get('speaker_verification'))}",
            f"- WER: {_format_metric(result.get('wer'))}",
            f"- CER: {_format_metric(result.get('cer'))}",
            f"- LaBSE mean: {_format_metric(result.get('labse_mean'))}",
            f"- Segments over timing window: {result.get('over_window', '')}",
            f"- SmartSync rewrites: {result.get('smart_sync_rewrites', '')}",
            f"- Babble guard trims: {result.get('babble_guard_trims', '')}",
            "",
            "## Files",
            "",
        ]
        if copied:
            profile_readme.extend(f"- `{name}`" for name in copied)
        else:
            profile_readme.append("- No artifacts copied.")
        profile_readme.append("")
        (profile_dir / "README.md").write_text(
            "\n".join(profile_readme),
            encoding="utf-8",
        )

        artifact_links = ", ".join(f"`{profile_name}/{name}`" for name in copied)
        if not artifact_links:
            artifact_links = "n/a"
        readme_lines.append(
            "| "
            f"{profile_name} | "
            f"`{_format_config_brief(result)}` | "
            f"{_format_metric(result.get('speaker_verification'))} | "
            f"{_format_metric(result.get('wer'))} | "
            f"{_format_metric(result.get('cer'))} | "
            f"{_format_metric(result.get('labse_mean'))} | "
            f"{artifact_links} |"
        )

    readme_lines.append("")
    (pack_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"Listen pack written to: {pack_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TTS profile settings on the same prepared short video.",
    )
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="Short input video.")
    parser.add_argument("--job-prefix", default=DEFAULT_JOB_PREFIX, help="Prefix for data/test jobs.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help="Profile names to run. Defaults to all built-in profiles.",
    )
    parser.add_argument("--list-profiles", action="store_true", help="Print available profiles and exit.")
    parser.add_argument("--subtitle-mode", choices=["soft", "hard", "both"], default="hard")
    parser.add_argument("--mt-model", default=None, help="Optional MT model override for the prep translate step.")
    parser.add_argument("--mt-strategy", choices=["per-segment", "sentence-boundary-aware"], default=None)
    parser.add_argument("--skip-prep", action="store_true", help="Use existing <prefix>_prep artifacts.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip profile jobs that already have metrics and run_report.")
    parser.add_argument("--no-clean-profiles", action="store_true", help="Do not remove existing profile job dirs before rerun.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_profiles:
        for profile in DEFAULT_PROFILES:
            print(f"{profile.name}: {profile.description} | overrides={dict(profile.overrides)}")
        return 0

    try:
        profiles = select_profiles(args.profiles)
        job_prefix = sanitize_job_name(args.job_prefix)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    video_path = _resolve_project_path(args.video)
    if not video_path.exists():
        print(f"Benchmark input video not found: {video_path}", file=sys.stderr)
        return 2

    prep_job_name = sanitize_job_name(f"{job_prefix}_prep")
    summary_job_name = sanitize_job_name(f"{job_prefix}_summary")
    summary_paths = build_pipeline_paths(
        video_path=str(video_path),
        job_name=summary_job_name,
        output_root=str(DEFAULT_OUTPUT_ROOT),
        test_output_root=str(DEFAULT_TEST_OUTPUT_ROOT),
        test=True,
    )

    prep_paths = prepare_upstream_artifacts(
        video_path=video_path,
        prep_job_name=prep_job_name,
        subtitle_mode=args.subtitle_mode,
        mt_model=args.mt_model,
        mt_strategy=args.mt_strategy,
        dry_run=args.dry_run,
        skip_prep=args.skip_prep,
    )

    results: list[dict[str, Any]] = []
    for profile in profiles:
        profile_job_name = sanitize_job_name(f"{job_prefix}_{profile.name}")
        profile_paths = build_pipeline_paths(
            video_path=str(video_path),
            job_name=profile_job_name,
            output_root=str(DEFAULT_OUTPUT_ROOT),
            test_output_root=str(DEFAULT_TEST_OUTPUT_ROOT),
            test=True,
        )

        metrics_ready = Path(profile_paths["metrics_summary"]).is_file()
        report_ready = Path(profile_paths["run_report"]).is_file()
        if args.skip_existing and metrics_ready and report_ready:
            print(f"Skipping existing profile: {profile.name}", flush=True)
            results.append(collect_profile_result(profile, profile_job_name, profile_paths))
            continue

        if not args.no_clean_profiles and not args.dry_run:
            _safe_delete_test_job_dir(Path(profile_paths["output"]))
        if not args.dry_run:
            copy_upstream_artifacts(prep_paths, profile_paths)

        for step in ("tts", "postprocess", "subtitles", "metrics"):
            run_main(
                main_args=build_main_args(
                    video_path=video_path,
                    job_name=profile_job_name,
                    step=step,
                    subtitle_mode=args.subtitle_mode,
                    mt_model=args.mt_model,
                    mt_strategy=args.mt_strategy,
                ),
                overrides=profile.overrides,
                dry_run=args.dry_run,
            )

        if not args.dry_run:
            results.append(collect_profile_result(profile, profile_job_name, profile_paths))

    if not args.dry_run:
        summary_dir = Path(summary_paths["output"])
        write_benchmark_summary(
            results=results,
            summary_dir=summary_dir,
            prep_job_name=prep_job_name,
            video_path=video_path,
        )
        write_listen_pack(
            results=results,
            summary_dir=summary_dir,
            prep_job_name=prep_job_name,
            video_path=video_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
