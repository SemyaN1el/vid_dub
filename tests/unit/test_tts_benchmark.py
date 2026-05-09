import json
from pathlib import Path

import pytest

from scripts.benchmark_tts_profiles import (
    build_main_args,
    collect_profile_result,
    rewrite_speaker_profile_paths,
    select_profiles,
    write_listen_pack,
)
from utils.pipeline_io import build_pipeline_paths


def _write_json(path: str | Path, payload: object) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload), encoding="utf-8")


def _write_file(path: str | Path, payload: bytes = b"x") -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(payload)


def test_select_profiles_defaults_to_builtin_order() -> None:
    profiles = select_profiles(None)

    assert profiles[0].name == "baseline"
    assert "SMART_SYNC_ENABLED" in select_profiles(["smart_sync_off"])[0].overrides


def test_select_profiles_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown TTS benchmark profile"):
        select_profiles(["missing"])


def test_build_main_args_includes_resume_and_overrides() -> None:
    args = build_main_args(
        video_path=Path("video.mp4"),
        job_name="demo",
        step="translate",
        subtitle_mode="soft",
        mt_model="model-a",
        mt_strategy="per-segment",
        resume=True,
        force_step="tts",
    )

    assert args[:6] == ["--video", "video.mp4", "--job-name", "demo", "--step", "translate"]
    assert "--resume" in args
    assert args[args.index("--force-step") + 1] == "tts"
    assert args[args.index("--mt-model") + 1] == "model-a"


def test_rewrite_speaker_profile_paths_moves_refs_to_profile_job(tmp_path: Path) -> None:
    prep_paths = build_pipeline_paths(
        video_path=str(tmp_path / "input.mp4"),
        job_name="bench_prep",
        output_root=str(tmp_path / "output"),
        test_output_root=str(tmp_path / "test"),
        test=True,
    )
    profile_paths = build_pipeline_paths(
        video_path=str(tmp_path / "input.mp4"),
        job_name="bench_baseline",
        output_root=str(tmp_path / "output"),
        test_output_root=str(tmp_path / "test"),
        test=True,
    )
    profile = {
        "merged_reference_path": prep_paths["speaker_ref"],
        "reference_audio_path": prep_paths["vocals"],
        "clips": [{"path": str(Path(prep_paths["speaker_refs_dir"]) / "ref_00.wav")}],
        "routing_clips": [{"path": str(Path(prep_paths["speaker_refs_dir"]) / "route_ref_00.wav")}],
    }

    rewritten = rewrite_speaker_profile_paths(profile, prep_paths, profile_paths)

    assert rewritten["merged_reference_path"] == str(Path(profile_paths["speaker_ref"]).resolve())
    assert rewritten["reference_audio_path"] == str(Path(profile_paths["vocals"]).resolve())
    assert Path(rewritten["clips"][0]["path"]).parts[-4:] == (
        "bench_baseline",
        "temp",
        "speaker_refs",
        "ref_00.wav",
    )
    assert Path(rewritten["routing_clips"][0]["path"]).parts[-4:] == (
        "bench_baseline",
        "temp",
        "speaker_refs",
        "route_ref_00.wav",
    )


def test_collect_profile_result_reads_metrics_and_tts_summary(tmp_path: Path) -> None:
    paths = build_pipeline_paths(
        video_path=str(tmp_path / "input.mp4"),
        job_name="bench_baseline",
        output_root=str(tmp_path / "output"),
        test_output_root=str(tmp_path / "test"),
        test=True,
    )
    profile = select_profiles(["baseline"])[0]
    _write_json(
        paths["metrics_summary"],
        {
            "tts_config": {
                "xtts_generation": {
                    "temperature": 0.55,
                    "top_p": 0.82,
                    "repetition_penalty": 2.35,
                },
                "smart_sync": {"enabled": True},
                "audio_level": {"segment_matching_enabled": False},
                "tail_guards": {"babble_guard_enabled": True},
            },
            "metrics": {
                "speaker_verification": 0.8,
                "wer": 0.2,
                "cer": 0.1,
                "labse_mean": 0.85,
            }
        },
    )
    _write_json(
        paths["translated_segments"],
        [
            {
                "text": "привет",
                "timing_window_ms": 1000,
                "corrected_duration_sec": 1.2,
                "babble_guard_trim_ms": 100,
            }
        ],
    )

    result = collect_profile_result(profile, "bench_baseline", paths)

    assert result["profile"] == "baseline"
    assert result["wer"] == 0.2
    assert result["smart_sync_enabled"] is True
    assert result["segment_matching_enabled"] is False
    assert result["babble_guard_enabled"] is True
    assert result["xtts_temperature"] == 0.55
    assert result["xtts_top_p"] == 0.82
    assert result["xtts_repetition_penalty"] == 2.35
    assert result["final_video"] == paths["final_video"]
    assert result["final_dubbing"] == paths["final_voice"]
    assert result["tts_config_json"] == paths["tts_config_snapshot"]
    assert result["over_window"] == 1
    assert result["babble_guard_trims"] == 1


def test_write_listen_pack_copies_profile_artifacts(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profile"
    summary_dir = tmp_path / "summary"
    final_video = profile_dir / "final_video.mp4"
    final_dubbing = profile_dir / "final_dubbing.wav"
    run_report = profile_dir / "run_report.md"
    metrics_json = profile_dir / "metrics.json"
    tts_config_json = profile_dir / "tts_config.json"
    for path in (final_video, final_dubbing, run_report, metrics_json, tts_config_json):
        _write_file(path)

    write_listen_pack(
        results=[
            {
                "profile": "baseline",
                "description": "Baseline",
                "speaker_verification": 0.8,
                "wer": 0.2,
                "cer": 0.1,
                "labse_mean": 0.85,
                "smart_sync_enabled": True,
                "segment_matching_enabled": False,
                "babble_guard_enabled": True,
                "xtts_temperature": 0.55,
                "xtts_top_p": 0.82,
                "xtts_repetition_penalty": 2.35,
                "over_window": 1,
                "smart_sync_rewrites": 1,
                "babble_guard_trims": 0,
                "final_video": str(final_video),
                "final_dubbing": str(final_dubbing),
                "run_report": str(run_report),
                "metrics_json": str(metrics_json),
                "tts_config_json": str(tts_config_json),
            }
        ],
        summary_dir=summary_dir,
        prep_job_name="bench_prep",
        video_path=tmp_path / "input.mp4",
    )

    pack_profile_dir = summary_dir / "listen_pack" / "baseline"
    assert (pack_profile_dir / "final_video.mp4").is_file()
    assert (pack_profile_dir / "final_dubbing.wav").is_file()
    assert (pack_profile_dir / "run_report.md").is_file()
    assert (pack_profile_dir / "metrics.json").is_file()
    assert (pack_profile_dir / "tts_config.json").is_file()
    readme = (summary_dir / "listen_pack" / "README.md").read_text(encoding="utf-8")
    assert "baseline" in readme
    assert "baseline/final_video.mp4" in readme
