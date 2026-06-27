import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.smoke_pipeline import SmokeValidationError, build_main_command, validate_artifacts
from utils.pipeline_io import build_job_artifact_paths


def _write_file(path: str | Path, content: bytes = b"x" * 256) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(content)


def _write_json(path: str | Path, payload: object) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload), encoding="utf-8")


def _write_smoke_tree(tmp_path: Path) -> dict[str, str]:
    paths = build_job_artifact_paths("smoke", str(tmp_path))

    _write_file(paths["final_voice"])
    _write_file(paths["final_mix"])
    _write_file(paths["final_video"])
    _write_file(paths["speaker_ref"])
    _write_json(
        paths["tts_config_snapshot"],
        {
            "xtts_generation": {"temperature": 0.55},
            "smart_sync": {"enabled": True},
        },
    )
    _write_file(paths["run_report"])
    _write_json(paths["segments"], [{"text": "hello", "start": 0.0, "end": 1.0}])
    _write_json(
        paths["translated_segments"],
        [{"text": "привет", "start": 0.0, "end": 1.0}],
    )
    _write_json(
        Path(paths["output"]) / "translated_segments.clean.json",
        [{"text": "привет", "start": 0.0, "end": 1.0}],
    )
    _write_json(
        Path(paths["output"]) / "translation_quality.json",
        {"issue_count": 0, "translated_count": 1},
    )
    _write_json(
        paths["metrics_summary"],
        {
            "tts_config": {
                "xtts_generation": {"temperature": 0.55},
                "smart_sync": {"enabled": True},
            },
            "metrics": {
                "speaker_verification": 0.9,
                "wer": 0.1,
                "cer": 0.05,
                "labse_mean": 0.8,
            }
        },
    )

    subtitles_dir = Path(paths["subtitles_dir"])
    srt = subtitles_dir / "subtitles_smoke_ru.srt"
    vtt = subtitles_dir / "subtitles_smoke_ru.vtt"
    ass = subtitles_dir / "subtitles_smoke_ru.ass"
    hard_video = subtitles_dir / "final_video_hard_subs_ru.mp4"
    for path in (srt, vtt, ass, hard_video):
        _write_file(path)
    _write_json(
        subtitles_dir / "subtitles_manifest.json",
        {
            "srt": str(srt),
            "vtt": str(vtt),
            "ass": str(ass),
            "video_hard": str(hard_video),
        },
    )
    return paths


def test_validate_artifacts_accepts_complete_smoke_tree(tmp_path: Path) -> None:
    paths = _write_smoke_tree(tmp_path)

    summary = validate_artifacts(paths, subtitle_mode="hard")

    assert summary.segment_count == 1
    assert summary.translated_count == 1
    assert summary.metrics["wer"] == 0.1


def test_validate_artifacts_fails_on_missing_final_video(tmp_path: Path) -> None:
    paths = _write_smoke_tree(tmp_path)
    Path(paths["final_video"]).unlink()

    with pytest.raises(SmokeValidationError, match="final_video.mp4"):
        validate_artifacts(paths, subtitle_mode="hard")


def test_validate_artifacts_requires_mode_specific_subtitle_video(tmp_path: Path) -> None:
    paths = _write_smoke_tree(tmp_path)

    with pytest.raises(SmokeValidationError, match="video_soft"):
        validate_artifacts(paths, subtitle_mode="soft")


def test_validate_artifacts_requires_elevenlabs_voice_manifest(tmp_path: Path) -> None:
    paths = _write_smoke_tree(tmp_path)

    with pytest.raises(SmokeValidationError, match="elevenlabs_voice.json"):
        validate_artifacts(paths, subtitle_mode="hard", tts_provider="elevenlabs")


def test_build_main_command_for_current_online_smoke(tmp_path: Path) -> None:
    args = SimpleNamespace(
        subtitle_mode="hard",
        mt_model="openai/gpt-5.4-mini",
        mt_provider="openai_compatible",
        mt_strategy="sentence-boundary-aware",
        tts_provider="elevenlabs",
        elevenlabs_voice_id="",
        elevenlabs_voice_name="",
        elevenlabs_no_clone=False,
        resume=True,
        force_step=["translate"],
    )

    cmd = build_main_command(args, tmp_path / "smoke.mp4", "smoke")

    assert "--mt-provider" in cmd
    assert "openai_compatible" in cmd
    assert "--tts-provider" in cmd
    assert "elevenlabs" in cmd
    assert "--resume" in cmd
    assert cmd[-2:] == ["--force-step", "translate"]
