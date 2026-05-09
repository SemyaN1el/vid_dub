import json
from pathlib import Path

from utils.pipeline_io import build_job_artifact_paths
from utils.pipeline_resume import step_resume_status


def _write_file(path: str | Path, content: bytes = b"x" * 256) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(content)


def _write_json(path: str | Path, payload: object) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload), encoding="utf-8")


def test_preprocess_resume_requires_all_audio_artifacts(tmp_path: Path) -> None:
    paths = build_job_artifact_paths("demo", str(tmp_path))
    for key in ("original_audio", "vocals", "background", "vocals_processed"):
        _write_file(paths[key])

    complete, missing = step_resume_status(paths, "preprocess")

    assert complete is True
    assert missing == []

    Path(paths["vocals"]).unlink()
    complete, missing = step_resume_status(paths, "preprocess")

    assert complete is False
    assert missing == ["vocals"]


def test_json_resume_artifacts_must_be_non_empty_valid_json(tmp_path: Path) -> None:
    paths = build_job_artifact_paths("demo", str(tmp_path))
    _write_json(paths["segments"], [])
    _write_file(paths["speaker_ref"])

    complete, missing = step_resume_status(paths, "asr")

    assert complete is False
    assert missing == ["segments"]

    _write_json(paths["segments"], [{"text": "hello"}])
    complete, missing = step_resume_status(paths, "asr")

    assert complete is True
    assert missing == []


def test_subtitles_resume_respects_requested_subtitle_mode(tmp_path: Path) -> None:
    paths = build_job_artifact_paths("demo", str(tmp_path))
    subtitles_dir = Path(paths["subtitles_dir"])
    srt = subtitles_dir / "subtitles_demo_ru.srt"
    vtt = subtitles_dir / "subtitles_demo_ru.vtt"
    ass = subtitles_dir / "subtitles_demo_ru.ass"
    soft_video = subtitles_dir / "final_video_soft_subs_ru.mp4"
    for path in (srt, vtt, ass, soft_video):
        _write_file(path)
    _write_json(
        subtitles_dir / "subtitles_manifest.json",
        {
            "srt": str(srt),
            "vtt": str(vtt),
            "ass": str(ass),
            "video_soft": str(soft_video),
        },
    )

    complete, missing = step_resume_status(paths, "subtitles", subtitle_mode="soft")

    assert complete is True
    assert missing == []

    complete, missing = step_resume_status(paths, "subtitles", subtitle_mode="hard")

    assert complete is False
    assert missing == ["subtitles_manifest.video_hard"]


def test_unknown_resume_contract_is_not_complete(tmp_path: Path) -> None:
    paths = build_job_artifact_paths("demo", str(tmp_path))

    complete, missing = step_resume_status(paths, "prepare_finetune")

    assert complete is False
    assert missing == ["prepare_finetune: resume contract is not defined"]
