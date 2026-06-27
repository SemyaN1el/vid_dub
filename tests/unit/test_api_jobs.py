from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import jobs
from api.app import app
from api.schemas import PipelineOptions


@pytest.fixture()
def isolated_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(jobs, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(jobs, "JOBS_ROOT", tmp_path / "data" / "jobs")
    monkeypatch.setattr(jobs, "UPLOAD_ROOT", tmp_path / "data" / "api_input")
    monkeypatch.setattr(jobs, "ALLOW_ABSOLUTE_PATHS", False)
    (tmp_path / "data" / "input").mkdir(parents=True)
    return tmp_path


def test_build_pipeline_command_includes_current_online_options(tmp_path: Path) -> None:
    options = PipelineOptions(
        step="all",
        resume=True,
        skip_metrics=True,
        force_step=["translate"],
        mt_provider="openai_compatible",
        mt_model="openai/gpt-5.4-mini",
        mt_strategy="sentence-boundary-aware",
        mt_style="compact",
        tts_provider="elevenlabs",
        subtitle_mode="hard",
    )

    command = jobs.build_pipeline_command(tmp_path / "talk.mp4", "talk_ru", options)

    assert command[:5] == [command[0], "main.py", "--step", "all", "--video"]
    assert "--job-name" in command
    assert "talk_ru" in command
    assert "--resume" in command
    assert "--skip-metrics" in command
    assert command[-2:] == ["--subtitle-mode", "hard"]
    assert "openai_compatible" in command
    assert "sentence-boundary-aware" in command
    assert "elevenlabs" in command


def test_api_defaults_to_user_facing_run_without_metrics(tmp_path: Path) -> None:
    options = PipelineOptions()

    command = jobs.build_pipeline_command(tmp_path / "talk.mp4", "talk_ru", options)

    assert options.skip_metrics is True
    assert "--skip-metrics" in command


def test_api_can_explicitly_enable_metrics(tmp_path: Path) -> None:
    options = PipelineOptions(skip_metrics=False)

    command = jobs.build_pipeline_command(tmp_path / "talk.mp4", "talk_ru", options)

    assert "--skip-metrics" not in command


def test_prepare_path_job_and_list_artifacts(isolated_api: Path) -> None:
    video_path = isolated_api / "data" / "input" / "talk.mp4"
    video_path.write_bytes(b"video")

    record = jobs.prepare_path_job(
        "data/input/talk.mp4",
        PipelineOptions(job_name="talk api", tts_provider="elevenlabs"),
    )

    assert record.status == "queued"
    assert record.output_job_name == "talk_api"
    assert Path(record.input_video) == video_path

    output_dir = Path(record.output_dir)
    output_dir.mkdir(parents=True)
    final_video = output_dir / "final_video.mp4"
    metrics = output_dir / "metrics.json"
    final_video.write_bytes(b"mp4")
    metrics.write_text("{}", encoding="utf-8")

    artifacts = jobs.list_artifacts(record.id)

    assert [item.path for item in artifacts] == ["final_video.mp4", "metrics.json"]
    assert jobs.resolve_artifact_path(record.id, "final_video.mp4") == final_video
    with pytest.raises(FileNotFoundError):
        jobs.resolve_artifact_path(record.id, "../job.json")


def test_path_endpoint_creates_job_without_running_pipeline(
    isolated_api: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = isolated_api / "data" / "input" / "talk.mp4"
    video_path.write_bytes(b"video")

    def fake_start(job_id: str) -> None:
        jobs.update_job(
            job_id,
            status="succeeded",
            started_at=jobs.utc_now(),
            finished_at=jobs.utc_now(),
            return_code=0,
        )

    monkeypatch.setattr(jobs, "start_job", fake_start)

    client = TestClient(app)
    response = client.post(
        "/jobs/from-path",
        json={
            "video_path": "data/input/talk.mp4",
            "options": {
                "job_name": "talk api",
                "mt_provider": "openai_compatible",
                "mt_model": "openai/gpt-5.4-mini",
                "mt_strategy": "sentence-boundary-aware",
                "tts_provider": "elevenlabs",
                "subtitle_mode": "hard",
            },
        },
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "succeeded"
    assert payload["output_job_name"] == "talk_api"
    assert "--skip-metrics" in payload["command"]

    status_response = client.get(f"/jobs/{payload['id']}")
    assert status_response.status_code == 200
    assert status_response.json()["return_code"] == 0
