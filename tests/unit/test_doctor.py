from pathlib import Path
from types import SimpleNamespace

from src.doctor import format_doctor_report, has_doctor_failures, run_project_doctor


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        INPUT_PATH=str(tmp_path / "data" / "input"),
        INPUT_VIDEO_EXTENSIONS=(".mp4", ".mov"),
        OUTPUT_PATH=str(tmp_path / "data" / "output"),
        TEST_OUTPUT_PATH=str(tmp_path / "data" / "test"),
        MODEL_TTS_DIR=str(tmp_path / "original_tts_model"),
        ASR_PROVIDER="local",
        METRICS_ASR_PROVIDER="local",
        MT_MODEL_NAME="facebook/nllb-200-distilled-1.3B",
        SMART_SYNC_ENABLED=False,
    )


def _write_required_files(tmp_path: Path) -> None:
    (tmp_path / "config.py").write_text("# local config\n", encoding="utf-8")
    input_dir = tmp_path / "data" / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "smoke.mp4").write_bytes(b"video")
    model_dir = tmp_path / "original_tts_model"
    model_dir.mkdir()
    for filename in ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth"):
        (model_dir / filename).write_bytes(b"x")


def test_run_project_doctor_accepts_complete_local_structure(tmp_path: Path) -> None:
    _write_required_files(tmp_path)

    checks = run_project_doctor(
        config=_config(tmp_path),
        project_root=tmp_path,
        include_cli_checks=False,
        include_dependency_checks=False,
        include_api_key_checks=False,
    )

    assert has_doctor_failures(checks) is False
    assert "Verdict: OK" in format_doctor_report(checks)


def test_run_project_doctor_fails_when_explicit_video_is_missing(tmp_path: Path) -> None:
    _write_required_files(tmp_path)

    checks = run_project_doctor(
        config=_config(tmp_path),
        project_root=tmp_path,
        video_path=str(tmp_path / "missing.mp4"),
        include_cli_checks=False,
        include_dependency_checks=False,
        include_api_key_checks=False,
    )

    assert has_doctor_failures(checks) is True
    input_check = next(check for check in checks if check.name == "Input video")
    assert input_check.status == "FAIL"
    assert "missing.mp4" in input_check.detail


def test_run_project_doctor_requires_explicit_video_when_multiple_exist(tmp_path: Path) -> None:
    _write_required_files(tmp_path)
    (tmp_path / "data" / "input" / "extra.mov").write_bytes(b"video")

    checks = run_project_doctor(
        config=_config(tmp_path),
        project_root=tmp_path,
        include_cli_checks=False,
        include_dependency_checks=False,
        include_api_key_checks=False,
    )

    assert has_doctor_failures(checks) is True
    input_check = next(check for check in checks if check.name == "Input video")
    assert input_check.status == "FAIL"
    assert "multiple videos found" in input_check.detail
