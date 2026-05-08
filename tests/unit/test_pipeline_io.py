from pathlib import Path

import pytest

from utils.pipeline_io import (
    build_pipeline_paths,
    derive_job_name,
    discover_input_videos,
    resolve_input_video,
    sanitize_job_name,
)


def test_sanitize_job_name_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        sanitize_job_name(" .. ")


def test_derive_job_name_strips_legacy_video_prefix() -> None:
    assert derive_job_name("data/input/video_demo.mp4") == "demo"


def test_discover_input_videos_filters_extensions(tmp_path: Path) -> None:
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.txt").write_text("x", encoding="utf-8")

    videos = discover_input_videos(str(tmp_path), (".mp4",))

    assert [Path(path).name for path in videos] == ["a.mp4"]


def test_resolve_input_video_requires_explicit_choice_for_multiple(tmp_path: Path) -> None:
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.mp4").write_bytes(b"")

    with pytest.raises(ValueError):
        resolve_input_video(None, str(tmp_path), (".mp4",))


def test_build_pipeline_paths_uses_job_directory(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"")

    paths = build_pipeline_paths(
        video_path=str(video),
        job_name="demo job",
        output_root=str(tmp_path / "output"),
        test_output_root=str(tmp_path / "test"),
        test=True,
    )

    assert paths["job_name"] == "demo_job"
    assert Path(paths["segments"]).parts[-2:] == ("demo_job", "segments.json")
    assert Path(paths["subtitles_dir"]).parts[-2:] == ("demo_job", "subtitles")
