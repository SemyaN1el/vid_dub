from pathlib import Path

from src.reporting import build_run_report, summarize_tts_segments, write_run_report
from utils.pipeline_io import build_job_artifact_paths


def test_summarize_tts_segments_counts_timing_and_guard_events() -> None:
    summary = summarize_tts_segments([
        {
            "tts_group_size": 2,
            "timing_window_ms": 1000,
            "corrected_duration_sec": 1.25,
            "cheap_tail_guard_trim_ms": 120,
            "smart_sync": {"mode": "shorter"},
        },
        {
            "tts_group_size": 1,
            "timing_window_ms": 2000,
            "corrected_duration_sec": 1.8,
            "babble_guard_trim_ms": 80,
            "tts_retry_text_used": "Привет.",
        },
    ])

    assert summary["translated_count"] == 2
    assert summary["grouped_segment_count"] == 1
    assert summary["grouped_source_segment_count"] == 2
    assert summary["over_window_count"] == 1
    assert summary["cheap_tail_trim_count"] == 1
    assert summary["babble_guard_trim_count"] == 1
    assert summary["smart_sync_count"] == 1
    assert summary["tts_retry_count"] == 1


def test_build_run_report_includes_verdict_metrics_and_artifacts(tmp_path: Path) -> None:
    paths = build_job_artifact_paths("demo", str(tmp_path))
    metrics_summary = {
        "job_name": "demo",
        "created_at": "2026-05-09T04:00:00",
        "metrics": {
            "speaker_verification": 0.86,
            "wer": 0.14,
            "cer": 0.05,
            "labse_mean": 0.84,
            "labse_min": 0.80,
            "labse_max": 0.90,
        },
    }

    report = build_run_report(
        paths=paths,
        segments=[{"text": "hello"}],
        translated_segments=[
            {
                "text": "привет",
                "timing_window_ms": 1000,
                "corrected_duration_sec": 0.9,
            }
        ],
        metrics_summary=metrics_summary,
        mode="test",
    )

    assert "# Pipeline Run Report" in report
    assert "Verdict: **OK**" in report
    assert "| WER | 0.1400 |" in report
    assert "final_video" in report
    assert "subtitles_manifest" in report


def test_write_run_report_creates_markdown_file(tmp_path: Path) -> None:
    paths = build_job_artifact_paths("demo", str(tmp_path))
    Path(paths["output"]).mkdir(parents=True, exist_ok=True)

    report_path = write_run_report(
        paths=paths,
        segments=[],
        translated_segments=[],
        metrics_summary={"job_name": "demo", "metrics": {}},
        mode="test",
    )

    assert Path(report_path).exists()
    assert Path(report_path).read_text(encoding="utf-8").startswith("# Pipeline Run Report")
