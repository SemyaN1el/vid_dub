from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import config as cfg
from api.schemas import ArtifactEntry, JobRecord, PipelineOptions
from utils.pipeline_io import sanitize_job_name


PROJECT_ROOT = Path(os.getenv("VIDEO_DUBBING_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
JOBS_ROOT = (PROJECT_ROOT / os.getenv("VIDEO_DUBBING_JOBS_DIR", "data/jobs")).resolve()
UPLOAD_ROOT = (PROJECT_ROOT / os.getenv("VIDEO_DUBBING_UPLOAD_DIR", "data/api_input")).resolve()
ALLOW_ABSOLUTE_PATHS = os.getenv("VIDEO_DUBBING_API_ALLOW_ABSOLUTE_PATHS", "0").lower() in {
    "1",
    "true",
    "yes",
}
MAX_WORKERS = max(1, int(os.getenv("VIDEO_DUBBING_API_MAX_WORKERS", "1")))
RUN_LOCK = threading.Semaphore(MAX_WORKERS)

ARTIFACT_GLOBS = (
    "final_video.mp4",
    "final_mix.wav",
    "final_dubbing.wav",
    "metrics.json",
    "run_report.md",
    "segments.json",
    "translated_segments.json",
    "translated_segments.clean.json",
    "translation_quality.json",
    "elevenlabs_voice.json",
    "subtitles/*",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def safe_filename(filename: str | None) -> str:
    name = Path(filename or "input.mp4").name
    stem = sanitize_job_name(Path(name).stem or "input")
    suffix = Path(name).suffix.lower() or ".mp4"
    return f"{stem}{suffix}"


def metadata_path(job_id: str) -> Path:
    return JOBS_ROOT / job_id / "job.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def load_job(job_id: str) -> JobRecord:
    path = metadata_path(job_id)
    if not path.is_file():
        raise FileNotFoundError(f"Job not found: {job_id}")
    return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))


def save_job(record: JobRecord) -> None:
    write_json(metadata_path(record.id), record.model_dump(mode="json"))


def update_job(job_id: str, **updates: object) -> JobRecord:
    record = load_job(job_id)
    payload = record.model_dump(mode="json")
    payload.update(updates)
    payload["updated_at"] = utc_now()
    updated = JobRecord.model_validate(payload)
    save_job(updated)
    return updated


def list_jobs(limit: int = 50) -> list[JobRecord]:
    if not JOBS_ROOT.is_dir():
        return []
    jobs: list[JobRecord] = []
    for path in JOBS_ROOT.glob("*/job.json"):
        try:
            jobs.append(JobRecord.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    jobs.sort(key=lambda item: item.created_at, reverse=True)
    return jobs[:limit]


def output_root_for_options(options: PipelineOptions) -> Path:
    root = getattr(cfg, "TEST_OUTPUT_PATH", "./data/test") if options.test else getattr(cfg, "OUTPUT_PATH", "./data/output")
    return resolve_project_path(root)


def build_pipeline_command(video_path: Path, output_job_name: str, options: PipelineOptions) -> list[str]:
    command = [
        sys.executable,
        "main.py",
        "--step",
        options.step,
        "--video",
        str(video_path),
        "--job-name",
        output_job_name,
    ]
    if options.test:
        command.append("--test")
    if options.resume:
        command.append("--resume")
    if options.skip_metrics:
        command.append("--skip-metrics")
    for step in options.force_step:
        command.extend(["--force-step", step])
    if options.mt_provider:
        command.extend(["--mt-provider", options.mt_provider])
    if options.mt_model:
        command.extend(["--mt-model", options.mt_model])
    if options.mt_strategy:
        command.extend(["--mt-strategy", options.mt_strategy])
    if options.mt_style:
        command.extend(["--mt-style", options.mt_style])
    if options.mt_profile:
        command.extend(["--mt-profile", options.mt_profile])
    for correction in options.mt_asr_correction:
        command.extend(["--mt-asr-correction", correction])
    if options.tts_provider:
        command.extend(["--tts-provider", options.tts_provider])
    if options.elevenlabs_voice_id:
        command.extend(["--elevenlabs-voice-id", options.elevenlabs_voice_id])
    if options.elevenlabs_voice_name:
        command.extend(["--elevenlabs-voice-name", options.elevenlabs_voice_name])
    if options.elevenlabs_no_clone:
        command.append("--elevenlabs-no-clone")
    if options.subtitle_mode:
        command.extend(["--subtitle-mode", options.subtitle_mode])
    if options.subtitle_original:
        command.append("--subtitle-original")
    return command


def create_job_record(
    input_video: Path,
    options: PipelineOptions,
    *,
    job_id: str | None = None,
) -> JobRecord:
    input_video = input_video.resolve()
    output_job_name = sanitize_job_name(options.job_name or uuid.uuid4().hex[:12])
    job_id = job_id or uuid.uuid4().hex
    now = utc_now()
    output_dir = output_root_for_options(options) / output_job_name
    log_path = JOBS_ROOT / job_id / "run.log"
    command = build_pipeline_command(input_video, output_job_name, options)
    record = JobRecord(
        id=job_id,
        status="queued",
        created_at=now,
        updated_at=now,
        command=command,
        input_video=str(input_video),
        output_job_name=output_job_name,
        output_dir=str(output_dir),
        log_path=str(log_path),
        test=options.test,
    )
    save_job(record)
    return record


def prepare_uploaded_job(filename: str | None, options: PipelineOptions) -> JobRecord:
    job_id = uuid.uuid4().hex
    upload_dir = UPLOAD_ROOT / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_video = upload_dir / safe_filename(filename)

    output_job_name = sanitize_job_name(options.job_name or job_id[:12])
    options = options.model_copy(update={"job_name": output_job_name})
    record = create_job_record(input_video, options, job_id=job_id)
    return record


def prepare_path_job(video_path: str, options: PipelineOptions) -> JobRecord:
    input_video = resolve_project_path(video_path)
    if not input_video.is_file():
        raise FileNotFoundError(f"Video not found: {input_video}")
    if not ALLOW_ABSOLUTE_PATHS and not is_relative_to(input_video, PROJECT_ROOT):
        raise ValueError(
            "Video path must be inside the project directory. "
            "Set VIDEO_DUBBING_API_ALLOW_ABSOLUTE_PATHS=1 to allow external paths."
        )
    return create_job_record(input_video, options)


def run_job(job_id: str) -> None:
    with RUN_LOCK:
        record = load_job(job_id)
        log_path = Path(record.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
                started_at = utc_now()
                process = subprocess.Popen(
                    record.command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                update_job(
                    job_id,
                    status="running",
                    started_at=started_at,
                    process_pid=process.pid,
                )
                return_code = process.wait()
            finished_at = utc_now()
            update_job(
                job_id,
                status="succeeded" if return_code == 0 else "failed",
                finished_at=finished_at,
                return_code=return_code,
            )
        except Exception as exc:
            update_job(
                job_id,
                status="failed",
                finished_at=utc_now(),
                error=str(exc),
            )


def start_job(job_id: str) -> None:
    thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
    thread.start()


def read_log(job_id: str, tail_bytes: int = 65536) -> str:
    record = load_job(job_id)
    path = Path(record.log_path)
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - tail_bytes), os.SEEK_SET)
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def list_artifacts(job_id: str) -> list[ArtifactEntry]:
    record = load_job(job_id)
    output_dir = Path(record.output_dir).resolve()
    artifacts: list[ArtifactEntry] = []
    if not output_dir.is_dir():
        return artifacts

    seen: set[Path] = set()
    for pattern in ARTIFACT_GLOBS:
        for path in sorted(output_dir.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            relative = path.relative_to(output_dir).as_posix()
            artifacts.append(
                ArtifactEntry(
                    name=path.name,
                    path=relative,
                    size_bytes=path.stat().st_size,
                    download_url=f"/jobs/{job_id}/download/{relative}",
                )
            )
    return artifacts


def resolve_artifact_path(job_id: str, relative_path: str) -> Path:
    record = load_job(job_id)
    output_dir = Path(record.output_dir).resolve()
    path = (output_dir / relative_path).resolve()
    if not is_relative_to(path, output_dir) or not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {relative_path}")
    return path
