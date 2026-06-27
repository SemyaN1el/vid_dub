from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from api import jobs as job_service
from api.schemas import (
    ArtifactListResponse,
    HealthResponse,
    JobListResponse,
    JobRecord,
    LogResponse,
    PathJobRequest,
    PipelineOptions,
)


app = FastAPI(
    title="Video Dubbing API",
    version="0.1.0",
    description="HTTP wrapper around the video dubbing CLI pipeline.",
)

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"
if WEB_ROOT.is_dir():
    app.mount("/ui", StaticFiles(directory=WEB_ROOT, html=True), name="ui")


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_form_options(
    *,
    step: str,
    job_name: str | None,
    test: bool,
    resume: bool,
    skip_metrics: bool,
    force_step: str | None,
    mt_provider: str | None,
    mt_model: str | None,
    mt_strategy: str | None,
    mt_style: str | None,
    mt_profile: str | None,
    mt_asr_correction: str | None,
    tts_provider: str | None,
    elevenlabs_voice_id: str | None,
    elevenlabs_voice_name: str | None,
    elevenlabs_no_clone: bool,
    subtitle_mode: str | None,
    subtitle_original: bool,
) -> PipelineOptions:
    return PipelineOptions.model_validate(
        {
            "step": step,
            "job_name": job_name or None,
            "test": test,
            "resume": resume,
            "skip_metrics": skip_metrics,
            "force_step": parse_csv(force_step),
            "mt_provider": mt_provider or None,
            "mt_model": mt_model or None,
            "mt_strategy": mt_strategy or None,
            "mt_style": mt_style or None,
            "mt_profile": mt_profile or None,
            "mt_asr_correction": parse_csv(mt_asr_correction),
            "tts_provider": tts_provider or None,
            "elevenlabs_voice_id": elevenlabs_voice_id or None,
            "elevenlabs_voice_name": elevenlabs_voice_name or None,
            "elevenlabs_no_clone": elevenlabs_no_clone,
            "subtitle_mode": subtitle_mode or None,
            "subtitle_original": subtitle_original,
        }
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", project_root=str(job_service.PROJECT_ROOT))


@app.post("/jobs", response_model=JobRecord, status_code=status.HTTP_202_ACCEPTED)
async def create_upload_job(
    video: UploadFile = File(...),
    step: str = Form("all"),
    job_name: str | None = Form(None),
    test: bool = Form(False),
    resume: bool = Form(False),
    skip_metrics: bool = Form(True),
    force_step: str | None = Form(None, description="Comma-separated steps."),
    mt_provider: str | None = Form(None),
    mt_model: str | None = Form(None),
    mt_strategy: str | None = Form(None),
    mt_style: str | None = Form(None),
    mt_profile: str | None = Form(None),
    mt_asr_correction: str | None = Form(None, description="Comma-separated correction hints."),
    tts_provider: str | None = Form(None),
    elevenlabs_voice_id: str | None = Form(None),
    elevenlabs_voice_name: str | None = Form(None),
    elevenlabs_no_clone: bool = Form(False),
    subtitle_mode: str | None = Form(None),
    subtitle_original: bool = Form(False),
) -> JobRecord:
    try:
        options = build_form_options(
            step=step,
            job_name=job_name,
            test=test,
            resume=resume,
            skip_metrics=skip_metrics,
            force_step=force_step,
            mt_provider=mt_provider,
            mt_model=mt_model,
            mt_strategy=mt_strategy,
            mt_style=mt_style,
            mt_profile=mt_profile,
            mt_asr_correction=mt_asr_correction,
            tts_provider=tts_provider,
            elevenlabs_voice_id=elevenlabs_voice_id,
            elevenlabs_voice_name=elevenlabs_voice_name,
            elevenlabs_no_clone=elevenlabs_no_clone,
            subtitle_mode=subtitle_mode,
            subtitle_original=subtitle_original,
        )
        record = job_service.prepare_uploaded_job(video.filename, options)
        input_path = Path(record.input_video)
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with input_path.open("wb") as output:
            while chunk := await video.read(1024 * 1024):
                output.write(chunk)
        job_service.start_job(record.id)
        return job_service.load_job(record.id)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await video.close()


@app.post("/jobs/from-path", response_model=JobRecord, status_code=status.HTTP_202_ACCEPTED)
def create_path_job(request: PathJobRequest) -> JobRecord:
    try:
        record = job_service.prepare_path_job(request.video_path, request.options)
        job_service.start_job(record.id)
        return job_service.load_job(record.id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/jobs", response_model=JobListResponse)
def list_jobs(limit: int = 50) -> JobListResponse:
    return JobListResponse(jobs=job_service.list_jobs(limit=limit))


@app.get("/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    try:
        return job_service.load_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/jobs/{job_id}/logs", response_model=LogResponse)
def get_job_logs(job_id: str, tail_bytes: int = 65536) -> LogResponse:
    try:
        return LogResponse(job_id=job_id, log=job_service.read_log(job_id, tail_bytes=tail_bytes))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/jobs/{job_id}/artifacts", response_model=ArtifactListResponse)
def get_job_artifacts(job_id: str) -> ArtifactListResponse:
    try:
        record = job_service.load_job(job_id)
        return ArtifactListResponse(
            job_id=job_id,
            output_dir=record.output_dir,
            artifacts=job_service.list_artifacts(job_id),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/jobs/{job_id}/download/{artifact_path:path}")
def download_artifact(job_id: str, artifact_path: str) -> FileResponse:
    try:
        path = job_service.resolve_artifact_path(job_id, artifact_path)
        return FileResponse(path, filename=path.name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
