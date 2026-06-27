from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


PipelineStep = Literal[
    "preprocess",
    "asr",
    "translate",
    "tts",
    "postprocess",
    "subtitles",
    "metrics",
    "prepare_finetune",
    "all",
]
ForceStep = PipelineStep
TranslationProvider = Literal[
    "hf",
    "gemini",
    "openai",
    "openrouter",
    "groq",
    "cerebras",
    "openai_compatible",
]
TranslationStrategy = Literal[
    "per-segment",
    "sentence-boundary-aware",
    "boundary-aware",
]
TranslationStyle = Literal["standard", "academic", "casual", "news", "compact"]
TtsProvider = Literal["xtts", "elevenlabs"]
SubtitleMode = Literal["soft", "hard", "both"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


class PipelineOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: PipelineStep = "all"
    job_name: str | None = None
    test: bool = False
    resume: bool = False
    skip_metrics: bool = True
    force_step: list[ForceStep] = Field(default_factory=list)
    mt_provider: TranslationProvider | None = None
    mt_model: str | None = None
    mt_strategy: TranslationStrategy | None = None
    mt_style: TranslationStyle | None = None
    mt_profile: str | None = None
    mt_asr_correction: list[str] = Field(default_factory=list)
    tts_provider: TtsProvider | None = None
    elevenlabs_voice_id: str | None = None
    elevenlabs_voice_name: str | None = None
    elevenlabs_no_clone: bool = False
    subtitle_mode: SubtitleMode | None = None
    subtitle_original: bool = False


class PathJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_path: str
    options: PipelineOptions = Field(default_factory=PipelineOptions)


class JobRecord(BaseModel):
    id: str
    status: JobStatus
    created_at: str
    updated_at: str
    command: list[str]
    input_video: str
    output_job_name: str
    output_dir: str
    log_path: str
    test: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    process_pid: int | None = None
    return_code: int | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    project_root: str


class JobListResponse(BaseModel):
    jobs: list[JobRecord]


class ArtifactEntry(BaseModel):
    name: str
    path: str
    size_bytes: int
    download_url: str


class ArtifactListResponse(BaseModel):
    job_id: str
    output_dir: str
    artifacts: list[ArtifactEntry]


class LogResponse(BaseModel):
    job_id: str
    log: str
