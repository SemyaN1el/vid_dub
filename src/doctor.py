"""Диагностика проекта и окружения (--doctor): пути, входное видео, модельные файлы, наличие CLI, зависимостей и API-ключей."""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from utils.pipeline_io import discover_input_videos


ConfigSource = Any


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str = ""
    required: bool = True

    @property
    def status(self) -> str:
        if self.ok:
            return "OK"
        return "FAIL" if self.required else "WARN"


def _module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _resolve_path(project_root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _add_path_exists(
    checks: list[DoctorCheck],
    *,
    name: str,
    path: Path,
    required: bool = True,
) -> None:
    checks.append(DoctorCheck(name, path.exists(), str(path), required))


def _check_writable_dir(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".doctor_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, str(path)
    except OSError as exc:
        return False, f"{path}: {exc}"


def _check_input_video(
    *,
    config: ConfigSource,
    project_root: Path,
    video_path: str | None,
    legacy_suffix: str | None,
) -> DoctorCheck:
    input_dir = _resolve_path(
        project_root,
        getattr(config, "INPUT_PATH", "./data/input"),
    )
    extensions = getattr(config, "INPUT_VIDEO_EXTENSIONS", (".mp4", ".mov", ".mkv"))

    if video_path:
        resolved = _resolve_path(project_root, video_path)
        return DoctorCheck("Input video", resolved.is_file(), str(resolved))

    normalized_exts = [
        ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
        for ext in extensions
    ]
    if legacy_suffix:
        for ext in normalized_exts:
            candidate = input_dir / f"video_{legacy_suffix}{ext}"
            if candidate.is_file():
                return DoctorCheck("Input video", True, str(candidate))
        return DoctorCheck(
            "Input video",
            False,
            f"video_{legacy_suffix}<ext> not found in {input_dir}",
        )

    videos = discover_input_videos(str(input_dir), normalized_exts)
    if len(videos) == 1:
        return DoctorCheck("Input video", True, videos[0])
    if not videos:
        return DoctorCheck(
            "Input video",
            False,
            f"no input video in {input_dir}; pass --video",
        )
    names = ", ".join(Path(path).name for path in videos)
    return DoctorCheck(
        "Input video",
        False,
        f"multiple videos found, pass --video: {names}",
    )


def _add_cli_checks(checks: list[DoctorCheck], executables: Iterable[str]) -> None:
    for executable in executables:
        path = shutil.which(executable)
        checks.append(
            DoctorCheck(
                f"CLI: {executable}",
                path is not None,
                path or "not found in PATH",
            )
        )


def _add_module_checks(
    checks: list[DoctorCheck],
    *,
    modules: Iterable[str],
    prefix: str,
    required: bool = True,
) -> None:
    for module_name in modules:
        exists = _module_exists(module_name)
        checks.append(
            DoctorCheck(
                f"{prefix}: {module_name}",
                exists,
                "importable" if exists else "not importable",
                required,
            )
        )


def _add_api_key_check(
    checks: list[DoctorCheck],
    *,
    label: str,
    env_name: str,
) -> None:
    value = os.getenv(env_name, "").strip()
    checks.append(DoctorCheck(label, bool(value), env_name))


def run_project_doctor(
    *,
    config: ConfigSource,
    project_root: Path,
    video_path: str | None = None,
    legacy_suffix: str | None = None,
    include_cli_checks: bool = True,
    include_dependency_checks: bool = True,
    include_api_key_checks: bool = True,
) -> list[DoctorCheck]:
    project_root = project_root.resolve()
    checks: list[DoctorCheck] = []

    _add_path_exists(
        checks,
        name="Local config.py",
        path=project_root / "config.py",
    )
    checks.append(_check_input_video(
        config=config,
        project_root=project_root,
        video_path=video_path,
        legacy_suffix=legacy_suffix,
    ))

    output_root = _resolve_path(
        project_root,
        getattr(config, "OUTPUT_PATH", "./data/output"),
    )
    test_output_root = _resolve_path(
        project_root,
        getattr(config, "TEST_OUTPUT_PATH", "./data/test"),
    )
    for label, path in (
        ("Output dir writable", output_root),
        ("Test output dir writable", test_output_root),
    ):
        ok, detail = _check_writable_dir(path)
        checks.append(DoctorCheck(label, ok, detail))

    tts_provider = str(getattr(config, "TTS_PROVIDER", "xtts")).strip().lower()
    if tts_provider == "xtts":
        model_root = _resolve_path(
            project_root,
            getattr(config, "MODEL_TTS_DIR", "./original_tts_model"),
        )
        _add_path_exists(checks, name="XTTS model dir", path=model_root)
        for filename in ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth"):
            _add_path_exists(checks, name=f"XTTS: {filename}", path=model_root / filename)
    elif tts_provider != "elevenlabs":
        checks.append(DoctorCheck("TTS provider", False, f"unsupported: {tts_provider}"))

    if include_cli_checks:
        _add_cli_checks(checks, ("ffmpeg", "demucs"))

    if include_dependency_checks:
        _add_module_checks(
            checks,
            modules=("numpy", "scipy", "soundfile", "pydub", "noisereduce", "tqdm"),
            prefix="Python core",
        )
        _add_module_checks(
            checks,
            modules=("torch", "whisper", "transformers"),
            prefix="Python ML",
        )
        _add_module_checks(
            checks,
            modules=("TTS",),
            prefix="Python TTS",
            required=tts_provider == "xtts",
        )
        _add_module_checks(
            checks,
            modules=(
                "jiwer",
                "resemblyzer",
                "sentence_transformers",
                "sklearn",
                "matplotlib",
            ),
            prefix="Python metrics",
        )
        needs_openai = (
            getattr(config, "ASR_PROVIDER", "") in {"groq", "openai"}
            or getattr(config, "METRICS_ASR_PROVIDER", "") in {"groq", "openai"}
            or str(os.getenv("MT_PROVIDER", getattr(config, "MT_PROVIDER", ""))).strip().lower() in {
                "openai",
                "chatgpt",
                "openrouter",
                "groq",
                "cerebras",
                "openai_compatible",
            }
            or str(getattr(config, "MT_MODEL_NAME", "")).strip().lower().startswith(("gpt-", "chat-latest"))
            or (
                bool(getattr(config, "SMART_SYNC_ENABLED", False))
                and getattr(config, "SMART_SYNC_PROVIDER", "")
                in {"groq", "openai", "openai_compatible"}
            )
        )
        checks.append(
            DoctorCheck(
                "Python optional: openai",
                _module_exists("openai"),
                (
                    "needed by configured API providers"
                    if needs_openai
                    else "not required by current providers"
                ),
                required=needs_openai,
            )
        )
        checks.append(
            DoctorCheck(
                "Python optional: requests",
                _module_exists("requests"),
                (
                    "needed by ElevenLabs TTS"
                    if tts_provider == "elevenlabs"
                    else "not required by current TTS provider"
                ),
                required=tts_provider == "elevenlabs",
            )
        )

    if include_api_key_checks:
        if getattr(config, "ASR_PROVIDER", "") in {"groq", "openai"}:
            _add_api_key_check(
                checks,
                label=f"ASR API key: {getattr(config, 'ASR_API_KEY_ENV', 'GROQ_API_KEY')}",
                env_name=getattr(config, "ASR_API_KEY_ENV", "GROQ_API_KEY"),
            )
        if getattr(config, "METRICS_ASR_PROVIDER", "") in {"groq", "openai"}:
            _add_api_key_check(
                checks,
                label=(
                    "Metrics ASR API key: "
                    f"{getattr(config, 'METRICS_ASR_API_KEY_ENV', 'GROQ_API_KEY')}"
                ),
                env_name=getattr(config, "METRICS_ASR_API_KEY_ENV", "GROQ_API_KEY"),
            )
        if str(getattr(config, "MT_MODEL_NAME", "")).strip().lower().startswith("gemini-"):
            _add_api_key_check(
                checks,
                label=(
                    "Gemini API key: "
                    f"{getattr(config, 'MT_GEMINI_API_KEY_ENV', 'GEMINI_API_KEY')}"
                ),
                env_name=getattr(config, "MT_GEMINI_API_KEY_ENV", "GEMINI_API_KEY"),
            )
        mt_provider = str(os.getenv("MT_PROVIDER", getattr(config, "MT_PROVIDER", ""))).strip().lower()
        mt_model = str(getattr(config, "MT_MODEL_NAME", "")).strip().lower()
        if mt_provider in {
            "openai",
            "chatgpt",
            "openrouter",
            "groq",
            "cerebras",
            "openai_compatible",
        } or mt_model.startswith(("gpt-", "chat-latest")):
            defaults = {
                "openai": "OPENAI_API_KEY",
                "chatgpt": "OPENAI_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "groq": "GROQ_API_KEY",
                "cerebras": "CEREBRAS_API_KEY",
                "openai_compatible": "OPENAI_API_KEY",
            }
            configured_env = str(os.getenv("MT_OPENAI_API_KEY_ENV", getattr(config, "MT_OPENAI_API_KEY_ENV", ""))).strip()
            inferred_provider = mt_provider or "openai"
            api_key_env = configured_env or defaults.get(inferred_provider, "OPENAI_API_KEY")
            _add_api_key_check(
                checks,
                label=f"MT API key: {api_key_env}",
                env_name=api_key_env,
            )
        if bool(getattr(config, "SMART_SYNC_ENABLED", False)):
            provider = getattr(config, "SMART_SYNC_PROVIDER", "")
            if provider in {"groq", "openai", "openai_compatible", "gemini"}:
                _add_api_key_check(
                    checks,
                    label=(
                        "SmartSync API key: "
                        f"{getattr(config, 'SMART_SYNC_API_KEY_ENV', 'GROQ_API_KEY')}"
                    ),
                    env_name=getattr(config, "SMART_SYNC_API_KEY_ENV", "GROQ_API_KEY"),
                )
        if tts_provider == "elevenlabs":
            api_key_env = str(getattr(config, "ELEVENLABS_API_KEY_ENV", "ELEVENLABS_API_KEY")).strip()
            _add_api_key_check(
                checks,
                label=f"ElevenLabs API key: {api_key_env}",
                env_name=api_key_env,
            )

    return checks


def has_doctor_failures(checks: Iterable[DoctorCheck]) -> bool:
    return any(check.required and not check.ok for check in checks)


def format_doctor_report(checks: Iterable[DoctorCheck]) -> str:
    rows = list(checks)
    name_width = max([len("Check"), *(len(check.name) for check in rows)])
    status_width = len("Status")
    lines = [
        "Project Doctor",
        "",
        f"{'Check'.ljust(name_width)}  {'Status'.ljust(status_width)}  Detail",
        f"{'-' * name_width}  {'-' * status_width}  {'-' * 40}",
    ]
    for check in rows:
        lines.append(
            f"{check.name.ljust(name_width)}  {check.status.ljust(status_width)}  {check.detail}"
        )
    lines.append("")
    if has_doctor_failures(rows):
        lines.append("Verdict: FAIL")
    else:
        lines.append("Verdict: OK")
    return "\n".join(lines)
