from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from src.diarization import (
    apply_diarization_to_segments,
    load_diarization_turns,
    run_pyannote_diarization,
    save_diarization_turns,
    summarize_diarization,
)
from utils.pipeline_io import (
    build_pipeline_paths,
    derive_job_name,
    resolve_input_video,
)


DEFAULT_MODEL = os.getenv(
    "DIARIZATION_MODEL",
    "pyannote/speaker-diarization-community-1",
)


class DiarizationProbeError(RuntimeError):
    pass


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise DiarizationProbeError(f"Missing {label}: {resolved}")
    if resolved.stat().st_size <= 0:
        raise DiarizationProbeError(f"{label} is empty: {resolved}")
    return resolved


def _load_segments(path: str | Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise DiarizationProbeError(f"Cannot read segments: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DiarizationProbeError(f"Invalid JSON in segments: {path}") from exc

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise DiarizationProbeError(f"Segments must be a JSON list of objects: {path}")
    if not data:
        raise DiarizationProbeError(f"Segments are empty: {path}")
    return data


def _write_json(path: str | Path, payload: Any) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_paths(args: argparse.Namespace) -> dict[str, str]:
    if args.job_dir:
        job_dir = _resolve_project_path(args.job_dir)
        temp_dir = job_dir / "temp"
        return {
            "job_name": job_dir.name,
            "job_dir": str(job_dir),
            "temp": str(temp_dir),
            "segments": str(job_dir / "segments.json"),
            "vocals": str(temp_dir / "vocals.wav"),
            "vocals_processed": str(temp_dir / "vocals_processed.wav"),
            "original_audio": str(temp_dir / "original_extracted_audio.wav"),
        }

    input_dir = _resolve_project_path(getattr(cfg, "INPUT_PATH", "./data/input"))
    output_root = _resolve_project_path(getattr(cfg, "OUTPUT_PATH", "./data/output"))
    test_output_root = _resolve_project_path(getattr(cfg, "TEST_OUTPUT_PATH", "./data/test"))

    video_path = resolve_input_video(
        args.video,
        str(input_dir),
        getattr(cfg, "INPUT_VIDEO_EXTENSIONS", (".mp4", ".mov", ".mkv")),
        legacy_suffix=args.suffix,
    )
    job_name = derive_job_name(
        video_path,
        explicit_job_name=args.job_name,
        legacy_suffix=args.suffix,
    )
    return build_pipeline_paths(
        video_path=video_path,
        job_name=job_name,
        output_root=str(output_root),
        test_output_root=str(test_output_root),
        test=args.test,
    )


def _audio_path(paths: dict[str, str], args: argparse.Namespace) -> Path:
    if args.audio:
        return _resolve_project_path(args.audio)

    source_key = args.source_audio
    preferred = Path(paths[source_key])
    if preferred.is_file():
        return preferred

    for key in ("vocals", "vocals_processed", "original_audio"):
        candidate = Path(paths[key])
        if candidate.is_file():
            return candidate

    return preferred


def _hf_token(args: argparse.Namespace) -> str:
    return (
        args.hf_token
        or os.getenv(args.hf_token_env, "").strip()
        or os.getenv("HUGGINGFACE_TOKEN", "").strip()
    )


def _export_probe_speaker_profiles(
    *,
    audio_path: Path,
    segments: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    from src.asr import export_speaker_profile

    profiles_dir = output_dir / "speaker_profiles"
    profiles: dict[str, Any] = {}
    speaker_ids = sorted({
        str(segment["speaker_id"])
        for segment in segments
        if segment.get("speaker_id")
    })

    for speaker_id in speaker_ids:
        speaker_dir = profiles_dir / speaker_id
        try:
            profile = export_speaker_profile(
                reference_audio_path=str(audio_path),
                segments=segments,
                output_ref_path=str(speaker_dir / "speaker_ref.wav"),
                output_refs_dir=str(speaker_dir / "refs"),
                output_profile_path=str(speaker_dir / "speaker_profile.json"),
                max_reference_clips=getattr(cfg, "SPEAKER_PROFILE_CLIPS", 5),
                max_routing_clips=getattr(cfg, "SPEAKER_ROUTING_POOL_CLIPS", 12),
                min_reference_sec=getattr(cfg, "SPEAKER_PROFILE_MIN_SEC", 2.0),
                max_reference_sec=min(
                    getattr(cfg, "MAX_AUDIO_LENGTH_FOR_REF", 15.0),
                    getattr(cfg, "SPEAKER_PROFILE_MAX_SEC", 10.0),
                ),
                target_reference_sec=getattr(cfg, "SPEAKER_PROFILE_TARGET_SEC", 6.0),
                min_reference_text_chars=getattr(cfg, "SPEAKER_PROFILE_MIN_TEXT_CHARS", 20),
                padding_ms=getattr(cfg, "SPEAKER_PROFILE_PADDING_MS", 120),
                min_gap_sec=getattr(cfg, "SPEAKER_PROFILE_MIN_GAP_SEC", 1.5),
                speaker_id=speaker_id,
            )
            profiles[speaker_id] = {
                "ok": True,
                "profile_path": str(speaker_dir / "speaker_profile.json"),
                "reference_path": profile.get("merged_reference_path"),
                "clip_count": len(profile.get("clips", [])),
                "routing_clip_count": len(profile.get("routing_clips", [])),
            }
        except Exception as exc:
            profiles[speaker_id] = {
                "ok": False,
                "error": str(exc),
            }

    return profiles


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    paths = _build_paths(args)
    segments_path = _require_file(args.segments or paths["segments"], "segments.json")
    audio_path = _require_file(_audio_path(paths, args), "diarization audio")

    output_dir = (
        _resolve_project_path(args.output_dir)
        if args.output_dir
        else Path(paths["temp"]) / "diarization_probe"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.diarization_json:
        turns = load_diarization_turns(str(_require_file(args.diarization_json, "diarization JSON")))
    else:
        turns = run_pyannote_diarization(
            str(audio_path),
            model_name=args.model,
            token=_hf_token(args),
            device=args.device,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )

    segments = _load_segments(segments_path)
    diarized_segments = apply_diarization_to_segments(
        segments,
        turns,
        default_speaker_id=args.default_speaker_id,
        max_pause_between_sentences=args.max_pause_between_sentences,
    )

    diarization_path = output_dir / "diarization.json"
    diarized_segments_path = output_dir / "segments_diarized.json"
    summary_path = output_dir / "summary.json"

    save_diarization_turns(str(diarization_path), turns)
    _write_json(diarized_segments_path, diarized_segments)

    summary = summarize_diarization(turns, diarized_segments)
    summary.update({
        "job_name": paths["job_name"],
        "audio_path": str(audio_path),
        "segments_path": str(segments_path),
        "output_dir": str(output_dir),
        "diarization_path": str(diarization_path),
        "diarized_segments_path": str(diarized_segments_path),
        "model": args.model if not args.diarization_json else None,
        "source": "json" if args.diarization_json else "pyannote",
    })

    if args.export_speaker_profiles:
        summary["speaker_profiles"] = _export_probe_speaker_profiles(
            audio_path=audio_path,
            segments=diarized_segments,
            output_dir=output_dir,
        )

    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental diarization probe. It never overwrites pipeline "
            "segments.json and writes only probe artifacts."
        )
    )
    parser.add_argument("--job-dir", default=None, help="Existing job directory with segments.json and temp/*.wav.")
    parser.add_argument("--video", default=None, help="Input video path, same resolution rules as main.py.")
    parser.add_argument("--job-name", default=None, help="Job name, same as main.py --job-name.")
    parser.add_argument("--suffix", default=None, help="Legacy suffix, same as main.py --suffix.")
    parser.add_argument("--test", action="store_true", help="Resolve artifacts under data/test instead of data/output.")
    parser.add_argument("--segments", default=None, help="Override path to source segments.json.")
    parser.add_argument("--audio", default=None, help="Override path to audio for diarization.")
    parser.add_argument(
        "--source-audio",
        choices=["vocals", "vocals_processed", "original_audio"],
        default="vocals",
        help="Artifact audio to diarize when --audio is not set.",
    )
    parser.add_argument("--output-dir", default=None, help="Probe output directory.")
    parser.add_argument("--diarization-json", default=None, help="Reuse existing diarization turns JSON instead of running pyannote.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="pyannote pipeline name.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token value. Prefer env vars for shell history safety.")
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="Env var containing Hugging Face token.")
    parser.add_argument("--device", default=getattr(cfg, "DEVICE", None), help="Torch device for pyannote, e.g. cuda or cpu.")
    parser.add_argument("--num-speakers", type=int, default=None, help="Known exact speaker count.")
    parser.add_argument("--min-speakers", type=int, default=None, help="Known lower speaker count bound.")
    parser.add_argument("--max-speakers", type=int, default=None, help="Known upper speaker count bound.")
    parser.add_argument(
        "--default-speaker-id",
        default=getattr(cfg, "DEFAULT_SPEAKER_ID", "spk_0"),
        help="Fallback speaker_id for words not covered by diarization.",
    )
    parser.add_argument(
        "--max-pause-between-sentences",
        type=float,
        default=getattr(cfg, "MAX_PAUSE_BETWEEN_SENTENCES", 0.3),
        help="Split diarized probe segments on pauses above this threshold.",
    )
    parser.add_argument(
        "--export-speaker-profiles",
        action="store_true",
        help="Also write per-speaker experimental speaker_profile files under the probe directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run_probe(args)
    except Exception as exc:
        print(f"diarization_probe failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
