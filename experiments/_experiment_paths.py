from pathlib import Path
from typing import Dict


def resolve_segments_path(output_root: str, job_name: str, test_root: str = "data/test") -> Path:
    """Finds segments.json across the current job layout and the legacy flat layout."""
    candidates = [
        Path(output_root) / job_name / "segments.json",
        Path(output_root) / f"segments_{job_name}.json",
        Path(test_root) / job_name / "segments.json",
        Path(test_root) / f"segments_{job_name}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_job_artifacts(base_dir: Path, job_name: str, backup_tag: str) -> Dict[str, Path]:
    """Returns current and backup artifact paths for new or legacy output layouts."""
    job_dir = base_dir / job_name
    if job_dir.exists():
        return {
            "segments": job_dir / "segments.json",
            "speaker_ref": job_dir / "temp" / "speaker_ref.wav",
            "current_translated": job_dir / "translated_segments.json",
            "current_final_voice": job_dir / "final_dubbing.wav",
            "backup_translated": job_dir / f"translated_segments.{backup_tag}.json",
            "backup_final_voice": job_dir / f"final_dubbing.{backup_tag}.wav",
        }

    return {
        "segments": base_dir / f"segments_{job_name}.json",
        "speaker_ref": base_dir / "temp" / f"speaker_ref_{job_name}.wav",
        "current_translated": base_dir / f"translated_segments_{job_name}.json",
        "current_final_voice": base_dir / f"final_dubbing_{job_name}.wav",
        "backup_translated": base_dir / f"translated_segments_{job_name}.{backup_tag}.json",
        "backup_final_voice": base_dir / f"final_dubbing_{job_name}.{backup_tag}.wav",
    }
