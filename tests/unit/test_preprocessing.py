from pathlib import Path
from types import SimpleNamespace

from src.preprocessing import separate_audio_sources


def _run_separation(tmp_path: Path, monkeypatch, stem_files: dict[str, bytes]) -> tuple[list, str, str]:
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    input_audio = temp_dir / "original_extracted_audio.wav"
    input_audio.write_bytes(b"RIFF")

    commands = []

    def fake_run(cmd, capture_output, text):
        commands.append(cmd)
        stems_dir = temp_dir / "htdemucs" / "original_extracted_audio"
        stems_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in stem_files.items():
            (stems_dir / name).write_bytes(payload)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("src.preprocessing.subprocess.run", fake_run)

    vocals_path, background_path = separate_audio_sources(
        input_audio_path=str(input_audio),
        temp_dir=str(temp_dir),
        model_name="htdemucs",
        device="cpu",
        suffix="demo",
    )
    return commands, vocals_path, background_path


def test_separate_audio_sources_requests_two_stems(tmp_path: Path, monkeypatch) -> None:
    commands, vocals_path, background_path = _run_separation(
        tmp_path,
        monkeypatch,
        {"vocals.wav": b"vocals", "no_vocals.wav": b"background"},
    )

    cmd = commands[0]
    assert cmd[cmd.index("--two-stems") + 1] == "vocals"
    assert Path(vocals_path).read_bytes() == b"vocals"
    assert Path(background_path).read_bytes() == b"background"


def test_separate_audio_sources_falls_back_to_legacy_other_stem(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, vocals_path, background_path = _run_separation(
        tmp_path,
        monkeypatch,
        {"vocals.wav": b"vocals", "other.wav": b"legacy-background"},
    )

    assert Path(vocals_path).read_bytes() == b"vocals"
    assert Path(background_path).read_bytes() == b"legacy-background"


def test_separate_audio_sources_replaces_stems_on_forced_rerun(
    tmp_path: Path,
    monkeypatch,
) -> None:
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    input_audio = temp_dir / "original_extracted_audio.wav"
    input_audio.write_bytes(b"RIFF")
    run_number = 0

    def fake_run(cmd, capture_output, text):
        nonlocal run_number
        run_number += 1
        stems_dir = temp_dir / "htdemucs" / "original_extracted_audio"
        stems_dir.mkdir(parents=True, exist_ok=True)
        (stems_dir / "vocals.wav").write_bytes(f"vocals-{run_number}".encode())
        (stems_dir / "no_vocals.wav").write_bytes(f"background-{run_number}".encode())
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("src.preprocessing.subprocess.run", fake_run)

    first_vocals, first_background = separate_audio_sources(
        input_audio_path=str(input_audio),
        temp_dir=str(temp_dir),
        model_name="htdemucs",
        device="cpu",
        suffix="demo",
    )
    second_vocals, second_background = separate_audio_sources(
        input_audio_path=str(input_audio),
        temp_dir=str(temp_dir),
        model_name="htdemucs",
        device="cpu",
        suffix="demo",
    )

    assert second_vocals == first_vocals
    assert second_background == first_background
    assert Path(second_vocals).read_bytes() == b"vocals-2"
    assert Path(second_background).read_bytes() == b"background-2"
