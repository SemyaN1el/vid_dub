from pathlib import Path

from src.subtitles import generate_ass, generate_srt, generate_vtt


SEGMENTS = [
    {
        "text": "Привет мир",
        "original_text": "Hello world",
        "start": 1.0,
        "end": 2.5,
    }
]


def test_generate_subtitle_formats(tmp_path: Path) -> None:
    srt_path = generate_srt(SEGMENTS, str(tmp_path / "out.srt"))
    vtt_path = generate_vtt(SEGMENTS, str(tmp_path / "out.vtt"))
    ass_path = generate_ass(SEGMENTS, str(tmp_path / "out.ass"))

    srt = Path(srt_path).read_text(encoding="utf-8")
    vtt = Path(vtt_path).read_text(encoding="utf-8")
    ass = Path(ass_path).read_text(encoding="utf-8-sig")

    assert "00:00:01,000 --> 00:00:02,500" in srt
    assert "00:00:01.000 --> 00:00:02.500" in vtt
    assert "Dialogue:" in ass
    assert "Привет мир" in ass
