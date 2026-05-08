from pydub import AudioSegment

from src.tts import _serialize_tts_segments


def test_serialize_tts_segments_drops_runtime_audio_objects() -> None:
    segments = _serialize_tts_segments([
        {
            "text": "Привет.",
            "corrected_duration_sec": 1.2,
            "corrected_audio": AudioSegment.silent(duration=100),
        }
    ])

    assert segments == [{"text": "Привет.", "corrected_duration_sec": 1.2}]
