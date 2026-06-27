from pydub import AudioSegment
from pydub.generators import Sine

from src.postprocessing import mix_audio_tracks


def test_mix_audio_tracks_handles_empty_background(tmp_path) -> None:
    voice_path = tmp_path / "voice.wav"
    background_path = tmp_path / "background.wav"
    output_path = tmp_path / "mix.wav"

    voice = Sine(440).to_audio_segment(duration=500).apply_gain(-18)
    voice.export(voice_path, format="wav")
    AudioSegment.silent(duration=0, frame_rate=44100).export(background_path, format="wav")

    mix_audio_tracks(
        voice_over_path=str(voice_path),
        background_path=str(background_path),
        output_path=str(output_path),
        original_audio_path=None,
    )

    mixed = AudioSegment.from_wav(output_path)
    assert len(mixed) == len(voice)
