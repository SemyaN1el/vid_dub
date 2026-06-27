import json

import numpy as np
import pytest

from src.tts_backends import ElevenLabsBackend, create_elevenlabs_tts_backend


class _FakeResponse:
    def __init__(self, *, json_data=None, content=b"", status_code=200, headers=None):
        self._json_data = json_data or {}
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            error = RuntimeError(f"HTTP {self.status_code}")
            error.response = self
            raise error


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_elevenlabs_backend_clones_voice_and_writes_manifest(tmp_path) -> None:
    ref_path = tmp_path / "speaker_ref.wav"
    ref_path.write_bytes(b"fake wav")
    manifest_path = tmp_path / "elevenlabs_voice.json"
    session = _FakeSession([
        _FakeResponse(json_data={"voice_id": "voice_123", "requires_verification": False}),
    ])

    backend = ElevenLabsBackend(
        api_key="test-key",
        voice_name="demo_voice",
        voice_manifest_path=str(manifest_path),
        session=session,
    )

    voice_id = backend.prepare_conditioning([str(ref_path)])

    assert voice_id == "voice_123"
    assert backend.prepare_conditioning([str(ref_path)]) == "voice_123"
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/voices/add")
    assert call["headers"]["xi-api-key"] == "test-key"
    assert call["data"]["name"] == "demo_voice"
    assert call["files"][0][0] == "files"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["provider"] == "elevenlabs"
    assert manifest["voice_id"] == "voice_123"
    assert manifest["reference_paths"] == [str(ref_path)]


def test_elevenlabs_backend_uses_existing_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "elevenlabs_voice.json"
    manifest_path.write_text(
        json.dumps({"voice_id": "cached_voice"}, ensure_ascii=False),
        encoding="utf-8",
    )
    session = _FakeSession([])
    backend = ElevenLabsBackend(
        api_key="test-key",
        voice_manifest_path=str(manifest_path),
        session=session,
    )

    assert backend.prepare_conditioning([]) == "cached_voice"
    assert session.calls == []


def test_elevenlabs_backend_synthesizes_pcm_audio() -> None:
    pcm = np.array([0, 32767, -32768], dtype="<i2").tobytes()
    session = _FakeSession([
        _FakeResponse(content=pcm),
    ])
    backend = ElevenLabsBackend(
        api_key="test-key",
        voice_id="voice_123",
        model_id="eleven_multilingual_v2",
        output_format="pcm_24000",
        stability=0.4,
        similarity_boost=0.8,
        style=0.1,
        use_speaker_boost=True,
        speed=1.0,
        session=session,
    )

    audio, sample_rate = backend.synthesize(
        text="Привет.",
        language="ru",
        conditioning="voice_123",
    )

    assert sample_rate == 24000
    assert audio.shape == (3,)
    assert audio.dtype == np.float32
    call = session.calls[0]
    assert call["url"].endswith("/text-to-speech/voice_123")
    assert call["params"]["output_format"] == "pcm_24000"
    assert call["json"]["text"] == "Привет."
    assert call["json"]["language_code"] == "ru"
    assert call["json"]["voice_settings"] == {
        "stability": 0.4,
        "similarity_boost": 0.8,
        "style": 0.1,
        "use_speaker_boost": True,
        "speed": 1.0,
    }


def test_elevenlabs_backend_rejects_html_audio_response() -> None:
    session = _FakeSession([
        _FakeResponse(
            content=b"<!DOCTYPE html><html><title>Restricted</title></html>",
            headers={"content-type": "text/html; charset=utf-8"},
        ),
    ])
    backend = ElevenLabsBackend(
        api_key="test-key",
        voice_id="voice_123",
        output_format="pcm_24000",
        session=session,
    )

    with pytest.raises(RuntimeError, match="HTML instead of audio"):
        backend.synthesize(
            text="Привет.",
            language="ru",
            conditioning="voice_123",
        )


def test_elevenlabs_backend_rejects_odd_pcm_payload() -> None:
    session = _FakeSession([
        _FakeResponse(content=b"\x00\x01\x02"),
    ])
    backend = ElevenLabsBackend(
        api_key="test-key",
        voice_id="voice_123",
        output_format="pcm_24000",
        session=session,
    )

    with pytest.raises(ValueError, match="invalid PCM payload"):
        backend.synthesize(
            text="Привет.",
            language="ru",
            conditioning="voice_123",
        )


def test_create_elevenlabs_backend_requires_api_key() -> None:
    with pytest.raises(EnvironmentError):
        create_elevenlabs_tts_backend(api_key="")
