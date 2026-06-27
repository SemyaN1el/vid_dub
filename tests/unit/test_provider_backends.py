import pytest

from src import asr_backend as ab
from src import translation as tr


class _FakeOpenAI:
    created_kwargs: list[dict] = []

    def __init__(self, **kwargs):
        type(self).created_kwargs.append(kwargs)
        self.chat = None
        self.audio = None


@pytest.fixture()
def fake_openai_clients(monkeypatch):
    _FakeOpenAI.created_kwargs = []
    monkeypatch.setattr(tr, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(ab, "OpenAI", _FakeOpenAI)
    return _FakeOpenAI.created_kwargs


def _set_smart_sync_config(monkeypatch, *, provider: str, base_url: str) -> None:
    monkeypatch.setattr(tr.cfg, "SMART_SYNC_ENABLED", True, raising=False)
    monkeypatch.setattr(tr.cfg, "SMART_SYNC_PROVIDER", provider, raising=False)
    monkeypatch.setattr(tr.cfg, "SMART_SYNC_MODEL_NAME", "gpt-5.4-mini", raising=False)
    monkeypatch.setattr(tr.cfg, "SMART_SYNC_API_KEY_ENV", "OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(tr.cfg, "SMART_SYNC_BASE_URL", base_url, raising=False)
    monkeypatch.setattr(tr.cfg, "SMART_SYNC_TEMPERATURE", 0.2, raising=False)
    monkeypatch.setattr(tr.cfg, "SMART_SYNC_MAX_OUTPUT_TOKENS", 256, raising=False)
    monkeypatch.setattr(tr.cfg, "SMART_SYNC_TIMEOUT_SEC", 30, raising=False)
    monkeypatch.setattr(tr.cfg, "SMART_SYNC_MIN_INTERVAL_SEC", 0.0, raising=False)
    monkeypatch.setattr(tr.cfg, "SMART_SYNC_MAX_RETRIES", 1, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


def test_smart_sync_supports_direct_openai(monkeypatch, fake_openai_clients) -> None:
    _set_smart_sync_config(monkeypatch, provider="openai", base_url="")

    backend = tr.load_smart_sync_backend()

    assert isinstance(backend, tr.OpenAICompatibleTranslationBackend)
    assert backend.backend == "openai"
    assert backend.base_url == ""
    assert "base_url" not in fake_openai_clients[0]


def test_smart_sync_openai_compatible_still_requires_base_url(
    monkeypatch,
    fake_openai_clients,
) -> None:
    _set_smart_sync_config(monkeypatch, provider="openai_compatible", base_url="")

    with pytest.raises(EnvironmentError):
        tr.load_smart_sync_backend()


def test_translation_direct_openai_uses_default_endpoint(
    monkeypatch,
    fake_openai_clients,
) -> None:
    monkeypatch.setenv("MT_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("MT_OPENAI_API_KEY_ENV", raising=False)
    monkeypatch.delenv("MT_OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(tr.cfg, "MT_OPENAI_API_KEY_ENV", "", raising=False)
    monkeypatch.setattr(tr.cfg, "MT_OPENAI_BASE_URL", "", raising=False)

    backend, tokenizer = tr.load_translation_model("gpt-5.4-mini", "cpu")

    assert tokenizer is None
    assert isinstance(backend, tr.OpenAICompatibleTranslationBackend)
    assert backend.backend == "openai"
    assert backend.base_url == ""
    assert "base_url" not in fake_openai_clients[0]


def test_asr_backend_supports_openai_provider(monkeypatch, fake_openai_clients) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    backend = ab._load_asr_model(
        provider="openai",
        local_model_name="small",
        api_model_name="whisper-1",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        timeout_sec=60,
        device="cpu",
        purpose="test",
    )

    assert isinstance(backend, ab.GroqASRBackend)
    assert backend.model_name == "whisper-1"
    assert fake_openai_clients[0]["base_url"] == "https://api.openai.com/v1"
