"""Бэкенды синтеза речи за единым интерфейсом: локальный XTTS-v2 и ElevenLabs (Instant Voice Cloning)."""

import io
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import logging

logger = logging.getLogger(__name__)


def _normalize_audio_array(audio: Any) -> Any:
    """Приводит выход backend'а к 1D numpy-like массиву для soundfile."""
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    if hasattr(audio, "squeeze"):
        audio = audio.squeeze()
    return audio


class XTTSBackend:
    name = "xtts"

    def __init__(
        self,
        model: Any,
        sample_rate: int = 24000,
        inference_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.sample_rate = sample_rate
        self.inference_kwargs = {
            key: value
            for key, value in (inference_kwargs or {}).items()
            if value is not None
        }

    def prepare_conditioning(
        self,
        reference_paths: Sequence[str],
        speaker_profile: Optional[dict[str, Any]] = None,
    ) -> Tuple[Any, Any]:
        audio_path: str | List[str]
        if len(reference_paths) == 1:
            audio_path = reference_paths[0]
        else:
            audio_path = list(reference_paths)
        return self.model.get_conditioning_latents(audio_path=audio_path)

    def synthesize(
        self,
        text: str,
        language: str,
        conditioning: Tuple[Any, Any],
        inference_overrides: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, int]:
        gpt_cond_latent, speaker_embedding = conditioning
        inference_kwargs = dict(self.inference_kwargs)
        for key, value in (inference_overrides or {}).items():
            if value is not None:
                inference_kwargs[key] = value
        output = self.model.inference(
            text=text,
            language=language,
            speaker_embedding=speaker_embedding,
            gpt_cond_latent=gpt_cond_latent,
            **inference_kwargs,
        )
        return _normalize_audio_array(output["wav"]), self.sample_rate


def _bool_param(value: bool) -> str:
    return "true" if value else "false"


def _sample_rate_from_output_format(output_format: str, default: int = 24000) -> int:
    parts = (output_format or "").split("_")
    if len(parts) >= 2 and parts[0] in {"pcm", "wav", "mp3"}:
        try:
            return int(parts[1])
        except ValueError:
            return default
    return default


def _audio_bytes_to_array(audio_bytes: bytes, output_format: str) -> tuple[Any, int]:
    sample_rate = _sample_rate_from_output_format(output_format)
    if output_format.startswith("pcm_"):
        import numpy as np

        if len(audio_bytes) % 2 != 0:
            raise ValueError(
                "ElevenLabs returned invalid PCM payload: byte length is not divisible by 2."
            )
        samples = np.frombuffer(audio_bytes, dtype="<i2").astype("float32") / 32768.0
        return samples, sample_rate

    from pydub import AudioSegment

    format_hint = output_format.split("_", 1)[0] if "_" in output_format else None
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=format_hint)
    audio = audio.set_channels(1)
    sample_rate = audio.frame_rate
    import numpy as np

    samples = np.array(audio.get_array_of_samples()).astype("float32")
    max_value = float(1 << (8 * audio.sample_width - 1))
    if max_value > 0:
        samples /= max_value
    return samples, sample_rate


class ElevenLabsBackend:
    name = "elevenlabs"

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "",
        voice_name: str = "video_dubbing_voice",
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "pcm_24000",
        base_url: str = "https://api.elevenlabs.io/v1",
        voice_manifest_path: str = "",
        clone_voice: bool = True,
        remove_background_noise: bool = False,
        timeout_sec: int = 120,
        min_interval_sec: float = 0.0,
        max_retries: int = 3,
        enable_logging: bool = True,
        language_code: str = "",
        apply_text_normalization: str = "auto",
        stability: float | None = None,
        similarity_boost: float | None = None,
        style: float | None = None,
        use_speaker_boost: bool | None = None,
        speed: float | None = None,
        session: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id.strip()
        self.voice_name = voice_name.strip() or "video_dubbing_voice"
        self.model_id = model_id
        self.output_format = output_format
        self.base_url = base_url.rstrip("/")
        self.voice_manifest_path = voice_manifest_path
        self.clone_voice = clone_voice
        self.remove_background_noise = remove_background_noise
        self.timeout_sec = timeout_sec
        self.min_interval_sec = min_interval_sec
        self.max_retries = max_retries
        self.enable_logging = enable_logging
        self.language_code = language_code.strip()
        self.apply_text_normalization = apply_text_normalization.strip() or "auto"
        self.voice_settings = {
            key: value
            for key, value in {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": use_speaker_boost,
                "speed": speed,
            }.items()
            if value is not None
        }
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.last_request_ts = 0.0

    def _headers(self) -> dict[str, str]:
        return {"xi-api-key": self.api_key}

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        wait_sec = self.min_interval_sec - (time.monotonic() - self.last_request_ts)
        if wait_sec > 0:
            time.sleep(wait_sec)

        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, url, timeout=self.timeout_sec, **kwargs)
                self.last_request_ts = time.monotonic()
                response.raise_for_status()
                return response
            except Exception as exc:
                last_error = exc
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                response = getattr(exc, "response", None)
                if status_code is None and response is not None:
                    status_code = getattr(response, "status_code", None)
                retriable = status_code in {408, 409, 429, 500, 502, 503, 504} or status_code is None
                if not retriable or attempt >= self.max_retries:
                    raise RuntimeError(f"ElevenLabs API error: {exc}") from exc
                delay = max(self.min_interval_sec, min(30.0, 2 ** attempt))
                logger.warning(
                    "ElevenLabs retry через %.1f сек (attempt=%s/%s): %s",
                    delay,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                time.sleep(delay)

        raise RuntimeError(f"ElevenLabs API error: {last_error}")

    def _validate_audio_response(self, response: Any) -> None:
        content_type = str(response.headers.get("content-type", "")).lower()
        content = response.content or b""
        if content_type.startswith("text/html") or content.lstrip().startswith(b"<!DOCTYPE"):
            raise RuntimeError(
                "ElevenLabs returned HTML instead of audio. "
                "This usually means the request was intercepted by an access or region restriction page."
            )
        if "json" in content_type or content.lstrip().startswith((b"{", b"[")):
            preview = content[:300].decode("utf-8", errors="replace")
            raise RuntimeError(f"ElevenLabs returned JSON instead of audio: {preview}")
        if not content:
            raise RuntimeError("ElevenLabs returned an empty audio response.")

    def _load_manifest_voice_id(self) -> str:
        if not self.voice_manifest_path:
            return ""
        path = Path(self.voice_manifest_path)
        if not path.exists():
            return ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Не удалось прочитать ElevenLabs manifest %s: %s", path, exc)
            return ""
        voice_id = str(data.get("voice_id", "")).strip()
        if voice_id:
            logger.info("ElevenLabs: используем voice_id из manifest: %s", voice_id)
        return voice_id

    def _save_manifest(
        self,
        *,
        voice_id: str,
        reference_paths: Sequence[str],
        requires_verification: bool = False,
    ) -> None:
        if not self.voice_manifest_path:
            return
        path = Path(self.voice_manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": "elevenlabs",
            "voice_id": voice_id,
            "voice_name": self.voice_name,
            "model_id": self.model_id,
            "output_format": self.output_format,
            "requires_verification": requires_verification,
            "reference_paths": [os.path.abspath(path) for path in reference_paths],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _clone_voice(self, reference_paths: Sequence[str]) -> str:
        paths = [path for path in reference_paths if path and os.path.exists(path)]
        if not paths:
            raise ValueError("ElevenLabs voice cloning needs at least one existing reference audio file.")

        logger.info("ElevenLabs: создаём Instant Voice Clone из %s reference clip(s)...", len(paths))
        files = []
        handles = []
        try:
            for path in paths:
                handle = open(path, "rb")
                handles.append(handle)
                mime_type = mimetypes.guess_type(path)[0] or "audio/wav"
                files.append(("files", (os.path.basename(path), handle, mime_type)))

            response = self._request(
                "POST",
                "/voices/add",
                headers=self._headers(),
                data={
                    "name": self.voice_name,
                    "description": "Instant voice clone created by video_dubbing pipeline.",
                    "remove_background_noise": _bool_param(self.remove_background_noise),
                },
                files=files,
            )
        finally:
            for handle in handles:
                handle.close()

        data = response.json()
        voice_id = str(data.get("voice_id", "")).strip()
        if not voice_id:
            raise RuntimeError(f"ElevenLabs did not return voice_id: {data}")
        requires_verification = bool(data.get("requires_verification", False))
        if requires_verification:
            logger.warning("ElevenLabs voice_id=%s requires verification before use.", voice_id)
        self._save_manifest(
            voice_id=voice_id,
            reference_paths=paths,
            requires_verification=requires_verification,
        )
        return voice_id

    def prepare_conditioning(
        self,
        reference_paths: Sequence[str],
        speaker_profile: Optional[dict[str, Any]] = None,
    ) -> str:
        del speaker_profile
        if self.voice_id:
            return self.voice_id

        manifest_voice_id = self._load_manifest_voice_id()
        if manifest_voice_id:
            self.voice_id = manifest_voice_id
            return self.voice_id

        if not self.clone_voice:
            raise ValueError(
                "ELEVENLABS_VOICE_ID is empty and ELEVENLABS_CLONE_VOICE=0, "
                "so ElevenLabs has no voice to synthesize with."
            )

        self.voice_id = self._clone_voice(reference_paths)
        return self.voice_id

    def synthesize(
        self,
        text: str,
        language: str,
        conditioning: str,
        inference_overrides: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, int]:
        del inference_overrides
        voice_id = conditioning or self.voice_id
        if not voice_id:
            raise ValueError("ElevenLabs voice_id is not prepared.")

        params = {
            "output_format": self.output_format,
            "enable_logging": _bool_param(self.enable_logging),
        }
        payload: dict[str, Any] = {
            "text": text,
            "model_id": self.model_id,
        }
        language_code = self.language_code or (language if language else "")
        if language_code:
            payload["language_code"] = language_code
        if self.voice_settings:
            payload["voice_settings"] = self.voice_settings
        if self.apply_text_normalization:
            payload["apply_text_normalization"] = self.apply_text_normalization

        response = self._request(
            "POST",
            f"/text-to-speech/{voice_id}",
            headers={**self._headers(), "Content-Type": "application/json"},
            params=params,
            json=payload,
        )
        self._validate_audio_response(response)
        return _audio_bytes_to_array(response.content, self.output_format)


def create_tts_backend(
    device: str,
    xtts_model_dir: str,
    *,
    provider: str = "xtts",
    temperature: float | None = None,
    length_penalty: float | None = None,
    repetition_penalty: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> XTTSBackend:
    provider = (provider or "xtts").strip().lower()
    if provider == "elevenlabs":
        raise ValueError(
            "ElevenLabs backend requires create_elevenlabs_tts_backend(...). "
            "Use create_tts_backend_from_config or pass provider-specific config from main.py."
        )
    if provider != "xtts":
        raise ValueError(f"Unsupported TTS_PROVIDER={provider!r}.")

    from TTS.tts.layers.xtts.trainer.gpt_trainer import XttsConfig
    from TTS.tts.models.xtts import Xtts

    xtts_config = XttsConfig()
    xtts_config.load_json(os.path.join(xtts_model_dir, "config.json"))

    model = Xtts.init_from_config(xtts_config)
    model.load_checkpoint(
        xtts_config,
        checkpoint_path=os.path.join(xtts_model_dir, "model.pth"),
        vocab_path=os.path.join(xtts_model_dir, "vocab.json"),
        speaker_file_path=os.path.join(xtts_model_dir, "speakers_xtts.pth"),
        eval=True,
    )
    model.to(device)
    inference_kwargs = {
        "temperature": temperature,
        "length_penalty": length_penalty,
        "repetition_penalty": repetition_penalty,
        "top_k": top_k,
        "top_p": top_p,
    }
    logger.info(
        "XTTS decoding params: temperature=%s length_penalty=%s repetition_penalty=%s top_k=%s top_p=%s",
        inference_kwargs["temperature"],
        inference_kwargs["length_penalty"],
        inference_kwargs["repetition_penalty"],
        inference_kwargs["top_k"],
        inference_kwargs["top_p"],
    )
    return XTTSBackend(model=model, inference_kwargs=inference_kwargs)


def create_elevenlabs_tts_backend(
    *,
    api_key: str,
    voice_id: str = "",
    voice_name: str = "video_dubbing_voice",
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "pcm_24000",
    base_url: str = "https://api.elevenlabs.io/v1",
    voice_manifest_path: str = "",
    clone_voice: bool = True,
    remove_background_noise: bool = False,
    timeout_sec: int = 120,
    min_interval_sec: float = 0.0,
    max_retries: int = 3,
    enable_logging: bool = True,
    language_code: str = "",
    apply_text_normalization: str = "auto",
    stability: float | None = None,
    similarity_boost: float | None = None,
    style: float | None = None,
    use_speaker_boost: bool | None = None,
    speed: float | None = None,
    session: Any | None = None,
) -> ElevenLabsBackend:
    if not api_key:
        raise EnvironmentError("Для ElevenLabs TTS установите ELEVENLABS_API_KEY.")
    logger.info(
        "ElevenLabs TTS: model=%s output_format=%s voice_id=%s clone_voice=%s",
        model_id,
        output_format,
        voice_id or "auto-clone",
        clone_voice,
    )
    return ElevenLabsBackend(
        api_key=api_key,
        voice_id=voice_id,
        voice_name=voice_name,
        model_id=model_id,
        output_format=output_format,
        base_url=base_url,
        voice_manifest_path=voice_manifest_path,
        clone_voice=clone_voice,
        remove_background_noise=remove_background_noise,
        timeout_sec=timeout_sec,
        min_interval_sec=min_interval_sec,
        max_retries=max_retries,
        enable_logging=enable_logging,
        language_code=language_code,
        apply_text_normalization=apply_text_normalization,
        stability=stability,
        similarity_boost=similarity_boost,
        style=style,
        use_speaker_boost=use_speaker_boost,
        speed=speed,
        session=session,
    )
