import os
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


def create_tts_backend(
    device: str,
    xtts_model_dir: str,
    *,
    temperature: float | None = None,
    length_penalty: float | None = None,
    repetition_penalty: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
) -> XTTSBackend:
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
