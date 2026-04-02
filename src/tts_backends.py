import os
from typing import Any, List, Optional, Sequence, Tuple


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

    def __init__(self, model: Any, sample_rate: int = 24000) -> None:
        self.model = model
        self.sample_rate = sample_rate

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
    ) -> tuple[Any, int]:
        gpt_cond_latent, speaker_embedding = conditioning
        output = self.model.inference(
            text=text,
            language=language,
            speaker_embedding=speaker_embedding,
            gpt_cond_latent=gpt_cond_latent,
        )
        return _normalize_audio_array(output["wav"]), self.sample_rate


def create_tts_backend(
    device: str,
    xtts_model_dir: str,
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
    return XTTSBackend(model=model)
