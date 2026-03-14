"""
Пайплайн автоматического дубляжа видео.

Запуск:
    python main.py --step all                        # весь пайплайн
    python main.py --step preprocess                 # один шаг
    python main.py --step asr --suffix woman         # другой спикер
    python main.py --step translate --test           # тестовый режим
    python main.py --step all --suffix woman --test  # всё вместе

Доступные шаги: preprocess, asr, translate, tts, postprocess, metrics, all
"""

import argparse
import gc
import json
import logging
import os
import sys

import torch

import config as cfg
from utils.helpers import seed_everything, manage_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─── Пути ────────────────────────────────────────────────────────────────────

def build_paths(suffix: str, test: bool = False) -> dict:
    """
    Формирует все пути пайплайна.
    test=True  → ./data/test/   (изолированная директория для тестов)
    test=False → ./data/output/ (production)
    """
    base_out = "./data/test" if test else cfg.OUTPUT_PATH
    base_in  = cfg.INPUT_PATH
    temp     = os.path.join(base_out, "temp")

    return {
        "input":               base_in,
        "output":              base_out,
        "temp":                temp,
        "original_video":      os.path.join(base_in,  f"video_{suffix}.mp4"),
        "original_audio":      os.path.join(temp,     f"original_extracted_audio_{suffix}.wav"),
        "speaker_ref":         os.path.join(temp,     f"speaker_ref_{suffix}.wav"),
        "vocals":              os.path.join(temp,     f"vocals_{suffix}.wav"),
        "vocals_processed":    os.path.join(temp,     f"vocals_processed_{suffix}.wav"),
        "background":          os.path.join(temp,     f"background_{suffix}.wav"),
        "final_voice":         os.path.join(base_out, f"final_dubbing_{suffix}.wav"),
        "final_mix":           os.path.join(base_out, f"final_mix_{suffix}.wav"),
        "final_video":         os.path.join(base_out, f"final_video_{suffix}.mp4"),
        "segments":            os.path.join(base_out, f"segments_{suffix}.json"),
        "translated_segments": os.path.join(base_out, f"translated_segments_{suffix}.json"),
        "audio_segments_dir":  os.path.join(temp,     "audio_segments"),
    }


def ensure_dirs(paths: dict) -> None:
    for key in ("input", "output", "temp", "audio_segments_dir"):
        manage_directory(paths[key], action="create")


def check_file(path: str, step_name: str) -> None:
    """Проверяет наличие файла, завершает с понятной ошибкой если нет."""
    if not os.path.exists(path):
        logger.error(f"Файл не найден: {path}")
        logger.error(f"Запустите предыдущий шаг перед '{step_name}'")
        sys.exit(1)


# ─── Шаги пайплайна ──────────────────────────────────────────────────────────

def step_preprocess(paths: dict, suffix: str) -> None:
    from src.preprocessing import (extract_audio_from_video,
                                    separate_audio_sources,
                                    preprocess_audio_for_asr)
    logger.info("╔══ ШАГ 1: ПРЕДОБРАБОТКА ══╗")

    extract_audio_from_video(paths["original_video"], paths["original_audio"])

    vocals, _ = separate_audio_sources(
        input_audio_path=paths["original_audio"],
        temp_dir=paths["temp"],
        model_name=cfg.DEMUCS_MODEL,
        device=cfg.DEVICE,
        suffix=suffix
    )

    preprocess_audio_for_asr(
        input_path=vocals,
        output_path=paths["vocals_processed"],
        target_sample_rate=cfg.ASR_SAMPLE_RATE,
        noise_reduce=cfg.NOISE_REDUCE,
        gain_increase=cfg.GAIN_INCREASE,
        prop_decrease=cfg.PROP_DECREASE,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH
    )
    logger.info("╚══ Предобработка завершена ══╝")


def step_asr(paths: dict, suffix: str) -> None:
    import whisper
    from src.asr import transcribe_and_segment

    logger.info("╔══ ШАГ 2: ASR ══╗")
    check_file(paths["vocals_processed"], "asr")

    model_asr = whisper.load_model(cfg.WHISPER_MODEL_NAME).to(cfg.DEVICE)

    segments = transcribe_and_segment(
        model_asr=model_asr,
        audio_path=paths["vocals_processed"],
        max_pause_between_sentences=cfg.MAX_PAUSE_BETWEEN_SENTENCES,
        max_audio_length_for_ref=cfg.MAX_AUDIO_LENGTH_FOR_REF,
        output_ref_path=paths["speaker_ref"]
    )

    with open(paths["segments"], "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    logger.info(f"Сегменты сохранены ({len(segments)} шт.): {paths['segments']}")

    del model_asr
    gc.collect(); torch.cuda.empty_cache()
    logger.info("╚══ ASR завершён ══╝")


def step_translate(paths: dict, suffix: str) -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from src.translation import translate_segments_with_context

    logger.info("╔══ ШАГ 3: ПЕРЕВОД ══╗")
    check_file(paths["segments"], "translate")

    with open(paths["segments"], "r", encoding="utf-8") as f:
        segments = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg.MT_MODEL_NAME)
    model_mt  = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.MT_MODEL_NAME,
        torch_dtype=torch.float16 if cfg.DEVICE == "cuda" else torch.float32,
        device_map="auto" if cfg.DEVICE == "cuda" else None
    )
    if cfg.DEVICE == "cuda":
        model_mt = model_mt.to("cuda")
    model_mt.eval()

    translated = translate_segments_with_context(
        model=model_mt,
        tokenizer=tokenizer,
        segments=segments,
        src_lang=cfg.MT_SRC_LANG,
        tgt_lang=cfg.MT_TGT_LANG,
        max_chunk_chars=cfg.MT_MAX_CHUNK_CHARS,
        batch_size=cfg.MT_BATCH_SIZE,
        max_length=cfg.MT_MAX_LENGTH
    )

    with open(paths["translated_segments"], "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)
    logger.info(f"Перевод сохранён ({len(translated)} сегментов): {paths['translated_segments']}")

    del model_mt, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    logger.info("╚══ Перевод завершён ══╝")


def step_tts(paths: dict, suffix: str) -> None:
    from TTS.tts.layers.xtts.trainer.gpt_trainer import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from src.tts import synthesize_segments_with_timing

    logger.info("╔══ ШАГ 4: TTS ══╗")
    check_file(paths["translated_segments"], "tts")
    check_file(paths["speaker_ref"],         "tts")

    with open(paths["translated_segments"], "r", encoding="utf-8") as f:
        translated = json.load(f)

    xtts_config = XttsConfig()
    xtts_config.load_json(os.path.join(cfg.MODEL_TTS_DIR, "config.json"))

    model_tts = Xtts.init_from_config(xtts_config)
    model_tts.load_checkpoint(
        xtts_config,
        checkpoint_path=os.path.join(cfg.MODEL_TTS_DIR, "model.pth"),
        vocab_path=os.path.join(cfg.MODEL_TTS_DIR, "vocab.json"),
        speaker_file_path=os.path.join(cfg.MODEL_TTS_DIR, "speakers_xtts.pth"),
        eval=True
    )
    model_tts.to(cfg.DEVICE)

    synthesize_segments_with_timing(
        model_tts=model_tts,
        segments=translated,
        output_audio_path=paths["final_voice"],
        speaker_wav=paths["speaker_ref"],
        language=cfg.LANGUAGE,
        segments_dir=paths["audio_segments_dir"],
        max_speedup_factor=cfg.MAX_SPEEDUP_FACTOR,
        min_pause_between_segments=cfg.MIN_PAUSE_SEGMENTS,
        fade_in_out_ms=cfg.FADE_IN_OUT_MS,
        crossfade_ms=cfg.CROSSFADE_MS,
        max_shift_left_seconds=cfg.MAX_SHIFT_LEFT_SEC,
        threshold_compression=cfg.THRESHOLD_COMPRESSION,
        ratio_compression=cfg.RATIO_COMPRESSION,
        attack_compression=cfg.ATTACK_COMPRESSION,
        release_compression=cfg.RELEASE_COMPRESSION,
        target_dBFS=cfg.TARGET_DBFS
    )

    del model_tts
    gc.collect(); torch.cuda.empty_cache()
    logger.info("╚══ TTS завершён ══╝")


def step_postprocess(paths: dict, suffix: str) -> None:
    from src.postprocessing import mix_audio_tracks, add_audio_to_video

    logger.info("╔══ ШАГ 5: ПОСТОБРАБОТКА ══╗")
    check_file(paths["final_voice"],    "postprocess")
    check_file(paths["background"],     "postprocess")
    check_file(paths["original_audio"], "postprocess")

    mix_audio_tracks(
        voice_over_path=paths["final_voice"],
        background_path=paths["background"],
        output_path=paths["final_mix"],
        original_audio_path=paths["original_audio"],
        voice_gain=cfg.VOICE_GAIN,
        background_gain=cfg.BACKGROUND_GAIN,
        original_gain=cfg.ORIGINAL_GAIN
    )

    add_audio_to_video(
        video_path=paths["original_video"],
        audio_path=paths["final_mix"],
        output_video_path=paths["final_video"]
    )
    logger.info("╚══ Постобработка завершена ══╝")


def step_metrics(paths: dict, suffix: str) -> None:
    import whisper
    from src.metrics import (speaker_verification_score, compute_wer_cer,
                              compute_labse_similarity, print_metrics_summary,
                              plot_labse)

    logger.info("╔══ ШАГ 6: МЕТРИКИ ══╗")
    check_file(paths["speaker_ref"],         "metrics")
    check_file(paths["final_voice"],         "metrics")
    check_file(paths["segments"],            "metrics")
    check_file(paths["translated_segments"], "metrics")

    with open(paths["segments"],            "r", encoding="utf-8") as f:
        segments = json.load(f)
    with open(paths["translated_segments"], "r", encoding="utf-8") as f:
        translated = json.load(f)

    spk_score = speaker_verification_score(paths["speaker_ref"], paths["final_voice"])

    model_asr = whisper.load_model(cfg.WHISPER_MODEL_NAME).to(cfg.DEVICE)
    wer_cer   = compute_wer_cer(model_asr, paths["final_voice"], translated)
    del model_asr
    gc.collect(); torch.cuda.empty_cache()

    labse = compute_labse_similarity(segments, translated)

    print_metrics_summary(spk_score, wer_cer, labse)
    plot_labse(labse, title=f"[{suffix}] Zero-Shot —")
    logger.info("╚══ Метрики подсчитаны ══╝")


# ─── Регистр шагов ───────────────────────────────────────────────────────────

STEPS = {
    "preprocess":  step_preprocess,
    "asr":         step_asr,
    "translate":   step_translate,
    "tts":         step_tts,
    "postprocess": step_postprocess,
    "metrics":     step_metrics,
}

ALL_STEPS = ["preprocess", "asr", "translate", "tts", "postprocess", "metrics"]


# ─── Точка входа ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Пайплайн автоматического дубляжа видео",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()) + ["all"],
        default="all",                          # ← запуск без аргументов = весь пайплайн
        help=(
            "Шаг пайплайна (по умолчанию: all):\n"
            "  preprocess  — извлечь аудио, Demucs, денойзинг\n"
            "  asr         — транскрипция + сегментация\n"
            "  translate   — перевод en→ru\n"
            "  tts         — синтез речи + синхронизация\n"
            "  postprocess — микширование + сборка видео\n"
            "  metrics     — подсчёт метрик качества\n"
            "  all         — весь пайплайн целиком"
        )
    )
    parser.add_argument(
        "--suffix",
        default=cfg.SUFFIX,
        help=f"Суффикс спикера (по умолчанию: '{cfg.SUFFIX}'). Пример: --suffix woman"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Тестовый режим: данные сохраняются в ./data/test/ (production не трогается)"
    )

    args = parser.parse_args()

    seed_everything(cfg.SEED)

    paths = build_paths(suffix=args.suffix, test=args.test)
    ensure_dirs(paths)

    mode = "ТЕСТ" if args.test else "PRODUCTION"
    logger.info(f"{'='*50}")
    logger.info(f"Режим:    {mode}")
    logger.info(f"Спикер:   {args.suffix}")
    logger.info(f"Шаг:      {args.step}")
    logger.info(f"Выходная: {paths['output']}")
    logger.info(f"{'='*50}")

    if args.step == "all":
        for step_name in ALL_STEPS:
            STEPS[step_name](paths, args.suffix)
    else:
        STEPS[args.step](paths, args.suffix)

    logger.info("Готово.")
