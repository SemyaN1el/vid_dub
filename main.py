"""
Пайплайн автоматического дубляжа видео.

Запуск:
    python main.py --step all                        # весь пайплайн
    python main.py --step preprocess                 # один шаг
    python main.py --video ./data/input/talk.mp4     # явное видео без suffix
    python main.py --video ./clips/ted.mp4 --job-name ted_ru
    python main.py --step asr --suffix woman         # legacy-режим
    python main.py --step translate --test           # тестовый режим
    python main.py --step translate --mt-model gemini-2.5-flash --mt-strategy per-segment
    python main.py --step all --suffix woman --test  # всё вместе

Доступные шаги: preprocess, asr, translate, tts, postprocess, subtitles, metrics,
                 prepare_finetune, all
"""

import argparse
import gc
import importlib.util
import json
import logging
import os
import shutil
import sys
from datetime import datetime

try:
    import torch
except ImportError:
    torch = None

import config as cfg
from utils.helpers import seed_everything, manage_directory
from utils.pipeline_io import build_pipeline_paths, derive_job_name, resolve_input_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clear_torch_cache() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dirs(paths: dict) -> None:
    for key in ("output", "temp", "audio_segments_dir", "speaker_refs_dir"):
        manage_directory(paths[key], action="create")


def check_file(path: str, step_name: str) -> None:
    """Проверяет наличие файла, завершает с понятной ошибкой если нет."""
    if not os.path.exists(path):
        logger.error(f"Файл не найден: {path}")
        logger.error(f"Запустите предыдущий шаг перед '{step_name}'")
        sys.exit(1)


def _module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_environment() -> bool:
    """Проверяет локальное окружение без загрузки тяжёлых моделей."""
    checks: list[tuple[str, bool, str, bool]] = []

    def add(name: str, ok: bool, detail: str = "", required: bool = True) -> None:
        checks.append((name, ok, detail, required))

    for executable in ("ffmpeg", "demucs"):
        path = shutil.which(executable)
        add(f"CLI: {executable}", path is not None, path or "не найден в PATH")

    required_modules = (
        "numpy",
        "soundfile",
        "pydub",
        "scipy",
        "noisereduce",
        "tqdm",
        "torch",
        "whisper",
        "transformers",
        "TTS",
    )
    for module_name in required_modules:
        add(f"Python: {module_name}", _module_exists(module_name))

    metrics_modules = (
        "jiwer",
        "resemblyzer",
        "sentence_transformers",
        "sklearn",
    )
    for module_name in metrics_modules:
        add(f"Python metrics: {module_name}", _module_exists(module_name))

    model_files = ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth")
    for filename in model_files:
        path = os.path.join(cfg.MODEL_TTS_DIR, filename)
        add(f"XTTS: {filename}", os.path.exists(path), path)

    if cfg.ASR_PROVIDER == "groq":
        add("Python API: openai", _module_exists("openai"))
        api_key = os.getenv(cfg.ASR_API_KEY_ENV, "").strip()
        add(f"ASR API key: {cfg.ASR_API_KEY_ENV}", bool(api_key))
    if cfg.METRICS_ASR_PROVIDER == "groq":
        add("Python API: openai", _module_exists("openai"))
        api_key = os.getenv(cfg.METRICS_ASR_API_KEY_ENV, "").strip()
        add(f"Metrics ASR API key: {cfg.METRICS_ASR_API_KEY_ENV}", bool(api_key))
    if cfg.MT_MODEL_NAME.strip().lower().startswith("gemini-"):
        api_key = os.getenv(cfg.MT_GEMINI_API_KEY_ENV, "").strip()
        add(f"Gemini API key: {cfg.MT_GEMINI_API_KEY_ENV}", bool(api_key))
    if cfg.SMART_SYNC_ENABLED:
        if cfg.SMART_SYNC_PROVIDER in {"groq", "openai", "openai_compatible"}:
            add("Python API: openai", _module_exists("openai"))
        api_key = os.getenv(cfg.SMART_SYNC_API_KEY_ENV, "").strip()
        add(f"SmartSync API key: {cfg.SMART_SYNC_API_KEY_ENV}", bool(api_key))

    failed = False
    for name, ok, detail, required in checks:
        status = "OK" if ok else ("FAIL" if required else "WARN")
        logger.info("%-42s %s %s", name, status, detail)
        failed = failed or (required and not ok)

    if failed:
        logger.error("Окружение неполное. Исправьте FAIL-пункты перед полным запуском.")
        return False

    logger.info("Окружение готово для полного пайплайна.")
    return True


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
    canonical_vocals = paths["vocals"]
    if os.path.abspath(vocals) != os.path.abspath(canonical_vocals):
        shutil.copyfile(vocals, canonical_vocals)
        vocals = canonical_vocals

    background_candidate = os.path.join(paths["temp"], f"background_{suffix}.wav")
    canonical_background = paths["background"]
    if (
        os.path.exists(background_candidate)
        and os.path.abspath(background_candidate) != os.path.abspath(canonical_background)
    ):
        shutil.copyfile(background_candidate, canonical_background)

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
    from src.asr import transcribe_and_segment
    from src.asr_backend import load_main_asr_model

    logger.info("╔══ ШАГ 2: ASR ══╗")
    check_file(paths["vocals_processed"], "asr")

    model_asr = load_main_asr_model()

    segments = transcribe_and_segment(
        model_asr=model_asr,
        audio_path=paths["vocals_processed"],
        max_pause_between_sentences=cfg.MAX_PAUSE_BETWEEN_SENTENCES,
        max_audio_length_for_ref=cfg.MAX_AUDIO_LENGTH_FOR_REF,
        output_ref_path=paths["speaker_ref"],
        default_speaker_id=cfg.DEFAULT_SPEAKER_ID,
        reference_audio_path=paths["vocals"],
        output_refs_dir=paths["speaker_refs_dir"],
        output_profile_path=paths["speaker_profile"],
        max_reference_clips=cfg.SPEAKER_PROFILE_CLIPS,
        max_routing_clips=cfg.SPEAKER_ROUTING_POOL_CLIPS,
        min_reference_sec=cfg.SPEAKER_PROFILE_MIN_SEC,
        max_reference_sec=cfg.SPEAKER_PROFILE_MAX_SEC,
        target_reference_sec=cfg.SPEAKER_PROFILE_TARGET_SEC,
        min_reference_text_chars=cfg.SPEAKER_PROFILE_MIN_TEXT_CHARS,
        reference_padding_ms=cfg.SPEAKER_PROFILE_PADDING_MS,
        min_reference_gap_sec=cfg.SPEAKER_PROFILE_MIN_GAP_SEC
    )

    with open(paths["segments"], "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    logger.info(f"Сегменты сохранены ({len(segments)} шт.): {paths['segments']}")

    del model_asr
    gc.collect(); clear_torch_cache()
    logger.info("╚══ ASR завершён ══╝")


def step_translate(paths: dict, suffix: str) -> None:
    from src.translation import (
        load_translation_model,
        split_segments_for_translation,
        normalize_translated_segments,
        translate_segments,
        translate_segments_as_sentences,
        translate_segments_sliding_window,
        translate_segments_with_context,
    )

    logger.info("╔══ ШАГ 3: ПЕРЕВОД ══╗")
    check_file(paths["segments"], "translate")

    with open(paths["segments"], "r", encoding="utf-8") as f:
        segments = json.load(f)

    prepared_segments = split_segments_for_translation(
        segments,
        cfg.MT_MAX_SEGMENT_CHARS
    )
    if len(prepared_segments) != len(segments):
        logger.info(
            "Сегменты подготовлены к переводу: %s -> %s",
            len(segments),
            len(prepared_segments)
        )

    model_mt, tokenizer = load_translation_model(cfg.MT_MODEL_NAME, cfg.DEVICE)

    logger.info(
        "Модель перевода: %s | стратегия: %s",
        cfg.MT_MODEL_NAME,
        cfg.MT_STRATEGY
    )

    translate_kwargs = {
        "model": model_mt,
        "tokenizer": tokenizer,
        "segments": prepared_segments,
        "src_lang": cfg.MT_SRC_LANG,
        "tgt_lang": cfg.MT_TGT_LANG,
        "batch_size": cfg.MT_BATCH_SIZE,
        "max_length": cfg.MT_MAX_LENGTH,
    }

    if cfg.MT_STRATEGY == "per-segment":
        translated = translate_segments(**translate_kwargs)
    elif cfg.MT_STRATEGY == "sentence-level":
        translated = translate_segments_as_sentences(**translate_kwargs)
    elif cfg.MT_STRATEGY == "sliding-window":
        translated = translate_segments_sliding_window(
            **translate_kwargs,
            window_size=2
        )
    elif cfg.MT_STRATEGY == "context-aware":
        translated = translate_segments_with_context(
            **translate_kwargs,
            max_chunk_chars=cfg.MT_MAX_CHUNK_CHARS
        )
    else:
        raise ValueError(
            f"Неизвестная стратегия перевода: {cfg.MT_STRATEGY}. "
            "Ожидается per-segment, sentence-level, sliding-window или context-aware."
        )

    translated = normalize_translated_segments(translated)

    with open(paths["translated_segments"], "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)
    logger.info(f"Перевод сохранён ({len(translated)} сегментов): {paths['translated_segments']}")

    del model_mt, tokenizer
    gc.collect(); clear_torch_cache()
    logger.info("╚══ Перевод завершён ══╝")


def step_tts(paths: dict, suffix: str) -> None:
    from src.tts_backends import create_tts_backend
    from src.tts import (
        AudioLevelConfig,
        SegmentRoutingConfig,
        SmartSyncConfig,
        TailGuardConfig,
        TTSRuntimeConfig,
        synthesize_segments_with_timing,
    )

    logger.info("╔══ ШАГ 4: TTS ══╗")
    check_file(paths["translated_segments"], "tts")
    check_file(paths["speaker_ref"],         "tts")

    with open(paths["translated_segments"], "r", encoding="utf-8") as f:
        translated = json.load(f)

    speaker_source: str | list[str] = paths["speaker_ref"]
    speaker_profile = None
    if os.path.exists(paths["speaker_profile"]):
        with open(paths["speaker_profile"], "r", encoding="utf-8") as f:
            speaker_profile = json.load(f)
        clip_paths = [
            clip["path"]
            for clip in speaker_profile.get("clips", [])
            if clip.get("path") and os.path.exists(clip["path"])
        ]
        if clip_paths:
            speaker_source = clip_paths
            logger.info(
                "Speaker profile: %s reference clips",
                len(clip_paths)
            )

    tts_backend = create_tts_backend(
        device=cfg.DEVICE,
        xtts_model_dir=cfg.MODEL_TTS_DIR,
        temperature=cfg.XTTS_TEMPERATURE,
        length_penalty=cfg.XTTS_LENGTH_PENALTY,
        repetition_penalty=cfg.XTTS_REPETITION_PENALTY,
        top_k=cfg.XTTS_TOP_K,
        top_p=cfg.XTTS_TOP_P,
    )

    runtime_config = TTSRuntimeConfig(
        max_speedup_factor=cfg.MAX_SPEEDUP_FACTOR,
        max_next_start_shift_sec=cfg.MAX_NEXT_START_SHIFT_SEC,
        speedup_tail_padding_ms=cfg.SPEEDUP_TAIL_PADDING_MS,
        min_pause_between_segments=cfg.MIN_PAUSE_SEGMENTS,
        fade_in_out_ms=cfg.FADE_IN_OUT_MS,
        crossfade_ms=cfg.CROSSFADE_MS,
        max_shift_left_seconds=cfg.MAX_SHIFT_LEFT_SEC,
        enable_grouping=cfg.TTS_GROUPING_ENABLED,
        grouping_max_gap_sec=cfg.TTS_GROUPING_MAX_GAP_SEC,
        grouping_max_segments=cfg.TTS_GROUPING_MAX_SEGMENTS,
        grouping_max_chars=cfg.TTS_GROUPING_MAX_CHARS,
        grouping_max_duration_sec=cfg.TTS_GROUPING_MAX_DURATION_SEC,
    )
    smart_sync_config = SmartSyncConfig(
        enabled=cfg.SMART_SYNC_ENABLED,
        device=cfg.DEVICE,
        src_lang=cfg.MT_SRC_LANG,
        tgt_lang=cfg.MT_TGT_LANG,
        max_rewrites=cfg.SMART_SYNC_MAX_REWRITES,
        trigger_speed_factor=cfg.SMART_SYNC_TRIGGER_SPEED_FACTOR,
        min_fill_ratio=cfg.SMART_SYNC_MIN_FILL_RATIO,
        min_improvement_ms=cfg.SMART_SYNC_MIN_IMPROVEMENT_MS,
        allow_lengthen=cfg.SMART_SYNC_ALLOW_LENGTHEN,
        accept_min_fill_ratio=cfg.SMART_SYNC_ACCEPT_MIN_FILL_RATIO,
        accept_min_text_similarity=cfg.SMART_SYNC_ACCEPT_MIN_TEXT_SIMILARITY,
        accept_min_word_ratio=cfg.SMART_SYNC_ACCEPT_MIN_WORD_RATIO,
        accept_min_token_precision=cfg.SMART_SYNC_ACCEPT_MIN_TOKEN_PRECISION,
        accept_min_asr_score=cfg.SMART_SYNC_ACCEPT_MIN_ASR_SCORE,
        accept_max_asr_drop=cfg.SMART_SYNC_ACCEPT_MAX_ASR_DROP,
    )
    tail_guard_config = TailGuardConfig(
        enable_cheap_tail_guard=cfg.ENABLE_TTS_CHEAP_TAIL_GUARD,
        cheap_tail_guard_max_segment_sec=cfg.TTS_CHEAP_TAIL_GUARD_MAX_SEGMENT_SEC,
        cheap_tail_guard_min_overhang_ms=cfg.TTS_CHEAP_TAIL_GUARD_MIN_OVERHANG_MS,
        cheap_tail_guard_min_gap_ms=cfg.TTS_CHEAP_TAIL_GUARD_MIN_GAP_MS,
        cheap_tail_guard_min_island_ms=cfg.TTS_CHEAP_TAIL_GUARD_MIN_ISLAND_MS,
        cheap_tail_guard_max_island_ms=cfg.TTS_CHEAP_TAIL_GUARD_MAX_ISLAND_MS,
        cheap_tail_guard_search_window_ms=cfg.TTS_CHEAP_TAIL_GUARD_SEARCH_WINDOW_MS,
        cheap_tail_guard_max_trim_ms=cfg.TTS_CHEAP_TAIL_GUARD_MAX_TRIM_MS,
        enable_babble_guard=cfg.ENABLE_TTS_BABBLE_GUARD,
        babble_guard_model_name=cfg.TTS_BABBLE_GUARD_MODEL_NAME,
        babble_guard_device=cfg.TTS_BABBLE_GUARD_DEVICE,
        babble_guard_max_segment_sec=cfg.TTS_BABBLE_GUARD_MAX_SEGMENT_SEC,
        babble_guard_min_gap_ms=cfg.TTS_BABBLE_GUARD_MIN_GAP_MS,
        babble_guard_min_island_ms=cfg.TTS_BABBLE_GUARD_MIN_ISLAND_MS,
        babble_guard_max_island_ms=cfg.TTS_BABBLE_GUARD_MAX_ISLAND_MS,
        babble_guard_search_window_ms=cfg.TTS_BABBLE_GUARD_SEARCH_WINDOW_MS,
        babble_guard_max_trim_ms=cfg.TTS_BABBLE_GUARD_MAX_TRIM_MS,
        babble_guard_anchor_words=cfg.TTS_BABBLE_GUARD_ANCHOR_WORDS,
        babble_guard_min_score_gain=cfg.TTS_BABBLE_GUARD_MIN_SCORE_GAIN,
        enable_asr_retry=cfg.ENABLE_TTS_ASR_RETRY,
        asr_retry_model_name=cfg.TTS_ASR_RETRY_MODEL_NAME,
        asr_retry_device=cfg.TTS_ASR_RETRY_DEVICE,
        asr_retry_max_segment_sec=cfg.TTS_ASR_RETRY_MAX_SEGMENT_SEC,
        asr_retry_attempts=cfg.TTS_ASR_RETRY_ATTEMPTS,
        asr_retry_min_score=cfg.TTS_ASR_RETRY_MIN_SCORE,
        enable_short_segment_tail_trim=cfg.SHORT_SEGMENT_TAIL_TRIM_ENABLED,
        short_segment_tail_trim_min_overhang_ms=cfg.SHORT_SEGMENT_TAIL_TRIM_MIN_OVERHANG_MS,
        short_segment_tail_trim_max_ms=cfg.SHORT_SEGMENT_TAIL_TRIM_MAX_MS,
        short_segment_tail_trim_max_ratio=cfg.SHORT_SEGMENT_TAIL_TRIM_MAX_RATIO,
    )
    segment_routing_config = SegmentRoutingConfig(
        enabled=cfg.SEGMENT_ROUTING_ENABLED,
        short_segment_sec=cfg.SEGMENT_ROUTING_SHORT_SEC,
        max_refs_per_segment=cfg.SEGMENT_ROUTING_MAX_REFS,
        min_segment_sec=cfg.SEGMENT_ROUTING_MIN_SEC,
        min_segment_words=cfg.SEGMENT_ROUTING_MIN_WORDS,
        confidence_margin=cfg.SEGMENT_ROUTING_CONFIDENCE_MARGIN,
    )
    audio_level_config = AudioLevelConfig(
        threshold_compression=cfg.THRESHOLD_COMPRESSION,
        ratio_compression=cfg.RATIO_COMPRESSION,
        attack_compression=cfg.ATTACK_COMPRESSION,
        release_compression=cfg.RELEASE_COMPRESSION,
        target_dbfs=cfg.TARGET_DBFS,
        reference_gain_offset_db=cfg.REFERENCE_GAIN_OFFSET_DB,
        max_segment_boost_db=cfg.MAX_SEGMENT_BOOST_DB,
        max_segment_cut_db=cfg.MAX_SEGMENT_CUT_DB,
        peak_ceiling_dbfs=cfg.PEAK_CEILING_DBFS,
        enable_final_compression=cfg.ENABLE_FINAL_COMPRESSION,
        enable_segment_matching=cfg.SEGMENT_MATCHING_ENABLED,
        segment_match_padding_ms=cfg.SEGMENT_MATCH_PADDING_MS,
        segment_match_strength=cfg.SEGMENT_MATCH_STRENGTH,
        segment_match_max_delta_db=cfg.SEGMENT_MATCH_MAX_DELTA_DB,
        segment_match_min_active_ratio=cfg.SEGMENT_MATCH_MIN_ACTIVE_RATIO,
    )

    tts_segments = synthesize_segments_with_timing(
        tts_backend=tts_backend,
        segments=translated,
        output_audio_path=paths["final_voice"],
        speaker_wav=speaker_source,
        speaker_profile=speaker_profile,
        reference_audio_path=paths["speaker_ref"],
        source_vocals_path=paths["vocals"] if os.path.exists(paths["vocals"]) else None,
        language=cfg.LANGUAGE,
        segments_dir=paths["audio_segments_dir"],
        runtime_config=runtime_config,
        smart_sync_config=smart_sync_config,
        tail_guard_config=tail_guard_config,
        segment_routing_config=segment_routing_config,
        audio_level_config=audio_level_config,
    )

    with open(paths["translated_segments"], "w", encoding="utf-8") as f:
        json.dump(tts_segments, f, ensure_ascii=False, indent=2)
    logger.info(
        "TTS-обновления сегментов сохранены: %s",
        paths["translated_segments"]
    )

    del tts_backend
    gc.collect(); clear_torch_cache()
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


def step_subtitles(paths: dict, suffix: str) -> None:
    from src.subtitles import add_subtitles_to_video

    logger.info("╔══ ШАГ 6: СУБТИТРЫ ══╗")
    check_file(paths["final_video"], "subtitles")
    check_file(paths["translated_segments"], "subtitles")

    with open(paths["translated_segments"], "r", encoding="utf-8") as f:
        translated = json.load(f)

    results = add_subtitles_to_video(
        video_path=paths["final_video"],
        segments=translated,
        output_dir=paths["subtitles_dir"],
        suffix=suffix,
        mode=getattr(cfg, "SUBTITLE_MODE", "soft"),
        use_original=getattr(cfg, "SUBTITLE_USE_ORIGINAL", False),
        ass_font=getattr(cfg, "SUBTITLE_ASS_FONT", "Arial"),
        ass_font_size=getattr(cfg, "SUBTITLE_ASS_FONT_SIZE", 24),
    )

    manifest_path = os.path.join(paths["subtitles_dir"], "subtitles_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Субтитры сохранены: %s", paths["subtitles_dir"])
    logger.info("╚══ Субтитры завершены ══╝")


def step_metrics(paths: dict, suffix: str) -> None:
    from src.asr_backend import load_metrics_asr_model
    from src.metrics import (speaker_verification_score, compute_wer_cer,
                              compute_labse_similarity, print_metrics_summary,
                              plot_labse)

    logger.info("╔══ ШАГ 7: МЕТРИКИ ══╗")
    check_file(paths["speaker_ref"],         "metrics")
    check_file(paths["final_voice"],         "metrics")
    check_file(paths["segments"],            "metrics")
    check_file(paths["translated_segments"], "metrics")

    with open(paths["segments"],            "r", encoding="utf-8") as f:
        segments = json.load(f)
    with open(paths["translated_segments"], "r", encoding="utf-8") as f:
        translated = json.load(f)

    spk_score = speaker_verification_score(paths["speaker_ref"], paths["final_voice"])

    model_asr = load_metrics_asr_model()
    wer_cer   = compute_wer_cer(model_asr, paths["final_voice"], translated)
    del model_asr
    gc.collect(); clear_torch_cache()

    labse = compute_labse_similarity(segments, translated)

    metrics_summary = {
        "job_name": suffix,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "original_video": paths["original_video"],
        "speaker_ref": paths["speaker_ref"],
        "final_voice": paths["final_voice"],
        "translated_segments": paths["translated_segments"],
        "translation": {
            "model_name": cfg.MT_MODEL_NAME,
            "strategy": cfg.MT_STRATEGY,
            "batch_size": cfg.MT_BATCH_SIZE,
            "max_length": cfg.MT_MAX_LENGTH,
            "max_segment_chars": cfg.MT_MAX_SEGMENT_CHARS,
        },
        "runtime": {
            "device": cfg.DEVICE,
            "whisper_model": cfg.WHISPER_MODEL_NAME,
            "metrics_asr_provider": cfg.METRICS_ASR_PROVIDER,
            "metrics_whisper_model": cfg.METRICS_WHISPER_MODEL_NAME,
            "metrics_asr_api_model": cfg.METRICS_ASR_API_MODEL if cfg.METRICS_ASR_PROVIDER == "groq" else None,
            "tts_backend": "xtts",
        },
        "metrics": {
            "speaker_verification": spk_score,
            "wer": wer_cer.get("wer"),
            "cer": wer_cer.get("cer"),
            "labse_mean": labse.get("mean"),
            "labse_min": labse.get("min"),
            "labse_max": labse.get("max"),
            "labse_n": len(labse.get("per_segment", [])),
        },
    }
    with open(paths["metrics_summary"], "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    print_metrics_summary(spk_score, wer_cer, labse)
    plot_labse(labse, title=f"[{suffix}] Zero-Shot —")
    logger.info("Сводка метрик сохранена: %s", paths["metrics_summary"])
    logger.info("╚══ Метрики подсчитаны ══╝")


def step_prepare_finetune(paths: dict, suffix: str) -> None:
    from src.finetune import prepare_finetune_dataset

    logger.info("╔══ ШАГ 8: ПОДГОТОВКА ДАТАСЕТА ДЛЯ FINETUNE ══╗")
    check_file(paths["segments"], "prepare_finetune")

    source_candidates = [
        paths.get(cfg.FINETUNE_SOURCE_AUDIO),
        paths.get("vocals"),
        paths.get("vocals_processed"),
        paths.get("original_audio"),
    ]
    source_audio = next(
        (candidate for candidate in source_candidates if candidate and os.path.exists(candidate)),
        None
    )
    if not source_audio:
        logger.error("Не найдено аудио для подготовки датасета.")
        logger.error("Сначала запустите preprocess и asr.")
        sys.exit(1)

    with open(paths["segments"], "r", encoding="utf-8") as f:
        segments = json.load(f)

    dataset_root = os.path.join(cfg.FINETUNE_DATA_ROOT, suffix)
    speaker_name = cfg.FINETUNE_SPEAKER_NAME or f"{suffix}_original"
    result = prepare_finetune_dataset(
        audio_path=source_audio,
        segments=segments,
        dataset_root=dataset_root,
        speaker_name=speaker_name,
        sample_rate=cfg.FINETUNE_SAMPLE_RATE,
        min_sec=cfg.FINETUNE_CLIP_MIN_SEC,
        max_sec=cfg.FINETUNE_CLIP_MAX_SEC,
        min_chars=cfg.FINETUNE_MIN_TEXT_CHARS,
        max_chars=cfg.FINETUNE_MAX_TEXT_CHARS,
        padding_ms=cfg.FINETUNE_PADDING_MS,
        eval_ratio=cfg.FINETUNE_EVAL_RATIO,
        max_eval_samples=cfg.FINETUNE_MAX_EVAL_SAMPLES,
        reference_clips=cfg.FINETUNE_REFERENCE_CLIPS,
        seed=cfg.SEED,
        target_speaker_id=cfg.DEFAULT_SPEAKER_ID
    )

    logger.info("Fine-tune dataset: %s", result["paths"]["root"])
    logger.info("Speaker name для обучения: %s", speaker_name)
    logger.info("╚══ Датасет для finetune подготовлен ══╝")


# ─── Регистр шагов ───────────────────────────────────────────────────────────

STEPS = {
    "preprocess":  step_preprocess,
    "asr":         step_asr,
    "translate":   step_translate,
    "tts":         step_tts,
    "postprocess": step_postprocess,
    "subtitles":   step_subtitles,
    "metrics":     step_metrics,
    "prepare_finetune": step_prepare_finetune,
}

ALL_STEPS = ["preprocess", "asr", "translate", "tts", "postprocess", "subtitles", "metrics"]


# ─── Точка входа ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Пайплайн автоматического дубляжа видео",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()) + ["all"],
        default="all",
        help=(
            "Шаг пайплайна (по умолчанию: all):\n"
            "  preprocess  - извлечь аудио, Demucs, денойзинг\n"
            "  asr         - транскрипция + сегментация\n"
            "  translate   - перевод en->ru\n"
            "  tts         - синтез речи + синхронизация\n"
            "  postprocess - микширование + сборка видео\n"
            "  subtitles   - генерация и встраивание субтитров\n"
            "  metrics     - подсчёт метрик качества\n"
            "  prepare_finetune - подготовка датасета для XTTS fine-tuning\n"
            "  all         - весь пайплайн целиком"
        )
    )
    parser.add_argument(
        "--video",
        default=None,
        help=(
            "Путь к входному видео. "
            "Если не указан, берётся единственное видео из ./data/input/ "
            "или legacy-файл по --suffix."
        )
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help=(
            "Имя задания для артефактов. "
            "По умолчанию берётся из имени файла видео."
        )
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help=(
            f"Legacy-режим старого нейминга: ищет video_<suffix>.mp4 "
            f"в {cfg.INPUT_PATH}. Если не указан, suffix больше не обязателен."
        )
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Тестовый режим: данные сохраняются в ./data/test/ (production не трогается)"
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Проверить зависимости, CLI и модельные файлы без запуска пайплайна"
    )
    parser.add_argument(
        "--mt-model",
        default=None,
        help=(
            "Переопределить модель перевода для текущего запуска. "
            "Пример: facebook/nllb-200-distilled-1.3B или gemini-2.5-flash."
        )
    )
    parser.add_argument(
        "--mt-strategy",
        choices=["per-segment", "sentence-level", "sliding-window", "context-aware"],
        default=None,
        help="Переопределить стратегию перевода для текущего запуска."
    )
    parser.add_argument(
        "--subtitle-mode",
        choices=["soft", "hard", "both"],
        default="soft",
        help="Режим субтитров для шага subtitles/all (по умолчанию: soft)."
    )
    parser.add_argument(
        "--subtitle-original",
        action="store_true",
        help="Генерировать субтитры из original_text вместо перевода."
    )
    args = parser.parse_args()

    if args.mt_model:
        cfg.MT_MODEL_NAME = args.mt_model
    if args.mt_strategy:
        cfg.MT_STRATEGY = args.mt_strategy
    cfg.SUBTITLE_MODE = args.subtitle_mode
    cfg.SUBTITLE_USE_ORIGINAL = args.subtitle_original

    if args.check_env:
        sys.exit(0 if check_environment() else 1)

    seed_everything(cfg.SEED)

    try:
        input_video = resolve_input_video(
            video_path=args.video,
            input_dir=cfg.INPUT_PATH,
            extensions=cfg.INPUT_VIDEO_EXTENSIONS,
            legacy_suffix=args.suffix
        )
        job_name = derive_job_name(
            video_path=input_video,
            explicit_job_name=args.job_name,
            legacy_suffix=args.suffix
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    paths = build_pipeline_paths(
        video_path=input_video,
        job_name=job_name,
        output_root=cfg.OUTPUT_PATH,
        test_output_root=cfg.TEST_OUTPUT_PATH,
        test=args.test
    )
    ensure_dirs(paths)

    mode = "ТЕСТ" if args.test else "PRODUCTION"
    logger.info(f"{'='*50}")
    logger.info(f"Режим:    {mode}")
    logger.info(f"Задание:  {job_name}")
    logger.info(f"Шаг:      {args.step}")
    logger.info(f"Видео:    {paths['original_video']}")
    logger.info(f"Выходная: {paths['output']}")
    logger.info(f"MT:       {cfg.MT_MODEL_NAME} | {cfg.MT_STRATEGY}")
    logger.info("TTS:      xtts")
    logger.info(f"{'='*50}")

    if args.step == "all":
        for step_name in ALL_STEPS:
            STEPS[step_name](paths, job_name)
    else:
        STEPS[args.step](paths, job_name)

    logger.info("Готово.")
