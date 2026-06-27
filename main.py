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
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

import config as cfg
from utils.helpers import seed_everything, manage_directory
from utils.pipeline_io import build_pipeline_paths, derive_job_name, resolve_input_video
from utils.pipeline_resume import step_resume_status

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

    tts_provider = getattr(cfg, "TTS_PROVIDER", "xtts").strip().lower()
    required_modules = [
        "numpy",
        "soundfile",
        "pydub",
        "scipy",
        "noisereduce",
        "tqdm",
        "torch",
        "whisper",
        "transformers",
    ]
    if tts_provider == "xtts":
        required_modules.append("TTS")
    if tts_provider == "elevenlabs":
        required_modules.append("requests")
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

    if tts_provider == "xtts":
        model_files = ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth")
        for filename in model_files:
            path = os.path.join(cfg.MODEL_TTS_DIR, filename)
            add(f"XTTS: {filename}", os.path.exists(path), path)
    elif tts_provider == "elevenlabs":
        api_key_env = getattr(cfg, "ELEVENLABS_API_KEY_ENV", "ELEVENLABS_API_KEY")
        api_key = os.getenv(api_key_env, "").strip()
        add(f"ElevenLabs API key: {api_key_env}", bool(api_key))
    else:
        add("TTS provider", False, f"unsupported: {tts_provider}")

    if cfg.ASR_PROVIDER in {"groq", "openai"}:
        add("Python API: openai", _module_exists("openai"))
        api_key = os.getenv(cfg.ASR_API_KEY_ENV, "").strip()
        add(f"ASR API key: {cfg.ASR_API_KEY_ENV}", bool(api_key))
    if cfg.METRICS_ASR_PROVIDER in {"groq", "openai"}:
        add("Python API: openai", _module_exists("openai"))
        api_key = os.getenv(cfg.METRICS_ASR_API_KEY_ENV, "").strip()
        add(f"Metrics ASR API key: {cfg.METRICS_ASR_API_KEY_ENV}", bool(api_key))
    mt_provider = os.getenv("MT_PROVIDER", getattr(cfg, "MT_PROVIDER", "")).strip().lower()
    mt_model = cfg.MT_MODEL_NAME.strip().lower()
    if mt_model.startswith("gemini-") or mt_provider == "gemini":
        api_key = os.getenv(cfg.MT_GEMINI_API_KEY_ENV, "").strip()
        add(f"Gemini API key: {cfg.MT_GEMINI_API_KEY_ENV}", bool(api_key))
    if mt_provider in {"openai", "chatgpt", "openrouter", "groq", "cerebras", "openai_compatible"} or mt_model.startswith(("gpt-", "chat-latest")):
        add("Python API: openai", _module_exists("openai"))
        provider_defaults = {
            "openai": "OPENAI_API_KEY",
            "chatgpt": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "groq": "GROQ_API_KEY",
            "cerebras": "CEREBRAS_API_KEY",
            "openai_compatible": "OPENAI_API_KEY",
        }
        api_key_env = os.getenv("MT_OPENAI_API_KEY_ENV", getattr(cfg, "MT_OPENAI_API_KEY_ENV", "")).strip()
        if not api_key_env:
            inferred_provider = mt_provider or "openai"
            api_key_env = api_key_env or provider_defaults.get(inferred_provider, "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env, "").strip()
        add(f"MT API key: {api_key_env}", bool(api_key))
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
        asr_language=getattr(cfg, "ASR_LANGUAGE", ""),
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
        analyze_translation_quality,
        load_translation_model,
        split_segments_for_translation,
        normalize_translated_segments,
        translate_segments,
        translate_segments_sentence_boundary_aware,
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
    elif cfg.MT_STRATEGY in {"sentence-boundary-aware", "boundary-aware"}:
        translated = translate_segments_sentence_boundary_aware(**translate_kwargs)
    else:
        raise ValueError(
            f"Неизвестная стратегия перевода: {cfg.MT_STRATEGY}. "
            "Ожидается per-segment, sentence-boundary-aware или boundary-aware."
        )

    translated = normalize_translated_segments(translated)

    with open(paths["translated_segments"], "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)
    clean_translation_path = os.path.join(paths["job_dir"], "translated_segments.clean.json")
    with open(clean_translation_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)
    logger.info(f"Перевод сохранён ({len(translated)} сегментов): {paths['translated_segments']}")

    translation_quality = analyze_translation_quality(
        translated,
        tgt_lang=cfg.MT_TGT_LANG,
    )
    quality_path = os.path.join(paths["job_dir"], "translation_quality.json")
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(translation_quality, f, ensure_ascii=False, indent=2)
    if translation_quality["issue_count"]:
        logger.warning(
            "Translation QA: %s/%s сегм. с предупреждениями (timing=%s, latin=%s, numbers=%s). См. %s",
            translation_quality["issue_count"],
            translation_quality["translated_count"],
            translation_quality["over_timing_soft_limit_count"],
            translation_quality["latin_word_count"],
            translation_quality["missing_number_count"],
            quality_path,
        )
    else:
        logger.info("Translation QA: предупреждений нет (%s)", quality_path)

    del model_mt, tokenizer
    gc.collect(); clear_torch_cache()
    logger.info("╚══ Перевод завершён ══╝")


def step_tts(paths: dict, suffix: str) -> None:
    from src.tts_backends import create_elevenlabs_tts_backend, create_tts_backend
    from src.config_snapshot import build_tts_config_snapshot
    from src.tts import synthesize_segments_with_timing
    from src.tts_config import (
        build_audio_level_config,
        build_segment_routing_config,
        build_smart_sync_config,
        build_tail_guard_config,
        build_tts_runtime_config,
    )

    logger.info("╔══ ШАГ 4: TTS ══╗")
    check_file(paths["translated_segments"], "tts")
    check_file(paths["speaker_ref"],         "tts")

    with open(paths["translated_segments"], "r", encoding="utf-8") as f:
        translated = json.load(f)
    clean_translation_path = os.path.join(paths["job_dir"], "translated_segments.clean.json")
    if os.path.exists(clean_translation_path) and any(
        "timing_fit" in segment
        or "corrected_duration_sec" in segment
        or "smart_sync" in segment
        for segment in translated
    ):
        with open(clean_translation_path, "r", encoding="utf-8") as f:
            translated = json.load(f)
        logger.info(
            "TTS использует чистую копию перевода без прошлой TTS metadata: %s",
            clean_translation_path,
        )

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

    tts_provider = getattr(cfg, "TTS_PROVIDER", "xtts").strip().lower()
    if tts_provider == "elevenlabs":
        api_key_env = getattr(cfg, "ELEVENLABS_API_KEY_ENV", "ELEVENLABS_API_KEY")
        api_key = os.getenv(api_key_env, "").strip()
        voice_name = getattr(cfg, "ELEVENLABS_VOICE_NAME", "").strip() or f"{paths['job_name']}_voice"
        tts_backend = create_elevenlabs_tts_backend(
            api_key=api_key,
            voice_id=getattr(cfg, "ELEVENLABS_VOICE_ID", ""),
            voice_name=voice_name,
            model_id=getattr(cfg, "ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
            output_format=getattr(cfg, "ELEVENLABS_OUTPUT_FORMAT", "pcm_24000"),
            base_url=getattr(cfg, "ELEVENLABS_BASE_URL", "https://api.elevenlabs.io/v1"),
            voice_manifest_path=paths["elevenlabs_voice"],
            clone_voice=getattr(cfg, "ELEVENLABS_CLONE_VOICE", True),
            remove_background_noise=getattr(cfg, "ELEVENLABS_REMOVE_BACKGROUND_NOISE", False),
            timeout_sec=getattr(cfg, "ELEVENLABS_TIMEOUT_SEC", 120),
            min_interval_sec=getattr(cfg, "ELEVENLABS_MIN_INTERVAL_SEC", 0.0),
            max_retries=getattr(cfg, "ELEVENLABS_MAX_RETRIES", 3),
            enable_logging=getattr(cfg, "ELEVENLABS_ENABLE_LOGGING", True),
            language_code=getattr(cfg, "ELEVENLABS_LANGUAGE_CODE", ""),
            apply_text_normalization=getattr(cfg, "ELEVENLABS_APPLY_TEXT_NORMALIZATION", "auto"),
            stability=getattr(cfg, "ELEVENLABS_STABILITY", None),
            similarity_boost=getattr(cfg, "ELEVENLABS_SIMILARITY_BOOST", None),
            style=getattr(cfg, "ELEVENLABS_STYLE", None),
            use_speaker_boost=getattr(cfg, "ELEVENLABS_USE_SPEAKER_BOOST", None),
            speed=getattr(cfg, "ELEVENLABS_SPEED", None),
        )
    elif tts_provider == "xtts":
        tts_backend = create_tts_backend(
            device=cfg.DEVICE,
            xtts_model_dir=cfg.MODEL_TTS_DIR,
            provider=tts_provider,
            temperature=cfg.XTTS_TEMPERATURE,
            length_penalty=cfg.XTTS_LENGTH_PENALTY,
            repetition_penalty=cfg.XTTS_REPETITION_PENALTY,
            top_k=cfg.XTTS_TOP_K,
            top_p=cfg.XTTS_TOP_P,
        )
    else:
        raise ValueError(f"Unsupported TTS_PROVIDER={tts_provider!r}. Use xtts or elevenlabs.")

    runtime_config = build_tts_runtime_config(cfg)
    smart_sync_config = build_smart_sync_config(cfg)
    tail_guard_config = build_tail_guard_config(cfg)
    segment_routing_config = build_segment_routing_config(
        cfg,
        enabled=False if tts_provider == "elevenlabs" else None,
    )
    audio_level_config = build_audio_level_config(cfg)
    tts_config_snapshot = build_tts_config_snapshot(cfg)
    if tts_provider == "elevenlabs":
        tts_config_snapshot.setdefault("segment_routing", {})["enabled"] = False
    with open(paths["tts_config_snapshot"], "w", encoding="utf-8") as f:
        json.dump(tts_config_snapshot, f, ensure_ascii=False, indent=2)

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
        mode=getattr(cfg, "SUBTITLE_MODE", "hard"),
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
    from src.config_snapshot import build_tts_config_snapshot
    from src.reporting import write_run_report

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

    tts_config_snapshot = None
    if os.path.exists(paths.get("tts_config_snapshot", "")):
        with open(paths["tts_config_snapshot"], "r", encoding="utf-8") as f:
            loaded_tts_config = json.load(f)
        if isinstance(loaded_tts_config, dict):
            tts_config_snapshot = loaded_tts_config
    if tts_config_snapshot is None:
        tts_config_snapshot = build_tts_config_snapshot(cfg)

    metrics_summary = {
        "job_name": suffix,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "original_video": paths["original_video"],
        "speaker_ref": paths["speaker_ref"],
        "final_voice": paths["final_voice"],
        "translated_segments": paths["translated_segments"],
        "translation": {
            "provider": getattr(cfg, "MT_PROVIDER", ""),
            "model_name": cfg.MT_MODEL_NAME,
            "strategy": cfg.MT_STRATEGY,
            "profile": getattr(cfg, "MT_PROFILE", ""),
            "style": getattr(cfg, "MT_STYLE", "standard"),
            "batch_size": cfg.MT_BATCH_SIZE,
            "max_length": cfg.MT_MAX_LENGTH,
            "max_segment_chars": cfg.MT_MAX_SEGMENT_CHARS,
        },
        "runtime": {
            "device": cfg.DEVICE,
            "whisper_model": cfg.WHISPER_MODEL_NAME,
            "asr_language": getattr(cfg, "ASR_LANGUAGE", ""),
            "metrics_asr_provider": cfg.METRICS_ASR_PROVIDER,
            "metrics_whisper_model": cfg.METRICS_WHISPER_MODEL_NAME,
            "metrics_asr_api_model": cfg.METRICS_ASR_API_MODEL if cfg.METRICS_ASR_PROVIDER == "groq" else None,
            "tts_backend": getattr(cfg, "TTS_PROVIDER", "xtts"),
        },
        "tts_config": tts_config_snapshot,
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

    run_mode = (
        "test"
        if os.path.abspath(paths["root_output"]) == os.path.abspath(cfg.TEST_OUTPUT_PATH)
        else "production"
    )
    report_path = write_run_report(
        paths=paths,
        segments=segments,
        translated_segments=translated,
        metrics_summary=metrics_summary,
        mode=run_mode,
    )

    print_metrics_summary(spk_score, wer_cer, labse)
    labse_plot_path = os.path.join(paths["job_dir"], "labse.png")
    plot_labse(labse, title=f"[{suffix}] Zero-Shot —", output_path=labse_plot_path)
    logger.info("Сводка метрик сохранена: %s", paths["metrics_summary"])
    logger.info("Отчёт запуска сохранён: %s", report_path)
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


def resolve_requested_steps(step: str, *, skip_metrics: bool = False) -> list[str]:
    """Resolve the public CLI step selection into an ordered execution plan.

    ``--skip-metrics`` only changes the aggregate ``all`` plan. An explicit
    ``--step metrics`` remains available for research and evaluation runs.
    """
    requested_steps = list(ALL_STEPS) if step == "all" else [step]
    if step == "all" and skip_metrics:
        requested_steps = [name for name in requested_steps if name != "metrics"]
    return requested_steps


def normalize_force_steps(force_steps: list[str] | None) -> set[str]:
    selected = set(force_steps or [])
    if "all" in selected:
        return set(STEPS.keys())
    return selected


def run_steps(
    *,
    step_names: list[str],
    paths: dict,
    job_name: str,
    resume: bool,
    force_steps: set[str],
    subtitle_mode: str,
) -> None:
    upstream_ran = False
    for step_name in step_names:
        forced = step_name in force_steps
        if resume and not upstream_ran and not forced:
            complete, missing = step_resume_status(
                paths,
                step_name,
                subtitle_mode=subtitle_mode,
            )
            if complete:
                logger.info("Resume: шаг '%s' пропущен, артефакты уже готовы.", step_name)
                continue
            logger.info(
                "Resume: шаг '%s' будет выполнен, отсутствует/невалидно: %s",
                step_name,
                ", ".join(missing),
            )
        elif resume and forced:
            logger.info("Resume: шаг '%s' пересчитывается принудительно.", step_name)
        elif resume and upstream_ran:
            logger.info(
                "Resume: шаг '%s' будет выполнен, потому что предыдущий шаг пересчитан.",
                step_name,
            )

        STEPS[step_name](paths, job_name)
        upstream_ran = True


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
        "--show-config",
        action="store_true",
        help="Показать эффективную конфигурацию запуска в JSON без запуска пайплайна"
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Проверить локальные пути, входное видео, модельные файлы, CLI и зависимости"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Пропускать уже готовые шаги по наличию валидных артефактов. "
            "Если шаг пересчитан, последующие шаги в --step all тоже выполняются."
        )
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help=(
            "При --step all завершить пайплайн после субтитров и не считать метрики. "
            "Явный --step metrics продолжает работать."
        ),
    )
    parser.add_argument(
        "--force-step",
        choices=list(STEPS.keys()) + ["all"],
        action="append",
        default=[],
        help=(
            "При --resume принудительно пересчитать указанный шаг. "
            "Можно передать несколько раз или указать all."
        )
    )
    parser.add_argument(
        "--mt-model",
        default=None,
        help=(
            "Переопределить модель перевода для текущего запуска. "
            "Пример: facebook/nllb-200-distilled-1.3B, gemini-2.5-flash или gpt-5.4-mini."
        )
    )
    parser.add_argument(
        "--mt-provider",
        choices=["hf", "gemini", "openai", "openrouter", "groq", "cerebras", "openai_compatible"],
        default=None,
        help=(
            "Переопределить provider перевода. "
            "Для своего OpenAI-compatible endpoint используйте openai_compatible."
        )
    )
    parser.add_argument(
        "--mt-strategy",
        choices=[
            "per-segment",
            "sentence-boundary-aware",
            "boundary-aware",
        ],
        default=None,
        help="Переопределить стратегию перевода для текущего запуска."
    )
    parser.add_argument(
        "--mt-style",
        choices=["standard", "academic", "casual", "news", "compact"],
        default=None,
        help="Переопределить стиль LLM-перевода для текущего запуска."
    )
    parser.add_argument(
        "--mt-profile",
        default=None,
        help=(
            "Профиль LLM-перевода из translation_profiles.yaml, например "
            "interview, lecture, podcast, news, movie или tech_review."
        )
    )
    parser.add_argument(
        "--mt-asr-correction",
        action="append",
        default=[],
        help=(
            "Добавить подсказку исправления ASR перед переводом, например "
            "'03 -> o3 (OpenAI model name)'. Можно передать несколько раз."
        )
    )
    parser.add_argument(
        "--tts-provider",
        choices=["xtts", "elevenlabs"],
        default=None,
        help="Переопределить TTS backend: локальный xtts или ElevenLabs API."
    )
    parser.add_argument(
        "--elevenlabs-voice-id",
        default=None,
        help="Использовать уже созданный ElevenLabs voice_id вместо создания клона."
    )
    parser.add_argument(
        "--elevenlabs-voice-name",
        default=None,
        help="Имя нового ElevenLabs Instant Voice Clone при автоклонировании."
    )
    parser.add_argument(
        "--elevenlabs-no-clone",
        action="store_true",
        help="Запретить создание нового ElevenLabs voice clone; требуется ELEVENLABS_VOICE_ID."
    )
    parser.add_argument(
        "--subtitle-mode",
        choices=["soft", "hard", "both"],
        default=None,
        help=(
            "Режим субтитров для шага subtitles/all. "
            "По умолчанию берётся SUBTITLE_MODE из config.py/config.example.py."
        )
    )
    parser.add_argument(
        "--subtitle-original",
        action="store_true",
        default=None,
        help="Генерировать субтитры из original_text вместо перевода."
    )
    args = parser.parse_args()

    # CLI-переопределения дублируются в os.environ, потому что часть
    # translation-настроек читает env с приоритетом над config.
    if args.mt_model:
        cfg.MT_MODEL_NAME = args.mt_model
    if args.mt_provider:
        cfg.MT_PROVIDER = args.mt_provider
        os.environ["MT_PROVIDER"] = args.mt_provider
    if args.mt_strategy:
        cfg.MT_STRATEGY = args.mt_strategy
    if args.mt_profile:
        cfg.MT_PROFILE = args.mt_profile.strip().lower()
        os.environ["MT_PROFILE"] = cfg.MT_PROFILE
    if args.mt_style:
        cfg.MT_STYLE = args.mt_style
        os.environ["MT_STYLE"] = args.mt_style
    if args.mt_asr_correction:
        cfg.MT_ASR_CORRECTIONS = "|".join(args.mt_asr_correction)
        os.environ["MT_ASR_CORRECTIONS"] = cfg.MT_ASR_CORRECTIONS
    if args.tts_provider:
        cfg.TTS_PROVIDER = args.tts_provider
    if args.elevenlabs_voice_id is not None:
        cfg.ELEVENLABS_VOICE_ID = args.elevenlabs_voice_id.strip()
    if args.elevenlabs_voice_name is not None:
        cfg.ELEVENLABS_VOICE_NAME = args.elevenlabs_voice_name.strip()
    if args.elevenlabs_no_clone:
        cfg.ELEVENLABS_CLONE_VOICE = False
    if args.subtitle_mode is not None:
        cfg.SUBTITLE_MODE = args.subtitle_mode
    if not hasattr(cfg, "SUBTITLE_MODE"):
        cfg.SUBTITLE_MODE = "hard"
    cfg.SUBTITLE_MODE = str(cfg.SUBTITLE_MODE).strip().lower()
    if cfg.SUBTITLE_MODE not in {"soft", "hard", "both"}:
        parser.error("SUBTITLE_MODE должен быть soft, hard или both.")
    if args.subtitle_original is not None:
        cfg.SUBTITLE_USE_ORIGINAL = args.subtitle_original
    if not hasattr(cfg, "SUBTITLE_USE_ORIGINAL"):
        cfg.SUBTITLE_USE_ORIGINAL = False
    if not hasattr(cfg, "SUBTITLE_ASS_FONT"):
        cfg.SUBTITLE_ASS_FONT = "Arial"
    if not hasattr(cfg, "SUBTITLE_ASS_FONT_SIZE"):
        cfg.SUBTITLE_ASS_FONT_SIZE = 24

    if args.check_env:
        sys.exit(0 if check_environment() else 1)
    if args.show_config:
        from src.config_snapshot import build_pipeline_config_snapshot

        print(json.dumps(
            build_pipeline_config_snapshot(cfg),
            ensure_ascii=False,
            indent=2,
        ))
        sys.exit(0)
    if args.doctor:
        from src.doctor import format_doctor_report, has_doctor_failures, run_project_doctor

        checks = run_project_doctor(
            config=cfg,
            project_root=Path(__file__).resolve().parent,
            video_path=args.video,
            legacy_suffix=args.suffix,
        )
        print(format_doctor_report(checks))
        sys.exit(1 if has_doctor_failures(checks) else 0)

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
    logger.info(f"Resume:   {'on' if args.resume else 'off'}")
    logger.info(f"Метрики:  {'skip' if args.step == 'all' and args.skip_metrics else 'run'}")
    logger.info(f"Видео:    {paths['original_video']}")
    logger.info(f"Выходная: {paths['output']}")
    logger.info(f"MT:       {cfg.MT_MODEL_NAME} | {cfg.MT_STRATEGY}")
    logger.info(f"TTS:      {cfg.TTS_PROVIDER}")
    logger.info(f"{'='*50}")

    requested_steps = resolve_requested_steps(args.step, skip_metrics=args.skip_metrics)
    run_steps(
        step_names=requested_steps,
        paths=paths,
        job_name=job_name,
        resume=args.resume,
        force_steps=normalize_force_steps(args.force_step),
        subtitle_mode=cfg.SUBTITLE_MODE,
    )

    logger.info("Готово.")
