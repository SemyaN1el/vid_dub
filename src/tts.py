"""Синтез русской речи и временная синхронизация: группировка сегментов в TTS-блоки, расчёт окон, SmartSync, ограниченное ускорение atempo, guard-проверки и сборка дорожки дубляжа."""

import os
import logging
import subprocess
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, List, Tuple

import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.effects import speedup
from tqdm.auto import tqdm
from src.translation import smart_sync_rewrite_segment_text
from src.tts_audio import (
    AudioLevelConfig,
    _active_speech_dbfs,
    _active_speech_stats,
    _compute_segment_target_level,
    _match_segment_level,
    apply_final_audio_processing,
)
from src.tts_guards import (
    _compute_safe_tail_trim_ms,
    _is_better_recognition_eval,
    _normalize_word_tokens,
    _normalized_text_similarity,
    _segment_recognition_score,
    _token_overlap_stats,
    _transcribe_short_audio,
    _trim_trailing_babble,
    _trim_trailing_speech_island_fast,
)
from src.tts_routing import _select_segment_references
from src.tts_resources import (
    existing_reference_paths,
    load_smart_sync_rewrite_backend,
    load_tts_aux_models,
)
from src.tts_timing import (
    _edge_silence_ms,
    _timing_speech_stats,
    build_segment_timing_window,
    update_next_segment_start,
)
from src.tts_text import (
    _build_tts_retry_text_variants,
    _clean_text,
    _ends_with_terminal_punctuation,
    _prepare_tts_segments,
    _same_speaker,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TTSRuntimeConfig:
    max_speedup_factor: float = 1.35
    max_next_start_shift_sec: float | None = None
    speedup_tail_padding_ms: int = 120
    speedup_trim_edge_silence: bool = True
    speedup_trim_keep_edge_ms: int = 90
    speedup_trim_min_edge_ms: int = 180
    min_pause_between_segments: float = 0.2
    fade_in_out_ms: int = 50
    crossfade_ms: int = 30
    max_shift_left_seconds: float = 0.5
    enable_grouping: bool = True
    grouping_max_gap_sec: float = 0.6
    grouping_max_segments: int = 2
    grouping_max_chars: int = 220
    grouping_max_duration_sec: float = 8.5


@dataclass(frozen=True)
class SmartSyncConfig:
    enabled: bool = False
    device: str = "cpu"
    src_lang: str = "eng_Latn"
    tgt_lang: str = "rus_Cyrl"
    max_rewrites: int = 1
    candidate_count: int = 3
    target_speedup_factor: float = 1.25
    trigger_speed_factor: float = 1.08
    min_fill_ratio: float = 0.82
    min_improvement_ms: int = 180
    allow_lengthen: bool = True
    accept_min_fill_ratio: float = 0.72
    accept_min_text_similarity: float = 0.42
    accept_min_word_ratio: float = 0.58
    accept_min_token_precision: float = 0.62
    accept_min_asr_score: float = 0.86
    accept_max_asr_drop: float = 0.05


@dataclass(frozen=True)
class TailGuardConfig:
    enable_cheap_tail_guard: bool = True
    cheap_tail_guard_max_segment_sec: float = 3.2
    cheap_tail_guard_min_overhang_ms: int = 180
    cheap_tail_guard_min_gap_ms: int = 80
    cheap_tail_guard_min_island_ms: int = 80
    cheap_tail_guard_max_island_ms: int = 450
    cheap_tail_guard_search_window_ms: int = 900
    cheap_tail_guard_max_trim_ms: int = 700
    enable_babble_guard: bool = False
    babble_guard_model_name: str = "small"
    babble_guard_device: str = "cpu"
    babble_guard_max_segment_sec: float = 4.0
    babble_guard_min_gap_ms: int = 80
    babble_guard_min_island_ms: int = 80
    babble_guard_max_island_ms: int = 450
    babble_guard_search_window_ms: int = 900
    babble_guard_max_trim_ms: int = 700
    babble_guard_anchor_words: int = 2
    babble_guard_min_score_gain: float = 0.08
    enable_asr_retry: bool = False
    asr_retry_model_name: str = "tiny"
    asr_retry_device: str = "cpu"
    asr_retry_max_segment_sec: float = 2.5
    asr_retry_attempts: int = 4
    asr_retry_min_score: float = 0.9
    enable_short_segment_tail_trim: bool = False
    short_segment_tail_trim_min_overhang_ms: int = 280
    short_segment_tail_trim_max_ms: int = 500
    short_segment_tail_trim_max_ratio: float = 0.22


@dataclass(frozen=True)
class SegmentRoutingConfig:
    enabled: bool = False
    short_segment_sec: float = 2.2
    max_refs_per_segment: int = 2
    min_segment_sec: float = 0.9
    min_segment_words: int = 3
    confidence_margin: float = 0.45


def _smart_sync_distance_ms(duration_ms: int, target_ms: int, mode: str) -> int:
    if mode == "shorter":
        return max(0, duration_ms - target_ms)
    return abs(duration_ms - target_ms)


def _smart_sync_acceptance_gate(
    *,
    source_text: str,
    rewritten_text: str,
    rewrite_mode: str,
    baseline_duration_ms: int,
    rewritten_duration_ms: int,
    target_duration_ms: int,
    baseline_eval: Dict[str, Any] | None,
    rewritten_eval: Dict[str, Any] | None,
    min_fill_ratio: float,
    min_text_similarity: float,
    min_word_ratio: float,
    min_token_precision: float,
    min_asr_score: float,
    max_asr_drop: float,
) -> tuple[bool, Dict[str, Any]]:
    source_words = _normalize_word_tokens(source_text)
    rewritten_words = _normalize_word_tokens(rewritten_text)
    text_similarity = _normalized_text_similarity(source_text, rewritten_text)
    _, _, token_precision = _token_overlap_stats(source_words, rewritten_words)
    word_ratio = len(rewritten_words) / max(1, len(source_words))
    fill_ratio = rewritten_duration_ms / max(1, target_duration_ms)
    duration_ratio = rewritten_duration_ms / max(1, baseline_duration_ms)

    baseline_score = None
    if baseline_eval is not None:
        baseline_score = float(baseline_eval.get("score") or 0.0)
    rewritten_score = None
    if rewritten_eval is not None:
        rewritten_score = float(rewritten_eval.get("score") or 0.0)
    baseline_has_extra_tail = bool(baseline_eval.get("has_extra_tail")) if baseline_eval is not None else None
    rewritten_has_extra_tail = bool(rewritten_eval.get("has_extra_tail")) if rewritten_eval is not None else None
    rewritten_has_suffix = bool(rewritten_eval.get("has_suffix")) if rewritten_eval is not None else None

    metrics = {
        "text_similarity": round(text_similarity, 4),
        "token_precision": round(token_precision, 4),
        "word_ratio": round(word_ratio, 4),
        "fill_ratio": round(fill_ratio, 4),
        "duration_ratio": round(duration_ratio, 4),
        "baseline_asr_score": round(baseline_score, 4) if baseline_score is not None else None,
        "rewritten_asr_score": round(rewritten_score, 4) if rewritten_score is not None else None,
        "asr_drop": (
            round(max(0.0, baseline_score - rewritten_score), 4)
            if baseline_score is not None and rewritten_score is not None
            else None
        ),
        "baseline_has_extra_tail": baseline_has_extra_tail,
        "rewritten_has_extra_tail": rewritten_has_extra_tail,
        "rewritten_has_suffix": rewritten_has_suffix,
    }

    if rewrite_mode == "shorter":
        if fill_ratio < min_fill_ratio:
            metrics["reject_reason"] = "underfill"
            return False, metrics
        if len(source_words) >= 5 and word_ratio < min_word_ratio:
            metrics["reject_reason"] = "word_ratio"
            return False, metrics
        if len(source_words) >= 4 and text_similarity < min_text_similarity:
            metrics["reject_reason"] = "text_similarity"
            return False, metrics
        if len(source_words) >= 4 and token_precision < min_token_precision:
            metrics["reject_reason"] = "token_precision"
            return False, metrics

    if rewritten_score is not None:
        if rewritten_has_extra_tail:
            metrics["reject_reason"] = "extra_tail"
            return False, metrics
        if rewritten_has_suffix is False:
            metrics["reject_reason"] = "missing_suffix"
            return False, metrics
        if rewritten_score < min_asr_score:
            metrics["reject_reason"] = "asr_score"
            return False, metrics
        if baseline_score is not None and rewritten_score < (baseline_score - max_asr_drop):
            metrics["reject_reason"] = "asr_drop"
            return False, metrics
        if rewrite_mode == "longer" and baseline_score is not None:
            if rewritten_score + 0.01 < baseline_score:
                metrics["reject_reason"] = "asr_not_preserved"
                return False, metrics

    metrics["reject_reason"] = None
    return True, metrics


def _smart_sync_candidate_options(
    rewritten_text: str | None,
    rewrite_info: Dict[str, Any] | None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Returns SmartSync candidates in preflight rank order, without duplicates."""
    if not rewritten_text:
        return []

    base_info = dict(rewrite_info or {})
    options: List[Tuple[str, Dict[str, Any]]] = []
    seen: set[str] = set()

    def add_option(text: Any, *, rank: int | None = None, metrics: Dict[str, Any] | None = None) -> None:
        candidate = _clean_text(str(text or ""))
        key = " ".join(candidate.lower().split())
        if not candidate or key in seen:
            return
        seen.add(key)
        info = dict(base_info)
        info["rewritten_text"] = candidate
        if rank is not None:
            info["candidate_rank"] = rank
        if metrics is not None:
            info["candidate_metrics"] = metrics
        options.append((candidate, info))

    add_option(rewritten_text, rank=0)

    ranked = ((rewrite_info or {}).get("preflight") or {}).get("ranked") or []
    for rank, entry in enumerate(ranked, start=1):
        if not isinstance(entry, dict):
            continue
        add_option(
            entry.get("candidate"),
            rank=rank,
            metrics=entry.get("metrics") if isinstance(entry.get("metrics"), dict) else None,
        )

    return options


def _build_atempo_chain(playback_speed: float) -> List[float]:
    """Разбивает коэффициент скорости на допустимую цепочку atempo."""
    if playback_speed <= 0:
        raise ValueError("playback_speed must be positive")

    factors: List[float] = []
    remainder = playback_speed

    while remainder > 2.0:
        factors.append(2.0)
        remainder /= 2.0

    while remainder < 0.5:
        factors.append(0.5)
        remainder /= 0.5

    factors.append(remainder)
    return factors


def _time_stretch_ffmpeg(
    audio: AudioSegment,
    playback_speed: float
) -> AudioSegment:
    """
    Меняет темп через ffmpeg/atempo.
    Это надёжнее для концовок фраз, чем pydub.speedup, который выкидывает чанки.
    """
    if len(audio) == 0 or abs(playback_speed - 1.0) < 0.01:
        return audio

    filter_chain = ",".join(
        f"atempo={factor:.5f}"
        for factor in _build_atempo_chain(playback_speed)
    )

    src_path = dst_path = None
    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as src_tmp:
            src_path = src_tmp.name
        with NamedTemporaryFile(suffix=".wav", delete=False) as dst_tmp:
            dst_path = dst_tmp.name

        audio.export(src_path, format="wav")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            src_path,
            "-filter:a",
            filter_chain,
            dst_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ffmpeg atempo failed")

        return AudioSegment.from_wav(dst_path)
    except Exception as exc:
        logger.warning(
            "FFmpeg atempo не сработал, fallback на pydub.speedup: %s",
            exc
        )
        return speedup(audio, playback_speed=playback_speed)
    finally:
        for path in (src_path, dst_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def _trim_edge_silence_for_timing_fit(
    audio: AudioSegment,
    *,
    keep_edge_ms: int,
    min_edge_ms: int,
    edge_silence: Dict[str, int] | None = None,
) -> tuple[AudioSegment, Dict[str, Any]]:
    """Trims only surplus edge silence before timing fit, preserving a small edge pad."""
    keep_edge_ms = max(0, int(keep_edge_ms))
    min_edge_ms = max(0, int(min_edge_ms))
    info: Dict[str, Any] = {
        "leading_silence_ms": 0,
        "trailing_silence_ms": 0,
        "trim_leading_ms": 0,
        "trim_trailing_ms": 0,
    }
    if len(audio) <= 0:
        return audio, info

    edge = edge_silence if edge_silence is not None else _edge_silence_ms(audio)
    trim_leading_ms = (
        max(0, edge["leading"] - keep_edge_ms)
        if edge["leading"] >= min_edge_ms
        else 0
    )
    trim_trailing_ms = (
        max(0, edge["trailing"] - keep_edge_ms)
        if edge["trailing"] >= min_edge_ms
        else 0
    )
    if trim_leading_ms + trim_trailing_ms >= max(0, len(audio) - 120):
        trim_leading_ms = 0
        trim_trailing_ms = 0

    info.update({
        "leading_silence_ms": int(edge["leading"]),
        "trailing_silence_ms": int(edge["trailing"]),
        "trim_leading_ms": int(trim_leading_ms),
        "trim_trailing_ms": int(trim_trailing_ms),
    })
    if not trim_leading_ms and not trim_trailing_ms:
        return audio, info

    end_ms = max(trim_leading_ms, len(audio) - trim_trailing_ms)
    return audio[trim_leading_ms:end_ms], info


def _fit_audio_to_timing_window(
    audio: AudioSegment,
    *,
    available_ms: int,
    max_speedup_factor: float,
    speedup_tail_padding_ms: int,
    trim_edge_silence: bool,
    trim_keep_edge_ms: int,
    trim_min_edge_ms: int,
) -> tuple[AudioSegment, Dict[str, Any]]:
    """
    Fits generated speech toward its timing window by trimming surplus edge silence
    first, then applying bounded ffmpeg atempo speedup.
    """
    available_ms = max(1, int(available_ms))
    max_speedup_factor = max(1.0, float(max_speedup_factor))
    fitted_audio = audio

    initial_stats = _timing_speech_stats(fitted_audio)
    initial_timing_ms = max(1, initial_stats["effective_duration_ms"] or len(fitted_audio))
    fit_info: Dict[str, Any] = {
        "available_ms": int(available_ms),
        "initial_duration_ms": int(len(audio)),
        "initial_timing_ms": int(initial_timing_ms),
        "edge_trim_enabled": bool(trim_edge_silence),
        "pre_trim_leading_ms": 0,
        "pre_trim_trailing_ms": 0,
        "post_trim_leading_ms": 0,
        "post_trim_trailing_ms": 0,
        "required_speedup_factor": round(initial_timing_ms / available_ms, 4),
        "applied_speedup_factor": 1.0,
        "speedup_capped": False,
        "final_duration_ms": int(len(audio)),
        "final_timing_ms": int(initial_timing_ms),
        "residual_timing_ratio": round(initial_timing_ms / available_ms, 4),
    }

    timing_ms = initial_timing_ms
    if trim_edge_silence:
        fitted_audio, trim_info = _trim_edge_silence_for_timing_fit(
            fitted_audio,
            keep_edge_ms=trim_keep_edge_ms,
            min_edge_ms=trim_min_edge_ms,
            edge_silence={
                "leading": int(initial_stats["leading_silence_ms"]),
                "trailing": int(initial_stats["trailing_silence_ms"]),
            },
        )
        fit_info["pre_trim_leading_ms"] = trim_info["trim_leading_ms"]
        fit_info["pre_trim_trailing_ms"] = trim_info["trim_trailing_ms"]
        if trim_info["trim_leading_ms"] or trim_info["trim_trailing_ms"]:
            trimmed_stats = _timing_speech_stats(fitted_audio)
            timing_ms = max(1, trimmed_stats["effective_duration_ms"] or len(fitted_audio))

    required_factor = timing_ms / available_ms
    fit_info["required_speedup_factor"] = round(required_factor, 4)
    speedup_applied = False

    if timing_ms > available_ms:
        applied_factor = min(required_factor, max_speedup_factor)
        fit_info["applied_speedup_factor"] = round(applied_factor, 4)
        fit_info["speedup_capped"] = required_factor > max_speedup_factor + 1e-6
        speedup_input = fitted_audio + AudioSegment.silent(
            duration=max(0, int(speedup_tail_padding_ms)),
            frame_rate=fitted_audio.frame_rate,
        )
        fitted_audio = _time_stretch_ffmpeg(
            speedup_input,
            playback_speed=applied_factor,
        )
        speedup_applied = True

        if trim_edge_silence:
            fitted_audio, post_trim_info = _trim_edge_silence_for_timing_fit(
                fitted_audio,
                keep_edge_ms=trim_keep_edge_ms,
                min_edge_ms=trim_min_edge_ms,
            )
            fit_info["post_trim_leading_ms"] = post_trim_info["trim_leading_ms"]
            fit_info["post_trim_trailing_ms"] = post_trim_info["trim_trailing_ms"]
            if not post_trim_info["trim_leading_ms"] and not post_trim_info["trim_trailing_ms"]:
                speedup_applied = True

    if speedup_applied:
        final_stats = _timing_speech_stats(fitted_audio)
        final_timing_ms = max(1, final_stats["effective_duration_ms"] or len(fitted_audio))
    else:
        final_timing_ms = timing_ms
    fit_info["final_duration_ms"] = int(len(fitted_audio))
    fit_info["final_timing_ms"] = int(final_timing_ms)
    fit_info["residual_timing_ratio"] = round(final_timing_ms / available_ms, 4)
    return fitted_audio, fit_info


def _audio_array_to_segment(wav: Any, sample_rate: int) -> AudioSegment:
    samples = np.asarray(wav)
    if samples.ndim == 0:
        samples = samples.reshape(1)
    if samples.ndim == 1:
        channels = 1
    else:
        channels = samples.shape[1]
    if np.issubdtype(samples.dtype, np.floating):
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    else:
        pcm = samples.astype("<i2")
    return AudioSegment(
        data=pcm.tobytes(),
        sample_width=2,
        frame_rate=int(sample_rate),
        channels=int(channels),
    )


def generate_audio_segment(
    tts_backend,
    text: str,
    output_path: str | None,
    speaker_wav,
    language: str,
    conditioning=None,
    speaker_profile: Dict[str, Any] | None = None,
    inference_overrides: Dict[str, Any] | None = None,
) -> Tuple[str | None, float, AudioSegment]:
    """
    Синтезирует аудио для одного текстового сегмента.

    Параметры:
        tts_backend: загруженный backend TTS
        text: текст для синтеза
        output_path: путь для сохранения сегмента; если None, файл не пишется
        speaker_wav: путь или список путей к референсам спикера
        language: язык синтеза ('ru', 'en', ...)
        conditioning: предвычисленное conditioning backend-а
        speaker_profile: профиль спикера с текстами референсов (опционально)

    Возвращает:
        Tuple[str | None, float, AudioSegment]: путь, длительность и готовый audio object
    """
    if conditioning is None:
        if isinstance(speaker_wav, list):
            reference_paths = [path for path in speaker_wav if path and os.path.exists(path)]
        else:
            reference_paths = [speaker_wav] if speaker_wav and os.path.exists(speaker_wav) else []
        conditioning = tts_backend.prepare_conditioning(
            reference_paths=reference_paths,
            speaker_profile=speaker_profile
        )

    wav, sample_rate = tts_backend.synthesize(
        text=text,
        language=language,
        conditioning=conditioning,
        inference_overrides=inference_overrides,
    )
    if output_path:
        sf.write(output_path, wav, sample_rate)

    audio = _audio_array_to_segment(wav, sample_rate)
    duration = len(audio) / 1000.0
    return output_path, duration, audio


def _serialize_tts_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Возвращает TTS metadata без runtime-only audio objects."""
    serializable: List[Dict[str, Any]] = []
    for segment in segments:
        item = {
            key: value
            for key, value in segment.items()
            if key != "corrected_audio"
        }
        serializable.append(item)
    return serializable


def _assemble_timeline(
    placements: List[Tuple[int, AudioSegment]],
    replacements: List[Tuple[int, AudioSegment]],
    min_duration_ms: int,
) -> AudioSegment:
    """
    Собирает дорожку одним numpy-проходом вместо повторного копирования
    всего таймлайна на каждом overlay: placements микшируются между собой,
    replacements перезаписывают свой регион (семантика кроссфейд-патчей).
    """
    min_duration_ms = max(0, int(min_duration_ms))
    parts = list(placements) + list(replacements)
    if not parts:
        return AudioSegment.silent(duration=min_duration_ms)

    frame_rate = max(audio.frame_rate for _, audio in parts)
    channels = max(audio.channels for _, audio in parts)

    def to_int32_samples(audio: AudioSegment) -> np.ndarray:
        normalized = (
            audio.set_frame_rate(frame_rate)
            .set_channels(channels)
            .set_sample_width(2)
        )
        return np.asarray(normalized.get_array_of_samples(), dtype=np.int32)

    def buffer_offset(position_ms: int) -> int:
        return int(round(max(0, int(position_ms)) * frame_rate / 1000.0)) * channels

    prepared_placements = [
        (buffer_offset(position_ms), to_int32_samples(audio))
        for position_ms, audio in placements
    ]
    prepared_replacements = [
        (buffer_offset(position_ms), to_int32_samples(audio))
        for position_ms, audio in replacements
    ]

    min_size = int(round(min_duration_ms * frame_rate / 1000.0)) * channels
    buffer_size = max(
        [min_size]
        + [
            offset + samples.size
            for offset, samples in prepared_placements + prepared_replacements
        ]
    )
    buffer = np.zeros(buffer_size, dtype=np.int32)

    for offset, samples in prepared_placements:
        buffer[offset:offset + samples.size] += samples
    np.clip(buffer, -32768, 32767, out=buffer)

    for offset, samples in prepared_replacements:
        buffer[offset:offset + samples.size] = samples

    return AudioSegment(
        data=buffer.astype("<i2").tobytes(),
        sample_width=2,
        frame_rate=frame_rate,
        channels=channels,
    )


def _synthesize_best_retry_take(
    *,
    retry_text_variants: List[str],
    fallback_text: str,
    attempts: int,
    synthesize_fn: Callable[[str], AudioSegment],
    transcribe_fn: Callable[[AudioSegment], str],
    expected_text: str,
    anchor_words: int,
) -> tuple[AudioSegment | None, str | None, Dict[str, Any] | None]:
    """
    Выбирает лучший retry-дубль по ASR-оценке сырого синтеза.
    Дорогой finalize (atempo + guards) потом выполняется один раз
    только для победителя, а не для каждой попытки.
    """
    best_audio: AudioSegment | None = None
    best_text: str | None = None
    best_eval: Dict[str, Any] | None = None

    for attempt in range(1, max(0, int(attempts)) + 1):
        text_idx = min(attempt - 1, max(0, len(retry_text_variants) - 1))
        retry_text = retry_text_variants[text_idx] if retry_text_variants else fallback_text
        retry_audio = synthesize_fn(retry_text)

        raw_eval = _segment_recognition_score(
            expected_text,
            transcribe_fn(retry_audio),
            anchor_words,
        )
        if _is_better_recognition_eval(raw_eval, best_eval):
            best_audio = retry_audio
            best_text = retry_text
            best_eval = raw_eval

        if (
            raw_eval["score"] >= 0.995
            and not raw_eval.get("has_extra_tail")
            and raw_eval.get("has_suffix", True)
        ):
            break

    return best_audio, best_text, best_eval


def synthesize_segments_with_timing(
    tts_backend,
    segments: List[Dict[str, Any]],
    output_audio_path: str,
    speaker_wav,
    language: str,
    speaker_profile: Dict[str, Any] | None = None,
    reference_audio_path: str | None = None,
    source_vocals_path: str | None = None,
    segments_dir: str = "./data/output/temp/audio_segments",
    runtime_config: TTSRuntimeConfig | None = None,
    smart_sync_config: SmartSyncConfig | None = None,
    tail_guard_config: TailGuardConfig | None = None,
    segment_routing_config: SegmentRoutingConfig | None = None,
    audio_level_config: AudioLevelConfig | None = None,
) -> List[Dict[str, Any]]:
    """
    Синтезирует дубляж с временно́й синхронизацией сегментов.

    Алгоритм:
        1. Предвычисляет conditioning latents один раз для всего пайплайна.
        2. Для каждого сегмента генерирует аудио и проверяет вписывается ли
           оно в отведённое время с учётом паузы до следующего сегмента.
        3. При необходимости ускоряет сегмент (не более max_speedup_factor).
        4. Применяет fade-in/out и кроссфейды между соседними сегментами.
        5. Нормализует итоговое аудио.
    """
    runtime_config = runtime_config or TTSRuntimeConfig()
    smart_sync_config = smart_sync_config or SmartSyncConfig()
    tail_guard_config = tail_guard_config or TailGuardConfig()
    segment_routing_config = segment_routing_config or SegmentRoutingConfig()
    audio_level_config = audio_level_config or AudioLevelConfig()

    max_speedup_factor = runtime_config.max_speedup_factor
    max_next_start_shift_sec = runtime_config.max_next_start_shift_sec
    speedup_tail_padding_ms = runtime_config.speedup_tail_padding_ms
    speedup_trim_edge_silence = runtime_config.speedup_trim_edge_silence
    speedup_trim_keep_edge_ms = runtime_config.speedup_trim_keep_edge_ms
    speedup_trim_min_edge_ms = runtime_config.speedup_trim_min_edge_ms
    min_pause_between_segments = runtime_config.min_pause_between_segments
    fade_in_out_ms = runtime_config.fade_in_out_ms
    crossfade_ms = runtime_config.crossfade_ms
    max_shift_left_seconds = runtime_config.max_shift_left_seconds
    enable_tts_grouping = runtime_config.enable_grouping
    tts_grouping_max_gap_sec = runtime_config.grouping_max_gap_sec
    tts_grouping_max_segments = runtime_config.grouping_max_segments
    tts_grouping_max_chars = runtime_config.grouping_max_chars
    tts_grouping_max_duration_sec = runtime_config.grouping_max_duration_sec

    enable_smart_sync = smart_sync_config.enabled
    smart_sync_src_lang = smart_sync_config.src_lang
    smart_sync_tgt_lang = smart_sync_config.tgt_lang
    smart_sync_max_rewrites = smart_sync_config.max_rewrites
    smart_sync_candidate_count = max(1, int(smart_sync_config.candidate_count))
    smart_sync_target_speedup_factor = min(
        max_speedup_factor,
        max(1.0, smart_sync_config.target_speedup_factor),
    )
    smart_sync_trigger_speed_factor = smart_sync_config.trigger_speed_factor
    smart_sync_min_fill_ratio = smart_sync_config.min_fill_ratio
    smart_sync_min_improvement_ms = smart_sync_config.min_improvement_ms
    smart_sync_allow_lengthen = smart_sync_config.allow_lengthen
    smart_sync_accept_min_fill_ratio = smart_sync_config.accept_min_fill_ratio
    smart_sync_accept_min_text_similarity = smart_sync_config.accept_min_text_similarity
    smart_sync_accept_min_word_ratio = smart_sync_config.accept_min_word_ratio
    smart_sync_accept_min_token_precision = smart_sync_config.accept_min_token_precision
    smart_sync_accept_min_asr_score = smart_sync_config.accept_min_asr_score
    smart_sync_accept_max_asr_drop = smart_sync_config.accept_max_asr_drop

    enable_tts_cheap_tail_guard = tail_guard_config.enable_cheap_tail_guard
    tts_cheap_tail_guard_max_segment_sec = tail_guard_config.cheap_tail_guard_max_segment_sec
    tts_cheap_tail_guard_min_overhang_ms = tail_guard_config.cheap_tail_guard_min_overhang_ms
    tts_cheap_tail_guard_min_gap_ms = tail_guard_config.cheap_tail_guard_min_gap_ms
    tts_cheap_tail_guard_min_island_ms = tail_guard_config.cheap_tail_guard_min_island_ms
    tts_cheap_tail_guard_max_island_ms = tail_guard_config.cheap_tail_guard_max_island_ms
    tts_cheap_tail_guard_search_window_ms = tail_guard_config.cheap_tail_guard_search_window_ms
    tts_cheap_tail_guard_max_trim_ms = tail_guard_config.cheap_tail_guard_max_trim_ms
    tts_babble_guard_max_segment_sec = tail_guard_config.babble_guard_max_segment_sec
    tts_babble_guard_min_gap_ms = tail_guard_config.babble_guard_min_gap_ms
    tts_babble_guard_min_island_ms = tail_guard_config.babble_guard_min_island_ms
    tts_babble_guard_max_island_ms = tail_guard_config.babble_guard_max_island_ms
    tts_babble_guard_search_window_ms = tail_guard_config.babble_guard_search_window_ms
    tts_babble_guard_max_trim_ms = tail_guard_config.babble_guard_max_trim_ms
    tts_babble_guard_anchor_words = tail_guard_config.babble_guard_anchor_words
    tts_babble_guard_min_score_gain = tail_guard_config.babble_guard_min_score_gain
    enable_tts_asr_retry = tail_guard_config.enable_asr_retry
    tts_asr_retry_max_segment_sec = tail_guard_config.asr_retry_max_segment_sec
    tts_asr_retry_attempts = tail_guard_config.asr_retry_attempts
    tts_asr_retry_min_score = tail_guard_config.asr_retry_min_score
    enable_short_segment_tail_trim = tail_guard_config.enable_short_segment_tail_trim
    short_segment_tail_trim_min_overhang_ms = tail_guard_config.short_segment_tail_trim_min_overhang_ms
    short_segment_tail_trim_max_ms = tail_guard_config.short_segment_tail_trim_max_ms
    short_segment_tail_trim_max_ratio = tail_guard_config.short_segment_tail_trim_max_ratio

    enable_segment_routing = segment_routing_config.enabled
    short_segment_sec = segment_routing_config.short_segment_sec
    max_refs_per_segment = segment_routing_config.max_refs_per_segment
    min_segment_routing_sec = segment_routing_config.min_segment_sec
    min_segment_routing_words = segment_routing_config.min_segment_words
    routing_confidence_margin = segment_routing_config.confidence_margin

    target_dBFS = audio_level_config.target_dbfs
    reference_gain_offset_db = audio_level_config.reference_gain_offset_db
    max_segment_boost_db = audio_level_config.max_segment_boost_db
    max_segment_cut_db = audio_level_config.max_segment_cut_db
    enable_segment_matching = audio_level_config.enable_segment_matching
    segment_match_padding_ms = audio_level_config.segment_match_padding_ms
    segment_match_strength = audio_level_config.segment_match_strength
    segment_match_max_delta_db = audio_level_config.segment_match_max_delta_db
    segment_match_min_active_ratio = audio_level_config.segment_match_min_active_ratio

    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

    original_segment_count = len(segments)
    segments = _prepare_tts_segments(
        segments=segments,
        enable_grouping=enable_tts_grouping,
        max_gap_sec=tts_grouping_max_gap_sec,
        max_group_segments=tts_grouping_max_segments,
        max_group_chars=tts_grouping_max_chars,
        max_group_duration_sec=tts_grouping_max_duration_sec
    )
    if len(segments) != original_segment_count:
        logger.info(
            "TTS grouping: %s -> %s сегментов",
            original_segment_count,
            len(segments)
        )

    aux_models = load_tts_aux_models(tail_guard_config, smart_sync_config)
    babble_guard_model = aux_models.babble_guard_model
    asr_eval_model = aux_models.asr_eval_model
    tail_guard_asr_model = aux_models.tail_guard_asr_model
    smart_sync_backend = load_smart_sync_rewrite_backend(smart_sync_config)

    default_reference_paths = existing_reference_paths(speaker_wav)
    if not default_reference_paths:
        raise ValueError("Не найдено ни одного speaker reference для TTS.")

    conditioning_cache: Dict[Tuple[str, ...], Any] = {}
    logged_routes: set[Tuple[str, ...]] = set()

    def get_conditioning(reference_paths: List[str]) -> Any:
        key = tuple(reference_paths)
        cached = conditioning_cache.get(key)
        if cached is not None:
            return cached

        logger.info(
            "Подготавливаем conditioning для %s backend-а на %s reference clip(s)...",
            getattr(tts_backend, "name", "tts"),
            len(key)
        )
        conditioning_cache[key] = tts_backend.prepare_conditioning(
            reference_paths=list(key),
            speaker_profile=speaker_profile
        )
        return conditioning_cache[key]

    reference_path = reference_audio_path
    if reference_path is None:
        reference_path = default_reference_paths[0]
    reference_audio = AudioSegment.from_wav(reference_path)
    reference_active_dbfs = _active_speech_dbfs(reference_audio)
    source_vocals_audio = None
    if source_vocals_path and os.path.exists(source_vocals_path):
        source_vocals_audio = AudioSegment.from_wav(source_vocals_path).set_channels(1)
    target_active_dbfs = target_dBFS
    if reference_active_dbfs is not None:
        target_active_dbfs = reference_active_dbfs + reference_gain_offset_db
        logger.info(
            "Целевой уровень активной речи: %.2f dBFS (ref %.2f + %.2f)",
            target_active_dbfs,
            reference_active_dbfs,
            reference_gain_offset_db
        )
    else:
        logger.warning(
            "Не удалось оценить громкость референса, используем fallback %.2f dBFS",
            target_dBFS
        )

    full_duration_ms = int((max((s["end"] for s in segments), default=0.0) + 5) * 1000)
    timeline_placements: List[Tuple[int, AudioSegment]] = []

    # Сохраняем оригинальные временные метки
    for seg in segments:
        if "original_start" not in seg:
            seg["original_start"] = seg.get("start", 0.0)
        if "original_end" not in seg:
            seg["original_end"] = seg.get("end", seg.get("start", 0.0) + 1.0)

    prev_end_sec = 0.0

    for i, seg in tqdm(enumerate(segments), total=len(segments), desc="Синтез"):
        orig_start  = float(seg["original_start"])
        orig_end    = float(seg["original_end"])
        cur_start   = float(seg.get("start", orig_start))
        cur_start_ms = int(cur_start * 1000)
        next_segment_hint = segments[i + 1] if i < len(segments) - 1 else None
        next_text_hint = str(next_segment_hint.get("text") or "") if next_segment_hint else ""
        next_start_hint = None
        gap_after_hint_sec = None
        if next_segment_hint is not None:
            next_start_hint = float(
                next_segment_hint.get(
                    "start",
                    next_segment_hint.get("original_start", orig_end)
                )
            )
            gap_after_hint_sec = max(0.0, next_start_hint - orig_end)
        pause_after_hint = seg.get("pause_after_sec")
        if pause_after_hint is not None:
            try:
                pause_after_hint = float(pause_after_hint)
            except (TypeError, ValueError):
                pause_after_hint = None
        tts_group_size = int(seg.get("tts_group_size", 1) or 1)

        # Очистка текста
        original_clean = _clean_text(seg["text"])
        clean = original_clean
        seg["cleaned_text"] = clean
        seg["tts_text_was_stabilized"] = False
        if not clean:
            logger.warning(f"[{i}] Пустой сегмент после очистки — пропуск.")
            continue

        selected_references = default_reference_paths
        if enable_segment_routing:
            routed_references = _select_segment_references(
                segment=seg,
                speaker_wav=default_reference_paths,
                speaker_profile=speaker_profile,
                max_refs_per_segment=max_refs_per_segment,
                short_segment_sec=short_segment_sec,
                min_segment_sec=min_segment_routing_sec,
                min_segment_words=min_segment_routing_words,
                confidence_margin=routing_confidence_margin
            )
            if routed_references:
                selected_references = routed_references

        route_key = tuple(selected_references)
        if route_key not in logged_routes:
            logged_routes.add(route_key)
            logger.info(
                "TTS route [%s]: %s",
                clean[:60],
                ", ".join(os.path.basename(path) for path in route_key)
            )

        conditioning = get_conditioning(selected_references)
        seg["tts_reference_paths"] = list(selected_references)
        seg["tts_reference_count"] = len(selected_references)

        # Генерация аудио
        seg_path = os.path.join(segments_dir, f"seg_{int(seg['start'] * 1000)}.wav")
        seg_path, _, seg_audio = generate_audio_segment(
            tts_backend=tts_backend,
            text=clean,
            output_path=seg_path,
            speaker_wav=selected_references,
            language=language,
            conditioning=conditioning,
            speaker_profile=speaker_profile
        )

        generated_ms    = len(seg_audio)

        timing_stats = _timing_speech_stats(seg_audio)
        timing_duration_ms = max(1, timing_stats["effective_duration_ms"] or generated_ms)
        seg["timing_effective_duration_ms"] = timing_duration_ms
        seg["timing_leading_silence_ms"] = timing_stats["leading_silence_ms"]
        seg["timing_trailing_silence_ms"] = timing_stats["trailing_silence_ms"]

        timing_window = build_segment_timing_window(
            segments=segments,
            index=i,
            original_start_sec=orig_start,
            prev_end_sec=prev_end_sec,
            timing_duration_ms=timing_duration_ms,
            min_pause_between_segments=min_pause_between_segments,
            max_shift_left_seconds=max_shift_left_seconds,
            max_next_start_shift_sec=max_next_start_shift_sec,
        )
        window_start_val = timing_window.window_start_sec
        window_end_val = timing_window.window_end_sec
        cur_start = timing_window.cur_start_sec
        cur_start_ms = timing_window.cur_start_ms
        original_ms = timing_window.original_ms
        borrowed_before_ms = timing_window.borrowed_before_ms
        borrowed_after_ms = timing_window.borrowed_after_ms
        available_ms = timing_window.available_ms

        seg["start"] = cur_start
        seg["timing_borrow_before_ms"] = borrowed_before_ms
        seg["timing_borrow_after_ms"] = borrowed_after_ms
        seg["timing_window_start_sec"] = timing_window.timing_window_start_sec
        seg["timing_window_end_sec"] = timing_window.timing_window_end_sec
        seg["timing_window_ms"] = available_ms

        if borrowed_before_ms or borrowed_after_ms:
            logger.info(
                "[%s] Timing borrow: -%s мс / +%s мс -> окно %.2fs",
                i,
                borrowed_before_ms,
                borrowed_after_ms,
                available_ms / 1000.0
            )
        target_sync_ms = original_ms

        if (
            smart_sync_backend is not None
            and enable_smart_sync
            and clean
            and smart_sync_max_rewrites > 0
        ):
            rewrite_mode = None
            shorter_target_ms = max(
                available_ms,
                int(round(available_ms * smart_sync_target_speedup_factor)),
            )
            if (
                timing_duration_ms > shorter_target_ms
                and (
                    timing_duration_ms >= shorter_target_ms + smart_sync_min_improvement_ms
                    or (timing_duration_ms / max(shorter_target_ms, 1)) >= smart_sync_trigger_speed_factor
                )
            ):
                rewrite_mode = "shorter"
                target_sync_ms = shorter_target_ms
            elif (
                smart_sync_allow_lengthen
                and original_ms > 0
                and timing_duration_ms <= max(1, int(original_ms * smart_sync_min_fill_ratio))
                and (original_ms - timing_duration_ms) >= smart_sync_min_improvement_ms
            ):
                rewrite_mode = "longer"
                target_sync_ms = original_ms

            if rewrite_mode:
                best_audio = seg_audio
                best_text = clean
                best_timing_duration_ms = timing_duration_ms
                best_distance = _smart_sync_distance_ms(best_timing_duration_ms, target_sync_ms, rewrite_mode)
                initial_smart_sync_duration_ms = len(seg_audio)
                initial_smart_sync_timing_ms = best_timing_duration_ms
                accepted_rewrite = None
                baseline_smart_sync_eval = None

                context_pause_threshold_sec = max(0.5, min_pause_between_segments)
                previous_context_text = ""
                if i > 0:
                    prev_seg = segments[i - 1]
                    prev_gap_sec = max(
                        0.0,
                        window_start_val - float(prev_seg.get("end", prev_seg.get("start", 0.0)))
                    )
                    if prev_gap_sec <= context_pause_threshold_sec and _same_speaker(prev_seg, seg):
                        previous_context_text = str(
                            prev_seg.get("cleaned_text") or prev_seg.get("text") or ""
                        ).strip()

                next_context_text = ""
                if i < len(segments) - 1:
                    next_seg = segments[i + 1]
                    next_gap_sec = max(
                        0.0,
                        float(next_seg.get("start", next_seg.get("original_start", next_seg.get("start", 0.0))))
                        - window_end_val
                    )
                    if next_gap_sec <= context_pause_threshold_sec and _same_speaker(seg, next_seg):
                        next_context_text = str(
                            next_seg.get("text") or next_seg.get("cleaned_text") or ""
                        ).strip()

                if asr_eval_model is not None:
                    try:
                        baseline_recognized = _transcribe_short_audio(
                            asr_eval_model,
                            seg_audio,
                            language
                        )
                        baseline_smart_sync_eval = _segment_recognition_score(
                            clean,
                            baseline_recognized,
                            tts_babble_guard_anchor_words
                        )
                    except Exception as error:
                        logger.debug(
                            "[%s] SmartSync baseline ASR check skipped: %s",
                            i,
                            error
                        )

                for rewrite_idx in range(smart_sync_max_rewrites):
                    try:
                        rewritten_text, rewrite_info = smart_sync_rewrite_segment_text(
                            segment=seg,
                            backend=smart_sync_backend,
                            src_lang=smart_sync_src_lang,
                            tgt_lang=smart_sync_tgt_lang,
                            current_duration_sec=best_timing_duration_ms / 1000.0,
                            target_duration_sec=target_sync_ms / 1000.0,
                            available_duration_sec=available_ms / 1000.0,
                            rewrite_mode=rewrite_mode,
                            candidate_count=smart_sync_candidate_count,
                            previous_text=previous_context_text,
                            next_text=next_context_text,
                        )
                    except Exception as error:
                        logger.warning(
                            "[%s] SmartSync rewrite недоступен, fallback на обычный TTS: %s",
                            i,
                            error
                        )
                        smart_sync_backend = None
                        break

                    candidate_options = _smart_sync_candidate_options(rewritten_text, rewrite_info)
                    if not candidate_options:
                        break

                    attempted_candidate = False
                    for candidate_idx, (rewritten_clean, candidate_rewrite_info) in enumerate(
                        candidate_options,
                        start=1,
                    ):
                        if not rewritten_clean or rewritten_clean == best_text:
                            continue
                        attempted_candidate = True

                        _, _, rewritten_audio = generate_audio_segment(
                            tts_backend=tts_backend,
                            text=rewritten_clean,
                            output_path=None,
                            speaker_wav=selected_references,
                            language=language,
                            conditioning=conditioning,
                            speaker_profile=speaker_profile
                        )

                        rewritten_eval = None
                        if asr_eval_model is not None:
                            try:
                                rewritten_recognized = _transcribe_short_audio(
                                    asr_eval_model,
                                    rewritten_audio,
                                    language
                                )
                                rewritten_eval = _segment_recognition_score(
                                    rewritten_clean,
                                    rewritten_recognized,
                                    tts_babble_guard_anchor_words
                                )
                            except Exception as error:
                                logger.debug(
                                    "[%s] SmartSync rewritten ASR check skipped: %s",
                                    i,
                                    error
                                )

                        rewritten_timing_duration_ms = max(
                            1,
                            _timing_speech_stats(rewritten_audio)["effective_duration_ms"] or len(rewritten_audio)
                        )
                        new_distance = _smart_sync_distance_ms(
                            rewritten_timing_duration_ms,
                            target_sync_ms,
                            rewrite_mode
                        )
                        boundary_hit = (
                            rewrite_mode == "shorter"
                            and rewritten_timing_duration_ms <= target_sync_ms < best_timing_duration_ms
                        )
                        improved_enough = new_distance <= max(0, best_distance - smart_sync_min_improvement_ms)
                        accepted_by_gate, gate_metrics = _smart_sync_acceptance_gate(
                            source_text=clean,
                            rewritten_text=rewritten_clean,
                            rewrite_mode=rewrite_mode,
                            baseline_duration_ms=best_timing_duration_ms,
                            rewritten_duration_ms=rewritten_timing_duration_ms,
                            target_duration_ms=target_sync_ms,
                            baseline_eval=baseline_smart_sync_eval,
                            rewritten_eval=rewritten_eval,
                            min_fill_ratio=smart_sync_accept_min_fill_ratio,
                            min_text_similarity=smart_sync_accept_min_text_similarity,
                            min_word_ratio=smart_sync_accept_min_word_ratio,
                            min_token_precision=smart_sync_accept_min_token_precision,
                            min_asr_score=smart_sync_accept_min_asr_score,
                            max_asr_drop=smart_sync_accept_max_asr_drop,
                        )

                        if accepted_by_gate and (boundary_hit or improved_enough):
                            best_audio = rewritten_audio
                            best_text = rewritten_clean
                            best_timing_duration_ms = rewritten_timing_duration_ms
                            best_distance = new_distance
                            accepted_rewrite = candidate_rewrite_info
                            accepted_rewrite["attempt"] = rewrite_idx + 1
                            accepted_rewrite["candidate_attempt"] = candidate_idx
                            accepted_rewrite["accepted_duration_sec"] = round(len(rewritten_audio) / 1000.0, 3)
                            accepted_rewrite["accepted_timing_duration_sec"] = round(
                                rewritten_timing_duration_ms / 1000.0,
                                3
                            )
                            accepted_rewrite["target_speedup_factor"] = round(smart_sync_target_speedup_factor, 3)
                            accepted_rewrite["gate_metrics"] = gate_metrics
                            break

                        logger.info(
                            "[%s] SmartSync candidate %s/%s rejected (%s): fill=%.2f sim=%.2f word_ratio=%.2f precision=%.2f asr=%s",
                            i,
                            candidate_idx,
                            len(candidate_options),
                            gate_metrics.get("reject_reason") or "timing",
                            gate_metrics.get("fill_ratio", 0.0),
                            gate_metrics.get("text_similarity", 0.0),
                            gate_metrics.get("word_ratio", 0.0),
                            gate_metrics.get("token_precision", 0.0),
                            gate_metrics.get("rewritten_asr_score"),
                        )

                    if accepted_rewrite is not None or not attempted_candidate:
                        break

                if accepted_rewrite is not None:
                    seg_audio = best_audio
                    generated_ms = len(seg_audio)
                    timing_duration_ms = best_timing_duration_ms
                    seg_audio.export(seg_path, format="wav")
                    seg["smart_sync"] = accepted_rewrite
                    seg["smart_sync"]["distance_ms"] = best_distance
                    seg["smart_sync"]["initial_duration_sec"] = round(initial_smart_sync_duration_ms / 1000.0, 3)
                    seg["smart_sync"]["initial_timing_duration_sec"] = round(
                        initial_smart_sync_timing_ms / 1000.0,
                        3
                    )
                    seg["text"] = best_text
                    clean = best_text
                    logger.info(
                        "[%s] SmartSync %s: %.2fs -> %.2fs (timing %.2fs -> %.2fs) | %s",
                        i,
                        rewrite_mode,
                        initial_smart_sync_duration_ms / 1000.0,
                        generated_ms / 1000.0,
                        initial_smart_sync_timing_ms / 1000.0,
                        timing_duration_ms / 1000.0,
                        clean
                    )

        seg["cleaned_text"] = clean

        segment_target_dbfs = target_active_dbfs
        if enable_segment_matching:
            segment_target_dbfs, source_stats = _compute_segment_target_level(
                source_audio=source_vocals_audio,
                segment_start_sec=orig_start,
                segment_end_sec=orig_end,
                default_target_active_dbfs=target_active_dbfs,
                reference_gain_offset_db=reference_gain_offset_db,
                strength=segment_match_strength,
                max_delta_db=segment_match_max_delta_db,
                padding_ms=segment_match_padding_ms,
                min_active_ratio=segment_match_min_active_ratio
            )
            seg["source_active_dbfs"] = source_stats["active_dbfs"]
            seg["source_active_ratio"] = source_stats["active_ratio"]
            seg["segment_target_dbfs"] = segment_target_dbfs

        retry_candidate = (
            enable_tts_asr_retry
            and asr_eval_model is not None
            and int(seg.get("tts_group_size", 1) or 1) == 1
            and max(0.0, window_end_val - window_start_val) <= tts_asr_retry_max_segment_sec
            and _ends_with_terminal_punctuation(clean)
        )

        def finalize_candidate(candidate_audio: AudioSegment) -> tuple[AudioSegment, Dict[str, Any] | None, Dict[str, Any] | None, Dict[str, Any] | None, Dict[str, Any]]:
            corrected_candidate, timing_fit_info = _fit_audio_to_timing_window(
                candidate_audio,
                available_ms=available_ms,
                max_speedup_factor=max_speedup_factor,
                speedup_tail_padding_ms=speedup_tail_padding_ms,
                trim_edge_silence=speedup_trim_edge_silence,
                trim_keep_edge_ms=speedup_trim_keep_edge_ms,
                trim_min_edge_ms=speedup_trim_min_edge_ms,
            )
            if enable_short_segment_tail_trim:
                edge_silence_candidate = _edge_silence_ms(corrected_candidate)
                tail_trim_ms = _compute_safe_tail_trim_ms(
                    original_ms=available_ms,
                    corrected_ms=len(corrected_candidate),
                    trailing_silence_ms=edge_silence_candidate["trailing"],
                    short_segment_sec=short_segment_sec,
                    min_overhang_ms=short_segment_tail_trim_min_overhang_ms,
                    max_trim_ms=short_segment_tail_trim_max_ms,
                    max_trim_ratio=short_segment_tail_trim_max_ratio
                )
                if tail_trim_ms:
                    corrected_candidate = corrected_candidate[:-tail_trim_ms]

            corrected_level_stats = _active_speech_stats(corrected_candidate)
            corrected_candidate = _match_segment_level(
                corrected_candidate,
                target_active_dbfs=segment_target_dbfs,
                max_boost_db=max_segment_boost_db,
                max_cut_db=max_segment_cut_db,
                speech_stats=corrected_level_stats,
            )
            cheap_tail_info = None
            if enable_tts_cheap_tail_guard:
                corrected_candidate, cheap_tail_info = _trim_trailing_speech_island_fast(
                    audio=corrected_candidate,
                    expected_text=clean,
                    original_ms=available_ms,
                    segment_duration_sec=available_ms / 1000.0,
                    tts_group_size=int(seg.get("tts_group_size", 1) or 1),
                    max_segment_sec=tts_cheap_tail_guard_max_segment_sec,
                    min_overhang_ms=tts_cheap_tail_guard_min_overhang_ms,
                    min_gap_ms=tts_cheap_tail_guard_min_gap_ms,
                    min_island_ms=tts_cheap_tail_guard_min_island_ms,
                    max_island_ms=tts_cheap_tail_guard_max_island_ms,
                    search_window_ms=tts_cheap_tail_guard_search_window_ms,
                    max_trim_ms=tts_cheap_tail_guard_max_trim_ms
                )
            corrected_candidate, candidate_babble_info = _trim_trailing_babble(
                audio=corrected_candidate,
                expected_text=clean,
                model_asr=tail_guard_asr_model,
                language=language,
                segment_duration_sec=available_ms / 1000.0,
                tts_group_size=int(seg.get("tts_group_size", 1) or 1),
                max_segment_sec=tts_babble_guard_max_segment_sec,
                min_gap_ms=tts_babble_guard_min_gap_ms,
                min_island_ms=tts_babble_guard_min_island_ms,
                max_island_ms=tts_babble_guard_max_island_ms,
                search_window_ms=tts_babble_guard_search_window_ms,
                max_trim_ms=tts_babble_guard_max_trim_ms,
                anchor_words=tts_babble_guard_anchor_words,
                min_score_gain=tts_babble_guard_min_score_gain,
            )

            candidate_eval = None
            if retry_candidate:
                recognized_candidate = _transcribe_short_audio(
                    asr_eval_model,
                    corrected_candidate,
                    language
                )
                candidate_eval = _segment_recognition_score(
                    clean,
                    recognized_candidate,
                    tts_babble_guard_anchor_words
                    )
            return corrected_candidate, cheap_tail_info, candidate_babble_info, candidate_eval, timing_fit_info

        corrected, cheap_tail_info, babble_guard_info, asr_eval, timing_fit_info = finalize_candidate(seg_audio)

        retry_needed = (
            retry_candidate
            and asr_eval is not None
            and (
                asr_eval["score"] < tts_asr_retry_min_score
                or asr_eval.get("has_extra_tail")
                or not asr_eval.get("has_suffix", True)
            )
        )

        if retry_needed and asr_eval:
            retry_reason = []
            if asr_eval["score"] < tts_asr_retry_min_score:
                retry_reason.append(f"score<{tts_asr_retry_min_score:.2f}")
            if asr_eval.get("has_extra_tail"):
                retry_reason.append("extra_tail")
            if not asr_eval.get("has_suffix", True):
                retry_reason.append("missing_suffix")
            logger.info(
                "[%s] TTS retry armed (%s) | %.3f | %s",
                i,
                ",".join(retry_reason) or "quality",
                asr_eval["score"],
                asr_eval["recognized_text"]
            )

            retry_text_variants = _build_tts_retry_text_variants(
                seg["text"],
                original_text=str(seg.get("original_text") or ""),
                next_text=next_text_hint,
                gap_after_sec=gap_after_hint_sec,
                pause_after_sec=pause_after_hint,
                tts_group_size=tts_group_size,
                tts_backend_name=getattr(tts_backend, "name", ""),
            )
            best_retry_audio, best_retry_text, _ = _synthesize_best_retry_take(
                retry_text_variants=retry_text_variants,
                fallback_text=clean,
                attempts=tts_asr_retry_attempts,
                synthesize_fn=lambda text: generate_audio_segment(
                    tts_backend=tts_backend,
                    text=text,
                    output_path=None,
                    speaker_wav=selected_references,
                    language=language,
                    conditioning=conditioning,
                    speaker_profile=speaker_profile
                )[2],
                transcribe_fn=lambda audio: _transcribe_short_audio(
                    asr_eval_model,
                    audio,
                    language
                ),
                expected_text=clean,
                anchor_words=tts_babble_guard_anchor_words,
            )

            if best_retry_audio is not None:
                retry_corrected, retry_cheap_tail_info, retry_babble_info, retry_eval, retry_timing_fit_info = finalize_candidate(best_retry_audio)
                if _is_better_recognition_eval(retry_eval, asr_eval):
                    corrected = retry_corrected
                    cheap_tail_info = retry_cheap_tail_info
                    babble_guard_info = retry_babble_info
                    asr_eval = retry_eval
                    timing_fit_info = retry_timing_fit_info
                    seg["tts_retry_text_used"] = best_retry_text
                    logger.info(
                        "[%s] TTS retry improved ASR score to %.3f | %s",
                        i,
                        asr_eval["score"],
                        asr_eval["recognized_text"]
                    )

        if timing_fit_info:
            seg["timing_fit"] = timing_fit_info
            seg["required_speedup_factor"] = timing_fit_info["required_speedup_factor"]
            seg["applied_speedup_factor"] = timing_fit_info["applied_speedup_factor"]
            seg["speedup_capped"] = timing_fit_info["speedup_capped"]
            seg["residual_timing_ratio"] = timing_fit_info["residual_timing_ratio"]
            if (
                timing_fit_info["applied_speedup_factor"] > 1.001
                or timing_fit_info["pre_trim_leading_ms"]
                or timing_fit_info["pre_trim_trailing_ms"]
                or timing_fit_info["post_trim_leading_ms"]
                or timing_fit_info["post_trim_trailing_ms"]
            ):
                logger.info(
                    (
                        "[%s] Timing fit: required %.2fx, applied %.2fx%s, "
                        "trim pre %s/%s ms, post %s/%s ms, residual %.2f"
                    ),
                    i,
                    timing_fit_info["required_speedup_factor"],
                    timing_fit_info["applied_speedup_factor"],
                    " capped" if timing_fit_info["speedup_capped"] else "",
                    timing_fit_info["pre_trim_leading_ms"],
                    timing_fit_info["pre_trim_trailing_ms"],
                    timing_fit_info["post_trim_leading_ms"],
                    timing_fit_info["post_trim_trailing_ms"],
                    timing_fit_info["residual_timing_ratio"],
                )

        if cheap_tail_info:
            seg["cheap_tail_guard_trim_ms"] = cheap_tail_info["trim_ms"]
            seg["cheap_tail_guard_overhang_ms"] = cheap_tail_info["overhang_ms"]
            logger.info(
                "[%s] Cheap tail guard trim %s мс (overhang %s мс)",
                i,
                cheap_tail_info["trim_ms"],
                cheap_tail_info["overhang_ms"]
            )
        if babble_guard_info:
            seg["babble_guard_trim_ms"] = babble_guard_info["trim_ms"]
            seg["babble_guard_before"] = babble_guard_info["recognized_before"]
            seg["babble_guard_after"] = babble_guard_info["recognized_after"]
            logger.info(
                "[%s] TTS babble guard trim %s мс | до: %s | после: %s",
                i,
                babble_guard_info["trim_ms"],
                babble_guard_info["recognized_before"],
                babble_guard_info["recognized_after"]
            )
        if asr_eval:
            seg["tts_asr_score"] = asr_eval["score"]
            seg["tts_asr_similarity"] = asr_eval["similarity"]
            seg["tts_asr_recognized"] = asr_eval["recognized_text"]
        corrected_ms = len(corrected)
        edge_silence = _edge_silence_ms(corrected)
        fade_in_ms = min(fade_in_out_ms, edge_silence["leading"])
        fade_out_ms = min(fade_in_out_ms, edge_silence["trailing"])
        if fade_in_ms > 0:
            corrected = corrected.fade_in(fade_in_ms)
        if fade_out_ms > 0:
            corrected = corrected.fade_out(fade_out_ms)
        seg["edge_leading_silence_ms"] = edge_silence["leading"]
        seg["edge_trailing_silence_ms"] = edge_silence["trailing"]
        seg["fade_in_ms"] = fade_in_ms
        seg["fade_out_ms"] = fade_out_ms
        seg["corrected_audio"] = corrected

        # Вставляем в итоговое аудио
        timeline_placements.append((cur_start_ms, corrected))

        actual_end = cur_start + corrected_ms / 1000.0
        seg["corrected_duration_sec"] = corrected_ms / 1000.0
        prev_end_sec = actual_end

        update_next_segment_start(
            segments=segments,
            index=i,
            actual_end_sec=actual_end,
            max_next_start_shift_sec=max_next_start_shift_sec,
        )

    # Кроссфейды между соседними сегментами
    timeline_replacements: List[Tuple[int, AudioSegment]] = []
    for i in range(len(segments) - 1):
        curr, nxt = segments[i], segments[i + 1]
        if "corrected_audio" not in curr or "corrected_audio" not in nxt:
            continue
        curr_ms   = int(curr["start"] * 1000)
        nxt_ms    = int(nxt["start"]  * 1000)
        overlap   = curr_ms + len(curr["corrected_audio"]) - nxt_ms
        if 0 < overlap < crossfade_ms:
            merged = curr["corrected_audio"].append(nxt["corrected_audio"], crossfade=overlap)
            timeline_replacements.append((curr_ms, merged))

    full_audio = _assemble_timeline(
        placements=timeline_placements,
        replacements=timeline_replacements,
        min_duration_ms=full_duration_ms,
    )
    full_audio = apply_final_audio_processing(full_audio, audio_level_config)

    full_audio.export(output_audio_path, format="wav")
    logger.info(f"Финальное аудио сохранено: {output_audio_path}")
    return _serialize_tts_segments(segments)
