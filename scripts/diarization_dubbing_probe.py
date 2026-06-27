from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydub import AudioSegment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from scripts.diarization_probe import _build_paths, _require_file, _resolve_project_path


class DiarizationDubbingProbeError(RuntimeError):
    pass


def _load_json(path: str | Path, label: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise DiarizationDubbingProbeError(f"Cannot read {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DiarizationDubbingProbeError(f"Invalid JSON in {label}: {path}") from exc


def _write_json(path: str | Path, payload: Any) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _cleanup_json_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    if "</think>" in cleaned.lower():
        cleaned = re.split(r"</think>", cleaned, maxsplit=1, flags=re.IGNORECASE)[-1].strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]
    return cleaned


def _parse_llm_translation_response(text: str, expected_ids: list[int]) -> list[str]:
    cleaned = _cleanup_json_text(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise DiarizationDubbingProbeError(f"LLM returned invalid JSON: {text}") from exc

    if isinstance(payload, dict):
        payload = payload.get("translations") or payload.get("batch") or payload.get("items")
    if not isinstance(payload, list):
        raise DiarizationDubbingProbeError(
            "LLM translation response must be a JSON array, or an object with translations/batch/items."
        )
    if len(payload) != len(expected_ids):
        raise DiarizationDubbingProbeError(
            f"LLM translation count mismatch: got {len(payload)}, expected {len(expected_ids)}."
        )

    translations: list[str] = []
    for expected_id, item in zip(expected_ids, payload):
        if not isinstance(item, dict):
            raise DiarizationDubbingProbeError("Each LLM translation item must be a JSON object.")
        if int(item.get("id", -1)) != expected_id:
            raise DiarizationDubbingProbeError(
                f"LLM translation id mismatch: got {item.get('id')}, expected {expected_id}."
            )
        text_value = item.get("translation")
        if not isinstance(text_value, str) or not text_value.strip():
            raise DiarizationDubbingProbeError(f"Empty translation for id={expected_id}.")
        translations.append(re.sub(r"\s+", " ", text_value).strip())
    return translations


def _load_segment_list(path: str | Path, label: str) -> list[dict[str, Any]]:
    data = _load_json(path, label)
    if not isinstance(data, list) or not data:
        raise DiarizationDubbingProbeError(f"{label} must be a non-empty JSON list: {path}")
    if not all(isinstance(item, dict) for item in data):
        raise DiarizationDubbingProbeError(f"{label} must contain JSON objects: {path}")
    return data


def _probe_dir(paths: dict[str, str], args: argparse.Namespace) -> Path:
    if args.probe_dir:
        return _resolve_project_path(args.probe_dir)
    return Path(paths["temp"]) / "diarization_probe"


def _default_probe_paths(probe_dir: Path) -> dict[str, Path]:
    return {
        "segments_diarized": probe_dir / "segments_diarized.json",
        "translated_segments_diarized": probe_dir / "translated_segments_diarized.json",
        "tts_segments_diarized": probe_dir / "tts_segments_diarized.json",
        "tts_summary": probe_dir / "tts_summary.json",
        "layers_dir": probe_dir / "tts_layers",
        "final_dubbing": probe_dir / "final_dubbing_diarized_probe.wav",
        "final_mix": probe_dir / "final_mix_diarized_probe.wav",
        "final_video": probe_dir / "final_video_diarized_probe.mp4",
        "speaker_profiles_dir": probe_dir / "speaker_profiles",
    }


def _group_segments_by_speaker(
    segments: list[dict[str, Any]],
    *,
    default_speaker_id: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        speaker_id = str(segment.get("speaker_id") or default_speaker_id)
        grouped[speaker_id].append(deepcopy(segment))
    return dict(sorted(grouped.items()))


def _language_name(lang_code: str) -> str:
    prefix = (lang_code or "").split("_", 1)[0].lower()
    return {
        "eng": "English",
        "en": "English",
        "rus": "Russian",
        "ru": "Russian",
    }.get(prefix, lang_code or "target language")


def _speaker_context_line(segment: dict[str, Any]) -> str:
    text = str(segment.get("text") or segment.get("original_text") or "").strip()
    return (
        f"[{segment.get('speaker_id', 'spk_0')} "
        f"{float(segment.get('start', 0.0)):.2f}-{float(segment.get('end', 0.0)):.2f}] "
        f"{text}"
    )


def _segment_duration_sec(segment: dict[str, Any]) -> float:
    source_duration = segment.get("source_duration_sec")
    if source_duration is not None:
        try:
            duration = float(source_duration)
            if duration > 0:
                return duration
        except (TypeError, ValueError):
            pass

    try:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, end - start)


def _normalized_char_count(text: str) -> int:
    return len(re.sub(r"\s+", " ", str(text or "").strip()))


def _build_llm_timing_constraints(
    batch: list[dict[str, Any]],
    *,
    target_chars_per_sec: float,
    min_chars: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    for index, segment in enumerate(batch):
        duration_sec = _segment_duration_sec(segment)
        raw_limit = int(duration_sec * target_chars_per_sec)
        char_limit = max(1, min(max_chars, max(min_chars, raw_limit)))
        constraints.append(
            {
                "id": index,
                "duration_sec": round(duration_sec, 3),
                "max_chars": char_limit,
                "target_chars_per_sec": target_chars_per_sec,
            }
        )
    return constraints


def _translation_constraint_violations(
    translations: list[str],
    constraints: list[dict[str, Any]] | None,
    *,
    tolerance_ratio: float,
) -> list[dict[str, Any]]:
    if not constraints:
        return []

    violations: list[dict[str, Any]] = []
    for text, constraint in zip(translations, constraints):
        actual_chars = _normalized_char_count(text)
        max_chars = int(constraint["max_chars"])
        tolerated_chars = max_chars * max(1.0, tolerance_ratio)
        if actual_chars > tolerated_chars:
            violations.append(
                {
                    "id": int(constraint["id"]),
                    "max_chars": max_chars,
                    "tolerated_chars": int(tolerated_chars),
                    "actual_chars": actual_chars,
                    "translation": text,
                }
            )
    return violations


def _build_llm_repair_messages(
    *,
    original_messages: list[dict[str, str]],
    previous_response: str,
    violations: list[dict[str, Any]],
) -> list[dict[str, str]]:
    repair_payload = {
        "problem": (
            "Some translations exceeded their max_chars timing contract. "
            "Rewrite the full JSON array for the same batch and ids, making the listed items shorter."
        ),
        "violations": violations,
        "rules": [
            "Return all items from the batch, not only the violations.",
            "Keep each translation under max_chars whenever possible.",
            "If the full meaning cannot fit, preserve the core intent and omit secondary detail.",
            "Return only valid JSON array with id and translation.",
        ],
    }
    return [
        *original_messages,
        {"role": "assistant", "content": previous_response},
        {"role": "user", "content": json.dumps(repair_payload, ensure_ascii=False, indent=2)},
    ]


def _attach_timing_constraint_metadata(
    segment: dict[str, Any],
    translation: str,
    constraint: dict[str, Any] | None,
    *,
    tolerance_ratio: float,
) -> None:
    if not constraint:
        return

    actual_chars = _normalized_char_count(translation)
    max_chars = int(constraint["max_chars"])
    tolerated_chars = int(max_chars * max(1.0, tolerance_ratio))
    segment["llm_timing_constraint"] = {
        "enabled": True,
        "duration_sec": constraint["duration_sec"],
        "target_chars_per_sec": constraint["target_chars_per_sec"],
        "max_chars": max_chars,
        "tolerated_chars": tolerated_chars,
        "actual_chars": actual_chars,
        "ratio": round(actual_chars / max_chars, 3) if max_chars else None,
        "within_limit": actual_chars <= max_chars,
        "within_tolerance": actual_chars <= tolerated_chars,
    }


def _summarize_llm_timing_constraints(
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    constraints = [
        segment.get("llm_timing_constraint")
        for segment in segments
        if isinstance(segment.get("llm_timing_constraint"), dict)
    ]
    if not constraints:
        return {"enabled": False}

    ratios = [
        float(item["ratio"])
        for item in constraints
        if item.get("ratio") is not None
    ]
    return {
        "enabled": True,
        "segment_count": len(constraints),
        "over_limit_count": sum(1 for item in constraints if not item.get("within_limit", True)),
        "over_tolerance_count": sum(
            1 for item in constraints if not item.get("within_tolerance", True)
        ),
        "avg_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
        "max_ratio": round(max(ratios), 4) if ratios else None,
    }


def _build_llm_translation_messages(
    *,
    batch: list[dict[str, Any]],
    previous_context: list[dict[str, Any]],
    next_context: list[dict[str, Any]],
    src_lang: str,
    tgt_lang: str,
    timing_constraints: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    src_name = _language_name(src_lang)
    tgt_name = _language_name(tgt_lang)
    system = (
        "You are a senior audiovisual translator adapting a two-speaker TED-style dialogue "
        f"from {src_name} into natural spoken {tgt_name} for dubbing. "
        "Translate meaning, intent, and tone, not word-for-word syntax. "
        "Keep each line concise enough for the original timing. "
        "Preserve numbers, names, places, and technical terms. "
        "Use idiomatic spoken Russian when the target language is Russian. "
        "Do not add explanations, analysis, markdown, chain-of-thought, or <think> blocks. "
        "Return only valid JSON."
    )

    constraints_by_id = {
        int(constraint["id"]): constraint
        for constraint in timing_constraints or []
    }
    payload = {
        "task": (
            "Translate only the items in `batch`. Use context only to resolve references, "
            "speaker continuity, pronouns, and tone."
        ),
        "return_only_this_json_array_shape": [
            {"id": 0, "translation": "translated line"}
        ],
        "previous_context": [_speaker_context_line(segment) for segment in previous_context],
        "batch": [
            {
                "id": index,
                "speaker_id": segment.get("speaker_id", "spk_0"),
                "start": round(float(segment.get("start", 0.0)), 3),
                "end": round(float(segment.get("end", 0.0)), 3),
                **(
                    {
                        "duration_sec": constraints_by_id[index]["duration_sec"],
                        "max_chars": constraints_by_id[index]["max_chars"],
                    }
                    if index in constraints_by_id
                    else {}
                ),
                "text": str(segment.get("text") or "").strip(),
            }
            for index, segment in enumerate(batch)
        ],
        "next_context": [_speaker_context_line(segment) for segment in next_context],
        "style_rules": [
            "Natural spoken dubbing, not written lecture prose.",
            "Return only the JSON array, not this full request object.",
            "Keep the same number of JSON items and the same ids.",
            "Do not merge, split, reorder, or omit lines.",
            "Do not output analysis, reasoning text, markdown, or <think> blocks.",
            "Avoid literal calques and stiff bureaucratic wording.",
            "Prefer shorter Russian phrasing when possible.",
        ],
    }
    if timing_constraints:
        payload["timing_contract"] = {
            "enabled": True,
            "max_chars_meaning": (
                "Hard ceiling for each translation, including spaces and punctuation. "
                "For very short windows, fragmentary spoken Russian is acceptable."
            ),
            "compression_priority": [
                "preserve core intent",
                "preserve speaker tone",
                "drop filler and secondary detail",
                "avoid explanatory additions",
            ],
        }
        payload["style_rules"].extend(
            [
                "Treat max_chars as a timing contract, not a suggestion.",
                "For segments under one second, use a very short utterance or fragment.",
                "Do not exceed max_chars unless the result would become unintelligible.",
            ]
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def _call_openai_compatible_chat(
    *,
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise DiarizationDubbingProbeError("OpenAI package is required for LLM translation.") from exc

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not content:
        raise DiarizationDubbingProbeError("LLM returned an empty translation response.")
    return content


def _translate_segments_with_llm_dialogue(
    *,
    segments: list[dict[str, Any]],
    provider: str,
    model: str,
    api_key_env: str,
    base_url: str,
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    context_segments: int,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    min_interval_sec: float,
    max_retries: int,
    timing_constraints_enabled: bool = False,
    target_chars_per_sec: float = 16.0,
    min_chars: int = 8,
    max_chars: int = 170,
    constraint_tolerance_ratio: float = 1.05,
    constraint_repair_retries: int = 1,
    constraint_strict: bool = False,
) -> list[dict[str, Any]]:
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise DiarizationDubbingProbeError(f"Missing API key env var for LLM translation: {api_key_env}")

    provider = provider.strip().lower()
    if not base_url:
        base_url = {
            "groq": "https://api.groq.com/openai/v1",
            "openai": "https://api.openai.com/v1",
        }.get(provider, "")
    if not base_url:
        raise DiarizationDubbingProbeError("Set --llm-base-url for this LLM provider.")

    translated: list[dict[str, Any]] = []
    last_request_ts = 0.0

    for start in range(0, len(segments), batch_size):
        batch = segments[start:start + batch_size]
        previous_context = segments[max(0, start - context_segments):start]
        next_context = segments[start + len(batch):start + len(batch) + context_segments]
        timing_constraints = (
            _build_llm_timing_constraints(
                batch,
                target_chars_per_sec=target_chars_per_sec,
                min_chars=min_chars,
                max_chars=max_chars,
            )
            if timing_constraints_enabled
            else None
        )
        messages = _build_llm_translation_messages(
            batch=batch,
            previous_context=previous_context,
            next_context=next_context,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            timing_constraints=timing_constraints,
        )

        wait_sec = min_interval_sec - (time.monotonic() - last_request_ts)
        if wait_sec > 0:
            time.sleep(wait_sec)

        expected_ids = list(range(len(batch)))
        last_error: Exception | None = None
        repair_attempts = 0
        messages_for_attempt = messages
        for attempt in range(max_retries + 1):
            try:
                content = _call_openai_compatible_chat(
                    messages=messages_for_attempt,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_sec=timeout_sec,
                )
                last_request_ts = time.monotonic()
                translations = _parse_llm_translation_response(content, expected_ids)
                violations = _translation_constraint_violations(
                    translations,
                    timing_constraints,
                    tolerance_ratio=constraint_tolerance_ratio,
                )
                if violations:
                    if repair_attempts < constraint_repair_retries and attempt < max_retries:
                        repair_attempts += 1
                        messages_for_attempt = _build_llm_repair_messages(
                            original_messages=messages,
                            previous_response=content,
                            violations=violations,
                        )
                        continue
                    if constraint_strict:
                        raise DiarizationDubbingProbeError(
                            "LLM timing constraints exceeded after repair attempts: "
                            f"{violations}"
                        )
                break
            except Exception as exc:
                last_error = exc
                if attempt >= max_retries:
                    raise
                time.sleep(max(min_interval_sec, min(30.0, 2 ** attempt)))
        else:
            raise DiarizationDubbingProbeError(f"LLM translation failed: {last_error}")

        for index, (segment, text) in enumerate(zip(batch, translations)):
            item = deepcopy(segment)
            item["original_text"] = str(segment.get("text") or "").strip()
            item["text"] = text
            item["translation_backend"] = (
                "llm_dialogue_timing_constrained"
                if timing_constraints_enabled
                else "llm_dialogue"
            )
            item["translation_model"] = model
            _attach_timing_constraint_metadata(
                item,
                text,
                timing_constraints[index] if timing_constraints else None,
                tolerance_ratio=constraint_tolerance_ratio,
            )
            translated.append(item)

    return translated


def _speaker_profile_path(profiles_dir: Path, speaker_id: str) -> Path:
    return profiles_dir / speaker_id / "speaker_profile.json"


def _reference_paths_from_profile(profile: dict[str, Any]) -> list[str]:
    paths = [
        str(clip["path"])
        for clip in profile.get("clips", [])
        if isinstance(clip, dict) and clip.get("path") and Path(str(clip["path"])).is_file()
    ]
    if paths:
        return paths

    merged = profile.get("merged_reference_path")
    if merged and Path(str(merged)).is_file():
        return [str(merged)]
    return []


def _load_speaker_profile(
    profiles_dir: Path,
    speaker_id: str,
) -> tuple[dict[str, Any], list[str]]:
    profile_path = _speaker_profile_path(profiles_dir, speaker_id)
    _require_file(profile_path, f"{speaker_id} speaker_profile.json")
    profile = _load_json(profile_path, f"{speaker_id} speaker_profile.json")
    if not isinstance(profile, dict):
        raise DiarizationDubbingProbeError(f"Speaker profile must be an object: {profile_path}")

    reference_paths = _reference_paths_from_profile(profile)
    if not reference_paths:
        raise DiarizationDubbingProbeError(f"No reference clips found for {speaker_id}: {profile_path}")
    return profile, reference_paths


def _mix_audio_layers(layer_paths: list[Path], output_path: Path) -> AudioSegment:
    if not layer_paths:
        raise DiarizationDubbingProbeError("No speaker audio layers to mix.")

    layers = [AudioSegment.from_wav(path) for path in layer_paths]
    duration_ms = max(len(layer) for layer in layers)
    mixed = AudioSegment.silent(duration=duration_ms)
    for layer in layers:
        mixed = mixed.overlay(layer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mixed.export(output_path, format="wav")
    return mixed


def run_translate_probe(args: argparse.Namespace) -> dict[str, Any]:
    paths = _build_paths(args)
    probe_dir = _probe_dir(paths, args)
    probe_paths = _default_probe_paths(probe_dir)
    source_segments_path = Path(args.segments_diarized or probe_paths["segments_diarized"])
    output_path = Path(args.translated_segments or probe_paths["translated_segments_diarized"])

    segments = _load_segment_list(source_segments_path, "segments_diarized.json")
    src_lang = args.src_lang or getattr(cfg, "MT_SRC_LANG", "eng_Latn")
    tgt_lang = args.tgt_lang or getattr(cfg, "MT_TGT_LANG", "rus_Cyrl")

    if args.translation_mode == "llm-dialogue":
        model_name = args.llm_model or getattr(cfg, "SMART_SYNC_MODEL_NAME", "llama-3.3-70b-versatile")
        translated = _translate_segments_with_llm_dialogue(
            segments=segments,
            provider=args.llm_provider,
            model=model_name,
            api_key_env=args.llm_api_key_env,
            base_url=args.llm_base_url,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            batch_size=args.llm_batch_size,
            context_segments=args.llm_context_segments,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
            timeout_sec=args.llm_timeout_sec,
            min_interval_sec=args.llm_min_interval_sec,
            max_retries=args.llm_max_retries,
            timing_constraints_enabled=args.llm_timing_constraints,
            target_chars_per_sec=args.llm_target_chars_per_sec,
            min_chars=args.llm_min_chars,
            max_chars=args.llm_max_chars,
            constraint_tolerance_ratio=args.llm_constraint_tolerance_ratio,
            constraint_repair_retries=args.llm_constraint_repair_retries,
            constraint_strict=args.llm_constraint_strict,
        )
    else:
        from src.translation import (
            load_translation_model,
            normalize_translated_segments,
            translate_segments,
        )

        model_name = args.mt_model or getattr(cfg, "MT_MODEL_NAME", "facebook/nllb-200-distilled-1.3B")
        model, tokenizer = load_translation_model(model_name, getattr(cfg, "DEVICE", "cpu"))

        translated = translate_segments(
            model=model,
            tokenizer=tokenizer,
            segments=segments,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            batch_size=args.batch_size or getattr(cfg, "MT_BATCH_SIZE", 8),
            max_length=args.max_length or getattr(cfg, "MT_MAX_LENGTH", 1024),
        )
        translated = normalize_translated_segments(translated)

    _write_json(output_path, translated)

    result = {
        "stage": "translate",
        "translation_mode": args.translation_mode,
        "source_segments": str(source_segments_path),
        "translated_segments": str(output_path),
        "segment_count": len(translated),
        "model": model_name,
    }
    if args.translation_mode == "llm-dialogue":
        result["llm_timing_constraints"] = _summarize_llm_timing_constraints(translated)
    return result


def run_tts_probe(args: argparse.Namespace) -> dict[str, Any]:
    from src.tts import synthesize_segments_with_timing
    from src.tts_audio import apply_final_audio_processing
    from src.tts_backends import create_tts_backend
    from src.tts_config import (
        build_audio_level_config,
        build_segment_routing_config,
        build_smart_sync_config,
        build_tail_guard_config,
        build_tts_runtime_config,
    )

    paths = _build_paths(args)
    probe_dir = _probe_dir(paths, args)
    probe_paths = _default_probe_paths(probe_dir)
    translated_path = Path(args.translated_segments or probe_paths["translated_segments_diarized"])
    profiles_dir = Path(args.speaker_profiles_dir or probe_paths["speaker_profiles_dir"])
    layers_dir = Path(args.layers_dir or probe_paths["layers_dir"])
    output_path = Path(args.output_audio or probe_paths["final_dubbing"])
    tts_segments_path = Path(args.tts_segments_output or probe_paths["tts_segments_diarized"])
    summary_path = Path(args.tts_summary or probe_paths["tts_summary"])

    translated = _load_segment_list(translated_path, "translated_segments_diarized.json")
    grouped = _group_segments_by_speaker(
        translated,
        default_speaker_id=getattr(cfg, "DEFAULT_SPEAKER_ID", "spk_0"),
    )
    if len(grouped) < 2 and not args.allow_single_speaker:
        raise DiarizationDubbingProbeError(
            f"Expected at least two speakers, got {len(grouped)}. "
            "Pass --allow-single-speaker to force a one-speaker probe."
        )

    tts_backend = create_tts_backend(
        device=getattr(cfg, "DEVICE", "cpu"),
        xtts_model_dir=getattr(cfg, "MODEL_TTS_DIR", "./original_tts_model"),
        temperature=getattr(cfg, "XTTS_TEMPERATURE", 0.55),
        length_penalty=getattr(cfg, "XTTS_LENGTH_PENALTY", 1.0),
        repetition_penalty=getattr(cfg, "XTTS_REPETITION_PENALTY", 2.35),
        top_k=getattr(cfg, "XTTS_TOP_K", 50),
        top_p=getattr(cfg, "XTTS_TOP_P", 0.82),
    )

    runtime_config = build_tts_runtime_config(
        cfg,
        enable_grouping=not args.disable_grouping and getattr(cfg, "TTS_GROUPING_ENABLED", True),
    )
    smart_sync_config = build_smart_sync_config(cfg, enabled=args.enable_smart_sync)
    tail_guard_config = build_tail_guard_config(
        cfg,
        enable_babble_guard=args.enable_babble_guard,
        enable_asr_retry=args.enable_asr_retry,
    )
    segment_routing_config = build_segment_routing_config(
        cfg,
        enabled=not args.disable_segment_routing and getattr(cfg, "SEGMENT_ROUTING_ENABLED", True),
    )
    audio_level_config = build_audio_level_config(cfg)

    layer_paths: list[Path] = []
    combined_segments: list[dict[str, Any]] = []
    speaker_summary: dict[str, Any] = {}

    for speaker_id, speaker_segments in grouped.items():
        profile, reference_paths = _load_speaker_profile(profiles_dir, speaker_id)
        speaker_dir = layers_dir / speaker_id
        layer_path = speaker_dir / f"{speaker_id}.wav"
        speaker_dir.mkdir(parents=True, exist_ok=True)

        reference_audio_path = profile.get("merged_reference_path") or reference_paths[0]
        tts_segments = synthesize_segments_with_timing(
            tts_backend=tts_backend,
            segments=speaker_segments,
            output_audio_path=str(layer_path),
            speaker_wav=reference_paths,
            speaker_profile=profile,
            reference_audio_path=str(reference_audio_path),
            source_vocals_path=paths.get("vocals") if Path(paths.get("vocals", "")).is_file() else None,
            language=args.language or getattr(cfg, "LANGUAGE", "ru"),
            segments_dir=str(speaker_dir / "audio_segments"),
            runtime_config=runtime_config,
            smart_sync_config=smart_sync_config,
            tail_guard_config=tail_guard_config,
            segment_routing_config=segment_routing_config,
            audio_level_config=audio_level_config,
        )
        layer_paths.append(layer_path)
        combined_segments.extend(tts_segments)
        speaker_summary[speaker_id] = {
            "segment_count": len(speaker_segments),
            "layer_path": str(layer_path),
            "reference_count": len(reference_paths),
            "profile_path": str(_speaker_profile_path(profiles_dir, speaker_id)),
        }

    combined_segments.sort(
        key=lambda segment: (
            float(segment.get("original_start", segment.get("start", 0.0))),
            str(segment.get("speaker_id", "")),
        )
    )
    _write_json(tts_segments_path, combined_segments)

    mixed = _mix_audio_layers(layer_paths, output_path)
    if args.final_process_mix:
        mixed = apply_final_audio_processing(mixed, audio_level_config)
        mixed.export(output_path, format="wav")

    summary = {
        "stage": "tts",
        "translated_segments": str(translated_path),
        "output_audio": str(output_path),
        "tts_segments": str(tts_segments_path),
        "speaker_count": len(grouped),
        "segment_count": len(combined_segments),
        "duration_sec": round(len(mixed) / 1000.0, 3),
        "speakers": speaker_summary,
        "smart_sync_enabled": smart_sync_config.enabled,
        "babble_guard_enabled": tail_guard_config.enable_babble_guard,
        "asr_retry_enabled": tail_guard_config.enable_asr_retry,
    }
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def run_postprocess_probe(args: argparse.Namespace) -> dict[str, Any]:
    from src.postprocessing import add_audio_to_video, mix_audio_tracks

    paths = _build_paths(args)
    probe_dir = _probe_dir(paths, args)
    probe_paths = _default_probe_paths(probe_dir)
    dubbing_path = Path(args.output_audio or probe_paths["final_dubbing"])
    mix_path = Path(args.output_mix or probe_paths["final_mix"])
    video_path = Path(args.output_video or probe_paths["final_video"])

    _require_file(dubbing_path, "final_dubbing_diarized_probe.wav")
    _require_file(paths["background"], "background.wav")
    _require_file(paths["original_audio"], "original_extracted_audio.wav")
    _require_file(paths["original_video"], "original video")

    mix_audio_tracks(
        voice_over_path=str(dubbing_path),
        background_path=paths["background"],
        output_path=str(mix_path),
        original_audio_path=paths["original_audio"],
        voice_gain=getattr(cfg, "VOICE_GAIN", -3.0),
        background_gain=getattr(cfg, "BACKGROUND_GAIN", -5.0),
        original_gain=getattr(cfg, "ORIGINAL_GAIN", -10.0),
    )
    add_audio_to_video(
        video_path=paths["original_video"],
        audio_path=str(mix_path),
        output_video_path=str(video_path),
    )

    return {
        "stage": "postprocess",
        "final_dubbing": str(dubbing_path),
        "final_mix": str(mix_path),
        "final_video": str(video_path),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    if args.stage in {"translate", "all"}:
        results.append(run_translate_probe(args))
    if args.stage in {"tts", "all"}:
        results.append(run_tts_probe(args))
    if args.stage in {"postprocess", "all"}:
        results.append(run_postprocess_probe(args))
    return {"results": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental multi-speaker dubbing probe. Uses diarization_probe "
            "artifacts and never overwrites main pipeline files."
        )
    )
    parser.add_argument("--stage", choices=["translate", "tts", "postprocess", "all"], default="all")
    parser.add_argument("--job-dir", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--suffix", default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--probe-dir", default=None)
    parser.add_argument("--segments-diarized", default=None)
    parser.add_argument("--translated-segments", default=None)
    parser.add_argument("--speaker-profiles-dir", default=None)
    parser.add_argument("--layers-dir", default=None)
    parser.add_argument("--output-audio", default=None)
    parser.add_argument("--output-mix", default=None)
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--tts-segments-output", default=None)
    parser.add_argument("--tts-summary", default=None)

    parser.add_argument("--mt-model", default=None)
    parser.add_argument("--translation-mode", choices=["nllb", "llm-dialogue"], default="nllb")
    parser.add_argument("--src-lang", default=None)
    parser.add_argument("--tgt-lang", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--llm-provider", default=getattr(cfg, "SMART_SYNC_PROVIDER", "groq"))
    parser.add_argument("--llm-model", default=getattr(cfg, "SMART_SYNC_MODEL_NAME", "llama-3.3-70b-versatile"))
    parser.add_argument("--llm-api-key-env", default=getattr(cfg, "SMART_SYNC_API_KEY_ENV", "GROQ_API_KEY"))
    parser.add_argument("--llm-base-url", default=getattr(cfg, "SMART_SYNC_BASE_URL", ""))
    parser.add_argument("--llm-batch-size", type=int, default=10)
    parser.add_argument("--llm-context-segments", type=int, default=3)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-max-tokens", type=int, default=1800)
    parser.add_argument("--llm-timeout-sec", type=int, default=120)
    parser.add_argument("--llm-min-interval-sec", type=float, default=1.0)
    parser.add_argument("--llm-max-retries", type=int, default=4)
    parser.add_argument(
        "--llm-timing-constraints",
        action="store_true",
        help="Ask the LLM to keep each translation within a duration-derived character budget.",
    )
    parser.add_argument("--llm-target-chars-per-sec", type=float, default=16.0)
    parser.add_argument("--llm-min-chars", type=int, default=8)
    parser.add_argument("--llm-max-chars", type=int, default=170)
    parser.add_argument("--llm-constraint-tolerance-ratio", type=float, default=1.05)
    parser.add_argument("--llm-constraint-repair-retries", type=int, default=1)
    parser.add_argument("--llm-constraint-strict", action="store_true")

    parser.add_argument("--language", default=None)
    parser.add_argument("--disable-grouping", action="store_true")
    parser.add_argument("--disable-segment-routing", action="store_true")
    parser.add_argument("--enable-smart-sync", action="store_true")
    parser.add_argument("--enable-babble-guard", action="store_true")
    parser.add_argument("--enable-asr-retry", action="store_true")
    parser.add_argument("--allow-single-speaker", action="store_true")
    parser.add_argument(
        "--no-final-process-mix",
        dest="final_process_mix",
        action="store_false",
        default=True,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = run_probe(args)
    except Exception as exc:
        print(f"diarization_dubbing_probe failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
