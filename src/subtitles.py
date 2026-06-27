"""
Модуль генерации субтитров.

Поддерживает три формата:
    SRT — универсальный, поддерживается везде
    VTT — для веб-плееров (HTML5 video)
    ASS — продвинутое форматирование (шрифт, цвет, позиция)

Три режима встраивания в видео:
    hard  — прожиг (навсегда, нельзя выключить), режим по умолчанию
    soft  — мягкая дорожка (можно включить/выключить в плеере)
    both  — оба варианта
"""

import os
import re
import subprocess
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Форматирование времени ────────────────────────────────────────────────────

def _fmt_srt(seconds: float) -> str:
    """00:01:23,456"""
    total_ms = max(0, int(round(seconds * 1000)))
    h,  rem = divmod(total_ms, 3_600_000)
    m,  rem = divmod(rem, 60_000)
    s,  ms  = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt(seconds: float) -> str:
    """00:01:23.456"""
    return _fmt_srt(seconds).replace(",", ".")


def _fmt_ass(seconds: float) -> str:
    """0:01:23.45"""
    total_cs = max(0, int(round(seconds * 100)))
    h,  rem = divmod(total_cs, 360_000)
    m,  rem = divmod(rem, 6_000)
    s,  cs  = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ─── Генерация файлов субтитров ───────────────────────────────────────────────

def generate_srt(
    segments: List[Dict[str, Any]],
    output_path: str,
    use_original: bool = False
) -> str:
    """
    Генерирует файл субтитров в формате SRT.

    Параметры:
        segments: список сегментов с полями text/original_text, start, end
        output_path: путь для сохранения .srt файла
        use_original: True — оригинальный текст, False — переведённый

    Возвращает:
        str: путь к созданному файлу
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    idx = 1
    for seg in segments:
        text = seg.get("original_text" if use_original else "text", "").strip()
        if not text:
            continue
        start = _fmt_srt(seg["start"])
        end   = _fmt_srt(seg["end"])
        lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
        idx += 1

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"SRT сохранён: {output_path} ({idx - 1} субтитров)")
    return output_path


def generate_vtt(
    segments: List[Dict[str, Any]],
    output_path: str,
    use_original: bool = False
) -> str:
    """
    Генерирует файл субтитров в формате WebVTT.

    Возвращает:
        str: путь к созданному файлу
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = ["WEBVTT\n"]
    for seg in segments:
        text = seg.get("original_text" if use_original else "text", "").strip()
        if not text:
            continue
        start = _fmt_vtt(seg["start"])
        end   = _fmt_vtt(seg["end"])
        lines.append(f"{start} --> {end}\n{text}\n")

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"VTT сохранён: {output_path}")
    return output_path


def generate_ass(
    segments: List[Dict[str, Any]],
    output_path: str,
    use_original: bool = False,
    font_name: str = "Arial",
    font_size: int = 24,
    primary_color: str = "&H00FFFFFF",   # белый
    outline_color: str = "&H00000000",   # чёрный контур
    margin_v: int = 30
) -> str:
    """
    Генерирует файл субтитров в формате ASS с кастомным оформлением.

    Параметры:
        font_name: название шрифта
        font_size: размер шрифта
        primary_color: цвет текста в формате &HAABBGGRR
        outline_color: цвет контура
        margin_v: отступ от нижнего края в пикселях

    Возвращает:
        str: путь к созданному файлу
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_color},&H000000FF,{outline_color},&H00000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []
    for seg in segments:
        text = seg.get("original_text" if use_original else "text", "").strip()
        if not text:
            continue
        # Экранируем спецсимволы ASS
        text = text.replace("\n", "\\N").replace("{", "\\{").replace("}", "\\}")
        start = _fmt_ass(seg["start"])
        end   = _fmt_ass(seg["end"])
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    content = header + "\n".join(events) + "\n"
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write(content)

    logger.info(f"ASS сохранён: {output_path}")
    return output_path


def generate_all_formats(
    segments: List[Dict[str, Any]],
    output_dir: str,
    suffix: str,
    use_original: bool = False
) -> Dict[str, str]:
    """
    Генерирует субтитры во всех трёх форматах.

    Возвращает:
        Dict: {'srt': path, 'vtt': path, 'ass': path}
    """
    os.makedirs(output_dir, exist_ok=True)
    lang_tag = "orig" if use_original else "ru"

    paths = {
        "srt": generate_srt(segments, os.path.join(output_dir, f"subtitles_{suffix}_{lang_tag}.srt"), use_original),
        "vtt": generate_vtt(segments, os.path.join(output_dir, f"subtitles_{suffix}_{lang_tag}.vtt"), use_original),
        "ass": generate_ass(segments, os.path.join(output_dir, f"subtitles_{suffix}_{lang_tag}.ass"), use_original),
    }

    logger.info(f"Все форматы субтитров сохранены в {output_dir}")
    return paths


# ─── Встраивание в видео ───────────────────────────────────────────────────────

def embed_subtitles_soft(
    video_path: str,
    subtitle_path: str,
    output_path: str,
    language: str = "rus",
    title: str = "Русский"
) -> str:
    """
    Встраивает субтитры как мягкую дорожку (soft subtitles).
    Можно включить/выключить в плеере.

    Поддерживает .srt, .vtt, .ass файлы.

    Параметры:
        video_path: исходное видео
        subtitle_path: файл субтитров
        output_path: выходное видео
        language: код языка для метаданных дорожки
        title: название дорожки в плеере
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ext = os.path.splitext(subtitle_path)[1].lower()
    output_ext = os.path.splitext(output_path)[1].lower()

    # MP4 understands mov_text; MKV can keep richer text subtitle codecs.
    if output_ext == ".mp4":
        codec = "mov_text"
    elif ext == ".ass":
        codec = "copy"
    elif ext == ".srt":
        codec = "srt"
    else:
        codec = "webvtt"

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-i", subtitle_path,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-map", "1:0",
        "-c:v", "copy",
        "-c:a", "copy",
        "-c:s:0", codec,
        "-metadata:s:s:0", f"language={language}",
        "-metadata:s:s:0", f"title={title}",
        "-disposition:s:0", "default+forced",
        "-y",
        output_path
    ]

    logger.info(f"Встраивание мягких субтитров: {subtitle_path} → {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FFmpeg STDERR: {result.stderr}")
        raise RuntimeError(f"Ошибка встраивания субтитров: {result.stderr}")

    logger.info(f"Видео с мягкими субтитрами: {output_path}")
    return output_path


def embed_subtitles_hard(
    video_path: str,
    subtitle_path: str,
    output_path: str,
    video_size: str = "1920x1080"
) -> str:
    """
    Прожигает субтитры в видео (hard subtitles).
    Нельзя выключить — субтитры становятся частью картинки.

    Поддерживает .srt и .ass. Для .ass сохраняется кастомное оформление.

    Параметры:
        video_path: исходное видео
        subtitle_path: файл субтитров (.srt или .ass)
        output_path: выходное видео
        video_size: разрешение видео (нужно для .ass)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Нормализуем путь для FFmpeg (Windows)
    sub_path_escaped = subtitle_path.replace("\\", "/").replace(":", "\\:")

    ext = os.path.splitext(subtitle_path)[1].lower()
    if ext == ".ass":
        vf_filter = f"ass='{sub_path_escaped}'"
    else:
        vf_filter = f"subtitles='{sub_path_escaped}'"

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-vf", vf_filter,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-y",
        output_path
    ]

    logger.info(f"Прожиг субтитров: {subtitle_path} → {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FFmpeg STDERR: {result.stderr}")
        raise RuntimeError(f"Ошибка прожига субтитров: {result.stderr}")

    logger.info(f"Видео с прожжёнными субтитрами: {output_path}")
    return output_path


def add_subtitles_to_video(
    video_path: str,
    segments: List[Dict[str, Any]],
    output_dir: str,
    suffix: str,
    mode: str = "hard",
    use_original: bool = False,
    ass_font: str = "Arial",
    ass_font_size: int = 24
) -> Dict[str, str]:
    """
    Полный цикл: генерация субтитров + встраивание в видео.

    Параметры:
        video_path: исходное или уже дублированное видео
        segments: переведённые сегменты
        output_dir: директория для файлов субтитров и видео
        suffix: суффикс спикера
        mode: 'soft' | 'hard' | 'both'
        use_original: True — субтитры на оригинальном языке
        ass_font: шрифт для ASS
        ass_font_size: размер шрифта для ASS

    Возвращает:
        Dict: пути к созданным файлам
    """
    mode = (mode or "hard").strip().lower()
    if mode not in {"soft", "hard", "both"}:
        raise ValueError(f"Unsupported subtitle mode: {mode!r}. Use soft, hard or both.")

    results = {}

    # Генерируем все форматы
    sub_paths = generate_all_formats(segments, output_dir, suffix, use_original)
    results.update(sub_paths)

    lang_tag = "orig" if use_original else "ru"
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    soft_out = os.path.join(output_dir, f"{base_name}_soft_subs_{lang_tag}.mp4")
    hard_out = os.path.join(output_dir, f"{base_name}_hard_subs_{lang_tag}.mp4")

    stale_outputs = []
    if mode not in ("soft", "both"):
        stale_outputs.append(soft_out)
    if mode not in ("hard", "both"):
        stale_outputs.append(hard_out)
    for stale_output in stale_outputs:
        if os.path.exists(stale_output):
            os.remove(stale_output)
            logger.info("Удалён устаревший subtitle output: %s", stale_output)

    if mode in ("soft", "both"):
        try:
            embed_subtitles_soft(
                video_path=video_path,
                subtitle_path=sub_paths["srt"],
                output_path=soft_out,
                language="rus" if not use_original else "eng",
                title="Русский" if not use_original else "English"
            )
            results["video_soft"] = soft_out
        except Exception as e:
            logger.warning(f"Мягкие субтитры не встроены: {e}")

    if mode in ("hard", "both"):
        try:
            embed_subtitles_hard(
                video_path=video_path,
                subtitle_path=sub_paths["ass"],
                output_path=hard_out
            )
            results["video_hard"] = hard_out
        except Exception as e:
            logger.warning(f"Прожиг субтитров не выполнен: {e}")

    return results
