import logging
from tempfile import NamedTemporaryFile
from typing import Dict, List, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from pydub import AudioSegment

logger = logging.getLogger(__name__)


# ─── Speaker Verification ────────────────────────────────────────────────────

def speaker_verification_score(reference_path: str, generated_path: str) -> float:
    """
    Вычисляет сходство голосов через resemblyzer (независимо от XTTS).

    Параметры:
        reference_path: путь к референсному аудио оригинального спикера
        generated_path: путь к сгенерированному аудио

    Возвращает:
        float: косинусное сходство [0, 1] (>0.75 — хорошо, >0.6 — приемлемо)
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        encoder = VoiceEncoder()
        emb_ref = encoder.embed_utterance(preprocess_wav(reference_path))
        emb_gen = encoder.embed_utterance(preprocess_wav(generated_path))
        score = float(np.dot(emb_ref, emb_gen))
        logger.info(f"Speaker Verification Score: {score:.4f}")
        return score
    except ImportError:
        logger.error("resemblyzer не установлен. Запустите: pip install resemblyzer")
        return 0.0


# ─── WER / CER ───────────────────────────────────────────────────────────────

def compute_wer_cer(
    model_asr,
    generated_audio_path: str,
    reference_segments: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Вычисляет WER и CER: распознаёт сгенерированное аудио и сравнивает
    с эталонным текстом.

    Параметры:
        model_asr: загруженная модель Whisper
        generated_audio_path: путь к сгенерированной аудиодорожке
        reference_segments: сегменты с полем 'text' (переведённый текст)

    Возвращает:
        Dict с ключами 'wer' и 'cer'
    """
    try:
        from jiwer import wer, cer
    except ImportError:
        logger.error("jiwer не установлен. Запустите: pip install jiwer")
        return {"wer": -1.0, "cer": -1.0}

    result    = model_asr.transcribe(generated_audio_path, language="ru")
    recognized = result["text"].strip()
    reference  = " ".join(s["text"].strip() for s in reference_segments if s.get("text"))

    word_err = wer(reference, recognized)
    char_err = cer(reference, recognized)

    logger.info(f"WER: {word_err:.4f} | CER: {char_err:.4f}")
    logger.info(f"Распознано:  {recognized[:150]}...")
    logger.info(f"Эталон:      {reference[:150]}...")

    return {"wer": word_err, "cer": char_err}


# ─── LaBSE — семантическое сходство перевода ─────────────────────────────────

def compute_labse_similarity(
    original_segments: List[Dict[str, Any]],
    translated_segments: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Вычисляет семантическое сходство между оригинальными и переведёнными
    сегментами с помощью мультиязычной модели LaBSE.

    Параметры:
        original_segments:   сегменты с оригинальным текстом
        translated_segments: сегменты с переведённым текстом

    Возвращает:
        Dict с ключами 'mean', 'min', 'max', 'per_segment'
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        logger.error("sentence-transformers не установлен. Запустите: pip install sentence-transformers")
        return {}

    model = SentenceTransformer("LaBSE")

    orig_texts  = [s["text"].strip() for s in original_segments   if s.get("text")]
    trans_texts = [s["text"].strip() for s in translated_segments if s.get("text")]

    min_len     = min(len(orig_texts), len(trans_texts))
    orig_texts  = orig_texts[:min_len]
    trans_texts = trans_texts[:min_len]

    emb_orig  = model.encode(orig_texts,  show_progress_bar=True)
    emb_trans = model.encode(trans_texts, show_progress_bar=True)

    sims = cosine_similarity(emb_orig, emb_trans).diagonal()

    result = {
        "mean":        float(sims.mean()),
        "min":         float(sims.min()),
        "max":         float(sims.max()),
        "per_segment": sims.tolist()
    }

    logger.info(f"LaBSE — среднее: {result['mean']:.4f}, "
                f"мин: {result['min']:.4f}, макс: {result['max']:.4f}")
    return result


# ─── Косинусное сходство XTTS-эмбеддингов ────────────────────────────────────

def cosine_similarity_vectors(a: torch.Tensor, b: torch.Tensor) -> float:
    """Косинусное сходство между двумя тензорами."""
    a_np = a.cpu().numpy().flatten() if isinstance(a, torch.Tensor) else a.flatten()
    b_np = b.cpu().numpy().flatten() if isinstance(b, torch.Tensor) else b.flatten()

    norm_a, norm_b = np.linalg.norm(a_np), np.linalg.norm(b_np)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / (norm_a * norm_b))


# ─── Сводная таблица и графики ───────────────────────────────────────────────

def print_metrics_summary(
    speaker_score: float,
    wer_cer: Dict[str, float],
    labse: Dict[str, Any]
) -> None:
    """Выводит итоговую таблицу метрик."""

    def grade(val, good, ok, higher_is_better=True):
        if higher_is_better:
            return "хорошо" if val >= good else ("приемлемо" if val >= ok else "плохо")
        else:
            return "хорошо" if val <= good else ("приемлемо" if val <= ok else "плохо")

    rows = [
        ("Speaker Verification", speaker_score,
         grade(speaker_score, 0.75, 0.6)),
        ("WER", wer_cer.get("wer", -1),
         grade(wer_cer.get("wer", 1), 0.2, 0.4, higher_is_better=False)),
        ("CER", wer_cer.get("cer", -1),
         grade(wer_cer.get("cer", 1), 0.15, 0.3, higher_is_better=False)),
        ("LaBSE (среднее)", labse.get("mean", -1),
         grade(labse.get("mean", 0), 0.8, 0.65)),
    ]

    print("\n" + "=" * 52)
    print("  ИТОГОВЫЕ МЕТРИКИ КАЧЕСТВА")
    print("=" * 52)
    print(f"{'Метрика':<28} {'Значение':<12} {'Оценка'}")
    print("-" * 52)
    for name, val, g in rows:
        print(f"{name:<28} {val:<12.4f} {g}")
    print("=" * 52)


def plot_labse(labse: Dict[str, Any], title: str = "") -> None:
    """Строит график LaBSE-сходства по сегментам."""
    sims = labse.get("per_segment", [])
    if not sims:
        logger.warning("Нет данных для графика LaBSE.")
        return

    mean_val = labse["mean"]
    plt.figure(figsize=(12, 4))
    plt.plot(sims, marker="o", color="steelblue", label="LaBSE сходство")
    plt.axhline(mean_val, color="red", linestyle="--",
                label=f"Среднее: {mean_val:.4f}")
    plt.title(f"{title} Семантическое сходство по сегментам (LaBSE)")
    plt.xlabel("Сегмент")
    plt.ylabel("Косинусное сходство")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
