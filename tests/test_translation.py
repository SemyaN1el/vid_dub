"""
Сравнительный тест стратегий перевода:
    - per-segment:      каждый сегмент независимо
    - sliding-window:   каждый сегмент с контекстом соседей
    - sentence-level:   склейка в полные предложения → перевод → один сегмент на предложение

Запуск:
    python tests/test_translation.py
    python tests/test_translation.py --suffix woman
    python tests/test_translation.py --segments path/to/segments.json
    python tests/test_translation.py --window 2 --pause 0.5

Результаты сохраняются в ./data/test/
"""

import argparse
import gc
import json
import logging
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
from src.translation import (translate_segments,
                              translate_segments_sliding_window,
                              translate_segments_as_sentences)
from utils.helpers import manage_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(device: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    logger.info(f"Загрузка модели: {cfg.MT_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MT_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.MT_MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cuda":
        model = model.to("cuda")
    model.eval()
    logger.info("Модель загружена.")
    return model, tokenizer


def compute_labse(original_segments, translated_segments) -> dict:
    """
    LaBSE сравнивает оригинальный текст с переводом.
    Для sentence-level: каждое предложение сравнивается со своим оригиналом.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    labse = SentenceTransformer("LaBSE")
    orig_texts  = [s.get("original_text", s["text"]).strip()
                   for s in translated_segments if s.get("text")]
    trans_texts = [s["text"].strip()
                   for s in translated_segments if s.get("text")]

    min_len   = min(len(orig_texts), len(trans_texts))
    emb_orig  = labse.encode(orig_texts[:min_len],  show_progress_bar=False)
    emb_trans = labse.encode(trans_texts[:min_len], show_progress_bar=False)
    sims      = cosine_similarity(emb_orig, emb_trans).diagonal()

    return {
        "mean":        float(sims.mean()),
        "min":         float(sims.min()),
        "max":         float(sims.max()),
        "per_segment": sims.tolist(),
        "n_segments":  len(sims)
    }


def print_comparison(results: dict) -> None:
    print("\n" + "=" * 72)
    print("  СРАВНЕНИЕ СТРАТЕГИЙ ПЕРЕВОДА (LaBSE)")
    print("=" * 72)
    print(f"{'Метрика':<22}", end="")
    for name in results:
        print(f"{name:>16}", end="")
    print()
    print("-" * 72)

    for key, label in [("mean", "Mean"), ("min", "Min"), ("max", "Max")]:
        print(f"{label:<22}", end="")
        for data in results.values():
            print(f"{data['labse'][key]:>16.4f}", end="")
        print()

    print("-" * 72)
    print(f"{'Сегментов (после)':<22}", end="")
    for data in results.values():
        print(f"{data['labse']['n_segments']:>16}", end="")
    print()

    print("=" * 72)
    best = max(results.items(), key=lambda x: x[1]["labse"]["mean"])
    print(f"  Лучший метод: {best[0]} (LaBSE mean: {best[1]['labse']['mean']:.4f})")
    print("=" * 72)


def plot_comparison(results: dict, save_path: str = None) -> None:
    colors = ["steelblue", "darkorange", "green", "purple"]
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (name, data) in enumerate(results.items()):
        sims = data["labse"]["per_segment"]
        x    = list(range(len(sims)))
        ax.plot(x, sims, marker="o", markersize=2,
                label=f"{name} (mean: {data['labse']['mean']:.4f}, N={data['labse']['n_segments']})",
                color=colors[i % len(colors)], alpha=0.8)
        ax.axhline(data["labse"]["mean"], color=colors[i % len(colors)],
                   linestyle="--", alpha=0.4)

    ax.set_title("Сравнение стратегий перевода — LaBSE")
    ax.set_xlabel("Сегмент / предложение")
    ax.set_ylabel("Косинусное сходство")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"График сохранён: {save_path}")
    plt.show()


def print_sentence_examples(segments, translated, n=5):
    """Показывает примеры склейки и перевода для sentence-level."""
    print(f"\n── Sentence-level: примеры склейки (первые {n} предложений) ──")
    for i, seg in enumerate(translated[:n]):
        merged = seg.get("merged_count", 1)
        print(f"\n[{i}] ({merged} сег.) Оригинал:  {seg['original_text']}")
        print(f"          Перевод:   {seg['text']}")
        print(f"          Время:     {seg['start']:.2f}с → {seg['end']:.2f}с "
              f"({seg['end'] - seg['start']:.2f}с)")


def run_test(segments_path: str, suffix: str,
             window_size: int, max_pause: float) -> None:
    out_dir = "./data/test"
    manage_directory(out_dir, action="create")

    if not os.path.exists(segments_path):
        logger.error(f"Файл не найден: {segments_path}")
        sys.exit(1)

    with open(segments_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    logger.info(f"Загружено сегментов: {len(segments)}")

    model, tokenizer = load_model(cfg.DEVICE)
    all_results = {}

    # ── Тест 1: Per-segment ───────────────────────────────────────────────────
    logger.info("▶ Тест 1: Per-segment...")
    t_ps = translate_segments(
        model=model, tokenizer=tokenizer, segments=segments,
        src_lang=cfg.MT_SRC_LANG, tgt_lang=cfg.MT_TGT_LANG,
        batch_size=cfg.MT_BATCH_SIZE, max_length=cfg.MT_MAX_LENGTH
    )
    path = os.path.join(out_dir, f"translated_per_segment_{suffix}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(t_ps, f, ensure_ascii=False, indent=2)
    all_results["per-segment"] = {"translated": t_ps}

    # ── Тест 2: Sliding window ────────────────────────────────────────────────
    logger.info(f"▶ Тест 2: Sliding window (w={window_size})...")
    t_sw = translate_segments_sliding_window(
        model=model, tokenizer=tokenizer, segments=segments,
        src_lang=cfg.MT_SRC_LANG, tgt_lang=cfg.MT_TGT_LANG,
        window_size=window_size,
        batch_size=cfg.MT_BATCH_SIZE, max_length=cfg.MT_MAX_LENGTH
    )
    path = os.path.join(out_dir, f"translated_sliding_window_{suffix}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(t_sw, f, ensure_ascii=False, indent=2)
    all_results[f"sliding-w{window_size}"] = {"translated": t_sw}

    # ── Тест 3: Sentence-level ────────────────────────────────────────────────
    logger.info(f"▶ Тест 3: Sentence-level (pause={max_pause}с)...")
    t_sent = translate_segments_as_sentences(
        model=model, tokenizer=tokenizer, segments=segments,
        src_lang=cfg.MT_SRC_LANG, tgt_lang=cfg.MT_TGT_LANG,
        max_pause_merge=max_pause,
        batch_size=cfg.MT_BATCH_SIZE, max_length=cfg.MT_MAX_LENGTH
    )
    path = os.path.join(out_dir, f"translated_sentence_level_{suffix}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(t_sent, f, ensure_ascii=False, indent=2)
    all_results["sentence-level"] = {"translated": t_sent}

    # Выгружаем модель
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()

    # ── LaBSE ─────────────────────────────────────────────────────────────────
    logger.info("▶ Считаем LaBSE...")
    for name, data in all_results.items():
        logger.info(f"  {name}...")
        data["labse"] = compute_labse(segments, data["translated"])

    # Сохраняем сводку
    summary = {k: {"labse": v["labse"]} for k, v in all_results.items()}
    results_path = os.path.join(out_dir, f"translation_comparison_{suffix}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── Вывод ─────────────────────────────────────────────────────────────────
    print_comparison({k: {"labse": v["labse"]} for k, v in all_results.items()})

    plot_path = os.path.join(out_dir, f"translation_comparison_{suffix}.png")
    plot_comparison({k: {"labse": v["labse"]} for k, v in all_results.items()},
                    save_path=plot_path)

    # Примеры sentence-level
    print_sentence_examples(segments, t_sent, n=5)

    # Сравнение первых 3 сегментов per-segment vs sentence-level
    print("\n── Per-segment vs Sentence-level (первые 3) ──")
    for i in range(min(3, len(t_ps))):
        print(f"\n[{i}] Оригинал:       {t_ps[i]['original_text']}")
        print(f"    per-segment:    {t_ps[i]['text']}")
    print()
    for i in range(min(3, len(t_sent))):
        print(f"[{i}] Предложение:    {t_sent[i]['original_text']}")
        print(f"    sentence-level: {t_sent[i]['text']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сравнение стратегий перевода")
    parser.add_argument("--suffix",   default=cfg.SUFFIX)
    parser.add_argument("--segments", default=None)
    parser.add_argument("--window",   type=int,   default=2,
                        help="Размер окна sliding window (по умолчанию: 2)")
    parser.add_argument("--pause",    type=float, default=0.5,
                        help="Макс. пауза для склейки в sentence-level (по умолчанию: 0.5)")
    args = parser.parse_args()

    segments_path = args.segments or os.path.join(
        cfg.OUTPUT_PATH, f"segments_{args.suffix}.json"
    )

    run_test(
        segments_path=segments_path,
        suffix=args.suffix,
        window_size=args.window,
        max_pause=args.pause
    )
