"""
Сравнение моделей перевода (per-segment стратегия):
    - facebook/nllb-200-distilled-600M  (~2.4 GB)
    - facebook/nllb-200-distilled-1.3B  (~5.4 GB)

Запуск:
    python tests/test_translation_models.py
    python tests/test_translation_models.py --segments path/to/segments.json
    python tests/test_translation_models.py --suffix woman

Результаты сохраняются в ./data/test/
"""
import argparse
import gc
import json
import logging
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
from src.translation import translate_segments
from utils.helpers import manage_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

MODELS = {
    "nllb-600M": "facebook/nllb-200-distilled-600M",
    "nllb-1.3B": "facebook/nllb-200-distilled-1.3B",
}


def load_model(model_name: str, device: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    logger.info(f"Загрузка модели: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cuda":
        model = model.to("cuda")
    model.eval()
    logger.info(f"Модель загружена: {model_name}")
    return model, tokenizer


def compute_labse(original_segments, translated_segments) -> dict:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    labse = SentenceTransformer("LaBSE")
    orig_texts  = [s["text"].strip() for s in original_segments  if s.get("text")]
    trans_texts = [s["text"].strip() for s in translated_segments if s.get("text")]
    min_len     = min(len(orig_texts), len(trans_texts))

    emb_orig  = labse.encode(orig_texts[:min_len],  show_progress_bar=False)
    emb_trans = labse.encode(trans_texts[:min_len], show_progress_bar=False)
    sims      = cosine_similarity(emb_orig, emb_trans).diagonal()

    return {
        "mean":        float(sims.mean()),
        "min":         float(sims.min()),
        "max":         float(sims.max()),
        "per_segment": sims.tolist()
    }


def print_comparison(results: dict) -> None:
    print("\n" + "=" * 68)
    print("  СРАВНЕНИЕ МОДЕЛЕЙ ПЕРЕВОДА (LaBSE, per-segment)")
    print("=" * 68)
    print(f"{'Метрика':<20}", end="")
    for name in results:
        print(f"{name:>20}", end="")
    print()
    print("-" * 68)

    for key in ("mean", "min", "max"):
        print(f"{key.capitalize():<20}", end="")
        for data in results.values():
            print(f"{data['labse'][key]:>20.4f}", end="")
        print()

    print("-" * 68)
    print(f"{'Время (сек)':<20}", end="")
    for data in results.values():
        print(f"{data['time']:>20.1f}", end="")
    print()

    print("=" * 68)
    best = max(results.items(), key=lambda x: x[1]["labse"]["mean"])
    print(f"  Лучшая модель: {best[0]} (LaBSE mean: {best[1]['labse']['mean']:.4f})")
    print("=" * 68)


def plot_comparison(results: dict, save_path: str = None) -> None:
    colors = ["steelblue", "darkorange"]
    names  = list(results.keys())
    n      = min(len(v["labse"]["per_segment"]) for v in results.values())
    x      = list(range(n))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Верхний — LaBSE по сегментам
    for i, (name, data) in enumerate(results.items()):
        sims = data["labse"]["per_segment"][:n]
        ax1.plot(x, sims, marker="o", markersize=2,
                 label=f"{name} (среднее: {data['labse']['mean']:.4f}, время: {data['time']:.0f}с)",
                 color=colors[i % len(colors)], alpha=0.8)
        ax1.axhline(data["labse"]["mean"], color=colors[i % len(colors)],
                    linestyle="--", alpha=0.4)

    ax1.set_title("Сравнение моделей перевода — LaBSE по сегментам")
    ax1.set_ylabel("Косинусное сходство")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Нижний — разница (1.3B - 600M)
    keys = list(results.keys())
    if len(keys) == 2:
        sims_a = results[keys[0]]["labse"]["per_segment"][:n]
        sims_b = results[keys[1]]["labse"]["per_segment"][:n]
        diff   = [sims_b[i] - sims_a[i] for i in range(n)]
        colors_bar = ["green" if d >= 0 else "red" for d in diff]
        ax2.bar(x, diff, color=colors_bar, alpha=0.7)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.axhline(float(np.mean(diff)), color="purple", linestyle="--",
                    label=f"Средняя разница: {np.mean(diff):+.4f}")
        ax2.set_title(f"Разница ({keys[1]} − {keys[0]}): зелёный = 1.3B лучше")
        ax2.set_xlabel("Сегмент")
        ax2.set_ylabel("Δ LaBSE")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"График сохранён: {save_path}")
    plt.show()


def run_test(segments_path: str, suffix: str) -> None:
    out_dir = "./data/test"
    manage_directory(out_dir, action="create")

    if not os.path.exists(segments_path):
        logger.error(f"Файл не найден: {segments_path}")
        sys.exit(1)

    with open(segments_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    logger.info(f"Загружено сегментов: {len(segments)}")

    all_results = {}

    for label, model_name in MODELS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"▶ Тестируем: {label} ({model_name})")
        logger.info(f"{'='*50}")

        # Загружаем модель
        model, tokenizer = load_model(model_name, cfg.DEVICE)

        # Переводим и замеряем время
        t_start = time.time()
        translated = translate_segments(
            model=model,
            tokenizer=tokenizer,
            segments=segments,
            src_lang=cfg.MT_SRC_LANG,
            tgt_lang=cfg.MT_TGT_LANG,
            batch_size=cfg.MT_BATCH_SIZE,
            max_length=cfg.MT_MAX_LENGTH
        )
        elapsed = time.time() - t_start
        logger.info(f"Время перевода: {elapsed:.1f} сек")

        # Сохраняем перевод
        out_path = os.path.join(out_dir, f"translated_{label}_{suffix}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(translated, f, ensure_ascii=False, indent=2)
        logger.info(f"Сохранено: {out_path}")

        # Выгружаем модель
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()

        # LaBSE
        logger.info(f"Считаем LaBSE для {label}...")
        labse = compute_labse(segments, translated)
        logger.info(f"LaBSE mean: {labse['mean']:.4f}")

        all_results[label] = {
            "labse": labse,
            "time":  elapsed
        }

    # Сохраняем сводные результаты
    results_path = os.path.join(out_dir, f"models_comparison_{suffix}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Вывод
    print_comparison(all_results)

    plot_path = os.path.join(out_dir, f"models_comparison_{suffix}.png")
    plot_comparison(all_results, save_path=plot_path)

    # Примеры
    translations_by_label = {}
    for label in MODELS:
        out_path = os.path.join(out_dir, f"translated_{label}_{suffix}.json")
        with open(out_path, "r", encoding="utf-8") as f:
            translations_by_label[label] = json.load(f)

    print("\n── Примеры (первые 5 сегментов) ──")
    for i in range(min(5, len(segments))):
        print(f"\n[{i}] Оригинал: {segments[i]['text']}")
        for label, translated in translations_by_label.items():
            text = translated[i]["text"] if i < len(translated) else "—"
            print(f"    {label:<15} {text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сравнение моделей перевода")
    parser.add_argument("--suffix",   default=cfg.SUFFIX)
    parser.add_argument("--segments", default=None)
    args = parser.parse_args()

    segments_path = args.segments or os.path.join(
        cfg.OUTPUT_PATH, f"segments_{args.suffix}.json"
    )

    run_test(segments_path=segments_path, suffix=args.suffix)
