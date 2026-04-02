"""
Сравнение моделей и стратегий перевода:
    nllb-600M  × {per-segment, sentence-level}
    nllb-1.3B  × {per-segment, sentence-level}
    qwen-3B    × {per-segment, sentence-level}

Запуск:
    python tests/test_translation_models.py
    python tests/test_translation_models.py --segments path/to/segments.json
    python tests/test_translation_models.py --pause 0.5
    python tests/test_translation_models.py --models qwen2.5-3B

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
from src.translation import (
    load_translation_model,
    translate_segments,
    translate_segments_as_sentences,
)
from utils.helpers import manage_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)

logger = logging.getLogger(__name__)

MODELS = {
    "nllb-600M": "facebook/nllb-200-distilled-600M",
    "nllb-1.3B": "facebook/nllb-200-distilled-1.3B",
    "qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite-preview-09-2025",
}


def compute_labse(translated_segments) -> dict:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    labse = SentenceTransformer("LaBSE")
    orig_texts  = [s.get("original_text", s["text"]).strip() for s in translated_segments if s.get("text")]
    trans_texts = [s["text"].strip() for s in translated_segments if s.get("text")]
    min_len     = min(len(orig_texts), len(trans_texts))

    emb_orig  = labse.encode(orig_texts[:min_len],  show_progress_bar=False)
    emb_trans = labse.encode(trans_texts[:min_len], show_progress_bar=False)
    sims      = cosine_similarity(emb_orig, emb_trans).diagonal()

    return {
        "mean":        float(sims.mean()),
        "min":         float(sims.min()),
        "max":         float(sims.max()),
        "per_segment": sims.tolist(),
        "n":           len(sims)
    }


def print_comparison(results: dict) -> None:
    names = list(results.keys())
    print("\n" + "=" * 80)
    print("  МОДЕЛЬ × СТРАТЕГИЯ (LaBSE)")
    print("=" * 80)
    print(f"{'Метрика':<20}", end="")
    for name in names:
        print(f"{name:>15}", end="")
    print()
    print("-" * 80)

    for key, label in [("mean", "Mean"), ("min", "Min"), ("max", "Max")]:
        print(f"{label:<20}", end="")
        for data in results.values():
            print(f"{data['labse'][key]:>15.4f}", end="")
        print()

    print("-" * 80)
    print(f"{'Сегментов':<20}", end="")
    for data in results.values():
        print(f"{data['labse']['n']:>15}", end="")
    print()

    print(f"{'Время (сек)':<20}", end="")
    for data in results.values():
        print(f"{data['time']:>15.1f}", end="")
    print()

    print("=" * 80)
    best_name, best_data = max(results.items(), key=lambda x: x[1]["labse"]["mean"])
    print(f"  Победитель: {best_name}  (LaBSE mean: {best_data['labse']['mean']:.4f})")
    print("=" * 80)


def plot_comparison(results: dict, save_path: str = None) -> None:
    # Цвета: per-segment синий/оранжевый, sentence-level зелёный/красный
    color_map = {
        "nllb-600M × per-segment":      "steelblue",
        "nllb-600M × sentence-level":   "deepskyblue",
        "nllb-1.3B × per-segment":      "darkorange",
        "nllb-1.3B × sentence-level":   "green",
        "qwen2.5-3B × per-segment":     "firebrick",
        "qwen2.5-3B × sentence-level":  "salmon",
        "gemini-2.5-flash-lite × per-segment":    "mediumpurple",
        "gemini-2.5-flash-lite × sentence-level": "orchid",
    }
    style_map = {
        "per-segment":    "-",
        "sentence-level": "--",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    # Верхний — все 4 кривые
    ax = axes[0]
    for name, data in results.items():
        sims   = data["labse"]["per_segment"]
        x      = list(range(len(sims)))
        color  = color_map.get(name, "gray")
        style  = "--" if "sentence" in name else "-"
        ax.plot(x, sims, marker="o", markersize=2, linestyle=style,
                label=f"{name}  (mean={data['labse']['mean']:.4f}, N={data['labse']['n']}, {data['time']:.0f}с)",
                color=color, alpha=0.85)
        ax.axhline(data["labse"]["mean"], color=color, linestyle=":", alpha=0.4)

    ax.set_title("Сравнение моделей и стратегий перевода — LaBSE")
    ax.set_ylabel("Косинусное сходство")
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Нижний — bar chart средних значений
    ax2 = axes[1]
    names  = list(results.keys())
    means  = [results[n]["labse"]["mean"] for n in names]
    colors = [color_map.get(n, "gray") for n in names]
    bars   = ax2.bar(names, means, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_title("Среднее LaBSE по методам")
    ax2.set_ylabel("LaBSE mean")
    ax2.set_ylim(min(means) - 0.02, max(means) + 0.02)
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"График сохранён: {save_path}")
    plt.show()


def run_test(
    segments_path: str,
    suffix: str,
    max_pause: float,
    selected_models: list[str] | None = None
) -> None:
    out_dir = "./data/test"
    manage_directory(out_dir, action="create")

    if not os.path.exists(segments_path):
        logger.error(f"Файл не найден: {segments_path}")
        sys.exit(1)

    with open(segments_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    logger.info(f"Загружено сегментов: {len(segments)}")

    all_results = {}

    available_models = MODELS
    if selected_models:
        requested = {name.strip() for name in selected_models if name.strip()}
        unknown = sorted(requested - set(MODELS))
        if unknown:
            logger.error("Неизвестные модели: %s", ", ".join(unknown))
            sys.exit(1)
        available_models = {
            label: name
            for label, name in MODELS.items()
            if label in requested
        }

    for model_label, model_name in available_models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"▶ Модель: {model_label} ({model_name})")
        logger.info(f"{'='*60}")

        model, tokenizer = load_translation_model(model_name, cfg.DEVICE)

        # Per-segment
        logger.info(f"  Стратегия: per-segment")
        t0 = time.time()
        t_ps = translate_segments(
            model=model, tokenizer=tokenizer, segments=segments,
            src_lang=cfg.MT_SRC_LANG, tgt_lang=cfg.MT_TGT_LANG,
            batch_size=cfg.MT_BATCH_SIZE, max_length=cfg.MT_MAX_LENGTH
        )
        elapsed_ps = time.time() - t0
        path = os.path.join(out_dir, f"translated_{model_label}_per_segment_{suffix}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(t_ps, f, ensure_ascii=False, indent=2)
        logger.info(f"  Сохранено: {path} ({elapsed_ps:.1f}с)")

        # Sentence-level
        logger.info(f"  Стратегия: sentence-level (pause={max_pause}с)")
        t0 = time.time()
        t_sent = translate_segments_as_sentences(
            model=model, tokenizer=tokenizer, segments=segments,
            src_lang=cfg.MT_SRC_LANG, tgt_lang=cfg.MT_TGT_LANG,
            max_pause_merge=max_pause,
            batch_size=cfg.MT_BATCH_SIZE, max_length=cfg.MT_MAX_LENGTH
        )
        elapsed_sent = time.time() - t0
        path = os.path.join(out_dir, f"translated_{model_label}_sentence_level_{suffix}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(t_sent, f, ensure_ascii=False, indent=2)
        logger.info(f"  Сохранено: {path} ({elapsed_sent:.1f}с)")

        # Выгружаем модель
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()

        # LaBSE
        logger.info("  Считаем LaBSE...")
        key_ps   = f"{model_label} × per-segment"
        key_sent = f"{model_label} × sentence-level"
        all_results[key_ps]   = {"labse": compute_labse(t_ps),   "time": elapsed_ps}
        all_results[key_sent] = {"labse": compute_labse(t_sent), "time": elapsed_sent}

        logger.info(f"  per-segment:    {all_results[key_ps]['labse']['mean']:.4f}")
        logger.info(f"  sentence-level: {all_results[key_sent]['labse']['mean']:.4f}")

    # Сохраняем сводку
    summary = {k: {"labse": v["labse"], "time": v["time"]} for k, v in all_results.items()}
    results_path = os.path.join(out_dir, f"models_strategies_comparison_{suffix}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Вывод
    print_comparison(all_results)

    plot_path = os.path.join(out_dir, f"models_strategies_comparison_{suffix}.png")
    plot_comparison(all_results, save_path=plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сравнение моделей и стратегий перевода")
    parser.add_argument("--suffix",   default=cfg.SUFFIX)
    parser.add_argument("--segments", default=None)
    parser.add_argument("--pause",    type=float, default=0.5,
                        help="Макс. пауза для склейки sentence-level (по умолчанию: 0.5)")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Какие модели считать. Пример: --models nllb-1.3B qwen2.5-3B gemini-2.5-flash-lite"
    )
    args = parser.parse_args()

    segments_path = args.segments or os.path.join(
        cfg.OUTPUT_PATH, f"segments_{args.suffix}.json"
    )

    run_test(
        segments_path=segments_path,
        suffix=args.suffix,
        max_pause=args.pause,
        selected_models=args.models
    )
