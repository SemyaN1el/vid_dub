"""
Тест Google Translate как baseline для сравнения с NLLB.

Использует googletrans (бесплатно, без API-ключа).
Установка: pip install googletrans==4.0.0rc1

Запуск:
    python tests/test_google_translate.py
    python tests/test_google_translate.py --segments data/test/segments_man.json
    python tests/test_google_translate.py --compare   # сравнить с уже готовыми NLLB-результатами

Результаты сохраняются в ./data/test/
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
from utils.helpers import manage_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)


import asyncio

async def _translate_async(segments: list, src: str = "en", tgt: str = "ru") -> list:
    """Асинхронный перевод через Google Translate."""
    from googletrans import Translator

    translator = Translator()
    translated = []
    errors = 0

    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if not text:
            continue

        try:
            result = await translator.translate(text, src=src, dest=tgt)  # ← await!
            translated.append({
                "text":          result.text,
                "original_text": text,
                "start":         seg["start"],
                "end":           seg["end"]
            })

            if i % 10 == 0:
                logger.info(f"  [{i+1}/{len(segments)}] {text[:50]}... → {result.text[:50]}...")

            await asyncio.sleep(0.3)  # ← asyncio.sleep вместо time.sleep

        except Exception as e:
            logger.warning(f"  [{i}] Ошибка перевода: {e}. Ставим оригинал.")
            translated.append({
                "text":          text,
                "original_text": text,
                "start":         seg["start"],
                "end":           seg["end"]
            })
            errors += 1
            await asyncio.sleep(1.0)

    logger.info(f"Google Translate: переведено {len(translated)} сегментов, ошибок: {errors}")
    return translated



def translate_with_google(segments: list, src: str = "en", tgt: str = "ru") -> list:
    """Синхронная обёртка для вызова из обычного кода."""
    try:
        from googletrans import Translator  # noqa: проверка установки
    except ImportError:
        logger.error("googletrans не установлен. Запустите: pip install googletrans==4.0.0rc1")
        sys.exit(1)

    return asyncio.run(_translate_async(segments, src, tgt))


def compute_labse(translated_segments: list) -> dict:
    """Считает LaBSE между original_text и text."""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    labse = SentenceTransformer("LaBSE")
    orig  = [s.get("original_text", s["text"]).strip() for s in translated_segments if s.get("text")]
    trans = [s["text"].strip() for s in translated_segments if s.get("text")]
    n = min(len(orig), len(trans))

    emb_o = labse.encode(orig[:n],  show_progress_bar=False)
    emb_t = labse.encode(trans[:n], show_progress_bar=False)
    sims  = cosine_similarity(emb_o, emb_t).diagonal()

    return {
        "mean": float(sims.mean()),
        "min":  float(sims.min()),
        "max":  float(sims.max()),
        "n":    n,
        "per_segment": sims.tolist()
    }


def print_comparison(results: dict) -> None:
    """Выводит сравнительную таблицу всех методов."""
    print("\n" + "=" * 72)
    print("  СРАВНЕНИЕ МЕТОДОВ ПЕРЕВОДА (LaBSE)")
    print("=" * 72)
    print(f"{'Метод':<35} {'Mean':>8} {'Min':>8} {'Max':>8} {'N':>5}")
    print("-" * 72)
    for name, data in results.items():
        labse = data["labse"]
        print(f"{name:<35} {labse['mean']:>8.4f} {labse['min']:>8.4f} {labse['max']:>8.4f} {labse['n']:>5}")
    print("=" * 72)
    best = max(results.items(), key=lambda x: x[1]["labse"]["mean"])
    print(f"  Победитель: {best[0]}  (LaBSE mean: {best[1]['labse']['mean']:.4f})")
    print("=" * 72)


def plot_comparison(results: dict, save_path: str = None) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    colors = ["steelblue", "darkorange", "green", "red", "purple"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

    # Верхний — кривые по сегментам
    for i, (name, data) in enumerate(results.items()):
        sims = data["labse"]["per_segment"]
        ax1.plot(range(len(sims)), sims, marker="o", markersize=2,
                 label=f"{name} (mean={data['labse']['mean']:.4f})",
                 color=colors[i % len(colors)], alpha=0.8)
        ax1.axhline(data["labse"]["mean"], color=colors[i % len(colors)],
                    linestyle="--", alpha=0.4)

    ax1.set_title("Сравнение методов перевода — LaBSE по сегментам")
    ax1.set_ylabel("Косинусное сходство")
    ax1.set_ylim(0.4, 1.0)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Нижний — bar chart средних
    names  = list(results.keys())
    means  = [results[n]["labse"]["mean"] for n in names]
    bars   = ax2.bar(names, means,
                     color=colors[:len(names)], alpha=0.8,
                     edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    ax2.set_title("Среднее LaBSE по методам")
    ax2.set_ylabel("LaBSE mean")
    ax2.set_ylim(min(means) - 0.03, max(means) + 0.03)
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"График сохранён: {save_path}")
    plt.show()


def run_test(segments_path: str, suffix: str, compare: bool) -> None:
    out_dir = "./data/test"
    manage_directory(out_dir, action="create")

    if not os.path.exists(segments_path):
        logger.error(f"Файл не найден: {segments_path}")
        sys.exit(1)

    with open(segments_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    logger.info(f"Загружено сегментов: {len(segments)}")

    all_results = {}

    # ── Google Translate ──────────────────────────────────────────────
    gt_path = os.path.join(out_dir, f"translated_google_{suffix}.json")

    if os.path.exists(gt_path):
        logger.info(f"Загружаем готовый перевод Google Translate: {gt_path}")
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_translated = json.load(f)
    else:
        logger.info("▶ Переводим через Google Translate...")
        gt_translated = translate_with_google(segments)
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_translated, f, ensure_ascii=False, indent=2)
        logger.info(f"Сохранено: {gt_path}")

    logger.info("▶ LaBSE для Google Translate...")
    all_results["Google Translate"] = {"labse": compute_labse(gt_translated)}

    # ── Загружаем готовые NLLB-результаты если есть ──────────────────
    if compare:
        candidates = {
            "NLLB-600M per-segment":    f"translated_nllb-600M_per_segment_{suffix}.json",
            "NLLB-1.3B per-segment":    f"translated_nllb-1.3B_per_segment_{suffix}.json",
            "NLLB-600M sentence-level": f"translated_nllb-600M_sentence_level_{suffix}.json",
            "NLLB-1.3B sentence-level": f"translated_nllb-1.3B_sentence_level_{suffix}.json",
        }
        for name, fname in candidates.items():
            fpath = os.path.join(out_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"▶ LaBSE для {name}...")
                all_results[name] = {"labse": compute_labse(data)}
            else:
                logger.warning(f"Файл не найден, пропуск: {fpath}")

    # ── Вывод ─────────────────────────────────────────────────────────
    print_comparison(all_results)

    # Сохраняем сводку
    summary = {k: {"labse": v["labse"]} for k, v in all_results.items()}
    out_json = os.path.join(out_dir, f"comparison_with_google_{suffix}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # График
    out_plot = os.path.join(out_dir, f"comparison_with_google_{suffix}.png")
    plot_comparison(all_results, save_path=out_plot)

    # Примеры
    print("\n── Примеры (первые 3 сегмента) ──")
    for i in range(min(3, len(segments))):
        print(f"\n[{i}] Оригинал:          {segments[i]['text']}")
        print(f"    Google Translate:  {gt_translated[i]['text'] if i < len(gt_translated) else '—'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Translate vs NLLB — сравнение LaBSE")
    parser.add_argument("--suffix",   default=cfg.SUFFIX)
    parser.add_argument("--segments", default=None, help="Путь к JSON с сегментами")
    parser.add_argument("--compare",  action="store_true",
                        help="Добавить в сравнение готовые NLLB-результаты из ./data/test/")
    args = parser.parse_args()

    segments_path = args.segments or os.path.join(
        f"data/test/" + f"segments_{args.suffix}.json"
    )

    run_test(segments_path=segments_path, suffix=args.suffix, compare=args.compare)
