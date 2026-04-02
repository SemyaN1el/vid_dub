"""
Строит наглядный график сравнения моделей перевода по summary JSON.

Пример:
    py -3 tests/plot_translation_metrics.py \
        --input data/test/models_strategies_comparison_man_plus_qwen.json \
        --output data/test/models_strategies_comparison_man_plus_qwen_metrics.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _color_for_name(name: str) -> str:
    lowered = name.lower()
    if "nllb-600m" in lowered:
        return "#4C78A8"
    if "nllb-1.3b" in lowered:
        return "#F58518"
    if "qwen" in lowered:
        return "#E45756"
    if "gemini" in lowered:
        return "#B279A2"
    return "#72B7B2"


def _annotate_bars(ax, bars, values, decimals: int = 4) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def build_plot(input_path: Path, output_path: Path) -> None:
    summary = json.loads(input_path.read_text(encoding="utf-8"))
    items = sorted(
        summary.items(),
        key=lambda pair: pair[1]["labse"]["mean"],
        reverse=True
    )

    labels = [name for name, _ in items]
    means = [item["labse"]["mean"] for _, item in items]
    mins = [item["labse"]["min"] for _, item in items]
    maxs = [item["labse"]["max"] for _, item in items]
    times = [float(item.get("time", 0.0)) for _, item in items]
    colors = [_color_for_name(name) for name in labels]

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle("Сравнение моделей перевода по метрикам", fontsize=18, fontweight="bold")

    plots = [
        (axes[0, 0], means, "LaBSE Mean", 0.78, 0.90, 4),
        (axes[0, 1], mins, "LaBSE Min", 0.50, 0.70, 4),
        (axes[1, 0], maxs, "LaBSE Max", 0.93, 0.97, 4),
        (axes[1, 1], times, "Время, сек", 0.0, max(times) * 1.15 if times else 1.0, 1),
    ]

    for ax, values, title, ymin, ymax, decimals in plots:
        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
        _annotate_bars(ax, bars, values, decimals=decimals)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Построение графика метрик перевода")
    parser.add_argument(
        "--input",
        default="data/test/models_strategies_comparison_man_plus_qwen.json",
        help="Путь к summary JSON",
    )
    parser.add_argument(
        "--output",
        default="data/test/models_strategies_comparison_man_plus_qwen_metrics.png",
        help="Куда сохранить PNG",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    build_plot(input_path=input_path, output_path=output_path)
    print(output_path)


if __name__ == "__main__":
    main()
