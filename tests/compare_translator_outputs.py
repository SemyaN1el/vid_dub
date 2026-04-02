"""
Сравнивает готовые pipeline-выходы двух переводчиков по сохранённым артефактам.

Сценарий:
    - в data/output уже лежат segments_<job>.json
    - текущие translated_segments_<job>.json / final_dubbing_<job>.wav
    - backup translated_segments_<job>.<tag>.json / final_dubbing_<job>.<tag>.wav

Пример:
    python tests/compare_translator_outputs.py ^
        --jobs man_tailfix speaking_skills_5min ^
        --current-label "Gemini 2.5 Flash per-segment" ^
        --backup-label "NLLB-1.3B per-segment" ^
        --backup-tag pre_gemini_backup
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config as cfg
from src.metrics import (
    compute_labse_similarity,
    compute_wer_cer,
    speaker_verification_score,
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_metrics(
    model_asr,
    segments_path: Path,
    translated_path: Path,
    speaker_ref_path: Path,
    final_voice_path: Path,
) -> Dict[str, Any]:
    segments = _load_json(segments_path)
    translated = _load_json(translated_path)
    wer_cer = compute_wer_cer(model_asr, str(final_voice_path), translated)
    labse = compute_labse_similarity(segments, translated)

    return {
        "speaker_verification": speaker_verification_score(
            str(speaker_ref_path),
            str(final_voice_path),
        ),
        "wer": wer_cer.get("wer"),
        "cer": wer_cer.get("cer"),
        "labse_mean": labse.get("mean"),
        "labse_min": labse.get("min"),
        "labse_max": labse.get("max"),
        "labse_n": len(labse.get("per_segment", [])),
        "translated_segments_path": str(translated_path),
        "final_voice_path": str(final_voice_path),
    }


def _plot_results(results: Dict[str, Any], output_path: Path) -> None:
    metric_specs = [
        ("labse_mean", "LaBSE mean", True),
        ("speaker_verification", "Speaker Verification", True),
        ("wer", "WER", False),
        ("cer", "CER", False),
    ]

    jobs = list(results["jobs"].keys())
    labels = list(next(iter(results["jobs"].values()))["runs"].keys())
    colors = ["#6C5CE7", "#F39C12", "#2E86DE", "#27AE60"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (metric_key, title, higher_is_better) in zip(axes, metric_specs):
        x = range(len(jobs))
        width = 0.35

        for idx, label in enumerate(labels):
            values = [
                results["jobs"][job]["runs"][label][metric_key]
                for job in jobs
            ]
            positions = [p + (idx - 0.5) * width for p in x]
            bars = ax.bar(
                positions,
                values,
                width=width,
                label=label,
                color=colors[idx % len(colors)],
                edgecolor="black",
                linewidth=0.6,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(jobs, rotation=12)
        ax.grid(True, axis="y", alpha=0.25)

        if metric_key == "labse_mean":
            ax.set_ylim(0.82, 0.91)
        elif metric_key == "speaker_verification":
            ax.set_ylim(0.84, 0.90)
        elif metric_key == "wer":
            ax.set_ylim(0.10, 0.20)
        elif metric_key == "cer":
            ax.set_ylim(0.03, 0.08)

        winner_direction = "↑" if higher_is_better else "↓"
        ax.set_ylabel(f"Лучше {winner_direction}")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(labels))
    fig.suptitle("Сравнение переводчиков по итоговым pipeline-метрикам", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение готовых output-артефактов двух переводчиков")
    parser.add_argument("--jobs", nargs="+", required=True, help="Имена заданий, например: man_tailfix speaking_skills_5min")
    parser.add_argument("--base-dir", default="data/output", help="Корневая директория output-артефактов")
    parser.add_argument("--backup-tag", default="pre_gemini_backup", help="Суффикс backup-версии перед расширением")
    parser.add_argument("--current-label", default="Current translator", help="Подпись для текущих output-файлов")
    parser.add_argument("--backup-label", default="Backup translator", help="Подпись для backup-файлов")
    parser.add_argument("--output-json", default="data/test/translator_outputs_comparison.json", help="Куда сохранить JSON")
    parser.add_argument("--output-plot", default="data/test/translator_outputs_comparison.png", help="Куда сохранить PNG")
    args = parser.parse_args()

    import whisper

    model_asr = whisper.load_model(cfg.WHISPER_MODEL_NAME).to(cfg.DEVICE)

    try:
        base_dir = Path(args.base_dir)
        results: Dict[str, Any] = {
            "jobs": {},
            "labels": {
                "current": args.current_label,
                "backup": args.backup_label,
            },
            "config": {
                "whisper_model": cfg.WHISPER_MODEL_NAME,
                "device": cfg.DEVICE,
                "backup_tag": args.backup_tag,
            },
        }

        for job in args.jobs:
            segments_path = base_dir / f"segments_{job}.json"
            speaker_ref_path = base_dir / "temp" / f"speaker_ref_{job}.wav"
            current_translated = base_dir / f"translated_segments_{job}.json"
            current_final_voice = base_dir / f"final_dubbing_{job}.wav"
            backup_translated = base_dir / f"translated_segments_{job}.{args.backup_tag}.json"
            backup_final_voice = base_dir / f"final_dubbing_{job}.{args.backup_tag}.wav"

            required_paths = [
                segments_path,
                speaker_ref_path,
                current_translated,
                current_final_voice,
                backup_translated,
                backup_final_voice,
            ]
            missing = [str(path) for path in required_paths if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Не найдены обязательные артефакты для job='{job}': {missing}"
                )

            results["jobs"][job] = {
                "runs": {
                    args.current_label: _run_metrics(
                        model_asr=model_asr,
                        segments_path=segments_path,
                        translated_path=current_translated,
                        speaker_ref_path=speaker_ref_path,
                        final_voice_path=current_final_voice,
                    ),
                    args.backup_label: _run_metrics(
                        model_asr=model_asr,
                        segments_path=segments_path,
                        translated_path=backup_translated,
                        speaker_ref_path=speaker_ref_path,
                        final_voice_path=backup_final_voice,
                    ),
                }
            }

        aggregate: Dict[str, Dict[str, float]] = {}
        metric_names = ("labse_mean", "speaker_verification", "wer", "cer")
        for label in (args.current_label, args.backup_label):
            aggregate[label] = {}
            for metric_name in metric_names:
                values = [
                    results["jobs"][job]["runs"][label][metric_name]
                    for job in args.jobs
                ]
                aggregate[label][metric_name] = sum(values) / len(values)
        results["aggregate_mean"] = aggregate

        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        _plot_results(results, Path(args.output_plot))
        print(output_json)
    finally:
        del model_asr


if __name__ == "__main__":
    main()
