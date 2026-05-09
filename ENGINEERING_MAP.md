# Инженерная карта проекта

## 1. Назначение системы

Проект решает задачу автоматического дубляжа видео с сохранением голосовой идентичности исходного спикера:

1. извлекает аудио из исходного видео;
2. отделяет голос от фона;
3. распознает речь и режет ее на сегменты;
4. переводит сегменты на русский;
5. синтезирует новый голос в темпе, близком к исходному;
6. микширует дубляж с фоном и собирает финальное видео;
7. генерирует субтитры;
8. считает метрики качества.

Кодовая база остается исследовательской, но после стабилизации у нее уже есть воспроизводимый entrypoint, шаблон конфигурации, файл зависимостей, unit-тесты и единая структура артефактов по `job_name`.

## 2. Верхнеуровневая архитектура

```mermaid
flowchart TD
    A["input video"] --> B["preprocess"]
    B --> C["temp/original_extracted_audio.wav"]
    B --> D["temp/vocals.wav"]
    B --> E["temp/background.wav"]
    D --> F["temp/vocals_processed.wav"]
    F --> G["asr"]
    G --> H["segments.json"]
    G --> I["temp/speaker_ref.wav"]
    G --> J["temp/speaker_profile.json"]
    H --> K["translate"]
    K --> L["translated_segments.json"]
    L --> M["tts"]
    I --> M
    J --> M
    M --> N["final_dubbing.wav"]
    N --> O["postprocess"]
    E --> O
    C --> O
    O --> P["final_mix.wav"]
    P --> Q["final_video.mp4"]
    Q --> R["subtitles"]
    L --> R
    H --> S["metrics"]
    L --> S
    I --> S
    N --> S
```

Артефакты каждого запуска лежат в отдельной папке:

```text
data/output/<job-name>/
  segments.json
  translated_segments.json
  final_dubbing.wav
  final_mix.wav
  final_video.mp4
  metrics.json
  subtitles/
  temp/
```

В тестовом режиме используется тот же layout внутри `data/test/<job-name>/`.

## 3. Карта модулей

### Точка входа

- `main.py`
  - Оркестрация шагов `preprocess`, `asr`, `translate`, `tts`, `postprocess`, `subtitles`, `metrics`, `prepare_finetune`.
  - Разрешение входного видео через `--video`, legacy `--suffix` или единственное видео в `data/input/`.
  - Построение путей через `utils/pipeline_io.py`.
  - Preflight-проверка окружения через `--check-env`.

- `scripts/smoke_pipeline.py`
  - Запуск короткого `--test` smoke-run через `main.py --step all`.
  - Валидация ключевых артефактов, JSON-контрактов, subtitle manifest и метрик.
  - Режим `--skip-run` для быстрой проверки уже готовой папки `data/test/<job-name>/`.

### Конфигурация

- `config.example.py`
  - Шаблон локального `config.py`.
  - Пути, ASR/MT/TTS/SmartSync/subtitle/fine-tune параметры.
- `config.py`
  - Локальная конфигурация, не хранится в git.

### Доменные модули

- `src/preprocessing.py`
  - `ffmpeg` для извлечения аудио.
  - `demucs` для source separation.
  - `pydub` + `noisereduce` для подготовки вокала под ASR.

- `src/asr_backend.py`
  - Выбор ASR backend: локальный Whisper или Groq-compatible API.
  - Отдельные загрузчики для основного ASR и ASR в метриках.

- `src/asr.py`
  - Сегментация по словам и паузам.
  - Формирование `words`, `words_with_silence`, длительностей и пауз.
  - Создание `speaker_ref.wav`, `speaker_refs/` и `speaker_profile.json`.

- `src/translation.py`
  - NLLB и Gemini backend.
  - Стратегии `per-segment`, `sentence-level`, `sliding-window`, `context-aware`.
  - SmartSync rewrite backend для TTS.

- `src/tts.py`
  - XTTS synthesis orchestration.
  - SmartSync/TTS retry orchestration и финальная сборка сегментов.
  - Сериализация TTS-таймингов обратно в `translated_segments.json`.

- `src/tts_audio.py`
  - Оценка active speech dBFS.
  - Segment level matching и локальная подстройка target dBFS по source vocals.
  - Финальная компрессия и peak ceiling.

- `src/tts_guards.py`
  - Cheap tail guard и safe tail trim.
  - Babble guard и ASR retry scoring.
  - Общие recognition helpers для SmartSync и TTS retry.

- `src/tts_timing.py`
  - Расчет timing window для сегмента.
  - Оценка эффективной речевой длительности без краевых пауз.
  - Сдвиг следующего сегмента с учетом лимита.

- `src/tts_routing.py`
  - Оценка соответствия reference-клипа текущему сегменту.
  - Выбор per-segment reference paths из `speaker_profile`.

- `src/tts_text.py`
  - Text cleanup для TTS.
  - Построение retry-вариантов текста.
  - Grouping соседних TTS-сегментов.

- `src/tts_backends.py`
  - XTTS backend factory и backend wrapper.

- `src/postprocessing.py`
  - Наложение дубляжа на фон и оригинальную дорожку.
  - Сборка финального видео через `ffmpeg`.

- `src/subtitles.py`
  - Генерация `SRT`, `VTT`, `ASS`.
  - Soft/hard subtitle embedding.
  - Подключено в `main.py` как шаг `subtitles` и часть `all`.

- `src/metrics.py`
  - Speaker verification через `resemblyzer`.
  - WER/CER через ASR + `jiwer`.
  - Семантическое сходство перевода через LaBSE.

### Утилиты

- `utils/pipeline_io.py`
  - `job_name` normalization.
  - Поиск входных видео.
  - Сборка путей `data/output/<job-name>/` и `data/test/<job-name>/`.

- `utils/helpers.py`
  - Seed, управление директориями, очистка GPU-памяти, нормализация путей.

### Эксперименты и тесты

- `experiments/`
  - Offline benchmark-скрипты сравнения переводчиков и стратегий.
  - Это не unit-тесты и не должны запускаться как быстрый CI-контур.

- `tests/unit/`
  - Быстрые unit-тесты для ASR metadata, pipeline paths, subtitles, TTS serialization и вынесенных TTS helpers.

## 4. Контракты между шагами

### `preprocess`

Вход:

- `--video <path>`;
- или единственное видео в `data/input/`;
- или legacy `data/input/video_<suffix>.<ext>`.

Выход:

- `temp/original_extracted_audio.wav`
- `temp/vocals.wav`
- `temp/background.wav`
- `temp/vocals_processed.wav`

### `asr`

Вход:

- `temp/vocals_processed.wav`

Выход:

- `segments.json`
- `temp/speaker_ref.wav`
- `temp/speaker_profile.json`
- `temp/speaker_refs/`

Контракт сегмента:

```json
{
  "text": "source text",
  "start": 12.34,
  "end": 15.67,
  "speaker_id": "spk_0",
  "words": [],
  "words_with_silence": [],
  "source_duration_sec": 3.33,
  "source_word_count": 5,
  "pause_before_sec": 0.2,
  "pause_after_sec": 0.4
}
```

### `translate`

Вход:

- `segments.json`

Выход:

- `translated_segments.json`

Контракт переведенного сегмента:

```json
{
  "text": "translated text",
  "original_text": "source text",
  "start": 12.34,
  "end": 15.67,
  "speaker_id": "spk_0"
}
```

Дополнительные поля могут появляться в зависимости от стратегии, например `merged_count`.

### `tts`

Вход:

- `translated_segments.json`
- `temp/speaker_ref.wav`
- опционально `temp/speaker_profile.json`

Выход:

- `final_dubbing.wav`
- обновленный `translated_segments.json` с TTS-таймингами;
- временные сегменты в `temp/audio_segments/`.

Важно: runtime-аудиообъекты не сериализуются; `_serialize_tts_segments` выкидывает `corrected_audio`.

### `postprocess`

Вход:

- `final_dubbing.wav`
- `temp/background.wav`
- `temp/original_extracted_audio.wav`

Выход:

- `final_mix.wav`
- `final_video.mp4`

### `subtitles`

Вход:

- `final_video.mp4`
- `translated_segments.json`

Выход:

- `subtitles/subtitles.srt`
- `subtitles/subtitles.vtt`
- `subtitles/subtitles.ass`
- `subtitles/subtitles_manifest.json`
- soft/hard video artifact в зависимости от `--subtitle-mode`.

### `metrics`

Вход:

- `temp/speaker_ref.wav`
- `final_dubbing.wav`
- `segments.json`
- `translated_segments.json`

Выход:

- `metrics.json`
- печать сводки в stdout;
- график LaBSE через matplotlib.

## 5. Внешние зависимости

### Python

Основной список закреплен в `requirements.txt`: `torch`, `openai-whisper`, `transformers`, `TTS`, `demucs`, `soundfile`, `pydub`, `noisereduce`, `sentence-transformers`, `jiwer`, `resemblyzer`, `pytest` и сопутствующие библиотеки.

### System CLI

- `ffmpeg` в `PATH`;
- `demucs` CLI в активном Python-окружении.

### Local model files

`original_tts_model/` должен содержать минимум:

```text
config.json
model.pth
vocab.json
speakers_xtts.pth
```

## 6. Текущий статус стабилизации

Уже сделано:

- добавлен `README.md` с runbook;
- добавлен `requirements.txt`;
- добавлен `config.example.py`;
- добавлен `python main.py --check-env`;
- добавлен `python scripts/smoke_pipeline.py`;
- подключен шаг `subtitles`;
- введена структура `data/output/<job-name>/`;
- удалены tracked `__pycache__`;
- benchmark-скрипты перенесены в `experiments/`;
- быстрые тесты находятся в `tests/unit/`;
- TTS runtime-настройки сгруппированы в dataclass-конфиги;
- text cleanup/grouping вынесены из `src/tts.py` в `src/tts_text.py`;
- reference routing вынесен из `src/tts.py` в `src/tts_routing.py`;
- timing/window logic вынесена из `src/tts.py` в `src/tts_timing.py`;
- tail/babble guards вынесены из `src/tts.py` в `src/tts_guards.py`;
- audio level/compression helpers вынесены из `src/tts.py` в `src/tts_audio.py`;
- smoke-run на коротком видео был успешно пройден перед audio-refactor;
- smoke-run формализован как отдельный скрипт с проверкой артефактов.

## 7. Текущие инженерные риски

### P1. `src/tts.py` слишком крупный

В одном файле все еще смешаны SmartSync, TTS retry и финальная сборка аудио. Настройки, text cleanup/grouping, reference routing, timing/window logic, guards и audio helpers уже вынесены отдельно, поэтому следующий безопасный шаг - продолжать дробление по зонам ответственности небольшими коммитами.

### P1. Мало тестов вокруг TTS-контрактов

Есть тесты сериализации TTS-сегментов, text/grouping helpers, routing helpers, timing-window rules, guards, audio level/compression helpers и smoke artifact validation. Следующий пробел - SmartSync acceptance edge cases и регулярный прогон smoke-check перед значимыми изменениями.

### P1. Эксперименты не формализованы как воспроизводимый benchmark

Скрипты лежат в `experiments/`, но их входные данные и ожидаемые результаты пока не описаны как единый сценарий.

### P2. Vendor-мусор в репозитории

`vendor_latex2mathml/` и похожие большие локальные папки лучше чистить отдельным коммитом. Это репозиторная гигиена, а не изменение pipeline-кода.

### P2. Конфигурация пока глобальная

`main.py` читает `config.py` напрямую. TTS-настройки уже передаются как config-объекты, но общий подход к конфигурации пока остается глобальным.

## 8. Следующая программа работ

### Этап 1. TTS config refactor

Статус: выполнено.

1. Введены dataclass-конфиги:
   - `TTSRuntimeConfig`
   - `SmartSyncConfig`
   - `TailGuardConfig`
   - `SegmentRoutingConfig`
   - `AudioLevelConfig`
2. Эти объекты передаются из `main.py` в `synthesize_segments_with_timing`.
3. Алгоритмы TTS не менялись.

### Этап 2. Дробление `src/tts.py`

После dataclass-конфигов выносить блоки небольшими коммитами:

- text cleanup/grouping - выполнено, `src/tts_text.py`;
- reference routing - выполнено, `src/tts_routing.py`;
- timing/window logic - выполнено, `src/tts_timing.py`;
- tail/babble guards - выполнено, `src/tts_guards.py`;
- audio level/compression helpers - выполнено, `src/tts_audio.py`.

### Этап 3. Репозиторная чистка

Отдельно решить судьбу:

- `vendor_latex2mathml/`;
- крупных локальных папок;
- неиспользуемых черновиков, если они появятся в tracked-файлах.

## 9. Практический вывод

Проект уже имеет рабочее ядро дубляжа, воспроизводимый запуск, базовые тесты и отдельный smoke-check. Основные TTS helper-зоны вынесены из `src/tts.py`; дальше разумнее улучшать качество запусков, отчеты и UX пайплайна, а не продолжать дробление ради дробления.
