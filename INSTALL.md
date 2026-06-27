# Installation

This file describes the local source checkout as a runnable software prototype.
The public showcase repository can intentionally omit the source code.

## Prerequisites

- Windows 10/11 or Linux.
- Python 3.10 or 3.11 recommended. Python 3.12 can work for API-only paths, but the local XTTS stack is usually safer on 3.10/3.11.
- FFmpeg available in `PATH`.
- A CUDA-capable GPU is recommended for local ASR/XTTS runs.

## Create Environment

```powershell
cd C:\Users\SemyaNiEl\Desktop\video_dubbing
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If PyTorch needs a specific CUDA build, install it first using the command from
the official PyTorch selector, then run `pip install -r requirements.txt`.

## Configure

```powershell
Copy-Item config.example.py config.py
```

For the current online configuration set these environment variables in the same
PowerShell session or in the system environment:

```powershell
$env:ROUTERAI_API_KEY = "<routerai-key>"
$env:MT_PROVIDER = "openai_compatible"
$env:MT_MODEL_NAME = "openai/gpt-5.4-mini"
$env:MT_STRATEGY = "sentence-boundary-aware"
$env:MT_OPENAI_API_KEY_ENV = "ROUTERAI_API_KEY"
$env:MT_OPENAI_BASE_URL = "https://<routerai-openai-compatible-endpoint>/v1"
$env:TTS_PROVIDER = "elevenlabs"
$env:ELEVENLABS_API_KEY = "<elevenlabs-key>"
$env:SMART_SYNC_ENABLED = "1"
$env:SMART_SYNC_PROVIDER = "openai_compatible"
$env:SMART_SYNC_MODEL_NAME = "openai/gpt-5.4-mini"
$env:SMART_SYNC_API_KEY_ENV = "ROUTERAI_API_KEY"
$env:SMART_SYNC_BASE_URL = "https://<routerai-openai-compatible-endpoint>/v1"
```

`.env.example` is a reference file only; the pipeline reads real environment
variables and does not auto-load `.env` files.

## Verify Installation

Fast product smoke:

```powershell
python scripts/product_smoke.py
```

Environment and dependency doctor:

```powershell
python main.py --doctor --video .\data\input\smoke_20s.mp4
python main.py --show-config
```

Unit tests used by the packaging smoke:

```powershell
pytest tests/unit/test_pipeline_io.py tests/unit/test_config_snapshot.py tests/unit/test_pipeline_resume.py tests/unit/test_reporting.py tests/unit/test_smoke_pipeline.py
```

## Run

Online RouterAI + ElevenLabs path:

```powershell
python main.py --step all --video .\data\input\talk.mp4 --job-name talk_ru --mt-provider openai_compatible --mt-model openai/gpt-5.4-mini --mt-strategy sentence-boundary-aware --tts-provider elevenlabs --subtitle-mode hard
```

For a user-facing dubbing result without the research metrics pass, add
`--skip-metrics`. The Desktop and web UI enable this option by default:

```powershell
python main.py --step all --skip-metrics --video .\data\input\talk.mp4 --job-name talk_ru --mt-provider openai_compatible --mt-model openai/gpt-5.4-mini --mt-strategy sentence-boundary-aware --tts-provider elevenlabs --subtitle-mode hard
```

An explicit `python main.py --step metrics ...` still calculates metrics for an
existing job when evaluation is needed.

Local fallback path:

```powershell
python main.py --step all --video .\data\input\talk.mp4 --job-name talk_ru --mt-provider hf --mt-model facebook/nllb-200-distilled-1.3B --mt-strategy per-segment --tts-provider xtts
```

Desktop wrapper:

```powershell
python desktop/app.py
```

## Full Smoke Pipeline

The full smoke run needs a short input video and either local models or API
credentials. It writes only to `data/test/<job-name>`.

```powershell
python scripts/smoke_pipeline.py --video .\data\input\smoke_20s.mp4 --job-name smoke_online --mt-provider openai_compatible --mt-model openai/gpt-5.4-mini --mt-strategy sentence-boundary-aware --tts-provider elevenlabs
```

To validate an existing smoke output without rerunning the pipeline:

```powershell
python scripts/smoke_pipeline.py --skip-run --job-name smoke_online --tts-provider elevenlabs
```

## Docker Online CLI

The online Docker profile packages the CLI pipeline, FFmpeg, local Whisper/Demucs
preprocessing, API-based translation, ElevenLabs TTS, SmartSync, and smoke
checks. It is headless; the PySide desktop window is for local Windows runs.

Create a local `.env` from `.env.example` and fill in API keys/base URLs:

```powershell
Copy-Item .env.example .env
notepad .env
```

Build the image:

```powershell
docker compose build dub
```

Run product smoke inside the container:

```powershell
docker compose run --rm dub scripts/product_smoke.py
```

Show the effective configuration:

```powershell
docker compose run --rm dub main.py --show-config
```

Run the current online configuration:

```powershell
docker compose run --rm dub main.py --step all --video /app/data/input/talk.mp4 --job-name talk_ru --mt-provider openai_compatible --mt-model openai/gpt-5.4-mini --mt-strategy sentence-boundary-aware --tts-provider elevenlabs --subtitle-mode hard
```

Start the FastAPI service:

```powershell
docker compose up api
```

Open the web UI:

```text
http://localhost:8000/ui
```

Check the API:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Create a job for a video already mounted under `./data/input`:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/jobs/from-path `
  -ContentType "application/json" `
  -Body '{"video_path":"/app/data/input/talk.mp4","options":{"job_name":"talk_ru","mt_provider":"openai_compatible","mt_model":"openai/gpt-5.4-mini","mt_strategy":"sentence-boundary-aware","tts_provider":"elevenlabs","subtitle_mode":"hard"}}'
```

Upload a video directly to the API:

```powershell
curl.exe -X POST http://localhost:8000/jobs `
  -F "video=@data/input/talk.mp4" `
  -F "job_name=talk_ru" `
  -F "mt_provider=openai_compatible" `
  -F "mt_model=openai/gpt-5.4-mini" `
  -F "mt_strategy=sentence-boundary-aware" `
  -F "tts_provider=elevenlabs" `
  -F "subtitle_mode=hard"
```

The image does not contain videos, generated artifacts, API keys, or local model
weights. They are mounted through `./data`, `./models`, and
`./original_tts_model`.
