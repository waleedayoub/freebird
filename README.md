# FreeBird

Bird feeder monitor that polls the VicoHome cloud API for motion events, identifies species with AI vision, and sends Telegram notifications.

## Setup

```bash
uv sync                              # Install dependencies
cp .env.example .env                  # Configure credentials
uv run python -m freebird.main       # Run locally
```

### Docker

```bash
docker compose up -d                  # Run in Docker
docker compose logs -f                # Tail logs
docker compose build && docker compose up -d  # Rebuild & restart
```

### Environment Variables

See `.env.example` for all options. Required:

| Variable | Description |
|----------|-------------|
| `VICOHOME_EMAIL` | VicoHome account email |
| `VICOHOME_PASSWORD` | VicoHome account password |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token from BotFather |
| `TELEGRAM_CHAT_ID` | Telegram chat to send notifications to |
| `VISION_MODEL` | PydanticAI model string (default: `google-gla:gemini-2.5-flash`) |
| `GOOGLE_API_KEY` | Google AI API key (for Gemini vision) |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional, for Claude Q&A chatbot) |

## How It Works

Single Python asyncio process running two concurrent tasks:

1. **Pipeline** -- polls VicoHome API every 15s for motion events, downloads keyshot images, runs AI vision to identify species, stores results in SQLite, sends Telegram notifications with lifer alerts.
2. **Telegram bot** -- handles `/today`, `/stats`, `/lifers`, `/show` commands and freeform Q&A via Claude.

## Vision

Species identification uses [PydanticAI](https://ai.pydantic.dev/) with structured output. The vision model is configurable via `VISION_MODEL`:

```bash
VISION_MODEL=google-gla:gemini-2.5-flash          # Gemini (default)
VISION_MODEL=anthropic:claude-sonnet-4-5-20250929  # Claude
VISION_MODEL=openai:gpt-4o                         # OpenAI
```

Prompts live as text files in `eval/prompts/`. Production uses `default.txt`.

## Eval Framework

Tools for comparing vision model accuracy against human-labeled ground truth.

### Label images

```bash
uv run python -m freebird.eval_label      # Opens web UI at localhost:8000
```

Web UI shows each keyshot image alongside a form to label it as bird (with species), critter, or empty. Labels save to `eval/ground_truth.json`.

### Run evals

```bash
# Test a model/prompt combo
uv run python -m freebird.eval_run --model google-gla:gemini-2.5-flash --prompt default

# Compare against another model
uv run python -m freebird.eval_run --model anthropic:claude-sonnet-4-5-20250929 --prompt default

# Review all results
cat eval/results.jsonl
```

Each run appends a summary line to `eval/results.jsonl` with accuracy scores.

### Backfill

Re-analyze all historical images with the current vision model:

```bash
uv run python -m freebird.vision_backfill          # Only missing
uv run python -m freebird.vision_backfill --rerun   # Re-do all
```

## Project Layout

```
src/freebird/
  main.py              # Entry point: asyncio loop, graceful shutdown
  config.py            # All env vars and paths
  pipeline.py          # Poll -> dedupe -> download -> analyze -> notify
  vicohome/            # VicoHome API client (auth, polling, models)
  media/               # ffmpeg: M3U8->MP4, image download, audio extraction
  analysis/
    vision.py          # PydanticAI vision agent, VisionAnalysis schema
    birdnet.py         # BirdNET audio species identification
  storage/
    database.py        # SQLite WAL, sighting CRUD, lifer detection
  bot/
    telegram.py        # Telegram bot commands + notifications
    claude.py          # Claude Q&A chatbot
  eval_label.py        # Web UI for labeling images (FastAPI)
  eval_run.py          # Eval runner (pydantic-evals)
  vision_backfill.py   # Batch re-analysis script
eval/
  prompts/             # Vision prompt templates
  ground_truth.json    # Human labels for eval
  results.jsonl        # Eval run history
```

## Requirements

- Python 3.12 (BirdNET/TensorFlow constraint)
- ffmpeg on PATH
- uv for dependency management
