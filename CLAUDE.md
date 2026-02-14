# FreeBird

Bird feeder monitor: polls VicoHome cloud API for motion events, downloads keyshot images, identifies species via AI vision (Gemini/OpenAI/Claude), sends Telegram notifications.

## Quick Reference

```bash
uv sync                              # Install dependencies
uv run python -m freebird.main       # Run locally (needs .env)
docker compose up -d                  # Run in Docker
docker compose logs -f                # Tail logs
```

## Architecture

Single Python asyncio process running two concurrent tasks:
1. **Pipeline** (`pipeline.py`): poll VicoHome API -> dedupe -> download media -> vision analysis -> store -> Telegram notify
2. **Telegram bot** (`bot/telegram.py`): handles `/today`, `/stats`, `/lifers`, `/show` commands + Claude Q&A

All components share in-memory references (no IPC). Entry point: `main.py` starts both via `asyncio`.

## Project Layout

```
src/freebird/
  main.py              # Entry point: asyncio loop, graceful shutdown
  config.py            # All env vars and paths (single source of truth)
  pipeline.py          # Poll -> dedupe -> download -> analyze -> notify
  vicohome/
    auth.py            # JWT login, 23h token cache, auto-refresh on -1024..-1027
    api.py             # Event polling with retry, _request() handles auth errors
    models.py          # Pydantic models: MotionEvent, SubcategoryInfo, Keyshot
  media/
    downloader.py      # Async ffmpeg: M3U8->MP4, image download, audio extraction
  analysis/
    vision.py          # PydanticAI vision agent, VisionAnalysis schema, prompt loading
    birdnet.py         # BirdNET audio species identification (legacy, not in pipeline)
  storage/
    database.py        # SQLite WAL, sighting CRUD, lifer detection, query methods
  bot/
    telegram.py        # python-telegram-bot handlers + notification methods
    claude.py          # Anthropic API for freeform Q&A
  eval_label.py        # Web UI for labeling keyshot images (FastAPI)
  eval_run.py          # Eval runner: pydantic-evals harness with IsBirdCorrect + SpeciesMatch
  vision_backfill.py   # Batch re-analysis of historical images
eval/
  prompts/             # Vision prompt templates (default.txt, default_v2.txt)
  ground_truth.json    # Human-labeled eval dataset
  results.jsonl        # Eval run history (generated, not committed)
  details/             # Per-case eval details (generated, not committed)
```

## Key Constraints

- **Python <=3.12** required. BirdNET depends on TensorFlow which has no 3.13+ wheels. `.python-version` must stay at `3.12`.
- **ffmpeg** must be available on PATH (installed in Docker image, `brew install ffmpeg` locally).
- **No tests yet.** When adding tests, use `pytest` and mock the VicoHome API / Telegram bot (they require real credentials).

## VicoHome API Gotchas

- Auth token goes in `Authorization` header with **no `Bearer` prefix** — just the raw JWT.
- `startTimestamp`/`endTimestamp` must be **strings**, not ints.
- Event list endpoint uses `"code"` field for status; login uses `"result"`. Both must be checked.
- Auth error codes -1024 to -1027 trigger one retry with fresh token (see `auth.py:AUTH_ERROR_CODES`).

## Vision

- Species ID uses PydanticAI with structured output (`VisionAnalysis` Pydantic model).
- Default model: `google-gla:gemini-3-flash-preview` (87% species accuracy on 115 labeled images).
- Prompts are text files in `eval/prompts/`. Production uses `default_v2.txt`.
- `load_prompt(name)` reads `eval/prompts/{name}.txt` and interpolates `{location}` from `FEEDER_LOCATION`.
- Supported model prefixes: `google-gla:` (Gemini), `openai:` (OpenAI), `anthropic:` (Claude).

## Eval Framework

- `eval_label.py`: FastAPI web UI at `localhost:8000` for labeling keyshots. Saves to `eval/ground_truth.json`.
- `eval_run.py`: pydantic-evals harness. Evaluators: `IsBirdCorrect` (bird vs non-bird) and `SpeciesMatch` (species/animal_type).
- `SpeciesMatch` logic: empty frames check `animal_type is None`, birds use case-insensitive species match, non-birds use case-insensitive containment (e.g., "squirrel" matches `animal_type: "squirrel"` or `species: "Eastern Gray Squirrel"`).
- Unlabeled birds (`is_bird: true`, `label: ""`) are skipped during dataset construction.
- Results append to `eval/results.jsonl`; per-case details write to `eval/details/`.

### Model Accuracy (115 labeled images)

| Model | Prompt | Bird/Non-bird | Species |
|-------|--------|:---:|:---:|
| Gemini 2.5 Flash | default_v2 | 91.3% | 75.7% |
| Gemini 3 Flash Preview | default_v2 | 93.9% | **87.0%** |
| GPT-5 Mini | default_v2 | **97.4%** | 77.4% |

## BirdNET (Legacy)

BirdNET audio analysis is no longer in the pipeline (replaced by vision). The module remains for reference.

- `birdnet.load("acoustic", "2.4", "tf")` — loads the TF model (downloads ~77MB on first run).
- `model.predict(path)` returns `AcousticFilePredictionResult`.
- Use `.to_structured_array()` to iterate results. **`.as_dict_list()` does not exist.**
- `species_name` field format: `"Scientific name_Common Name"` — split on `_` with maxsplit=1.

## Environment Variables

Required in `.env` (see `.env.example`):
- `VICOHOME_EMAIL`, `VICOHOME_PASSWORD` — VicoHome account
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — Telegram bot
- `GOOGLE_API_KEY` — Google AI API key (for Gemini vision)
- `ANTHROPIC_API_KEY` — optional, for Claude Q&A

Optional:
- `VISION_MODEL` — defaults to `google-gla:gemini-3-flash-preview`
- `VISION_PROMPT` — defaults to `default_v2`
- `FEEDER_LOCATION` — e.g. `Toronto, Ontario, Canada` (used in vision prompt)
- `FREEBIRD_DATA_DIR` — defaults to `/data` (Docker) or override for local dev
- `POLL_INTERVAL_SECONDS` — defaults to `15`

## Docker

- `Dockerfile`: python:3.12-slim + ffmpeg + uv. Deps cached in separate layer. Copies `eval/prompts/` for prompt loading.
- `compose.yaml`: bind mount `./data` at `/data` for SQLite + media persistence.
- Build & deploy: `docker compose build && docker compose up -d`
- Remote deploy: `docker context create macbook --docker "host=ssh://user@host"` then `docker compose --context macbook up -d --build`

## Plan Files

- **NEVER overwrite the original plan file** (`.claude/plans/vectorized-wandering-snail.md`). It is the master architecture document.
- Always create **new plan files** in `.claude/plans/` for new tasks (e.g., `fix-vision-accuracy.md`, `vision-backfill.md`).

## Conventions

- All modules use `from __future__ import annotations` for modern type syntax.
- Pydantic models use `Field(alias=...)` with `populate_by_name=True` for camelCase API responses.
- Async operations (ffmpeg, network) use `asyncio.create_subprocess_exec`, not `subprocess.run`.
- Database uses raw SQL with `sqlite3` (not an ORM). WAL mode for concurrent reads.
- Logging via stdlib `logging`, not print statements.
- Config is module-level constants in `config.py`, not a settings class.
