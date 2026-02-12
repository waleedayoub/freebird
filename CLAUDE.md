# FreeBird

Bird feeder monitor: polls VicoHome cloud API for motion events, downloads video, identifies species via BirdNET, sends Telegram notifications.

## Quick Reference

```bash
uv sync                              # Install dependencies
uv run python -m freebird.main       # Run locally (needs .env)
docker compose up -d                  # Run in Docker
docker compose logs -f                # Tail logs
```

## Architecture

Single Python asyncio process running two concurrent tasks:
1. **Pipeline** (`pipeline.py`): poll VicoHome API -> dedupe -> download media -> BirdNET -> store -> Telegram notify
2. **Telegram bot** (`bot/telegram.py`): handles `/today`, `/stats`, `/lifers`, `/species` commands + Claude Q&A

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
    birdnet.py         # BirdNET wrapper: predict -> to_structured_array() -> best species
  storage/
    database.py        # SQLite WAL, sighting CRUD, lifer detection, query methods
  bot/
    telegram.py        # python-telegram-bot handlers + notification methods
    claude.py          # Anthropic API for freeform Q&A
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

## BirdNET API

- `birdnet.load("acoustic", "2.4", "tf")` — loads the TF model (downloads ~77MB on first run).
- `model.predict(path)` returns `AcousticFilePredictionResult`.
- Use `.to_structured_array()` to iterate results. **`.as_dict_list()` does not exist.**
- `species_name` field format: `"Scientific name_Common Name"` — split on `_` with maxsplit=1.
- Audio must be WAV. Pipeline extracts at 48kHz mono via ffmpeg.

## Environment Variables

Required in `.env` (see `.env.example`):
- `VICOHOME_EMAIL`, `VICOHOME_PASSWORD` — VicoHome account
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — Telegram bot
- `ANTHROPIC_API_KEY` — optional, for Claude Q&A

Optional:
- `FREEBIRD_DATA_DIR` — defaults to `/data` (Docker) or override for local dev
- `POLL_INTERVAL_SECONDS` — defaults to `15`
- `BIRDNET_CONFIDENCE_THRESHOLD` — defaults to `0.5`

## Docker

- `Dockerfile`: python:3.12-slim + ffmpeg + uv. Deps cached in separate layer.
- `compose.yaml`: named volume `freebird-data` at `/data` for SQLite + media persistence.
- Build & deploy: `docker compose build && docker compose up -d`

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
