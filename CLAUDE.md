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
2. **Telegram bot** (`bot/telegram.py`): handles `/today`, `/stats`, `/lifers`, `/show`, `/latest` commands + Claude Q&A

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
  eval_label.py        # The one labeling web UI (FastAPI): contested-first, species buttons, undo
  eval_run.py          # Eval runner: pydantic-evals harness with IsBirdCorrect + SpeciesMatch
  vision_backfill.py   # Batch re-analysis of historical images
eval/
  prompts/             # Vision prompt templates (default.txt, default_v2.txt)
  ground_truth.json    # Human-labeled eval dataset
  results.jsonl        # Eval run history (generated, not committed)
  details/             # Per-case eval details (generated, not committed)
```

## Key Constraints

- **Python 3.12** required. `.python-version` must stay at `3.12`.
- **ffmpeg** must be available on PATH (`brew install ffmpeg` locally).
- **No tests yet.** When adding tests, use `pytest` and mock the VicoHome API / Telegram bot (they require real credentials).
- **Single instance only.** Only one FreeBird process can run at a time — multiple instances fight over the Telegram bot `getUpdates` and VicoHome auth token.
- **VicoHome single-session auth.** Each login invalidates the previous token. Use a **separate VicoHome account** for FreeBird to avoid getting logged out of the mobile app.

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
- Claude 5.x models reject forced tool calls, so `vision.output_spec()` uses `PromptedOutput` for them. Use it everywhere (production and evals).
- **Second opinion + failover:** set `VISION_REVIEW_MODEL` to run a second model only on lifer candidates and low-confidence birds. If the two disagree, the lifer alert becomes a Telegram message with buttons (`rv:<sighting>:<n>`), and the tap is stored in `human_labels` (future gold labels). The same model is the failover when the primary errors.
- Species names are normalized by `species.py` (aliases, case, parentheticals) for lifer checks and scoring. It does not restrict what a model may say; fix naming in the prompt.
- Prompts: `default_v3.txt` adds naming rules and look-alike guidance; `default_v4.txt` requires a best guess instead of "Unidentified bird".
- `feeder.py` sends a daily 10:00 nudge when bird visits fall to under 25% of the prior two weeks. `/refilled` silences it for 4 days. State lives in the `kv` table.
- Polling catches up to 48 h after an outage (`kv.last_poll_ok`).
- Q&A (`bot/claude.py`) uses `QA_MODEL` (default `claude-sonnet-5-5`) with DB tools (search_sightings, species_counts, daily_activity, get_sighting, send_media).

## Eval Framework

- `eval_label.py`: the single labeling UI. `uv run python -m freebird.eval_label [--set sample]` then open `localhost:8000`. Labels a named set (`eval/sets/<set>.json`) and saves to `eval/sets/<set>_gold.json` (legacy `ground_truth` saves to `eval/ground_truth.json`). If model runs exist for the set, the most contested images come first with the models' species as buttons.
- `eval_run.py`: pydantic-evals harness. Evaluators: `IsBirdCorrect` (bird vs non-bird) and `SpeciesMatch` (species/animal_type).
- `SpeciesMatch` logic: empty frames check `animal_type is None`, birds use case-insensitive species match, non-birds use case-insensitive containment (e.g., "squirrel" matches `animal_type: "squirrel"` or `species: "Eastern Gray Squirrel"`).
- Unlabeled birds (`is_bird: true`, `label: ""`) are skipped during dataset construction.
- Results append to `eval/results.jsonl`; per-case details write to `eval/details/`.
- **Multi-model evals (Oct 2026):** `eval_compare.py` runs many models over a set (resumable, writes `eval/runs/<set>/<prompt>/<model>.jsonl`); `eval_score.py` scores against gold labels (`--gold`) or by agreement, and prints cost per image; `eval/sets/sample.json` is a 381-image stratified set labeled with `eval_label`. The 115-image set above is mostly juncos, squirrels and empty frames, so it overstates accuracy.

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
- `TIMEZONE` — defaults to `America/Toronto` (DST-aware via `zoneinfo`)
- `FREEBIRD_DATA_DIR` — defaults to `/data` (Docker) or override for local dev
- `POLL_INTERVAL_SECONDS` — defaults to `15`
- `VISION_REVIEW_MODEL` — optional second-opinion/failover model (empty = off)
- `QA_MODEL` — defaults to `claude-sonnet-5-5`

## Deployment

Production runs on `eleven` (Debian 13, headless 11" MacBook Air) as the systemd user unit `freebird.service` (`ssh eleven`; `systemctl --user restart freebird`; logs via `journalctl --user -u freebird`). Deploy = push to GitHub, then `git pull` and restart on eleven. The commands below are the old `screen` setup on `thirteen`, kept for reference.

```bash
# Start
ssh thirteen "screen -dmS freebird bash -c 'cd ~/development/freebird && uv run python -m freebird.main 2>&1 | tee freebird.log'"

# Check status
ssh thirteen "screen -ls"

# View logs
ssh thirteen "tail -50 ~/development/freebird/freebird.log"

# Attach to session (interactive)
ssh thirteen -t "screen -r freebird"

# Stop
ssh thirteen "screen -S freebird -X quit"

# Restart (stop + start)
ssh thirteen "screen -S freebird -X quit"; sleep 2
ssh thirteen "screen -dmS freebird bash -c 'cd ~/development/freebird && uv run python -m freebird.main 2>&1 | tee freebird.log'"
```

**Deploy workflow**: develop locally → push to GitHub → `ssh thirteen "cd ~/development/freebird && git pull"` → restart screen session.

### Docker (alternative, not currently used)

- `Dockerfile`: python:3.12-slim + ffmpeg + uv. Deps cached in separate layer. Copies `eval/prompts/` for prompt loading.
- `compose.yaml`: bind mount `./data` at `/data` for SQLite + media persistence.
- `.dockerignore`: excludes `data/`, `.venv/`, `.git/` to keep build context small.
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
