from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

# VicoHome
VICOHOME_EMAIL: str = os.environ["VICOHOME_EMAIL"]
VICOHOME_PASSWORD: str = os.environ["VICOHOME_PASSWORD"]
VICOHOME_REGION: str = os.getenv("VICOHOME_REGION", "us")
VICOHOME_API_BASE: str | None = os.getenv("VICOHOME_API_BASE")

# Telegram
TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID: str = os.environ["TELEGRAM_CHAT_ID"]

# Claude
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# Paths — /data inside Docker, ./data locally
_default_data = "/data" if Path("/data").exists() else str(Path(__file__).resolve().parents[2] / "data")
DATA_DIR: Path = Path(os.getenv("FREEBIRD_DATA_DIR", _default_data))
DB_PATH: Path = DATA_DIR / "birds.db"
MEDIA_DIR: Path = DATA_DIR / "media"

# Polling
POLL_INTERVAL_SECONDS: int = int(os.getenv("POLL_INTERVAL_SECONDS", "15"))

# BirdNET
BIRDNET_CONFIDENCE_THRESHOLD: float = float(os.getenv("BIRDNET_CONFIDENCE_THRESHOLD", "0.5"))

# Vision
VISION_MODEL: str = os.getenv("VISION_MODEL", "google-gla:gemini-3-flash-preview")
VISION_PROMPT: str = os.getenv("VISION_PROMPT", "default_v2")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# Feeder location and timezone
FEEDER_LOCATION: str = os.getenv("FEEDER_LOCATION", "your city")
TIMEZONE: ZoneInfo = ZoneInfo(os.getenv("TIMEZONE", "America/Toronto"))


def format_local_time(utc_str: str, fmt: str = "%I:%M %p") -> str:
    """Convert a UTC ISO timestamp string to local time display."""
    dt = datetime.fromisoformat(utc_str).astimezone(TIMEZONE)
    return dt.strftime(fmt).lstrip("0")


def local_today() -> str:
    """Return today's date in local timezone as YYYY-MM-DD."""
    return datetime.now(TIMEZONE).strftime("%Y-%m-%d")

# API region mapping
API_BASES: dict[str, str] = {
    "us": "https://api-us.vicohome.io",
    "eu": "https://api-eu.vicoo.tech",
}


def get_api_base() -> str:
    if VICOHOME_API_BASE:
        return VICOHOME_API_BASE.rstrip("/")
    return API_BASES.get(VICOHOME_REGION.lower(), API_BASES["us"])


def get_country_no() -> str:
    base = get_api_base()
    if "-eu" in base or "vicoo.tech" in base:
        return "EU"
    return "US"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
