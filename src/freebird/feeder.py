"""Detect a probably-empty feeder from a drop in bird activity."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from freebird.bot.telegram import TelegramBot
from freebird.config import TELEGRAM_CHAT_ID, local_today
from freebird.storage.database import Database

logger = logging.getLogger(__name__)

RECENT_DAYS = 3        # completed days compared against the baseline
BASELINE_DAYS = 14     # completed days before the recent window
MIN_BASELINE = 8.0     # birds/day needed before a drop means anything
DROP_RATIO = 0.25      # alert when recent activity falls to this share of baseline
COOLDOWN_DAYS = 7
REFILL_GRACE_DAYS = 4  # stay quiet this long after /refilled


def feeder_message(db: Database, today: str | None = None) -> str | None:
    """Return a nudge if activity has collapsed, else None. Pure read of the DB plus kv state."""
    today = today or local_today()
    now = datetime.now(timezone.utc)

    def recent(key: str, days: int) -> bool:
        v = db.kv_get(key)
        return bool(v) and now - datetime.fromisoformat(v) < timedelta(days=days)

    if recent("refilled_at", REFILL_GRACE_DAYS) or recent("feeder_alert_at", COOLDOWN_DAYS):
        return None

    counts = db.daily_bird_counts(RECENT_DAYS + BASELINE_DAYS + 1, today)
    day0 = datetime.strptime(today, "%Y-%m-%d")
    days = [(day0 - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, RECENT_DAYS + BASELINE_DAYS + 1)]
    recent_avg = sum(counts.get(d, 0) for d in days[:RECENT_DAYS]) / RECENT_DAYS
    base_avg = sum(counts.get(d, 0) for d in days[RECENT_DAYS:]) / BASELINE_DAYS
    if base_avg < MIN_BASELINE or recent_avg > DROP_RATIO * base_avg:
        return None
    return (
        f"Feeder check: only {recent_avg:.0f} bird visits a day over the last {RECENT_DAYS} days, "
        f"down from {base_avg:.0f} a day before that. It may be empty. "
        "Send /refilled once you've topped it up and I'll stop asking."
    )


async def check_feeder(bot: TelegramBot, db: Database) -> None:
    msg = feeder_message(db)
    if not msg:
        return
    await bot.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
    db.kv_set("feeder_alert_at", datetime.now(timezone.utc).isoformat())
    logger.info("Sent feeder-empty nudge")
