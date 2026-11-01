from __future__ import annotations

import asyncio
import logging
import signal
from datetime import datetime, time, timedelta, timezone

from freebird.bot.telegram import TelegramBot
from freebird.config import TIMEZONE, ensure_dirs
from freebird.feeder import check_feeder
from freebird.pipeline import Pipeline
from freebird.storage.database import Database
from freebird.vicohome.api import VicoHomeAPI
from freebird.vicohome.auth import AuthManager

DAILY_SUMMARY_TIME = time(18, 0)  # 6:00 PM local
FEEDER_CHECK_TIME = time(10, 0)   # 10:00 AM local

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# httpx logs full request URLs at INFO, which include the Telegram bot token
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("freebird")


async def _run() -> None:
    ensure_dirs()

    # Initialize components
    logger.info("Initializing FreeBird...")
    auth = AuthManager()
    api = VicoHomeAPI(auth)
    db = Database()
    bot = TelegramBot(db)

    pipeline = Pipeline(api=api, db=db, bot=bot)

    # Set up graceful shutdown
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    # Initialize the bot application
    await bot.app.initialize()
    await bot.app.start()
    await bot.app.updater.start_polling(drop_pending_updates=True)

    logger.info("FreeBird is running! Polling for events + listening for Telegram commands.")

    # Run pipeline + daily summary scheduler
    pipeline_task = asyncio.create_task(pipeline.run())
    summary_task = asyncio.create_task(_daily_summary_loop(bot, stop_event))
    feeder_task = asyncio.create_task(_feeder_check_loop(bot, db, stop_event))

    await stop_event.wait()

    # Graceful shutdown
    logger.info("Shutting down...")
    pipeline_task.cancel()
    summary_task.cancel()
    feeder_task.cancel()
    for task in (pipeline_task, summary_task, feeder_task):
        try:
            await task
        except asyncio.CancelledError:
            pass

    await bot.app.updater.stop()
    await bot.app.stop()
    await bot.app.shutdown()
    db.close()
    logger.info("FreeBird stopped.")


async def _daily_summary_loop(bot: TelegramBot, stop_event: asyncio.Event) -> None:
    """Send daily summary at 6pm EST. Sleeps until the next occurrence."""
    while not stop_event.is_set():
        now = datetime.now(TIMEZONE)
        target = datetime.combine(now.date(), DAILY_SUMMARY_TIME, tzinfo=TIMEZONE)
        if now >= target:
            target += timedelta(days=1)
        delay = (target - now).total_seconds()
        logger.info("Next daily summary in %.0f minutes", delay / 60)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=delay)
            return  # stop_event was set, exit
        except asyncio.TimeoutError:
            pass  # Timer fired, send summary

        try:
            await bot.send_daily_summary()
            logger.info("Daily summary sent")
        except Exception:
            logger.exception("Failed to send daily summary")


async def _feeder_check_loop(bot: TelegramBot, db: Database, stop_event: asyncio.Event) -> None:
    """Once a day, nudge if bird activity has collapsed (feeder probably empty)."""
    while not stop_event.is_set():
        now = datetime.now(TIMEZONE)
        target = datetime.combine(now.date(), FEEDER_CHECK_TIME, tzinfo=TIMEZONE)
        if now >= target:
            target += timedelta(days=1)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=(target - now).total_seconds())
            return
        except asyncio.TimeoutError:
            pass
        try:
            await check_feeder(bot, db)
        except Exception:
            logger.exception("Feeder check failed")


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
