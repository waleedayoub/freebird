from __future__ import annotations

import asyncio
import logging
import signal
from datetime import datetime, time, timedelta, timezone

from freebird.analysis.birdnet import BirdAnalyzer
from freebird.bot.telegram import TelegramBot
from freebird.config import ensure_dirs
from freebird.pipeline import Pipeline
from freebird.storage.database import Database
from freebird.vicohome.api import VicoHomeAPI
from freebird.vicohome.auth import AuthManager

EST = timezone(timedelta(hours=-5))
DAILY_SUMMARY_TIME = time(18, 0)  # 6:00 PM EST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("freebird")


async def _run() -> None:
    ensure_dirs()

    # Initialize components
    logger.info("Initializing FreeBird...")
    auth = AuthManager()
    api = VicoHomeAPI(auth)
    db = Database()
    bot = TelegramBot(db)
    analyzer = BirdAnalyzer()

    pipeline = Pipeline(api=api, db=db, bot=bot, analyzer=analyzer)

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

    await stop_event.wait()

    # Graceful shutdown
    logger.info("Shutting down...")
    pipeline_task.cancel()
    summary_task.cancel()
    for task in (pipeline_task, summary_task):
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
        now = datetime.now(EST)
        target = datetime.combine(now.date(), DAILY_SUMMARY_TIME, tzinfo=EST)
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


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
