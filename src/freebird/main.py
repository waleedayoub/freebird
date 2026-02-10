from __future__ import annotations

import asyncio
import logging
import signal
import sys

from freebird.analysis.birdnet import BirdAnalyzer
from freebird.bot.telegram import TelegramBot
from freebird.config import ensure_dirs
from freebird.pipeline import Pipeline
from freebird.storage.database import Database
from freebird.vicohome.api import VicoHomeAPI
from freebird.vicohome.auth import AuthManager

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

    # Run pipeline and wait for shutdown
    pipeline_task = asyncio.create_task(pipeline.run())

    await stop_event.wait()

    # Graceful shutdown
    logger.info("Shutting down...")
    pipeline_task.cancel()
    try:
        await pipeline_task
    except asyncio.CancelledError:
        pass

    await bot.app.updater.stop()
    await bot.app.stop()
    await bot.app.shutdown()
    db.close()
    logger.info("FreeBird stopped.")


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
