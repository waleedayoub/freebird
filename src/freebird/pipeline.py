from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from freebird.analysis.vision import analyze_image
from freebird.bot.telegram import TelegramBot
from freebird.config import POLL_INTERVAL_SECONDS
from freebird.media.downloader import download_image, download_video
from freebird.storage.database import Database
from freebird.vicohome.api import VicoHomeAPI
from freebird.vicohome.models import MotionEvent

logger = logging.getLogger(__name__)

# Alert if pipeline hasn't processed successfully for this many seconds
ERROR_ALERT_THRESHOLD = 5 * 60


class Pipeline:
    def __init__(
        self,
        api: VicoHomeAPI,
        db: Database,
        bot: TelegramBot,
    ) -> None:
        self.api = api
        self.db = db
        self.bot = bot
        self._last_success: float = time.time()
        self._error_alerted: bool = False

    async def run(self) -> None:
        logger.info("Pipeline started (polling every %ds)", POLL_INTERVAL_SECONDS)

        # Bootstrap: check last hour on first run
        await self._poll_cycle()

        while True:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            try:
                await self._poll_cycle()
                self._last_success = time.time()
                self._error_alerted = False
            except Exception:
                logger.exception("Pipeline poll cycle failed")
                elapsed = time.time() - self._last_success
                if elapsed > ERROR_ALERT_THRESHOLD and not self._error_alerted:
                    await self.bot.send_error_alert(
                        f"Pipeline has been failing for {int(elapsed)}s. Check logs."
                    )
                    self._error_alerted = True

    async def _poll_cycle(self) -> None:
        now = int(time.time())
        # Look back 1 hour to catch any events we might have missed
        start = now - 3600
        events = self.api.get_events(start_timestamp=start, end_timestamp=now)

        new_count = 0
        for event in events:
            if self.db.has_trace_id(event.trace_id):
                continue
            new_count += 1
            await self._process_event(event)

        if new_count:
            logger.info("Processed %d new events", new_count)

    async def _process_event(self, event: MotionEvent) -> None:
        logger.info("Processing event %s from %s", event.trace_id, event.device_name)

        # Step 1: Download keyshot image
        image_path = download_image(event.keyshot_url, event.trace_id)

        # Step 2: Store initial sighting (for dedup)
        sighting_id = self.db.insert_sighting(
            trace_id=event.trace_id,
            timestamp=event.timestamp,
            device_name=event.device_name,
            image_path=str(image_path) if image_path else None,
        )

        # Step 3: Vision analysis on keyshot (primary species source)
        species = None
        species_latin = None
        confidence = None
        is_lifer = False

        if image_path:
            vision = analyze_image(image_path, sighting_id, self.db)
            if vision and vision.is_bird and vision.species:
                species = vision.species
                species_latin = vision.species_latin
                confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.4}
                confidence = confidence_map.get(vision.confidence or "", 0.5)
                is_lifer = self.db.is_lifer(species)

        # Step 4: Download video for archive
        video_path = await download_video(event.video_url, event.trace_id)
        if video_path:
            self.db.update_media_paths(sighting_id, video_path=str(video_path))

        # Step 5: Check VicoHome's own bird ID as fallback
        if not species and event.bird_name:
            species = event.bird_name
            species_latin = event.bird_latin
            confidence = event.bird_confidence
            is_lifer = self.db.is_lifer(species)
            logger.info("Using VicoHome ID: %s (%.0f%%)", species,
                        (confidence or 0) * 100)

        # Step 6: Update DB with species info
        self.db.update_species(sighting_id, species, species_latin, confidence, is_lifer)

        # Step 7: Notify only on new lifers
        if is_lifer:
            logger.info("NEW LIFER: %s", species)
            await self.bot.send_lifer_alert(species, confidence, image_path)
