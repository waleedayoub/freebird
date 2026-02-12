"""One-time backfill: load up to 3 days of VicoHome history into the sightings table."""
from __future__ import annotations

import asyncio
import logging
import time

from freebird.analysis.birdnet import BirdAnalyzer
from freebird.config import ensure_dirs
from freebird.media.downloader import download_image, download_video, extract_audio
from freebird.storage.database import Database
from freebird.vicohome.api import VicoHomeAPI
from freebird.vicohome.auth import AuthManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("freebird.backfill")


async def backfill() -> None:
    ensure_dirs()

    auth = AuthManager()
    api = VicoHomeAPI(auth)
    db = Database()
    analyzer = BirdAnalyzer()

    now = int(time.time())
    start = now - (3 * 24 * 3600)  # 3 days ago

    logger.info("Fetching events from last 3 days...")
    events = api.get_events(start_timestamp=start, end_timestamp=now)
    logger.info("Found %d total events", len(events))

    new = 0
    skipped = 0
    for event in events:
        if db.has_trace_id(event.trace_id):
            skipped += 1
            continue

        new += 1
        logger.info("[%d/%d] Processing %s (%s)", new, len(events) - skipped,
                     event.trace_id, event.device_name)

        # Download keyshot
        image_path = download_image(event.keyshot_url, event.trace_id)

        # Insert sighting
        sighting_id = db.insert_sighting(
            trace_id=event.trace_id,
            timestamp=event.timestamp,
            device_name=event.device_name,
            image_path=str(image_path) if image_path else None,
        )

        # Download video + extract audio + BirdNET
        species = None
        species_latin = None
        confidence = None
        is_lifer = False

        video_path = await download_video(event.video_url, event.trace_id)
        if video_path:
            audio_path = await extract_audio(video_path, event.trace_id)
            if audio_path:
                detection = analyzer.analyze(audio_path)
                if detection:
                    species = detection.species
                    species_latin = detection.species_latin
                    confidence = detection.confidence
                    is_lifer = db.is_lifer(species)

                db.update_media_paths(
                    sighting_id,
                    video_path=str(video_path),
                    audio_path=str(audio_path),
                )

        # Fallback to VicoHome's own bird ID
        if not species and event.bird_name:
            species = event.bird_name
            species_latin = event.bird_latin
            confidence = event.bird_confidence
            is_lifer = db.is_lifer(species)

        db.update_species(sighting_id, species, species_latin, confidence, is_lifer)

        if species:
            logger.info("  -> %s (%.0f%%)%s", species, (confidence or 0) * 100,
                         " [LIFER]" if is_lifer else "")
        else:
            logger.info("  -> No species identified")

    db.close()
    logger.info("Backfill complete: %d new, %d already existed", new, skipped)


if __name__ == "__main__":
    asyncio.run(backfill())
