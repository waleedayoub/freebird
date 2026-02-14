"""Run Claude Vision analysis on all existing sightings that have keyshot images."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from freebird.analysis.vision import analyze_image
from freebird.config import ensure_dirs
from freebird.storage.database import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("freebird.vision_backfill")


def run() -> None:
    rerun = "--rerun" in sys.argv
    ensure_dirs()
    db = Database()

    if rerun:
        logger.info("Re-run mode: clearing old vision data...")
        db.conn.execute("DELETE FROM vision_analyses")
        db.conn.execute(
            "UPDATE sightings SET species = NULL, species_latin = NULL, "
            "confidence = NULL, is_lifer = 0"
        )
        db.conn.commit()

    # Find sightings with images but no vision analysis yet
    rows = db.conn.execute(
        """SELECT s.id, s.image_path
           FROM sightings s
           LEFT JOIN vision_analyses v ON s.id = v.sighting_id
           WHERE s.image_path IS NOT NULL AND v.id IS NULL
           ORDER BY s.timestamp ASC"""
    ).fetchall()

    total = len(rows)
    logger.info("Found %d sightings needing vision analysis", total)

    analyzed = 0
    updated = 0
    for i, row in enumerate(rows, 1):
        sighting_id = row["id"]
        image_path = Path(row["image_path"])

        logger.info("[%d/%d] Analyzing %s", i, total, image_path.name)

        vision = analyze_image(image_path, sighting_id, db)

        if vision and vision.is_bird and vision.species:
            analyzed += 1
            confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.4}
            confidence = confidence_map.get(vision.confidence or "", 0.5)
            is_lifer = db.is_lifer(vision.species)
            db.update_species(
                sighting_id, vision.species, vision.species_latin,
                confidence, is_lifer,
            )
            updated += 1
            logger.info("  -> %s (%s)%s", vision.species, vision.confidence,
                        " [LIFER]" if is_lifer else "")
        elif vision and vision.animal_type:
            analyzed += 1
            logger.info("  -> %s (not a bird)", vision.animal_type)
        else:
            logger.info("  -> No animal detected")

        # Delay for Gemini free tier (10 RPM)
        time.sleep(7)

    db.close()
    logger.info(
        "Vision backfill complete: %d/%d had animals, %d sightings updated with species",
        analyzed, total, updated,
    )


if __name__ == "__main__":
    run()
