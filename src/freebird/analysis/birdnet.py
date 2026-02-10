from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from freebird.config import BIRDNET_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class BirdDetection:
    species: str
    species_latin: str
    confidence: float


class BirdAnalyzer:
    def __init__(self) -> None:
        logger.info("Loading BirdNET model...")
        import birdnet
        self._model = birdnet.load("acoustic", "2.4", "tf")
        logger.info("BirdNET model loaded")

    def analyze(self, audio_path: Path) -> BirdDetection | None:
        if not audio_path.exists():
            logger.warning("Audio file not found: %s", audio_path)
            return None

        try:
            predictions = self._model.predict(str(audio_path))
        except Exception:
            logger.exception("BirdNET prediction failed for %s", audio_path)
            return None

        # Find the highest-confidence bird detection
        best: BirdDetection | None = None
        for row in predictions.to_structured_array():
            confidence = float(row["confidence"])
            species_name = str(row["species_name"])

            if confidence < BIRDNET_CONFIDENCE_THRESHOLD:
                continue

            # species_name format: "Scientific name_Common Name"
            parts = species_name.split("_", 1)
            if len(parts) == 2:
                latin, common = parts
            else:
                latin, common = species_name, species_name

            if best is None or confidence > best.confidence:
                best = BirdDetection(
                    species=common,
                    species_latin=latin,
                    confidence=confidence,
                )

        if best:
            logger.info("Detected: %s (%.0f%%)", best.species, best.confidence * 100)
        else:
            logger.info("No confident bird detection in %s", audio_path.name)

        return best
