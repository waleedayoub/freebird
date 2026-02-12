from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from freebird.config import ANTHROPIC_API_KEY
from freebird.storage.database import Database

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-5-20250929"

VISION_PROMPT = """\
Analyze this bird feeder camera image. Respond with ONLY a JSON object (no markdown):

{
  "is_bird": true/false,
  "animal_type": "bird" | "squirrel" | "chipmunk" | "cat" | "unknown" | null,
  "species": "Common Name" or null,
  "species_latin": "Scientific name" or null,
  "confidence": "high" | "medium" | "low",
  "count": number of animals visible,
  "sex": "male" | "female" | "unknown" | null,
  "age": "adult" | "juvenile" | "unknown" | null,
  "behavior": brief description of what the animal is doing,
  "notable": any notable observations (unusual markings, weather, multiple species, etc.) or null
}

If no animal is visible (just the feeder/yard), set is_bird to false and animal_type to null.
If you can see an animal but can't identify the species, still describe what you see."""


@dataclass
class VisionResult:
    is_bird: bool
    animal_type: str | None
    species: str | None
    species_latin: str | None
    confidence: str | None
    count: int | None
    sex: str | None
    age: str | None
    behavior: str | None
    notable: str | None
    raw_response: str


def analyze_image(image_path: Path, sighting_id: str, db: Database) -> VisionResult | None:
    if not ANTHROPIC_API_KEY:
        logger.warning("No ANTHROPIC_API_KEY, skipping vision analysis")
        return None

    if not image_path.exists():
        logger.warning("Image not found: %s", image_path)
        return None

    try:
        import anthropic

        image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")
        media_type = "image/jpeg"

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": VISION_PROMPT,
                        },
                    ],
                },
            ],
        )

        raw = message.content[0].text
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        data = json.loads(text)

        result = VisionResult(
            is_bird=data.get("is_bird", False),
            animal_type=data.get("animal_type"),
            species=data.get("species"),
            species_latin=data.get("species_latin"),
            confidence=data.get("confidence"),
            count=data.get("count"),
            sex=data.get("sex"),
            age=data.get("age"),
            behavior=data.get("behavior"),
            notable=data.get("notable"),
            raw_response=raw,
        )

        db.insert_vision_analysis(
            sighting_id=sighting_id,
            is_bird=result.is_bird,
            species=result.species,
            species_latin=result.species_latin,
            confidence=result.confidence,
            animal_type=result.animal_type,
            count=result.count,
            sex=result.sex,
            age=result.age,
            behavior=result.behavior,
            notable=result.notable,
            raw_response=raw,
            model=MODEL,
        )

        if result.is_bird and result.species:
            logger.info("Vision: %s (%s, %s)", result.species, result.confidence, result.behavior)
        elif result.animal_type:
            logger.info("Vision: %s detected (%s)", result.animal_type, result.behavior)
        else:
            logger.info("Vision: no animal detected")

        return result

    except json.JSONDecodeError as e:
        logger.error("Vision: failed to parse JSON response: %s", e)
        db.insert_vision_analysis(
            sighting_id=sighting_id, is_bird=False, species=None,
            species_latin=None, confidence=None, animal_type=None,
            count=None, sex=None, age=None, behavior=None, notable=None,
            raw_response=raw, model=MODEL, error=str(e),
        )
        return None
    except Exception as e:
        logger.exception("Vision analysis failed")
        db.insert_vision_analysis(
            sighting_id=sighting_id, is_bird=False, species=None,
            species_latin=None, confidence=None, animal_type=None,
            count=None, sex=None, age=None, behavior=None, notable=None,
            raw_response="", model=MODEL, error=str(e),
        )
        return None
