from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from freebird.config import FEEDER_LOCATION, VISION_MODEL, VISION_PROMPT
from freebird.storage.database import Database

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "eval" / "prompts"


def load_prompt(name: str = "default") -> str:
    """Load a named prompt template from eval/prompts/{name}.txt."""
    text = (PROMPTS_DIR / f"{name}.txt").read_text()
    return text.format(location=FEEDER_LOCATION)


class VisionAnalysis(BaseModel):
    """Structured output schema for vision models (enforced by PydanticAI)."""

    is_bird: bool
    animal_type: str | None = None
    species: str | None = None
    species_latin: str | None = None
    confidence: str | None = None
    count: int | None = None
    sex: str | None = None
    age: str | None = None
    behavior: str | None = None
    notable: str | None = None


@dataclass
class VisionResult:
    """Public interface consumed by pipeline.py and backfill."""

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


def _build_agent(model: str | None = None, prompt_name: str | None = None):
    """Create a PydanticAI Agent for vision analysis."""
    from pydantic_ai import Agent

    return Agent(
        model or VISION_MODEL,
        output_type=VisionAnalysis,
        system_prompt=load_prompt(prompt_name or VISION_PROMPT),
    )


# Default agent for production use (lazy-initialized)
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


async def analyze_image(image_path: Path, sighting_id: str, db: Database) -> VisionResult | None:
    if not image_path.exists():
        logger.warning("Image not found: %s", image_path)
        return None

    try:
        from pydantic_ai import BinaryContent

        agent = _get_agent()
        ai_result = await agent.run([
            "Analyze this bird feeder camera image.",
            BinaryContent(data=image_path.read_bytes(), media_type="image/jpeg"),
        ])
        analysis = ai_result.output
        raw = analysis.model_dump_json()

        result = VisionResult(
            is_bird=analysis.is_bird,
            animal_type=analysis.animal_type,
            species=analysis.species,
            species_latin=analysis.species_latin,
            confidence=analysis.confidence,
            count=analysis.count,
            sex=analysis.sex,
            age=analysis.age,
            behavior=analysis.behavior,
            notable=analysis.notable,
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
            model=VISION_MODEL,
        )

        if result.is_bird and result.species:
            logger.info("Vision: %s (%s, %s)", result.species, result.confidence, result.behavior)
        elif result.animal_type:
            logger.info("Vision: %s detected (%s)", result.animal_type, result.behavior)
        else:
            logger.info("Vision: no animal detected")

        return result

    except Exception as e:
        logger.exception("Vision analysis failed")
        db.insert_vision_analysis(
            sighting_id=sighting_id, is_bird=False, species=None,
            species_latin=None, confidence=None, animal_type=None,
            count=None, sex=None, age=None, behavior=None, notable=None,
            raw_response="", model=VISION_MODEL, error=str(e),
        )
        return None
