from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from freebird.config import FEEDER_LOCATION, VISION_MODEL, VISION_PROMPT, VISION_REVIEW_MODEL
from freebird.species import display_name, is_real_species, norm_species
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
    alternatives: list[str] = []  # other plausible species when look-alikes, best first


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
    alternatives: list[str] = field(default_factory=list)
    model: str = ""
    reviewed_by: str | None = None
    disputed: bool = False  # the second model named a different species


def output_spec(model: str):
    """Claude 5.x rejects forced tool_choice, so use prompted JSON output there; other
    models keep PydanticAI's default tool-call mode."""
    from pydantic_ai import PromptedOutput

    if model.startswith("anthropic:claude-") and not model.startswith("anthropic:claude-haiku-5-5"):
        return PromptedOutput(VisionAnalysis)
    return VisionAnalysis


_agents: dict[tuple[str, str], object] = {}


def _get_agent(model: str, prompt_name: str | None = None):
    from pydantic_ai import Agent

    key = (model, prompt_name or VISION_PROMPT)
    if key not in _agents:
        _agents[key] = Agent(model, output_type=output_spec(model),
                             system_prompt=load_prompt(key[1]))
    return _agents[key]


async def _run_model(model: str, image_bytes: bytes, attempts: int = 2) -> VisionAnalysis:
    from pydantic_ai import BinaryContent

    last: Exception | None = None
    for i in range(attempts):
        try:
            res = await _get_agent(model).run([
                "Analyze this bird feeder camera image.",
                BinaryContent(data=image_bytes, media_type="image/jpeg"),
            ])
            return res.output
        except Exception as e:  # noqa: BLE001 -- provider outages (503/504) are common
            last = e
            logger.warning("Vision call to %s failed (attempt %d/%d): %s", model, i + 1, attempts, e)
            if i + 1 < attempts:
                await asyncio.sleep(2 * (i + 1))
    assert last is not None
    raise last


def _store(db: Database, sighting_id: str, a: VisionAnalysis, model: str, role: str) -> None:
    db.insert_vision_analysis(
        sighting_id=sighting_id, is_bird=a.is_bird, species=a.species,
        species_latin=a.species_latin, confidence=a.confidence, animal_type=a.animal_type,
        count=a.count, sex=a.sex, age=a.age, behavior=a.behavior, notable=a.notable,
        raw_response=a.model_dump_json(), model=model, role=role,
    )


def _merge_alternatives(final: str | None, chosen: VisionAnalysis, other: VisionAnalysis | None) -> list[str]:
    """Up to 3 other plausible species: the winner's own alternatives, then the other model's answer."""
    seen = {norm_species(final)}
    out: list[str] = []
    cands = list(chosen.alternatives)
    if other is not None:
        cands += [other.species or "", *other.alternatives]
    for c in cands:
        n = norm_species(c)
        if n and n != "unknown" and n not in seen:
            seen.add(n)
            out.append(display_name(c))
    return out[:3]


async def analyze_image(image_path: Path, sighting_id: str, db: Database) -> VisionResult | None:
    if not image_path.exists():
        logger.warning("Image not found: %s", image_path)
        return None

    data = image_path.read_bytes()
    used = VISION_MODEL
    role = "primary"
    try:
        try:
            analysis = await _run_model(VISION_MODEL, data)
        except Exception as e:
            if not VISION_REVIEW_MODEL:
                raise
            logger.error("Primary model failed (%s); failing over to %s", e, VISION_REVIEW_MODEL)
            analysis = await _run_model(VISION_REVIEW_MODEL, data)
            used = VISION_REVIEW_MODEL
    except Exception as e:
        logger.exception("Vision analysis failed")
        db.insert_vision_analysis(
            sighting_id=sighting_id, is_bird=False, species=None,
            species_latin=None, confidence=None, animal_type=None,
            count=None, sex=None, age=None, behavior=None, notable=None,
            raw_response="", model=VISION_MODEL, error=str(e),
        )
        return None

    _store(db, sighting_id, analysis, used, role)

    species = None
    if analysis.is_bird:
        species = display_name(analysis.species) if is_real_species(analysis.species) else "Unidentified bird"

    # Second opinion where a wrong answer is most visible: a new or unusual species (what a cheap model
    # tends to invent), or a shaky call.
    review: VisionAnalysis | None = None
    disputed = False
    reviewed_by = None
    if (VISION_REVIEW_MODEL and used != VISION_REVIEW_MODEL and analysis.is_bird
            and (db.is_uncommon(species) or analysis.confidence == "low")):
        try:
            review = await _run_model(VISION_REVIEW_MODEL, data)
            _store(db, sighting_id, review, VISION_REVIEW_MODEL, "review")
            reviewed_by = VISION_REVIEW_MODEL
            disputed = (not review.is_bird) or norm_species(review.species) != norm_species(species)
        except Exception:
            logger.exception("Review model failed; keeping primary result")

    # The reviewer is the stronger model, so when the two disagree on a bird its answer stands and the
    # primary's guess becomes an alternative. The Telegram confirm message still lets a human overrule it.
    chosen, chosen_model = analysis, used
    if disputed and review is not None and review.is_bird and is_real_species(review.species):
        chosen, chosen_model = review, VISION_REVIEW_MODEL
        species = display_name(review.species)

    result = VisionResult(
        is_bird=chosen.is_bird,
        animal_type=chosen.animal_type,
        species=species,
        species_latin=chosen.species_latin,
        confidence=chosen.confidence,
        count=chosen.count,
        sex=chosen.sex,
        age=chosen.age,
        behavior=chosen.behavior,
        notable=chosen.notable,
        raw_response=chosen.model_dump_json(),
        alternatives=_merge_alternatives(species, chosen, review if chosen is analysis else analysis),
        model=chosen_model,
        reviewed_by=reviewed_by,
        disputed=disputed,
    )

    if result.is_bird and result.species:
        logger.info("Vision: %s (%s, %s)%s", result.species, result.confidence, result.behavior,
                    f" DISPUTED, {reviewed_by} says {review.species if review else None}" if disputed else "")
    elif result.animal_type:
        logger.info("Vision: %s detected (%s)", result.animal_type, result.behavior)
    else:
        logger.info("Vision: no animal detected")
    return result
