"""Run vision eval against ground truth labels.

Usage:
  uv run python -m freebird.eval_run --model google-gla:gemini-2.5-flash --prompt default
  uv run python -m freebird.eval_run --model anthropic:claude-sonnet-4-5-20250929 --prompt default
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic_ai import Agent, BinaryContent
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from freebird.analysis.vision import VisionAnalysis, load_prompt
from freebird.config import VISION_MODEL, ensure_dirs
from freebird.storage.database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("freebird.eval_run")

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
GROUND_TRUTH_PATH = EVAL_DIR / "ground_truth.json"
RESULTS_PATH = EVAL_DIR / "results.jsonl"
DETAILS_DIR = EVAL_DIR / "details"


class IsBirdCorrect(Evaluator[str, VisionAnalysis, dict]):
    """Did the model correctly distinguish bird vs non-bird?"""

    def evaluate(self, ctx: EvaluatorContext[str, VisionAnalysis, dict]) -> bool:
        return ctx.output.is_bird == ctx.metadata["is_bird"]


def _species_eq(a: str | None, b: str | None) -> bool:
    """Case-insensitive species name equality, None-safe."""
    if a is None or b is None:
        return False
    return a.strip().lower() == b.strip().lower()


def _contains_ci(text: str | None, substring: str | None) -> bool:
    """Case-insensitive substring check, None-safe."""
    if text is None or substring is None:
        return False
    return substring.strip().lower() in text.strip().lower()


class SpeciesMatch(Evaluator[str, VisionAnalysis, dict]):
    """Did the model get the species/animal_type right?"""

    def evaluate(self, ctx: EvaluatorContext[str, VisionAnalysis, dict]) -> bool:
        expected = ctx.expected_output
        output = ctx.output

        # Empty frame expected — nothing detected is correct
        if expected == "empty":
            return not output.is_bird and output.animal_type is None

        # Bird expected — exact species match
        if ctx.metadata["is_bird"]:
            return _species_eq(output.species, expected)

        # Non-bird animal expected (e.g., "squirrel")
        return (_contains_ci(output.animal_type, expected) or
                _contains_ci(output.species, expected))


def _build_task(agent: Agent):
    """Create the task function for pydantic-evals."""
    def vision_task(image_path_str: str) -> VisionAnalysis:
        image_path = Path(image_path_str)
        result = agent.run_sync([
            "Analyze this bird feeder camera image.",
            BinaryContent(data=image_path.read_bytes(), media_type="image/jpeg"),
        ])
        time.sleep(7)  # Gemini free tier rate limit
        return result.output
    return vision_task


def run() -> None:
    parser = argparse.ArgumentParser(description="Run vision eval")
    parser.add_argument("--model", default=VISION_MODEL, help="PydanticAI model string")
    parser.add_argument("--prompt", default="default", help="Prompt name from eval/prompts/")
    args = parser.parse_args()

    ensure_dirs()

    if not GROUND_TRUTH_PATH.exists():
        logger.error("No ground truth file at %s — run eval_label.py first", GROUND_TRUTH_PATH)
        return

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())
    if not ground_truth:
        logger.error("Ground truth is empty — label some images first")
        return

    db = Database()

    # Build cases from ground truth
    cases = []
    skipped = 0
    for sighting_id, label in ground_truth.items():
        # Skip unlabeled birds — user saw a bird but didn't ID the species
        if not label["label"] and label["is_bird"]:
            skipped += 1
            continue
        row = db.conn.execute(
            "SELECT image_path FROM sightings WHERE id = ?", (sighting_id,)
        ).fetchone()
        if not row or not row["image_path"] or not Path(row["image_path"]).exists():
            skipped += 1
            continue
        cases.append(Case(
            name=sighting_id,
            inputs=row["image_path"],
            expected_output=label["label"],
            metadata={"is_bird": label["is_bird"]},
        ))
    db.close()

    if skipped:
        logger.warning("Skipped %d sightings (missing images)", skipped)

    logger.info("Running eval: model=%s, prompt=%s, cases=%d", args.model, args.prompt, len(cases))

    # Create agent and dataset
    agent = Agent(
        args.model,
        output_type=VisionAnalysis,
        system_prompt=load_prompt(args.prompt),
    )

    dataset = Dataset(
        cases=cases,
        evaluators=[IsBirdCorrect(), SpeciesMatch()],
    )

    report = dataset.evaluate_sync(_build_task(agent), max_concurrency=1)
    report.print(include_input=True, include_output=True)

    # Compute summary stats over successful cases only
    succeeded = len(report.cases)
    failures = len(report.failures)
    n = succeeded + failures

    is_bird_correct = sum(
        1 for c in report.cases
        if "IsBirdCorrect" in c.assertions and c.assertions["IsBirdCorrect"].value
    )
    species_correct = sum(
        1 for c in report.cases
        if "SpeciesMatch" in c.assertions and c.assertions["SpeciesMatch"].value
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    model_slug = args.model.replace(":", "_").replace("/", "_")

    summary = {
        "timestamp": timestamp,
        "model": args.model,
        "prompt": args.prompt,
        "is_bird_pct": round(is_bird_correct / succeeded * 100, 1) if succeeded else 0,
        "species_pct": round(species_correct / succeeded * 100, 1) if succeeded else 0,
        "n": n,
        "succeeded": succeeded,
        "failures": failures,
    }

    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(summary) + "\n")

    logger.info("Results appended to %s", RESULTS_PATH)
    logger.info(
        "Summary: is_bird=%s/%s (%.1f%%), species=%s/%s (%.1f%%), failures=%s",
        is_bird_correct, succeeded, summary["is_bird_pct"],
        species_correct, succeeded, summary["species_pct"],
        failures,
    )

    # Write per-case details
    DETAILS_DIR.mkdir(exist_ok=True)
    details_path = DETAILS_DIR / f"{timestamp}_{model_slug}.jsonl"

    with open(details_path, "w") as f:
        for c in report.cases:
            output: VisionAnalysis = c.output
            is_bird_ok = c.assertions.get("IsBirdCorrect")
            species_ok = c.assertions.get("SpeciesMatch")
            f.write(json.dumps({
                "id": c.name,
                "expected": c.expected_output,
                "is_bird_expected": c.metadata["is_bird"] if c.metadata else None,
                "predicted_species": output.species,
                "predicted_animal_type": output.animal_type,
                "predicted_is_bird": output.is_bird,
                "confidence": output.confidence,
                "is_bird_correct": is_bird_ok.value if is_bird_ok else None,
                "species_correct": species_ok.value if species_ok else None,
                "error": None,
            }) + "\n")

        for fail in report.failures:
            f.write(json.dumps({
                "id": fail.name,
                "expected": fail.expected_output,
                "is_bird_expected": fail.metadata["is_bird"] if fail.metadata else None,
                "predicted_species": None,
                "predicted_animal_type": None,
                "predicted_is_bird": None,
                "confidence": None,
                "is_bird_correct": None,
                "species_correct": None,
                "error": fail.error_message,
            }) + "\n")

    logger.info("Per-case details written to %s", details_path)


if __name__ == "__main__":
    run()
