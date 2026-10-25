"""Run many vision models over a set of keyshots and save raw predictions (resumable).

Usage:
  uv run python -m freebird.eval_compare --set ground_truth --prompt default_v2 \
      --models google-gla:gemini-3-flash-preview anthropic:claude-sonnet-5-5
  uv run python -m freebird.eval_compare --set sample --db eval/work/birds.db ...

Predictions land in eval/runs/<set>/<prompt>/<model-slug>.jsonl, one line per sighting
(id, prediction, latency, tokens, error). Re-running skips ids already done OK, so an
interrupted run just resumes. Scoring happens separately (eval_score.py).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path

from pydantic_ai import Agent, BinaryContent, NativeOutput, PromptedOutput

from freebird.analysis.vision import VisionAnalysis, load_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("freebird.eval_compare")

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"


def slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def load_set(name: str, db_path: Path) -> list[tuple[str, Path]]:
    """Return [(sighting_id, image_path)] for a named eval set."""
    if name == "ground_truth":
        ids = list(json.loads((EVAL_DIR / "ground_truth.json").read_text()))
    else:
        ids = json.loads((EVAL_DIR / "sets" / f"{name}.json").read_text())["ids"]
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = []
    for sid in ids:
        row = conn.execute("SELECT image_path FROM sightings WHERE id=?", (sid,)).fetchone()
        if row and row[0]:
            p = Path(row[0])
            if not p.exists():  # sample images may have been copied next to the db
                p = db_path.parent / "media" / Path(row[0]).parent.name / Path(row[0]).name
            if p.exists():
                out.append((sid, p))
    return out


def output_spec(model: str):
    """Claude 5.x rejects forced tool_choice, so use prompted JSON output there; others
    keep PydanticAI's default tool-call mode (what production uses)."""
    if model.startswith("anthropic:claude-") and not model.startswith("anthropic:claude-haiku-5-5"):
        return PromptedOutput(VisionAnalysis)
    return VisionAnalysis


async def run_model(model: str, items: list[tuple[str, Path]], prompt: str, out_path: Path,
                    concurrency: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        recs = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        done = {r["id"] for r in recs if not r.get("error")}  # errored ids are retried
    todo = [(i, p) for i, p in items if i not in done]
    if not todo:
        logger.info("%s: already complete (%d)", model, len(done))
        return
    agent = Agent(model, output_type=output_spec(model), system_prompt=load_prompt(prompt), retries=2)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    t0 = time.time()
    n = 0

    async def one(sid: str, path: Path) -> None:
        nonlocal n
        async with sem:
            start = time.time()
            rec: dict = {"id": sid, "model": model, "prompt": prompt}
            try:
                res = await agent.run([
                    "Analyze this bird feeder camera image.",
                    BinaryContent(data=path.read_bytes(), media_type="image/jpeg"),
                ])
                u = res.usage()
                rec |= {"pred": res.output.model_dump(), "in_tok": u.input_tokens,
                        "out_tok": u.output_tokens, "error": None}
            except Exception as e:  # noqa: BLE001 -- record and continue
                rec |= {"pred": None, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            rec["secs"] = round(time.time() - start, 2)
        async with lock:
            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            n += 1
            if n % 25 == 0:
                logger.info("%s: %d/%d (%.0fs)", model, n, len(todo), time.time() - t0)

    await asyncio.gather(*(one(i, p) for i, p in todo))
    logger.info("%s: finished %d in %.0fs", model, len(todo), time.time() - t0)


async def main_async(args: argparse.Namespace) -> None:
    db = Path(args.db)
    items = load_set(args.set, db)
    if args.limit:
        items = items[: args.limit]
    logger.info("set=%s images=%d models=%d", args.set, len(items), len(args.models))
    base = EVAL_DIR / "runs" / args.set / args.prompt
    await asyncio.gather(*(
        run_model(m, items, args.prompt, base / f"{slug(m)}.jsonl", args.concurrency)
        for m in args.models
    ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="ground_truth")
    ap.add_argument("--db", default=str(Path(__file__).resolve().parents[2] / "data" / "birds.db"))
    ap.add_argument("--prompt", default="default_v2")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
