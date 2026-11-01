from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from freebird.config import (
    ANTHROPIC_API_KEY, FEEDER_LOCATION, QA_MODEL, TIMEZONE, format_local_time, local_today,
)
from freebird.species import display_name, norm_species
from freebird.storage.database import Database

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 8
MAX_ROWS = 50

SYSTEM_PROMPT = """\
You are FreeBird, a friendly bird-watching assistant for one backyard feeder camera in {location}.
Today is {today} (timezone {tz}). All dates and times you read or write are local time.

Use the tools to look up facts in the sighting database. Never guess counts, dates or species
from memory. Species were identified from camera stills by an AI model, so they are sometimes
wrong (look-alike finches and sparrows are the usual mix-ups); mention that briefly when it matters,
for example when someone asks about a rare bird. A sighting with a human confirmation is reliable.
If the database has nothing, say so, then offer general bird knowledge if it helps.

When someone asks to see a bird, call send_media with the sighting id. Keep answers short and
warm, plain text only (no markdown), and under about 120 words unless asked for more."""

TOOLS = [
    {
        "name": "search_sightings",
        "description": "List individual sightings, newest first. Filter by species name (partial, "
                       "case-insensitive), local date range, or lifers only. Returns id, local time, "
                       "species, confidence and the camera's behavior note.",
        "input_schema": {
            "type": "object",
            "properties": {
                "species": {"type": "string", "description": "e.g. 'cardinal' or 'house finch'"},
                "date_from": {"type": "string", "description": "YYYY-MM-DD, inclusive"},
                "date_to": {"type": "string", "description": "YYYY-MM-DD, inclusive"},
                "lifers_only": {"type": "boolean"},
                "limit": {"type": "integer", "description": f"max {MAX_ROWS}, default 10"},
            },
        },
    },
    {
        "name": "species_counts",
        "description": "Visits per species between two local dates, with first and last seen. "
                       "Use for 'what came by this week', 'how many X', 'most common bird'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "YYYY-MM-DD, inclusive; omit for all time"},
                "date_to": {"type": "string", "description": "YYYY-MM-DD, inclusive"},
                "limit": {"type": "integer", "description": "default 20"},
            },
        },
    },
    {
        "name": "daily_activity",
        "description": "Bird visits per local day for the last N days. Use for trends or "
                       "'is the feeder empty / busy lately'.",
        "input_schema": {
            "type": "object",
            "properties": {"days": {"type": "integer", "description": "default 14, max 120"}},
        },
    },
    {
        "name": "get_sighting",
        "description": "Full details for one sighting id: species, alternatives considered, behavior, "
                       "notes, whether a human confirmed it.",
        "input_schema": {
            "type": "object",
            "properties": {"sighting_id": {"type": "string"}},
            "required": ["sighting_id"],
        },
    },
    {
        "name": "send_media",
        "description": "Attach the photo (and optionally the video clip) of a sighting to the reply.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sighting_id": {"type": "string"},
                "include_video": {"type": "boolean"},
            },
            "required": ["sighting_id"],
        },
    },
]


@dataclass
class Answer:
    text: str
    media: list[tuple[Path, str]] = field(default_factory=list)  # (path, "photo" | "video")


def _utc_bound(day: str | None, end: bool = False) -> str | None:
    if not day:
        return None
    d = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=TIMEZONE)
    if end:
        d += timedelta(days=1)
    return d.astimezone(timezone.utc).isoformat()


def _range_sql(args: dict) -> tuple[str, list]:
    clauses, params = [], []
    lo, hi = _utc_bound(args.get("date_from")), _utc_bound(args.get("date_to"), end=True)
    if lo:
        clauses.append("s.timestamp >= ?"); params.append(lo)
    if hi:
        clauses.append("s.timestamp < ?"); params.append(hi)
    return "".join(f" AND {c}" for c in clauses), params


def _sighting_row(r) -> dict:
    return {
        "id": r["id"], "when": format_local_time(r["timestamp"], "%Y-%m-%d %H:%M"),
        "species": r["species"], "confidence": r["confidence"],
        "has_video": bool(r["video_path"]), "lifer": bool(r["is_lifer"]),
    }


def run_tool(name: str, args: dict, db: Database, media: list[tuple[Path, str]]) -> str:
    c = db.conn
    if name == "search_sightings":
        where, params = _range_sql(args)
        if args.get("species"):
            where += " AND (s.species LIKE ? OR s.species_latin LIKE ?)"
            params += [f"%{args['species']}%"] * 2
        if args.get("lifers_only"):
            where += " AND s.is_lifer = 1"
        limit = max(1, min(int(args.get("limit") or 10), MAX_ROWS))
        rows = c.execute(
            f"SELECT s.* FROM sightings s WHERE s.species IS NOT NULL{where} "
            "ORDER BY s.timestamp DESC LIMIT ?", [*params, limit]).fetchall()
        total = c.execute(
            f"SELECT COUNT(*) FROM sightings s WHERE s.species IS NOT NULL{where}", params).fetchone()[0]
        return json.dumps({"total_matches": total, "shown": [_sighting_row(r) for r in rows]})

    if name == "species_counts":
        where, params = _range_sql(args)
        rows = c.execute(
            f"SELECT s.species, s.timestamp FROM sightings s WHERE s.species IS NOT NULL{where}", params
        ).fetchall()
        agg: dict[str, dict] = {}
        for r in rows:
            key = norm_species(r["species"])
            if key == "unknown":
                continue
            a = agg.setdefault(key, {"species": display_name(r["species"]), "visits": 0,
                                     "first": r["timestamp"], "last": r["timestamp"]})
            a["visits"] += 1
            a["first"], a["last"] = min(a["first"], r["timestamp"]), max(a["last"], r["timestamp"])
        top = sorted(agg.values(), key=lambda a: -a["visits"])[: int(args.get("limit") or 20)]
        for a in top:
            a["first"], a["last"] = (format_local_time(a["first"], "%Y-%m-%d"),
                                     format_local_time(a["last"], "%Y-%m-%d"))
        return json.dumps({"species_total": len(agg), "species": top})

    if name == "daily_activity":
        days = max(1, min(int(args.get("days") or 14), 120))
        counts = db.daily_bird_counts(days)
        today = datetime.strptime(local_today(), "%Y-%m-%d")
        series = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
        return json.dumps({"bird_visits_per_day": {d: counts.get(d, 0) for d in series},
                           "note": "today is partial"})

    if name == "get_sighting":
        s = db.get_sighting(args["sighting_id"])
        if not s:
            return json.dumps({"error": "no such sighting"})
        v = db.get_vision_for_sighting(s.id) or {}
        label = c.execute("SELECT label FROM human_labels WHERE sighting_id = ?", (s.id,)).fetchone()
        alts = c.execute(
            "SELECT species, model FROM vision_analyses WHERE sighting_id = ? AND role = 'review'",
            (s.id,)).fetchall()
        return json.dumps({
            **_sighting_row(c.execute("SELECT * FROM sightings WHERE id = ?", (s.id,)).fetchone()),
            "latin": s.species_latin, "behavior": v.get("behavior"), "notable": v.get("notable"),
            "count": v.get("count"), "sex": v.get("sex"), "age": v.get("age"),
            "second_opinion": [dict(a) for a in alts],
            "human_confirmed": label["label"] if label else None,
        })

    if name == "send_media":
        s = db.get_sighting(args["sighting_id"])
        if not s:
            return json.dumps({"error": "no such sighting"})
        sent = []
        if s.image_path and Path(s.image_path).exists():
            media.append((Path(s.image_path), "photo")); sent.append("photo")
        if args.get("include_video") and s.video_path and Path(s.video_path).exists():
            media.append((Path(s.video_path), "video")); sent.append("video")
        return json.dumps({"queued": sent or "no media on disk for this sighting"})

    return json.dumps({"error": f"unknown tool {name}"})


async def ask_claude(question: str, db: Database, user_name: str = "", previous: str = "") -> Answer:
    if not ANTHROPIC_API_KEY:
        return Answer("Claude Q&A is not configured. Set ANTHROPIC_API_KEY in .env")

    import anthropic

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    system = SYSTEM_PROMPT.format(location=FEEDER_LOCATION, today=local_today(), tz=str(TIMEZONE))
    content = f"Previous exchange (for follow-ups):\n{previous}\n\nQuestion: {question}" if previous else question
    messages: list[dict] = [{"role": "user", "content": content}]
    media: list[tuple[Path, str]] = []
    transcript: list[str] = []

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            resp = await client.messages.create(
                model=QA_MODEL, max_tokens=2000, system=system, tools=TOOLS, messages=messages,
                extra_body={"output_config": {"effort": "low"}},
            )
            if resp.stop_reason == "refusal":
                text = "Sorry, I can't help with that one."
                break
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not tool_uses:
                text = "".join(b.text for b in resp.content if b.type == "text").strip()
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for tu in tool_uses:
                try:
                    out = run_tool(tu.name, tu.input, db, media)
                except Exception as e:  # noqa: BLE001 -- report to the model, don't crash the bot
                    logger.exception("Tool %s failed", tu.name)
                    out = json.dumps({"error": str(e)})
                transcript.append(f"{tu.name}({json.dumps(tu.input)})")
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out})
            messages.append({"role": "user", "content": results})
        else:
            text = "That took too many steps. Could you ask it more simply?"
        text = text or "I couldn't come up with an answer."
        db.log_conversation(user_name, question, "; ".join(transcript), text, QA_MODEL)
        return Answer(text, media)
    except Exception as e:  # noqa: BLE001
        logger.exception("Claude API error")
        db.log_conversation(user_name, question, "; ".join(transcript), "", QA_MODEL, error=str(e))
        return Answer("Sorry, I couldn't process that question right now.")
