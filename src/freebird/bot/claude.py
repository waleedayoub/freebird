from __future__ import annotations

import base64
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
HISTORY_HOURS = 12      # how far back the conversation is replayed to the model
HISTORY_EXCHANGES = 20  # and at most this many question/answer pairs
NO_REPLY = "NO_REPLY"   # what the model says when an unaddressed chat message is not for it
IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

SYSTEM_PROMPT = """\
You are FreeBird, a friendly bird-watching assistant for one backyard feeder camera in {location}.
Today is {today} (timezone {tz}). All dates and times you read or write are local time.

Use the tools to look up facts in the sighting database. Never guess counts, dates or species
from memory. The camera also catches non-birds (squirrels, raccoons, cats, insects, people); the
tools list those as animals, so search them before saying the feeder has no data on one. Species were identified from camera stills by an AI model, so they are sometimes
wrong (look-alike finches and sparrows are the usual mix-ups); mention that briefly when it matters,
for example when someone asks about a rare bird. A sighting with a human confirmation is reliable.
If the database has nothing, say so, then offer general bird knowledge if it helps.

When someone asks to see a bird, call send_media with the sighting id. To check or describe what the
camera saw, call view_sighting and look at the photo yourself; the stored species comes from a cheaper
model, so give your own read and say how sure you are. If someone sends you a photo, look at it the same way.
Ignore the date and time printed in the corner of camera photos; use the database for times.

Waleed and Marissa both use this chat. Each message starts with the speaker's name. The earlier turns of
this conversation are above, so follow-ups like "show me that one" or "and last week?" refer to them;
sighting ids you looked up earlier are noted in brackets. Keep answers short and warm, plain text only
(no markdown), and under about 120 words unless asked for more."""

AMBIENT_NOTE = (
    "\n\nThe latest message was sent in the family chat without mentioning you. Reply only if it is "
    "clearly a question or request for you (about the feeder, a bird, a photo, or a follow-up to what you "
    f"just said). Otherwise reply with exactly {NO_REPLY} and nothing else. When unsure, reply {NO_REPLY}.")

TOOLS = [
    {
        "name": "search_sightings",
        "description": "List individual sightings, newest first. Filter by species or animal name "
                       "(partial, case-insensitive, e.g. 'cardinal' or 'squirrel'), local date range, or "
                       "lifers only. Returns id, local time, species (or animal for non-birds) and confidence.",
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
        "description": "Visits per bird species between two local dates, with first and last seen, plus "
                       "a count of other animals (squirrel, raccoon, ...). Use for 'what came by this week', "
                       "'how many X', 'most common bird'.",
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
        "name": "view_sighting",
        "description": "Look at the camera photo of a sighting. Use it to check or describe what a bird "
                       "looks like or is doing, or to give your own species read.",
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


# The non-bird animal the camera saw, from the sighting's primary vision row (NULL for birds).
_ANIMAL = ("(SELECT v.animal_type FROM vision_analyses v WHERE v.sighting_id = s.id AND v.is_bird = 0 "
           "AND v.animal_type IS NOT NULL AND v.role = 'primary' ORDER BY v.id DESC LIMIT 1)")


def _sighting_row(r) -> dict:
    row = {
        "id": r["id"], "when": format_local_time(r["timestamp"], "%Y-%m-%d %H:%M"),
        "species": r["species"], "confidence": r["confidence"],
        "has_video": bool(r["video_path"]), "lifer": bool(r["is_lifer"]),
    }
    if not r["species"] and r["animal"]:
        row["animal"] = r["animal"]
    return row


def run_tool(name: str, args: dict, db: Database, media: list[tuple[Path, str]]) -> str | list[dict]:
    """Run one tool. Returns JSON text, or content blocks (a photo plus text) for view_sighting."""
    c = db.conn
    if name == "search_sightings":
        where, params = _range_sql(args)
        if args.get("species"):
            where += " AND (s.species LIKE ? OR s.species_latin LIKE ? OR animal LIKE ?)"
            params += [f"%{args['species']}%"] * 3
        if args.get("lifers_only"):
            where += " AND s.is_lifer = 1"
        limit = max(1, min(int(args.get("limit") or 10), MAX_ROWS))
        rows = c.execute(
            f"SELECT s.*, {_ANIMAL} AS animal FROM sightings s WHERE (s.species IS NOT NULL OR animal IS NOT NULL){where} "
            "ORDER BY s.timestamp DESC LIMIT ?", [*params, limit]).fetchall()
        total = c.execute(
            f"SELECT COUNT(*) FROM (SELECT s.id, {_ANIMAL} AS animal FROM sightings s "
            f"WHERE (s.species IS NOT NULL OR animal IS NOT NULL){where})", params).fetchone()[0]
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
        animals = c.execute(
            f"SELECT animal, COUNT(*) AS visits FROM (SELECT s.id, {_ANIMAL} AS animal FROM sightings s "
            f"WHERE s.species IS NULL{where}) WHERE animal IS NOT NULL GROUP BY animal ORDER BY visits DESC",
            params).fetchall()
        return json.dumps({"species_total": len(agg), "species": top,
                           "other_animals": [dict(a) for a in animals]})

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
            **_sighting_row(c.execute(f"SELECT s.*, {_ANIMAL} AS animal FROM sightings s WHERE s.id = ?",
                                      (s.id,)).fetchone()),
            "latin": s.species_latin, "behavior": v.get("behavior"), "notable": v.get("notable"),
            "count": v.get("count"), "sex": v.get("sex"), "age": v.get("age"),
            "second_opinion": [dict(a) for a in alts],
            "human_confirmed": label["label"] if label else None,
        })

    if name == "view_sighting":
        s = db.get_sighting(args["sighting_id"])
        if not s:
            return json.dumps({"error": "no such sighting"})
        p = Path(s.image_path) if s.image_path else None
        if not p or not p.exists():
            return json.dumps({"error": "no photo saved for this sighting"})
        info = {"id": s.id, "when": format_local_time(s.timestamp, "%Y-%m-%d %H:%M"),
                "stored_species": s.species, "stored_confidence": s.confidence}
        return [
            {"type": "image", "source": {"type": "base64", "media_type": IMAGE_TYPES.get(p.suffix.lower(), "image/jpeg"),
                                         "data": base64.standard_b64encode(p.read_bytes()).decode()}},
            {"type": "text", "text": json.dumps(info)},
        ]

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


def _history(db: Database) -> list[dict]:
    """Earlier exchanges in this chat as alternating user/assistant turns (text only)."""
    msgs: list[dict] = []
    for r in db.recent_conversations(HISTORY_HOURS, HISTORY_EXCHANGES):
        msgs.append({"role": "user", "content": f"{r['user_name'] or 'Someone'}: {r['question']}"})
        reply = r["response"] + (f"\n[sightings I looked up: {r['refs']}]" if r["refs"] else "")
        msgs.append({"role": "assistant", "content": reply})
    return msgs


def _note_refs(name: str, args: dict, out: str | list[dict], refs: dict[str, str]) -> None:
    """Remember which sightings a tool call touched so a follow-up can point at them."""
    try:
        if name in ("view_sighting", "send_media") and args.get("sighting_id"):
            refs.setdefault(args["sighting_id"], "")
        d = json.loads(out) if isinstance(out, str) else {}
        for row in d.get("shown", []):
            refs[row["id"]] = f"{row.get('species') or row.get('animal')}, {row['when']}"
        if name == "get_sighting" and d.get("id"):
            refs[d["id"]] = f"{d.get('species') or d.get('animal')}, {d['when']}"
    except (ValueError, KeyError, TypeError, AttributeError):
        pass


async def ask_claude(
    question: str,
    db: Database,
    user_name: str = "",
    images: list[tuple[bytes, str]] | None = None,
    ambient: bool = False,
    quoted: str = "",
) -> Answer:
    """Answer one chat message, with the recent conversation as memory.

    `images` are photos the person sent, as (bytes, media_type). `ambient` means nobody addressed
    the bot, so it may stay silent (Answer with empty text). `quoted` is the message being replied to.
    """
    if not ANTHROPIC_API_KEY:
        return Answer("Claude Q&A is not configured. Set ANTHROPIC_API_KEY in .env")

    import anthropic

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    system = SYSTEM_PROMPT.format(location=FEEDER_LOCATION, today=local_today(), tz=str(TIMEZONE))
    if ambient:
        system += AMBIENT_NOTE
    text = f"{user_name or 'Someone'}: {question}"
    if quoted:
        text += f"\n(replying to this earlier message: {quoted[:500]})"
    blocks: list[dict] = [{"type": "image", "source": {"type": "base64", "media_type": mt,
                                                       "data": base64.standard_b64encode(b).decode()}}
                          for b, mt in (images or [])]
    messages: list[dict] = _history(db) + [
        {"role": "user", "content": [*blocks, {"type": "text", "text": text}] if blocks else text}]
    media: list[tuple[Path, str]] = []
    transcript: list[str] = []
    refs: dict[str, str] = {}

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            resp = await client.messages.create(
                model=QA_MODEL, max_tokens=2000, system=system, tools=TOOLS, messages=messages,
                extra_body={"output_config": {"effort": "low"}},
            )
            if resp.stop_reason == "refusal":
                reply = "Sorry, I can't help with that one."
                break
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not tool_uses:
                reply = "".join(b.text for b in resp.content if b.type == "text").strip()
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for tu in tool_uses:
                try:
                    out = run_tool(tu.name, tu.input, db, media)
                except Exception as e:  # noqa: BLE001 -- report to the model, don't crash the bot
                    logger.exception("Tool %s failed", tu.name)
                    out = json.dumps({"error": str(e)})
                _note_refs(tu.name, tu.input, out, refs)
                transcript.append(f"{tu.name}({json.dumps(tu.input)})")
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out})
            messages.append({"role": "user", "content": results})
        else:
            reply = "That took too many steps. Could you ask it more simply?"
        if ambient and reply.strip().startswith(NO_REPLY):
            return Answer("")  # not for the bot: say nothing and keep no record of the message
        reply = reply or "I couldn't come up with an answer."
        shown = "; ".join(f"{k} ({v})" if v else k for k, v in list(refs.items())[-10:])
        db.log_conversation(user_name, question, "; ".join(transcript), reply, QA_MODEL, refs=shown)
        return Answer(reply, media)
    except Exception as e:  # noqa: BLE001
        logger.exception("Claude API error")
        db.log_conversation(user_name, question, "; ".join(transcript), "", QA_MODEL, error=str(e))
        return Answer("Sorry, I couldn't process that question right now.")
