from __future__ import annotations

import logging

from freebird.config import ANTHROPIC_API_KEY
from freebird.storage.database import Database

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-5-20250929"

SYSTEM_PROMPT = """\
You are FreeBird, a friendly bird-watching assistant. You help answer questions
about birds spotted at the user's bird feeder. You have access to recent sighting
data from the database. Be concise and enthusiastic about birds.
If asked about birds not in the data, share general bird knowledge."""


async def ask_claude(question: str, db: Database, user_name: str = "") -> str:
    if not ANTHROPIC_API_KEY:
        return "Claude Q&A is not configured. Set ANTHROPIC_API_KEY in .env"

    context = db.get_recent_summary(days=7)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Recent bird feeder data:\n{context}\n\nQuestion: {question}",
                },
            ],
        )
        response = message.content[0].text
        db.log_conversation(user_name, question, context, response, MODEL)
        return response
    except Exception as e:
        logger.exception("Claude API error")
        error_msg = str(e)
        db.log_conversation(user_name, question, context, "", MODEL, error=error_msg)
        return "Sorry, I couldn't process that question right now."
