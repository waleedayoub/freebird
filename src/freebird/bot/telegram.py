from __future__ import annotations

import logging
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from freebird.bot.claude import ask_claude
from freebird.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from freebird.storage.database import Database

logger = logging.getLogger(__name__)


class TelegramBot:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        self._register_handlers()

    def _register_handlers(self) -> None:
        self.app.add_handler(CommandHandler("today", self._cmd_today))
        self.app.add_handler(CommandHandler("stats", self._cmd_stats))
        self.app.add_handler(CommandHandler("lifers", self._cmd_lifers))
        self.app.add_handler(CommandHandler("species", self._cmd_species))
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("help", self._cmd_start))
        self.app.add_handler(CallbackQueryHandler(self._callback_species_detail))
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_freeform)
        )

    # -- Notification methods (called by pipeline) --

    async def send_motion_alert(self, image_path: Path | None) -> int | None:
        """Send initial motion alert with keyshot. Returns message_id for editing."""
        try:
            if image_path and image_path.exists():
                msg = await self.app.bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=image_path.open("rb"),
                    caption="Motion detected at feeder!",
                )
            else:
                msg = await self.app.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text="Motion detected at feeder!",
                )
            return msg.message_id
        except Exception:
            logger.exception("Failed to send motion alert")
            return None

    async def update_with_species(
        self,
        message_id: int,
        species: str | None,
        confidence: float | None,
        is_lifer: bool,
    ) -> None:
        """Edit the initial notification with species identification."""
        if species and confidence is not None:
            pct = int(confidence * 100)
            if is_lifer:
                caption = f"NEW LIFER: {species}! (BirdNET: {pct}%)"
            elif confidence >= 0.5:
                caption = f"{species} spotted! (BirdNET: {pct}%)"
            else:
                caption = f"Motion detected -- best guess: {species} ({pct}%)"
        else:
            caption = "Motion detected -- species not identified"

        try:
            await self.app.bot.edit_message_caption(
                chat_id=TELEGRAM_CHAT_ID,
                message_id=message_id,
                caption=caption,
            )
        except Exception:
            # If the original was a text message (no image), edit text instead
            try:
                await self.app.bot.edit_message_text(
                    chat_id=TELEGRAM_CHAT_ID,
                    message_id=message_id,
                    text=caption,
                )
            except Exception:
                logger.exception("Failed to update message %s", message_id)

    async def send_error_alert(self, error_msg: str) -> None:
        try:
            await self.app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=f"FreeBird error: {error_msg}",
            )
        except Exception:
            logger.exception("Failed to send error alert")

    # -- Bot command handlers --

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "Welcome to FreeBird!\n\n"
            "/today -- Today's birds\n"
            "/stats -- Summary statistics\n"
            "/lifers -- All first-ever sightings\n"
            "/species <name> -- Search by species\n\n"
            "Or just ask me anything about your birds!"
        )

    async def _cmd_today(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        sightings = self.db.get_today_sightings()
        if not sightings:
            await update.message.reply_text("No bird sightings today yet!")
            return

        # Group by species
        species_counts: dict[str, int] = {}
        for s in sightings:
            name = s.species or "Unknown"
            species_counts[name] = species_counts.get(name, 0) + 1

        lines = ["Today's birds:"]
        buttons = []
        for name, count in sorted(species_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {count} visit{'s' if count > 1 else ''}")
            buttons.append(
                [InlineKeyboardButton(f"{name} ({count})", callback_data=f"species:{name}")]
            )

        await update.message.reply_text(
            "\n".join(lines),
            reply_markup=InlineKeyboardMarkup(buttons) if buttons else None,
        )

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        stats = self.db.get_stats()
        lines = [
            "FreeBird Stats:",
            f"  Total events: {stats['total_events']}",
            f"  Species identified: {stats['identified']}",
            f"  Unique species: {stats['unique_species']}",
            f"  Lifers: {stats['lifers']}",
        ]
        if stats["top_species"]:
            lines.append("\nTop visitors:")
            for name, count in stats["top_species"]:
                lines.append(f"  {name}: {count}")
        await update.message.reply_text("\n".join(lines))

    async def _cmd_lifers(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        lifers = self.db.get_lifers()
        if not lifers:
            await update.message.reply_text("No lifers recorded yet!")
            return
        lines = ["First-ever sightings:"]
        for s in lifers:
            ts = s.timestamp[:10]  # date portion
            conf = f" ({int(s.confidence * 100)}%)" if s.confidence else ""
            lines.append(f"  {s.species}{conf} -- {ts}")
        await update.message.reply_text("\n".join(lines))

    async def _cmd_species(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /species <name>\nExample: /species cardinal")
            return
        query = " ".join(args)
        results = self.db.search_species(query)
        if not results:
            await update.message.reply_text(f'No sightings matching "{query}"')
            return
        lines = [f'Sightings matching "{query}":']
        for s in results[:10]:
            ts = s.timestamp[:16]
            conf = f" ({int(s.confidence * 100)}%)" if s.confidence else ""
            lines.append(f"  {s.species}{conf} -- {ts}")
        if len(results) > 10:
            lines.append(f"  ... and {len(results) - 10} more")
        await update.message.reply_text("\n".join(lines))

    async def _callback_species_detail(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await query.answer()
        data = query.data or ""
        if not data.startswith("species:"):
            return
        species_name = data[len("species:"):]
        results = self.db.search_species(species_name)
        if not results:
            await query.edit_message_text(f"No details found for {species_name}")
            return
        latest = results[0]
        lines = [
            f"{latest.species}",
            f"  Latin: {latest.species_latin or 'N/A'}",
            f"  Last seen: {latest.timestamp[:16]}",
            f"  Total sightings: {len(results)}",
        ]
        if latest.confidence:
            lines.append(f"  Last confidence: {int(latest.confidence * 100)}%")
        await query.edit_message_text("\n".join(lines))

    async def _handle_freeform(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        # Only respond to messages from the configured chat
        if str(update.effective_chat.id) != TELEGRAM_CHAT_ID:
            return
        question = update.message.text
        await update.message.reply_chat_action("typing")
        answer = await ask_claude(question, self.db)
        await update.message.reply_text(answer)
