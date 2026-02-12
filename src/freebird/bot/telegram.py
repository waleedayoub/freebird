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
        self.app.add_handler(CommandHandler("show", self._cmd_show))
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("help", self._cmd_start))
        self.app.add_handler(CallbackQueryHandler(self._callback_species_detail))
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_freeform)
        )

    # -- Notification methods (called by pipeline / scheduler) --

    async def send_lifer_alert(
        self,
        species: str | None,
        confidence: float | None,
        image_path: Path | None,
    ) -> None:
        """Send a notification only when a brand new species is detected."""
        pct = int(confidence * 100) if confidence else 0
        caption = f"NEW LIFER: {species}! (BirdNET: {pct}%)"
        try:
            if image_path and image_path.exists():
                await self.app.bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=image_path.open("rb"),
                    caption=caption,
                )
            else:
                await self.app.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=caption,
                )
        except Exception:
            logger.exception("Failed to send lifer alert")

    async def send_daily_summary(self) -> None:
        """Send the 6pm daily summary: visits, unique species, per-species counts."""
        sightings = self.db.get_today_sightings()
        total_visits = len(sightings)

        if total_visits == 0:
            await self.app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text="Daily summary: No bird sightings today.",
            )
            return

        species_counts: dict[str, int] = {}
        for s in sightings:
            name = s.species or "Unknown"
            species_counts[name] = species_counts.get(name, 0) + 1

        lines = [
            "Daily Summary:",
            f"  Total visits: {total_visits}",
            f"  Unique species: {len(species_counts)}",
            "",
            "Visits per species:",
        ]
        for name, count in sorted(species_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {count}")

        try:
            await self.app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text="\n".join(lines),
            )
        except Exception:
            logger.exception("Failed to send daily summary")

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
            "/species <name> -- Search by species\n"
            "/show <name> -- Show photo/video of a species\n\n"
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

    async def _cmd_show(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /show <name>\nExample: /show rock pigeon")
            return
        query = " ".join(args)
        results = self.db.search_species(query)
        if not results:
            await update.message.reply_text(f'No sightings matching "{query}"')
            return

        # Find the most recent sighting with media
        sighting = results[0]
        caption = f"{sighting.species or query}"
        if sighting.confidence:
            caption += f" ({int(sighting.confidence * 100)}%)"
        caption += f"\n{sighting.timestamp[:16]}"
        if sighting.device_name:
            caption += f" - {sighting.device_name}"

        # Add vision details if available
        vision = self.db.get_vision_for_sighting(sighting.id)
        if vision:
            if vision["behavior"]:
                caption += f"\n{vision['behavior']}"
            if vision["notable"]:
                caption += f"\n{vision['notable']}"

        sent_media = False
        if sighting.image_path:
            img = Path(sighting.image_path)
            if img.exists():
                await update.message.reply_photo(photo=img.open("rb"), caption=caption)
                sent_media = True

        if sighting.video_path:
            vid = Path(sighting.video_path)
            if vid.exists():
                await update.message.reply_video(video=vid.open("rb"))
                sent_media = True

        if not sent_media:
            await update.message.reply_text(f"No media available for {query}")

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

        # In group chats, only respond when the bot is mentioned
        text = update.message.text or ""
        bot_username = (await self.app.bot.get_me()).username
        if update.effective_chat.type in ("group", "supergroup"):
            if f"@{bot_username}" not in text:
                return
            # Strip the mention from the question
            question = text.replace(f"@{bot_username}", "").strip()
        else:
            question = text

        if not question:
            return

        user_name = update.effective_user.first_name or ""
        await update.message.reply_chat_action("typing")
        answer = await ask_claude(question, self.db, user_name=user_name)
        await update.message.reply_text(answer)
