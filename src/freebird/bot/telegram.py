from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from freebird.config import format_local_time, local_today

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

CONVERSATION_WINDOW = 300  # seconds after the bot's last answer during which it reads unaddressed messages


def how_addressed(chat_type: str, text: str, bot_username: str, replied_to_bot: bool,
                  last_answer_at: float, now: float) -> str | None:
    """'direct' when the message is for the bot (private chat, @mention or a reply to it), 'ambient'
    during an open conversation (the model may stay silent), else None, and the message is ignored."""
    if chat_type not in ("group", "supergroup"):
        return "direct"
    if replied_to_bot or f"@{bot_username}".lower() in text.lower():
        return "direct"
    if last_answer_at and now - last_answer_at < CONVERSATION_WINDOW:
        return "ambient"
    return None


class TelegramBot:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        self._last_answer_at = 0.0  # when the bot last answered a question; opens the conversation window
        self._register_handlers()

    def _register_handlers(self) -> None:
        self.app.add_handler(CommandHandler("today", self._cmd_today))
        self.app.add_handler(CommandHandler("stats", self._cmd_stats))
        self.app.add_handler(CommandHandler("lifers", self._cmd_lifers))
        self.app.add_handler(CommandHandler("species", self._cmd_species))
        self.app.add_handler(CommandHandler("show", self._cmd_show))
        self.app.add_handler(CommandHandler("latest", self._cmd_latest))
        self.app.add_handler(CommandHandler("refilled", self._cmd_refilled))
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("help", self._cmd_start))
        self.app.add_handler(CallbackQueryHandler(self._callback_review, pattern=r"^rv:"))
        self.app.add_handler(CallbackQueryHandler(self._callback_species_detail, pattern=r"^species:"))
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_freeform)
        )
        self.app.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))

    # -- Notification methods (called by pipeline / scheduler) --

    async def send_lifer_alert(
        self,
        species: str | None,
        confidence: float | None,
        image_path: Path | None,
        video_path: Path | None = None,
    ) -> None:
        """Send a notification only when a brand new species is detected."""
        pct = int(confidence * 100) if confidence else 0
        caption = f"NEW LIFER: {species}! ({pct}% confidence)"
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
            if video_path and video_path.exists():
                await self.app.bot.send_video(
                    chat_id=TELEGRAM_CHAT_ID,
                    video=video_path.open("rb"),
                )
        except Exception:
            logger.exception("Failed to send lifer alert")

    async def send_lifer_review(
        self,
        sighting_id: str,
        options: list[str],
        image_path: Path | None,
        video_path: Path | None = None,
    ) -> None:
        """Possible new species the models weren't sure about: ask which it is."""
        options = options[:4]
        caption = (f"Possible NEW LIFER: {options[0]}? The two models were not sure. Tap the right one (Merlin can help), "
                   f"or ignore this and I'll keep {options[0]}.")
        rows = [[InlineKeyboardButton(o, callback_data=f"rv:{sighting_id}:{i}")]
                for i, o in enumerate(options)]
        rows.append([InlineKeyboardButton("None of these", callback_data=f"rv:{sighting_id}:x")])
        markup = InlineKeyboardMarkup(rows)
        self.db.add_pending_review(sighting_id, options)
        try:
            if image_path and image_path.exists():
                await self.app.bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID, photo=image_path.open("rb"),
                    caption=caption, reply_markup=markup)
            else:
                await self.app.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID, text=caption, reply_markup=markup)
            if video_path and video_path.exists():
                await self.app.bot.send_video(chat_id=TELEGRAM_CHAT_ID, video=video_path.open("rb"))
        except Exception:
            logger.exception("Failed to send lifer review")

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
            "/show <name> -- Show photo/video of a species\n"
            "/latest -- Latest video from today\n"
            "/refilled -- Tell me you topped up the feeder\n\n"
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
        if stats["critters"]:
            lines.append("\nOther critters:")
            for name, count in stats["critters"]:
                lines.append(f"  {name}: {count}")
        await update.message.reply_text("\n".join(lines))

    async def _cmd_lifers(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        lifers = self.db.get_lifers()
        if not lifers:
            await update.message.reply_text("No lifers recorded yet!")
            return
        lines = ["First-ever sightings:"]
        for s in lifers:
            ts = format_local_time(s.timestamp, "%b %d")
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
            ts = format_local_time(s.timestamp, "%b %d %I:%M %p")
            conf = f" ({int(s.confidence * 100)}%)" if s.confidence else ""
            lines.append(f"  {s.species}{conf} -- {ts}")
        if len(results) > 10:
            lines.append(f"  ... and {len(results) - 10} more")
        await update.message.reply_text("\n".join(lines))

    async def _cmd_show(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /show <name>\nExample: /show rock pigeon\nAlso works for critters: /show squirrel")
            return
        query = " ".join(args)
        results = self.db.search_species(query)

        if results:
            sighting = results[0]
            caption = f"{sighting.species or query}"
            if sighting.confidence:
                caption += f" ({int(sighting.confidence * 100)}%)"
            caption += f"\n{format_local_time(sighting.timestamp, '%b %d %I:%M %p')}"
            if sighting.device_name:
                caption += f" - {sighting.device_name}"

            vision = self.db.get_vision_for_sighting(sighting.id)
            if vision:
                if vision["behavior"]:
                    caption += f"\n{vision['behavior']}"
                if vision["notable"]:
                    caption += f"\n{vision['notable']}"

            image_path = sighting.image_path
            video_path = sighting.video_path
        else:
            # Fall back to non-bird critter search
            critters = self.db.search_critters(query)
            if not critters:
                await update.message.reply_text(f'No sightings matching "{query}"')
                return

            critter = critters[0]
            caption = f"{critter['animal_type'] or query}"
            caption += f"\n{format_local_time(critter['timestamp'], '%b %d %I:%M %p')}"
            if critter["device_name"]:
                caption += f" - {critter['device_name']}"
            if critter["behavior"]:
                caption += f"\n{critter['behavior']}"
            if critter["notable"]:
                caption += f"\n{critter['notable']}"

            image_path = critter["image_path"]
            video_path = critter["video_path"]

        sent_media = False
        if image_path:
            img = Path(image_path)
            if img.exists():
                await update.message.reply_photo(photo=img.open("rb"), caption=caption)
                sent_media = True

        if video_path:
            vid = Path(video_path)
            if vid.exists():
                await update.message.reply_video(video=vid.open("rb"))
                sent_media = True

        if not sent_media:
            await update.message.reply_text(f"No media available for {query}")

    async def _cmd_latest(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        today = local_today()
        row = self.db.conn.execute(
            """SELECT s.*, v.species as v_species, v.animal_type, v.behavior, v.notable
               FROM sightings s
               LEFT JOIN vision_analyses v ON s.id = v.sighting_id
               WHERE s.timestamp >= ? AND s.video_path IS NOT NULL
               ORDER BY s.timestamp DESC LIMIT 1""",
            (today,),
        ).fetchone()

        if not row or not row["video_path"]:
            await update.message.reply_text("No videos from today yet!")
            return

        vid = Path(row["video_path"])
        if not vid.exists():
            await update.message.reply_text("Video file not found on disk.")
            return

        species = row["species"] or row["v_species"] or row["animal_type"] or "Unknown"
        ts = format_local_time(row["timestamp"])
        caption = f"{species} at {ts}"
        if row["behavior"]:
            caption += f"\n{row['behavior']}"

        if row["image_path"] and Path(row["image_path"]).exists():
            await update.message.reply_photo(
                photo=Path(row["image_path"]).open("rb"), caption=caption
            )
        await update.message.reply_video(video=vid.open("rb"))

    async def _cmd_refilled(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        from datetime import datetime, timezone
        self.db.kv_set("refilled_at", datetime.now(timezone.utc).isoformat())
        await update.message.reply_text("Thanks! I'll watch for birds and stay quiet about the feeder for a few days.")

    async def _callback_review(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        try:
            _, sighting_id, choice = (query.data or "").split(":", 2)
        except ValueError:
            return
        options = self.db.get_pending_review(sighting_id)
        if options is None:
            await self._edit_review(query, "Already answered, thanks!")
            return
        if choice == "x":
            self.db.resolve_review(sighting_id, "other")
            self.db.set_species(sighting_id, None)
            await self._edit_review(query, "Okay, not counting that one as a new species.")
            return
        try:
            label = options[int(choice)]
        except (ValueError, IndexError):
            return
        lifer = self.db.set_species(sighting_id, label)
        self.db.resolve_review(sighting_id, label)
        await self._edit_review(
            query, f"Confirmed NEW LIFER: {label}!" if lifer else f"Logged as {label} (already on your list).")

    @staticmethod
    async def _edit_review(query, text: str) -> None:
        if query.message and query.message.photo:
            await query.edit_message_caption(caption=text)
        else:
            await query.edit_message_text(text)

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
            f"  Last seen: {format_local_time(latest.timestamp, '%b %d %I:%M %p')}",
            f"  Total sightings: {len(results)}",
        ]
        if latest.confidence:
            lines.append(f"  Last confidence: {int(latest.confidence * 100)}%")
        await query.edit_message_text("\n".join(lines))

    async def _handle_freeform(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await self._answer(update, update.message.text or "")

    async def _handle_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await self._answer(update, update.message.caption or "", photo=update.message.photo[-1])

    async def _answer(self, update: Update, text: str, photo=None) -> None:
        """Answer a chat message (and photo) when it is for the bot; ignore it otherwise."""
        msg = update.message
        # Only respond to messages from the configured chat
        if msg is None or str(update.effective_chat.id) != TELEGRAM_CHAT_ID:
            return
        me = await self.app.bot.get_me()
        replied = msg.reply_to_message
        replied_to_bot = bool(replied and replied.from_user and replied.from_user.id == me.id)
        mode = how_addressed(update.effective_chat.type, text, me.username or "", replied_to_bot,
                             self._last_answer_at, time.time())
        if mode is None:
            return  # nothing outside an @mention, a reply or an open conversation is read or stored
        question = re.sub(f"@{re.escape(me.username or '')}", "", text, flags=re.I).strip()

        images: list[tuple[bytes, str]] = []
        for p in (photo, replied.photo[-1] if replied and replied.photo else None):
            if p is not None:
                images.append((bytes(await (await p.get_file()).download_as_bytearray()), "image/jpeg"))
        if images and not question:
            question = ("What bird or animal is in this photo?" if mode == "direct"
                        else "(sent a photo with no caption)")
        if not question:
            return
        quoted = (replied.text or replied.caption or "") if replied else ""

        if mode == "direct":
            await msg.reply_chat_action("typing")
        answer = await ask_claude(question, self.db, user_name=update.effective_user.first_name or "",
                                  images=images, ambient=(mode == "ambient"), quoted=quoted)
        if not answer.text:
            return  # an unaddressed message that was not for the bot
        self._last_answer_at = time.time()
        await msg.reply_text(answer.text)
        for path, kind in answer.media:
            try:
                if kind == "video":
                    await msg.reply_video(video=path.open("rb"))
                else:
                    await msg.reply_photo(photo=path.open("rb"))
            except Exception:
                logger.exception("Failed to send %s", path)
