from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from freebird.config import DB_PATH, local_today
from freebird.species import is_real_species, norm_species

logger = logging.getLogger(__name__)

SCHEMA = """\
CREATE TABLE IF NOT EXISTS sightings (
    id TEXT PRIMARY KEY,
    trace_id TEXT UNIQUE NOT NULL,
    species TEXT,
    species_latin TEXT,
    confidence REAL,
    timestamp TEXT NOT NULL,
    device_name TEXT,
    video_path TEXT,
    image_path TEXT,
    audio_path TEXT,
    is_lifer INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sightings_species ON sightings(species);
CREATE INDEX IF NOT EXISTS idx_sightings_timestamp ON sightings(timestamp);
CREATE INDEX IF NOT EXISTS idx_sightings_trace_id ON sightings(trace_id);

CREATE TABLE IF NOT EXISTS vision_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sighting_id TEXT NOT NULL REFERENCES sightings(id),
    is_bird INTEGER NOT NULL,
    species TEXT,
    species_latin TEXT,
    confidence TEXT,
    animal_type TEXT,
    count INTEGER,
    sex TEXT,
    age TEXT,
    behavior TEXT,
    notable TEXT,
    raw_response TEXT NOT NULL,
    model TEXT,
    error TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_vision_sighting ON vision_analyses(sighting_id);

CREATE TABLE IF NOT EXISTS human_labels (
    sighting_id TEXT PRIMARY KEY REFERENCES sightings(id),
    label TEXT NOT NULL,
    source TEXT DEFAULT 'telegram',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pending_reviews (
    sighting_id TEXT PRIMARY KEY REFERENCES sightings(id),
    options TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_name TEXT,
    question TEXT NOT NULL,
    context TEXT,
    response TEXT NOT NULL,
    model TEXT,
    error TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


@dataclass
class Sighting:
    id: str
    trace_id: str
    species: str | None
    species_latin: str | None
    confidence: float | None
    timestamp: str
    device_name: str | None
    video_path: str | None
    image_path: str | None
    audio_path: str | None
    is_lifer: bool
    created_at: str


class Database:
    def __init__(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(DB_PATH))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(vision_analyses)")}
        if "role" not in cols:
            self.conn.execute("ALTER TABLE vision_analyses ADD COLUMN role TEXT DEFAULT 'primary'")
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def has_trace_id(self, trace_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sightings WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        return row is not None

    def is_lifer(self, species: str, exclude_id: str | None = None) -> bool:
        """True if no earlier sighting has this species, comparing canonical names so
        "House sparrow" and "House Sparrow" are the same bird."""
        if not is_real_species(species):
            return False
        key = norm_species(species)
        rows = self.conn.execute(
            "SELECT DISTINCT species FROM sightings WHERE species IS NOT NULL"
            + (" AND id != ?" if exclude_id else ""),
            (exclude_id,) if exclude_id else (),
        ).fetchall()
        return all(norm_species(r["species"]) != key for r in rows)

    def set_species(self, sighting_id: str, species: str | None) -> bool:
        """Overwrite a sighting's species (human correction) and recompute its lifer flag."""
        lifer = bool(species) and self.is_lifer(species, exclude_id=sighting_id)
        self.conn.execute(
            "UPDATE sightings SET species = ?, is_lifer = ?, confidence = 1.0 WHERE id = ?",
            (species, int(lifer), sighting_id),
        )
        self.conn.commit()
        return lifer

    # -- small key/value state --

    def kv_get(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def kv_set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
        self.conn.commit()

    # -- human review --

    def add_pending_review(self, sighting_id: str, options: list[str]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO pending_reviews (sighting_id, options) VALUES (?, ?)",
            (sighting_id, json.dumps(options)))
        self.conn.commit()

    def get_pending_review(self, sighting_id: str) -> list[str] | None:
        row = self.conn.execute(
            "SELECT options FROM pending_reviews WHERE sighting_id = ?", (sighting_id,)).fetchone()
        return json.loads(row["options"]) if row else None

    def resolve_review(self, sighting_id: str, label: str) -> None:
        """Record the human's answer (also usable later as a gold label for evals)."""
        self.conn.execute(
            "INSERT OR REPLACE INTO human_labels (sighting_id, label) VALUES (?, ?)",
            (sighting_id, label))
        self.conn.execute("DELETE FROM pending_reviews WHERE sighting_id = ?", (sighting_id,))
        self.conn.commit()

    def get_sighting(self, sighting_id: str) -> Sighting | None:
        row = self.conn.execute("SELECT * FROM sightings WHERE id = ?", (sighting_id,)).fetchone()
        return self._row_to_sighting(row) if row else None

    def insert_sighting(
        self,
        trace_id: str,
        timestamp: float,
        device_name: str = "",
        image_path: str | None = None,
    ) -> str:
        sighting_id = uuid.uuid4().hex[:16]
        ts_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO sightings (id, trace_id, timestamp, device_name, image_path)
               VALUES (?, ?, ?, ?, ?)""",
            (sighting_id, trace_id, ts_str, device_name, image_path),
        )
        self.conn.commit()
        return sighting_id

    def update_species(
        self,
        sighting_id: str,
        species: str | None,
        species_latin: str | None,
        confidence: float | None,
        is_lifer: bool,
    ) -> None:
        self.conn.execute(
            """UPDATE sightings
               SET species = ?, species_latin = ?, confidence = ?, is_lifer = ?
               WHERE id = ?""",
            (species, species_latin, confidence, int(is_lifer), sighting_id),
        )
        self.conn.commit()

    def update_media_paths(
        self,
        sighting_id: str,
        video_path: str | None = None,
        audio_path: str | None = None,
        image_path: str | None = None,
    ) -> None:
        updates = []
        params: list = []
        if video_path is not None:
            updates.append("video_path = ?")
            params.append(video_path)
        if audio_path is not None:
            updates.append("audio_path = ?")
            params.append(audio_path)
        if image_path is not None:
            updates.append("image_path = ?")
            params.append(image_path)
        if not updates:
            return
        params.append(sighting_id)
        self.conn.execute(
            f"UPDATE sightings SET {', '.join(updates)} WHERE id = ?", params
        )
        self.conn.commit()

    def get_today_sightings(self) -> list[Sighting]:
        today = local_today()
        rows = self.conn.execute(
            """SELECT * FROM sightings
               WHERE timestamp >= ? AND species IS NOT NULL
               ORDER BY timestamp DESC""",
            (today,),
        ).fetchall()
        return [self._row_to_sighting(r) for r in rows]

    def daily_bird_counts(self, days: int, today: str | None = None) -> dict[str, int]:
        """Bird visits per local calendar day for the last `days` days up to `today` (default now)."""
        from datetime import timedelta
        from freebird.config import TIMEZONE
        end = (datetime.strptime(today, "%Y-%m-%d").replace(tzinfo=TIMEZONE) + timedelta(days=1)
               if today else datetime.now(TIMEZONE))
        since = (end - timedelta(days=days + 1)).astimezone(timezone.utc).isoformat()
        counts: dict[str, int] = {}
        for r in self.conn.execute(
            "SELECT timestamp FROM sightings WHERE species IS NOT NULL AND timestamp >= ?"
            " AND timestamp < ?", (since, end.astimezone(timezone.utc).isoformat())
        ):
            d = datetime.fromisoformat(r["timestamp"]).astimezone(TIMEZONE).strftime("%Y-%m-%d")
            counts[d] = counts.get(d, 0) + 1
        return counts

    def get_stats(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) FROM sightings").fetchone()[0]
        with_species = self.conn.execute(
            "SELECT COUNT(*) FROM sightings WHERE species IS NOT NULL"
        ).fetchone()[0]
        unique = self.conn.execute(
            "SELECT COUNT(DISTINCT species) FROM sightings WHERE species IS NOT NULL"
        ).fetchone()[0]
        top = self.conn.execute(
            """SELECT species, COUNT(*) as cnt FROM sightings
               WHERE species IS NOT NULL
               GROUP BY species ORDER BY cnt DESC LIMIT 5"""
        ).fetchall()
        lifers = self.conn.execute(
            "SELECT COUNT(*) FROM sightings WHERE is_lifer = 1"
        ).fetchone()[0]
        critters = self.conn.execute(
            """SELECT animal_type, COUNT(*) as cnt FROM vision_analyses
               WHERE is_bird = 0 AND animal_type IS NOT NULL
               GROUP BY animal_type ORDER BY cnt DESC"""
        ).fetchall()
        return {
            "total_events": total,
            "identified": with_species,
            "unique_species": unique,
            "lifers": lifers,
            "top_species": [(r["species"], r["cnt"]) for r in top],
            "critters": [(r["animal_type"], r["cnt"]) for r in critters],
        }

    def get_lifers(self) -> list[Sighting]:
        rows = self.conn.execute(
            """SELECT * FROM sightings WHERE is_lifer = 1
               ORDER BY timestamp ASC"""
        ).fetchall()
        return [self._row_to_sighting(r) for r in rows]

    def search_species(self, query: str) -> list[Sighting]:
        rows = self.conn.execute(
            """SELECT * FROM sightings
               WHERE species LIKE ? OR species_latin LIKE ?
               ORDER BY timestamp DESC LIMIT 20""",
            (f"%{query}%", f"%{query}%"),
        ).fetchall()
        return [self._row_to_sighting(r) for r in rows]

    def search_critters(self, query: str) -> list[dict]:
        """Search vision_analyses for non-bird animals matching query."""
        rows = self.conn.execute(
            """SELECT va.animal_type, va.behavior, va.notable, va.sighting_id,
                      s.image_path, s.video_path, s.timestamp, s.device_name
               FROM vision_analyses va
               JOIN sightings s ON s.id = va.sighting_id
               WHERE va.is_bird = 0
                 AND va.animal_type LIKE ?
               ORDER BY s.timestamp DESC LIMIT 20""",
            (f"%{query}%",),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_vision_for_sighting(self, sighting_id: str) -> dict | None:
        row = self.conn.execute(
            """SELECT species, animal_type, confidence, count, sex, age,
                      behavior, notable
               FROM vision_analyses
               WHERE sighting_id = ? AND COALESCE(role, 'primary') = 'primary'
               ORDER BY id DESC LIMIT 1""",
            (sighting_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_recent_summary(self, days: int = 7) -> str:
        rows = self.conn.execute(
            """SELECT species, COUNT(*) as cnt,
                      MAX(timestamp) as last_seen
               FROM sightings
               WHERE species IS NOT NULL
                 AND timestamp >= datetime('now', ?)
               GROUP BY species ORDER BY cnt DESC""",
            (f"-{days} days",),
        ).fetchall()
        if not rows:
            return "No bird sightings in the last week."
        lines = [f"Last {days} days:"]
        for r in rows:
            lines.append(f"- {r['species']}: {r['cnt']} visits (last: {r['last_seen']})")
        return "\n".join(lines)

    def insert_vision_analysis(
        self,
        sighting_id: str,
        is_bird: bool,
        species: str | None,
        species_latin: str | None,
        confidence: str | None,
        animal_type: str | None,
        count: int | None,
        sex: str | None,
        age: str | None,
        behavior: str | None,
        notable: str | None,
        raw_response: str,
        model: str,
        error: str | None = None,
        role: str = "primary",
    ) -> None:
        self.conn.execute(
            """INSERT INTO vision_analyses
               (sighting_id, is_bird, species, species_latin, confidence,
                animal_type, count, sex, age, behavior, notable,
                raw_response, model, error, role)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sighting_id, int(is_bird), species, species_latin, confidence,
             animal_type, count, sex, age, behavior, notable,
             raw_response, model, error, role),
        )
        self.conn.commit()

    def log_conversation(
        self,
        user_name: str,
        question: str,
        context: str,
        response: str,
        model: str,
        error: str | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO conversations
               (user_name, question, context, response, model, error)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_name, question, context, response, model, error),
        )
        self.conn.commit()

    @staticmethod
    def _row_to_sighting(row: sqlite3.Row) -> Sighting:
        return Sighting(
            id=row["id"],
            trace_id=row["trace_id"],
            species=row["species"],
            species_latin=row["species_latin"],
            confidence=row["confidence"],
            timestamp=row["timestamp"],
            device_name=row["device_name"],
            video_path=row["video_path"],
            image_path=row["image_path"],
            audio_path=row["audio_path"],
            is_lifer=bool(row["is_lifer"]),
            created_at=row["created_at"],
        )
