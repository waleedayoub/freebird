import json
from datetime import datetime, timedelta, timezone

import pytest

from freebird.bot.claude import run_tool
from freebird.feeder import feeder_message
from freebird.species import display_name, is_real_species, norm_species
from freebird.storage.database import Database


@pytest.fixture
def db(tmp_path, monkeypatch):
    import freebird.storage.database as dbmod
    monkeypatch.setattr(dbmod, "DB_PATH", tmp_path / "birds.db")
    d = Database()
    yield d
    d.close()


def add(db, trace, species, when, lifer=False):
    sid = db.insert_sighting(trace, when.timestamp(), "cam", None)
    db.update_species(sid, species, None, 0.9, lifer)
    return sid


def test_name_variants_are_one_species():
    assert norm_species("House sparrow") == norm_species("House Sparrow Pachysylvia") == "house sparrow"
    assert norm_species("Dark-eyed Junco (Slate-colored)") == "dark-eyed junco"
    assert display_name("cooper's hawk") == "Cooper's Hawk"
    assert not is_real_species("Unidentified bird") and not is_real_species("Unknown bird")


def test_lifer_uses_canonical_names(db):
    now = datetime.now(timezone.utc)
    add(db, "a", "House Sparrow", now)
    assert not db.is_lifer("House sparrow")      # same bird, different spelling
    assert db.is_lifer("Purple Finch")
    assert not db.is_lifer("Unidentified bird")  # never a lifer


def test_human_correction_recomputes_lifer(db):
    now = datetime.now(timezone.utc)
    add(db, "a", "House Finch", now - timedelta(days=1))
    sid = add(db, "b", "Purple Finch", now, lifer=True)
    assert db.set_species(sid, "House Finch") is False
    assert db.get_sighting(sid).is_lifer is False
    assert db.set_species(sid, None) is False


def test_feeder_nudge_only_after_a_real_drop(db):
    today = datetime(2026, 8, 14, 12, tzinfo=timezone.utc)
    for d in range(4, 18):                      # busy baseline, 20 visits a day
        for i in range(20):
            add(db, f"b{d}-{i}", "Junco", today - timedelta(days=d))
    assert feeder_message(db, "2026-08-14") is not None   # last 3 days empty
    db.kv_set("refilled_at", datetime.now(timezone.utc).isoformat())
    assert feeder_message(db, "2026-08-14") is None       # quiet after /refilled


def test_feeder_quiet_when_feeder_was_never_busy(db):
    today = datetime(2026, 8, 14, 12, tzinfo=timezone.utc)
    add(db, "x", "Junco", today - timedelta(days=10))
    assert feeder_message(db, "2026-08-14") is None


def test_qa_tools_search_and_counts(db):
    now = datetime.now(timezone.utc)
    sid = add(db, "a", "Northern Cardinal", now)
    add(db, "b", "northern cardinal", now - timedelta(days=1))
    out = json.loads(run_tool("search_sightings", {"species": "cardinal"}, db, []))
    assert out["total_matches"] == 2 and out["shown"][0]["id"] == sid
    counts = json.loads(run_tool("species_counts", {}, db, []))
    assert counts["species"][0]["species"] == "Northern Cardinal" and counts["species"][0]["visits"] == 2
    assert "error" in json.loads(run_tool("get_sighting", {"sighting_id": "nope"}, db, []))


def _bird(species, confidence="high", alternatives=()):
    from freebird.analysis.vision import VisionAnalysis
    return VisionAnalysis(is_bird=True, species=species, confidence=confidence, alternatives=list(alternatives))


def _analyze(monkeypatch, db, tmp_path, primary, reviewer):
    """Run analyze_image with fake primary/reviewer answers; returns (result, models called)."""
    import asyncio
    from freebird.analysis import vision

    monkeypatch.setattr(vision, "VISION_MODEL", "p")
    monkeypatch.setattr(vision, "VISION_REVIEW_MODEL", "r")
    called = []

    async def fake(model, data, attempts=2):
        called.append(model)
        return primary if model == "p" else reviewer

    monkeypatch.setattr(vision, "_run_model", fake)
    img = tmp_path / "k.jpg"
    img.write_bytes(b"x")
    sid = db.insert_sighting("t", datetime.now(timezone.utc).timestamp(), "cam", None)
    return asyncio.run(vision.analyze_image(img, sid, db)), called


def test_reviewer_overrules_a_wrong_new_species(db, tmp_path, monkeypatch):
    add(db, "a", "House Finch", datetime.now(timezone.utc))
    r, called = _analyze(monkeypatch, db, tmp_path, _bird("Purple Finch", alternatives=["Cassin's Finch"]),
                         _bird("House Finch"))
    assert called == ["p", "r"]  # Purple Finch is a lifer candidate, so it was reviewed
    assert r.species == "House Finch" and r.disputed and r.reviewed_by == "r"
    assert "Purple Finch" in r.alternatives  # the primary's guess stays available as a choice
    assert not db.is_lifer(r.species)


def test_no_review_for_a_confident_known_species(db, tmp_path, monkeypatch):
    add(db, "a", "House Finch", datetime.now(timezone.utc))
    r, called = _analyze(monkeypatch, db, tmp_path, _bird("House Finch"), _bird("Purple Finch"))
    assert called == ["p"] and r.species == "House Finch" and not r.disputed


def test_low_confidence_gets_reviewed_and_agreement_is_quiet(db, tmp_path, monkeypatch):
    add(db, "a", "House Finch", datetime.now(timezone.utc))
    r, called = _analyze(monkeypatch, db, tmp_path, _bird("House Finch", confidence="low"), _bird("House Finch"))
    assert called == ["p", "r"] and r.species == "House Finch" and not r.disputed


def test_uncommon_species_get_reviewed_even_if_seen_before(db, tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    for i in range(400):
        add(db, f"h{i}", "House Sparrow", now)
    add(db, "p", "Purple Finch", now)  # seen once: under 1% of 401 sightings
    assert db.is_uncommon("Purple Finch") and not db.is_uncommon("House Sparrow")
    r, called = _analyze(monkeypatch, db, tmp_path, _bird("Purple Finch"), _bird("Northern Cardinal"))
    assert called == ["p", "r"] and r.species == "Northern Cardinal"


def test_qa_tools_see_squirrels_not_just_birds(db):
    now = datetime.now(timezone.utc)
    add(db, "b1", "Northern Cardinal", now)
    sq = db.insert_sighting("s1", now.timestamp(), "cam", None)  # a squirrel: no species, animal in the vision row
    db.insert_vision_analysis(sighting_id=sq, is_bird=False, species=None, species_latin=None, confidence=None,
                              animal_type="squirrel", count=1, sex=None, age=None, behavior=None, notable=None,
                              raw_response="{}", model="m")
    found = json.loads(run_tool("search_sightings", {"species": "squirrel"}, db, []))
    assert found["total_matches"] == 1 and found["shown"][0]["animal"] == "squirrel"
    counts = json.loads(run_tool("species_counts", {}, db, []))
    assert counts["other_animals"] == [{"animal": "squirrel", "visits": 1}]
    assert [s["species"] for s in counts["species"]] == ["Northern Cardinal"]
    assert json.loads(run_tool("get_sighting", {"sighting_id": sq}, db, []))["animal"] == "squirrel"


def test_group_chat_addressing():
    from freebird.bot.telegram import how_addressed
    ha = lambda **k: how_addressed(**{"chat_type": "supergroup", "text": "hi", "bot_username": "FreeBirdBot",
                                      "replied_to_bot": False, "last_answer_at": 0.0, "now": 1000.0, **k})
    assert ha(chat_type="private") == "direct"
    assert ha(text="what's new @freebirdbot") == "direct"
    assert ha(replied_to_bot=True) == "direct"
    assert ha() is None  # ordinary chat is ignored
    assert ha(last_answer_at=800.0) == "ambient"  # 200s after the bot answered
    assert ha(last_answer_at=600.0) is None  # window (300s) has closed


def test_conversation_history_is_recent_answered_and_ordered(db):
    for i, (q, a, err) in enumerate([("first?", "one", None), ("broken?", "", "boom"), ("second?", "two", None)]):
        db.log_conversation("Waleed", q, "", a, "m", error=err, refs="abc (Cardinal, 2026-10-23)" if i == 2 else "")
    db.conn.execute("INSERT INTO conversations (user_name, question, response, created_at) "
                    "VALUES ('Marissa', 'ancient?', 'old', datetime('now', '-30 hours'))")
    db.conn.commit()
    rows = db.recent_conversations(hours=12, limit=20)
    assert [r["question"] for r in rows] == ["first?", "second?"]
    from freebird.bot.claude import _history
    h = _history(db)
    assert [m["role"] for m in h] == ["user", "assistant", "user", "assistant"]
    assert h[0]["content"] == "Waleed: first?"
    assert "abc (Cardinal" in h[3]["content"]  # ids the bot looked up ride along for follow-ups


def test_view_sighting_returns_the_photo(db, tmp_path):
    img = tmp_path / "k.jpg"
    img.write_bytes(b"\xff\xd8fakejpeg")
    sid = db.insert_sighting("v1", datetime.now(timezone.utc).timestamp(), "cam", str(img))
    out = run_tool("view_sighting", {"sighting_id": sid}, db, [])
    assert out[0]["type"] == "image" and out[0]["source"]["media_type"] == "image/jpeg"
    assert json.loads(out[1]["text"])["id"] == sid
    missing = db.insert_sighting("v2", datetime.now(timezone.utc).timestamp(), "cam", str(tmp_path / "gone.jpg"))
    assert "no photo" in run_tool("view_sighting", {"sighting_id": missing}, db, [])


def test_bot_remembers_and_stays_quiet_when_not_addressed(db, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    import anthropic
    from freebird.bot import claude as qa

    seen, replies = [], iter(["It was a cardinal.", "NO_REPLY", "Yes, the same one."])

    class Stub:
        def __init__(self, **kw):
            self.messages = self

        async def create(self, **kw):
            seen.append(kw["messages"])
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=next(replies))])

    monkeypatch.setattr(anthropic, "AsyncAnthropic", Stub)
    monkeypatch.setattr(qa, "ANTHROPIC_API_KEY", "k")
    a = asyncio.run(qa.ask_claude("what visited?", db, "Waleed"))
    assert a.text == "It was a cardinal."
    quiet = asyncio.run(qa.ask_claude("anyway, dinner at 7?", db, "Marissa", ambient=True))
    assert quiet.text == "" and len(db.recent_conversations()) == 1  # NO_REPLY is not stored
    b = asyncio.run(qa.ask_claude("was it the same one as Tuesday?", db, "Marissa", ambient=True))
    assert b.text == "Yes, the same one."
    last = seen[-1]
    assert last[0] == {"role": "user", "content": "Waleed: what visited?"}
    assert last[1]["content"] == "It was a cardinal." and last[-1]["content"].startswith("Marissa: was it")


def _fake_update(text="", *, chat_type="group", chat_id=None, reply_to_bot=False, user="Marissa", photo=None):
    """A stand-in for a Telegram group message; records what the bot sends back."""
    from types import SimpleNamespace
    sent = []

    async def reply_text(t): sent.append(("text", t))
    async def noop(*a, **k): pass

    reply = SimpleNamespace(from_user=SimpleNamespace(id=99 if reply_to_bot else 5), text="NEW LIFER: Purple Finch!",
                            caption=None, photo=None) if reply_to_bot or text == "REPLY" else None
    msg = SimpleNamespace(text=text, caption=None, photo=[photo] if photo else None, reply_to_message=reply,
                          reply_text=reply_text, reply_chat_action=noop)
    import freebird.bot.telegram as tg
    chat = SimpleNamespace(id=int(chat_id or tg.TELEGRAM_CHAT_ID), type=chat_type)
    return SimpleNamespace(message=msg, effective_chat=chat, effective_user=SimpleNamespace(first_name=user)), sent


def test_bot_handler_routes_group_messages(db, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    import freebird.bot.telegram as tg
    from freebird.bot.claude import Answer

    calls = []

    async def fake_ask(question, db, user_name="", images=None, ambient=False, quoted=""):
        calls.append({"q": question, "who": user_name, "ambient": ambient, "quoted": quoted, "images": len(images or [])})
        return Answer("" if question == "dinner?" else f"re: {question}")

    async def get_me(): return SimpleNamespace(id=99, username="freeeebird_bot")

    monkeypatch.setattr(tg, "ask_claude", fake_ask)
    bot = object.__new__(tg.TelegramBot)
    bot.db, bot.app, bot._last_answer_at = db, SimpleNamespace(bot=SimpleNamespace(get_me=get_me)), 0.0

    def send(*a, **k):
        u, sent = _fake_update(*a, **k)
        asyncio.run(bot._answer(u, u.message.text or ""))
        return sent

    assert send("any birds today?") == [] and not calls  # not addressed, no open conversation: ignored unread
    assert send("@freeeebird_bot any birds today?") == [("text", "re: any birds today?")]  # mention stripped
    assert calls[-1]["ambient"] is False and calls[-1]["who"] == "Marissa"
    assert send("and yesterday?") == [("text", "re: and yesterday?")]  # inside the window, no @ needed
    assert calls[-1]["ambient"] is True
    assert send("dinner?") == []  # the model said it was not for it
    bot._last_answer_at = 0.0
    assert send("is that real?", reply_to_bot=True) == [("text", "re: is that real?")]  # reply to the bot, no @
    assert calls[-1]["quoted"] == "NEW LIFER: Purple Finch!"
    assert send("hello", chat_id=-12345) == []  # a different chat is never answered
    assert send("hi", chat_type="private") == [("text", "re: hi")]
