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
