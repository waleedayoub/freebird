"""The one labeling UI for eval sets.

Run: uv run python -m freebird.eval_label [--set sample] [--port 8000]   then open http://localhost:8000

Labels a named set (eval/sets/<set>.json, or the legacy ground_truth) and saves to
eval/sets/<set>_gold.json (legacy: eval/ground_truth.json) as
{id: {label, is_bird, unsure, notes?}}, the format eval_score reads.

If model runs exist for the set (eval/runs/<set>/<prompt>/*.jsonl) the queue runs in three tiers:
clear birds the models disagree on first, then photos the models call empty or a non-bird, then
photos most models flag as a poor view (tail only, cropped, blurry), which are mostly "unsure". The
species the models proposed become buttons (alphabetical, no model names, to avoid anchoring).
Without runs it labels in id order with a species autocomplete.

Keys: 1-9 pick a candidate species, E empty, S squirrel, U unsure (excluded from scoring), Z undo.
For anything else type a species and choose bird or critter.

--presence is for labelers who cannot tell species apart: no species buttons, just B bird (any
species), E empty, S squirrel, typed other animals, U unsure. Birds are saved without a species and
count for bird/not-bird scoring only. Hard views are left out unless --include-hard.
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import sqlite3
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from freebird.eval_compare import load_set
from freebird.eval_score import klass, load_runs, view_tier
from freebird.species import display_name, is_real_species, norm_species

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"

TIER_NAMES = {0: "clear bird", 1: "models say empty or non-bird", 2: "hard view (tail, cropped, blurry)"}



class Labeler:
    def __init__(self, set_name: str, db_path: Path, order_prompt: str, gold_path: Path | None = None,
                 presence: bool = False, include_hard: bool = False) -> None:
        self.set_name = set_name
        self.presence = presence
        self.gold_path = gold_path or (
            EVAL_DIR / "ground_truth.json" if set_name == "ground_truth"
            else EVAL_DIR / "sets" / f"{set_name}_gold.json")
        items = load_set(set_name, db_path)
        self.paths = dict(items)
        self.num = {sid: i + 1 for i, sid in enumerate(sorted(self.paths))}  # matches eval_jury --make-check files
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.when = dict(conn.execute("SELECT id, timestamp FROM sightings"))
        self.known = sorted({display_name(r[0]) for r in conn.execute(
            "SELECT DISTINCT species FROM sightings WHERE species IS NOT NULL") if is_real_species(r[0])})
        conn.close()
        self.queue = self._build_queue([i for i, _ in items], order_prompt)
        self.has_runs = any(q["n_runs"] for q in self.queue)
        if presence and not include_hard:
            self.queue = [q for q in self.queue if q["tier"] != 2]

    def _build_queue(self, ids: list[str], order_prompt: str) -> list[dict]:
        order_runs = load_runs(self.set_name, order_prompt)
        root = EVAL_DIR / "runs" / self.set_name
        cand_runs = [r for d in sorted(root.glob("*")) if d.is_dir()
                     for r in load_runs(self.set_name, d.name).values()]

        def ok(r, sid):
            return sid in r and not r[sid].get("error")

        queue = []
        for sid in ids:
            preds = [r[sid]["pred"] for r in cand_runs if ok(r, sid)]
            if self.presence:
                votes = collections.Counter(p["is_bird"] for p in preds)
            else:
                votes = collections.Counter(klass(r[sid]["pred"]) for r in order_runs.values() if ok(r, sid))
            agree = max(votes.values()) / sum(votes.values()) if votes else 1.0
            queue.append({"id": sid, "agree": agree, "cands": {klass(p) for p in preds},
                          "tier": view_tier(preds), "n_runs": len(preds)})
        # Clear birds first (most contested first), then empties, then the unidentifiable.
        # Presence mode instead puts the images the models disagree on bird/not-bird first.
        if self.presence:
            queue.sort(key=lambda q: (q["agree"], q["id"]))
        else:
            queue.sort(key=lambda q: (q["tier"], q["agree"], q["id"]))
        return queue

    def gold(self) -> dict:
        return json.loads(self.gold_path.read_text()) if self.gold_path.exists() else {}

    def write(self, g: dict) -> None:
        self.gold_path.parent.mkdir(parents=True, exist_ok=True)
        self.gold_path.write_text(json.dumps(g, indent=2) + "\n")


STYLE = """
body{font-family:system-ui;background:#1a1a2e;color:#eee;margin:0;padding:20px;max-width:1000px;margin:auto}
img{width:100%;border-radius:8px}
button,select{margin:4px;padding:10px 14px;border:0;border-radius:6px;background:#2a3a5e;color:#fff;font-size:1em;cursor:pointer}
button:hover{background:#5b8def}.m{color:#888}.sub{background:#222}
input[type=text],textarea{padding:9px;border-radius:6px;border:1px solid #444;background:#2a2a3e;color:#eee;font-size:1em}
input[type=text]{width:16em}textarea{width:100%;height:3.2em;box-sizing:border-box}
"""

SCRIPT = """
function k(b){document.getElementById('kind').value=b.dataset.kind}
function saveTyped(){const t=document.getElementById('typed').value.trim();if(!t)return;
  document.getElementById('kind').value=document.getElementById('tk').value;
  document.getElementById('f').requestSubmit()}
document.getElementById('typed').addEventListener('keydown',e=>{if(e.key=='Enter'){e.preventDefault();saveTyped()}});
document.addEventListener('keydown',e=>{
  if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;
  const b=[...document.querySelectorAll('#f button[data-key]')];
  const key=e.key.toLowerCase();
  if(key=='z'){document.getElementById('fu').submit();return}
  const t=b.find(x=>x.dataset.key==key);if(t)t.click()});
"""


def make_app(lab: Labeler) -> FastAPI:
    app = FastAPI()

    @app.get("/img/{sid}")
    async def img(sid: str):
        p = lab.paths.get(sid)
        return FileResponse(p) if p and p.exists() else HTMLResponse("not found", 404)

    @app.post("/save")
    async def save(sid: str = Form(...), kind: str = Form(...), pick: str = Form(""),
                   typed: str = Form(""), notes: str = Form("")):
        label = (pick or typed).strip()
        if sid not in lab.paths or not (label or kind in ("empty", "unsure", "bird_any")):
            return RedirectResponse("/", status_code=303)
        if kind == "bird":
            entry = {"label": display_name(label) or label, "is_bird": True, "unsure": False}
        elif kind == "critter":
            entry = {"label": norm_species(label) or label.lower(), "is_bird": False, "unsure": False}
        elif kind == "bird_any":
            entry = {"label": "", "is_bird": True, "unsure": False}
        elif kind == "unsure":
            entry = {"label": "unsure", "is_bird": False, "unsure": True}
        else:
            entry = {"label": "empty", "is_bird": False, "unsure": False}
        if notes.strip():
            entry["notes"] = notes.strip()
        g = lab.gold()
        g[sid] = entry
        lab.write(g)
        return RedirectResponse("/", status_code=303)

    @app.post("/undo")
    async def undo():
        g = lab.gold()
        if g:
            g.popitem()  # json keeps insertion order, so this is the most recent label
            lab.write(g)
        return RedirectResponse("/", status_code=303)

    @app.get("/")
    async def index():
        g = lab.gold()
        todo = [q for q in lab.queue if q["id"] not in g]
        if not todo:
            return HTMLResponse(f"<style>{STYLE}</style><h1>All done: {len(g)} labeled</h1>"
                                '<form method=post action=/undo><button class=sub>Z. Undo last label</button></form>')
        q = todo[0]
        birds = sorted({k.split(":", 1)[1] for k in q["cands"] if k.startswith("bird:") and k != "bird:unknown"})
        critters = sorted({k.split(":", 1)[1] for k in q["cands"]
                           if k.startswith("animal:") and k != "animal:squirrel"})
        if lab.presence:  # no species at all: the labeler only says what they can see
            btns = '<button name=pick value=bird data-kind=bird_any data-key=b onclick="k(this)">B. Bird</button>'
        else:
            btns = "".join(
                f'<button name=pick value="{html.escape(c)}" data-kind=bird data-key={i + 1} onclick="k(this)">'
                f'{i + 1}. {html.escape(display_name(c) or c)}</button>' for i, c in enumerate(birds[:9]))
        btns += "".join(
            f'<button name=pick value="{html.escape(a)}" data-kind=critter onclick="k(this)">{html.escape(a)}</button>'
            for a in critters)
        if lab.presence:
            typed_box = ('<input type=text id=typed name=typed placeholder="other animal (chipmunk, cat...), Enter" '
                         'autocomplete=off><input type=hidden id=tk value=critter>')
        else:
            species_opts = "".join(f'<option value="{html.escape(s)}">' for s in lab.known)
            typed_box = ('<input type=text id=typed name=typed list=sp placeholder="or type a species / animal, Enter" '
                         f'autocomplete=off><datalist id=sp>{species_opts}</datalist>'
                         '<select id=tk><option value=bird>bird</option><option value=critter>critter</option></select>')
        when = html.escape(str(lab.when.get(q["id"], "")))
        if lab.has_runs:
            tiers = collections.Counter(t["tier"] for t in todo)
            where = (f'{TIER_NAMES[q["tier"]]} ({tiers[q["tier"]]} left in this group, '
                     f'{len(todo) - tiers[q["tier"]]} in later groups)')
            info = f"{where} &middot; models agreed {q['agree']:.0%}"
        else:
            info = f"{len(todo)} left"  # no model runs for this set, so nothing to anchor on
        sid = html.escape(q["id"])
        return HTMLResponse(f"""<!DOCTYPE html><meta charset=utf-8><title>FreeBird labeler</title>
<style>{STYLE}</style>
<p class=m>{lab.set_name} photo {lab.num[q['id']]:02d} &middot; {len(g)} labeled &middot; {info} &middot; {when}</p>
<img src="/img/{sid}">
<form method=post action=/save id=f><input type=hidden name=sid value="{sid}"><input type=hidden name=kind id=kind value=bird>
{btns}
<button name=pick value=empty data-kind=empty data-key=e onclick="k(this)">E. Empty</button>
<button name=pick value=squirrel data-kind=critter data-key=s onclick="k(this)">S. Squirrel</button>
<button name=pick value=unsure data-kind=unsure data-key=u onclick="k(this)">U. Unsure</button>
<div style="margin:8px 4px">{typed_box}
<button type=button onclick="saveTyped()">Save typed</button></div>
<textarea name=notes placeholder="Notes (optional)"></textarea>
</form>
<form method=post action=/undo id=fu><button class=sub>Z. Undo last label</button></form>
<script>{SCRIPT}</script>""")

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="sample", help="eval set name (eval/sets/<set>.json, or ground_truth)")
    ap.add_argument("--db", default=str(EVAL_DIR / "work" / "birds.db"),
                    help="sightings DB; images are found via image_path or <db dir>/media/")
    ap.add_argument("--prompt", default="default_v2", help="prompt whose model runs order the queue")
    ap.add_argument("--gold", help="override the gold-label file path")
    ap.add_argument("--presence", action="store_true",
                    help="only bird / empty / critter, no species (for labelers who cannot tell species apart)")
    ap.add_argument("--include-hard", action="store_true", help="with --presence, also show hard views")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    lab = Labeler(args.set, Path(args.db), args.prompt, Path(args.gold) if args.gold else None,
                  args.presence, args.include_hard)
    print(f"{len(lab.queue)} images in '{args.set}', labels -> {lab.gold_path}")
    uvicorn.run(make_app(lab), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
