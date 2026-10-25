"""Fast labeling UI for the eval sample: contested images first, candidate species as buttons.

Run: uv run python -m freebird.eval_label_sample      then open http://localhost:8001
Saves to eval/sets/sample_gold.json in the same {id: {label, is_bird}} format as ground_truth.json.
Candidates are the species the models proposed (shown alphabetically, no model names, to avoid
anchoring). Keys: 1-9 pick a candidate, E = empty, S = squirrel, X = skip, U = unsure (excluded).
"""
from __future__ import annotations

import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from freebird.eval_score import klass, load_runs

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
GOLD = EVAL_DIR / "sets" / "sample_gold.json"
MEDIA = EVAL_DIR / "work" / "media"

app = FastAPI()


def _build_queue() -> list[dict]:
    import sqlite3

    runs = load_runs("sample", "default_v2")
    ids = sorted(set.intersection(*(set(k for k, v in r.items() if not v.get("error")) for r in runs.values())))
    conn = sqlite3.connect(f"file:{EVAL_DIR / 'work' / 'birds.db'}?mode=ro", uri=True)
    paths = dict(conn.execute("SELECT id, image_path FROM sightings"))
    queue = []
    for sid in ids:
        votes: dict[str, int] = {}
        for r in runs.values():
            k = klass(r[sid]["pred"])
            votes[k] = votes.get(k, 0) + 1
        top = max(votes.values())
        queue.append({"id": sid, "votes": votes, "agree": top / len(runs),
                      "img": Path(paths[sid]).parent.name + "/" + Path(paths[sid]).name})
    queue.sort(key=lambda q: (q["agree"], q["id"]))  # most contested first
    return queue


QUEUE = _build_queue()


def _gold() -> dict:
    return json.loads(GOLD.read_text()) if GOLD.exists() else {}


@app.get("/img/{path:path}")
async def img(path: str):
    p = (MEDIA / path).resolve()
    return FileResponse(p) if p.is_relative_to(MEDIA.resolve()) and p.exists() else HTMLResponse("nf", 404)


@app.post("/save")
async def save(sid: str = Form(...), label: str = Form(...), kind: str = Form(...)):
    g = _gold()
    g[sid] = {"label": label, "is_bird": kind == "bird", "unsure": kind == "unsure"}
    GOLD.write_text(json.dumps(g, indent=2) + "\n")
    return RedirectResponse("/", status_code=303)


@app.get("/")
async def index():
    g = _gold()
    todo = [q for q in QUEUE if q["id"] not in g]
    if not todo:
        return HTMLResponse(f"<h1>All done: {len(g)} labeled</h1>")
    q = todo[0]
    cands = sorted({k.split(":", 1)[1] for k in q["votes"] if k.startswith("bird:")})
    other_animals = sorted({k.split(":", 1)[1] for k in q["votes"] if k.startswith("animal:")})
    btns = "".join(
        f'<button name="label" value="{c}" onclick="k(\'bird\')">{i + 1}. {c.title()}</button>'
        for i, c in enumerate(cands[:9])
    ) + "".join(
        f'<button name="label" value="{a}" onclick="k(\'critter\')">{a}</button>' for a in other_animals
    )
    return HTMLResponse(f"""<!DOCTYPE html><meta charset=utf-8><title>Sample labeler</title>
<style>body{{font-family:system-ui;background:#1a1a2e;color:#eee;margin:0;padding:20px;max-width:1000px;margin:auto}}
img{{width:100%;border-radius:8px}}button{{margin:4px;padding:10px 14px;border:0;border-radius:6px;background:#2a3a5e;color:#fff;font-size:1em;cursor:pointer}}
button:hover{{background:#5b8def}}.m{{color:#888}}input[type=text]{{padding:9px;border-radius:6px;border:1px solid #444;background:#2a2a3e;color:#eee}}</style>
<p class=m>{len(g)} labeled &middot; {len(todo)} left &middot; models agreed {q['agree']:.0%}</p>
<img src="/img/{q['img']}">
<form method=post action=/save id=f><input type=hidden name=sid value="{q['id']}"><input type=hidden name=kind id=kind value=bird>
{btns}
<button name=label value=empty onclick="k('empty')">E. Empty</button>
<button name=label value=squirrel onclick="k('critter')">S. Squirrel</button>
<button name=label value=unsure onclick="k('unsure')">U. Unsure</button>
<span style="margin-left:12px"><input type=text name=label id=other placeholder="other species..." form=f2>
</span></form>
<form method=post action=/save id=f2><input type=hidden name=sid value="{q['id']}"><input type=hidden name=kind value=bird>
<input type=text name=label placeholder="or type another species, Enter" autocomplete=off></form>
<script>function k(v){{document.getElementById('kind').value=v}}
document.addEventListener('keydown',e=>{{if(e.target.tagName=='INPUT')return;const b=[...document.querySelectorAll('#f button')];
const n=parseInt(e.key);let t=null;if(n>=1&&n<=9)t=b.find(x=>x.textContent.startsWith(n+'.'));
else if(e.key=='e')t=b.find(x=>x.textContent.startsWith('E.'));else if(e.key=='s')t=b.find(x=>x.textContent.startsWith('S.'));
else if(e.key=='u')t=b.find(x=>x.textContent.startsWith('U.'));if(t)t.click()}});</script>""")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
