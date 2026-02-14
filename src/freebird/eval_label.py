"""Web UI for labeling keyshot images with ground truth species.

Run: uv run python -m freebird.eval_label
Then open http://localhost:8000
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from freebird.config import MEDIA_DIR, ensure_dirs
from freebird.storage.database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("freebird.eval_label")

GROUND_TRUTH_PATH = Path(__file__).resolve().parents[2] / "eval" / "ground_truth.json"

app = FastAPI()


def _load_ground_truth() -> dict:
    if GROUND_TRUTH_PATH.exists():
        return json.loads(GROUND_TRUTH_PATH.read_text())
    return {}


def _save_ground_truth(data: dict) -> None:
    GROUND_TRUTH_PATH.write_text(json.dumps(data, indent=2) + "\n")


def _get_sightings_to_label(db: Database, ground_truth: dict) -> list[dict]:
    """Get sightings with images that haven't been labeled yet."""
    rows = db.conn.execute(
        """SELECT id, image_path, timestamp, device_name, species
           FROM sightings
           WHERE image_path IS NOT NULL
           ORDER BY timestamp ASC"""
    ).fetchall()
    return [
        {"id": r["id"], "image_path": r["image_path"], "timestamp": r["timestamp"],
         "device_name": r["device_name"], "species": r["species"]}
        for r in rows if r["id"] not in ground_truth
    ]


def _get_known_species(db: Database) -> list[str]:
    """Get distinct species names for autocomplete."""
    rows = db.conn.execute(
        "SELECT DISTINCT species FROM sightings WHERE species IS NOT NULL ORDER BY species"
    ).fetchall()
    return [r["species"] for r in rows]


@app.get("/media/{path:path}")
async def serve_media(path: str):
    file_path = (MEDIA_DIR / path).resolve()
    if not file_path.is_relative_to(MEDIA_DIR.resolve()):
        return HTMLResponse("Forbidden", status_code=403)
    if file_path.exists():
        return FileResponse(file_path)
    return HTMLResponse("Not found", status_code=404)


@app.post("/label")
async def save_label(
    sighting_id: str = Form(...),
    category: str = Form(...),
    species: str = Form(""),
    animal_type: str = Form(""),
    notes: str = Form(""),
):
    gt = _load_ground_truth()

    if category == "bird":
        gt[sighting_id] = {"label": species.strip(), "is_bird": True}
    elif category == "critter":
        gt[sighting_id] = {"label": animal_type.strip(), "is_bird": False}
    else:
        gt[sighting_id] = {"label": "empty", "is_bird": False}

    if notes.strip():
        gt[sighting_id]["notes"] = notes.strip()

    _save_ground_truth(gt)
    logger.info("Labeled %s: %s", sighting_id, gt[sighting_id])
    return RedirectResponse("/", status_code=303)


@app.post("/skip")
async def skip(sighting_id: str = Form(...)):
    return RedirectResponse("/", status_code=303)


@app.get("/")
async def index():
    ensure_dirs()
    db = Database()
    gt = _load_ground_truth()
    unlabeled = _get_sightings_to_label(db, gt)
    species_list = _get_known_species(db)
    total = db.conn.execute("SELECT COUNT(*) FROM sightings WHERE image_path IS NOT NULL").fetchone()[0]
    labeled_count = len(gt)
    db.close()

    if not unlabeled:
        return HTMLResponse(f"<h1>All done!</h1><p>{labeled_count} / {total} labeled.</p>")

    s = unlabeled[0]
    # Convert absolute image path to relative media URL
    image_path = Path(s["image_path"])
    try:
        rel_path = image_path.relative_to(MEDIA_DIR)
    except ValueError:
        rel_path = image_path.name
    image_url = f"/media/{rel_path}"

    species_options = "\n".join(f'<option value="{sp}">' for sp in species_list)

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>FreeBird Labeler</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; }}
  .container {{ max-width: 900px; margin: 0 auto; padding: 20px; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }}
  h1 {{ font-size: 1.4em; }}
  .progress {{ color: #888; font-size: 0.95em; }}
  .main {{ display: flex; gap: 24px; }}
  .image-panel {{ flex: 1; }}
  .image-panel img {{ width: 100%; border-radius: 8px; border: 2px solid #333; }}
  .form-panel {{ width: 300px; flex-shrink: 0; }}
  .meta {{ font-size: 0.85em; color: #888; margin-bottom: 12px; }}
  .radio-group {{ display: flex; gap: 8px; margin-bottom: 16px; }}
  .radio-group label {{
    flex: 1; text-align: center; padding: 10px 0; border-radius: 6px;
    background: #2a2a3e; cursor: pointer; transition: all 0.15s;
    border: 2px solid transparent;
  }}
  .radio-group input {{ display: none; }}
  .radio-group input:checked + span {{ font-weight: 600; }}
  .radio-group label:has(input:checked) {{ border-color: #5b8def; background: #2a3a5e; }}
  .field {{ margin-bottom: 12px; }}
  .field label {{ display: block; font-size: 0.85em; color: #aaa; margin-bottom: 4px; }}
  .field input, .field select, .field textarea {{
    width: 100%; padding: 8px 10px; border-radius: 6px; border: 1px solid #444;
    background: #2a2a3e; color: #e0e0e0; font-size: 0.95em;
  }}
  .field textarea {{ resize: vertical; height: 60px; }}
  .buttons {{ display: flex; gap: 8px; margin-top: 16px; }}
  .buttons button {{
    flex: 1; padding: 10px; border: none; border-radius: 6px;
    font-size: 1em; cursor: pointer; font-weight: 500;
  }}
  .btn-save {{ background: #5b8def; color: #fff; }}
  .btn-save:hover {{ background: #4a7de0; }}
  .btn-skip {{ background: #3a3a4e; color: #ccc; }}
  .btn-skip:hover {{ background: #4a4a5e; }}
  .hint {{ font-size: 0.8em; color: #666; margin-top: 8px; text-align: center; }}
  #species-field, #animal-field {{ display: none; }}
</style>
</head><body>
<div class="container">
  <div class="header">
    <h1>FreeBird Labeler</h1>
    <div class="progress">{labeled_count} / {total} labeled &middot; {len(unlabeled)} remaining</div>
  </div>
  <div class="main">
    <div class="image-panel">
      <img src="{image_url}" alt="keyshot">
      <div class="meta">{s['timestamp']} &middot; {s['device_name'] or ''} &middot; Current: {s['species'] or 'none'}</div>
    </div>
    <div class="form-panel">
      <form id="label-form" method="post" action="/label">
        <input type="hidden" name="sighting_id" value="{s['id']}">

        <div class="radio-group">
          <label><input type="radio" name="category" value="bird" onchange="toggleFields()"><span>Bird</span></label>
          <label><input type="radio" name="category" value="critter" onchange="toggleFields()"><span>Critter</span></label>
          <label><input type="radio" name="category" value="empty" onchange="toggleFields()"><span>Empty</span></label>
        </div>

        <div class="field" id="species-field">
          <label>Species</label>
          <input type="text" name="species" list="species-list" placeholder="e.g. Dark-eyed Junco" autocomplete="off">
          <datalist id="species-list">{species_options}</datalist>
        </div>

        <div class="field" id="animal-field">
          <label>Animal type</label>
          <select name="animal_type">
            <option value="squirrel">Squirrel</option>
            <option value="chipmunk">Chipmunk</option>
            <option value="cat">Cat</option>
            <option value="unknown">Unknown</option>
          </select>
        </div>

        <div class="field">
          <label>Notes (optional)</label>
          <textarea name="notes" placeholder="Any observations..."></textarea>
        </div>

        <div class="buttons">
          <button type="button" class="btn-skip" onclick="document.getElementById('skip-form').submit()">Skip</button>
          <button type="submit" class="btn-save">Save &amp; Next</button>
        </div>
        <div class="hint">Enter = save &middot; &rarr; = skip</div>
      </form>
      <form id="skip-form" method="post" action="/skip" style="display:none">
        <input type="hidden" name="sighting_id" value="{s['id']}">
      </form>
    </div>
  </div>
</div>
<script>
function toggleFields() {{
  const cat = document.querySelector('input[name="category"]:checked');
  document.getElementById('species-field').style.display = cat && cat.value === 'bird' ? 'block' : 'none';
  document.getElementById('animal-field').style.display = cat && cat.value === 'critter' ? 'block' : 'none';
}}
document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowRight') document.getElementById('skip-form').submit();
}});
</script>
</body></html>"""

    return HTMLResponse(html)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
