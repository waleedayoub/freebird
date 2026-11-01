"""Score eval_compare runs.

  uv run python -m freebird.eval_score --set ground_truth      # accuracy vs hand labels
  uv run python -m freebird.eval_score --set sample            # agreement / consensus (no labels)
  uv run python -m freebird.eval_score --set sample --gold eval/sets/sample_gold.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"

from freebird.species import norm_species  # noqa: E402


def klass(pred: dict) -> str:
    """Collapse a prediction to one comparable class: 'empty', 'animal:<x>' or 'bird:<species>'."""
    if pred["is_bird"]:
        return "bird:" + (norm_species(pred.get("species")) or "unknown")
    at = norm_species(pred.get("animal_type"))
    return f"animal:{at}" if at else "empty"


def load_runs(set_name: str, prompt: str) -> dict[str, dict[str, dict]]:
    runs: dict[str, dict[str, dict]] = {}
    for f in sorted((EVAL_DIR / "runs" / set_name / prompt).glob("*.jsonl")):
        recs: dict[str, dict] = {}
        for line in f.read_text().splitlines():
            r = json.loads(line)
            recs[r["id"]] = r  # last record wins (retries)
        if any(not r.get("error") for r in recs.values()):
            runs[f.stem] = recs
    return runs


def expected_class(label: dict) -> str:
    if label["label"] == "empty":
        return "empty"
    if label["is_bird"]:
        return "bird:" + (norm_species(label["label"]) or "unknown")
    return "animal:" + norm_species(label["label"])


def score_gold(runs, gold: dict[str, dict]) -> None:
    print(f"{'model':42} {'n':>4} {'err':>3} {'bird/not':>8} {'class':>7} {'species*':>8} {'sec':>5} {'in_tok':>6} {'out_tok':>7}")
    for name, recs in runs.items():
        n = ok_bird = ok_class = sp_n = sp_ok = errs = 0
        secs = []; it = []; ot = []
        for sid, lab in gold.items():
            if lab["is_bird"] and not lab["label"]:
                continue  # unlabeled bird
            r = recs.get(sid)
            if r is None:
                continue
            if r.get("error"):
                errs += 1
                continue
            n += 1
            p = r["pred"]
            ok_bird += p["is_bird"] == lab["is_bird"]
            ok_class += klass(p) == expected_class(lab)
            if lab["is_bird"]:
                sp_n += 1
                sp_ok += klass(p) == expected_class(lab)
            secs.append(r["secs"]); it.append(r.get("in_tok") or 0); ot.append(r.get("out_tok") or 0)
        if n:
            print(f"{name:42} {n:4} {errs:3} {100*ok_bird/n:7.1f}% {100*ok_class/n:6.1f}% "
                  f"{(100*sp_ok/sp_n if sp_n else 0):7.1f}% {sum(secs)/n:5.1f} {sum(it)/n:6.0f} {sum(ot)/n:7.0f}")
    print("* species* = accuracy on images whose label is a bird species")


def score_consensus(runs, prod_labels: dict[str, str]) -> dict:
    """Majority vote across models; report each model's agreement and list contested images."""
    ids = sorted(set.intersection(*(set(k for k, v in r.items() if not v.get("error")) for r in runs.values())))
    votes: dict[str, collections.Counter] = {}
    for sid in ids:
        votes[sid] = collections.Counter(klass(r[sid]["pred"]) for r in runs.values())
    consensus = {sid: v.most_common(1)[0] for sid, v in votes.items()}
    print(f"{len(ids)} images complete across {len(runs)} models\n")
    print(f"{'model':42} {'agree w/ majority':>17} {'agree w/ production':>19} {'bird/not vs maj':>15}")
    for name, recs in runs.items():
        a = b = c = 0
        for sid in ids:
            k = klass(recs[sid]["pred"])
            maj = consensus[sid][0]
            a += k == maj
            b += k == prod_labels.get(sid)
            c += k.startswith("bird:") == maj.startswith("bird:")
        print(f"{name:42} {100*a/len(ids):16.1f}% {100*b/len(ids):18.1f}% {100*c/len(ids):14.1f}%")
    contested = [sid for sid in ids if consensus[sid][1] < 0.6 * len(runs)]
    unanimous = sum(1 for sid in ids if consensus[sid][1] == len(runs))
    print(f"\nunanimous: {unanimous}/{len(ids)}   contested (<60% agree): {len(contested)}")
    return {"ids": ids, "votes": votes, "consensus": consensus, "contested": contested}


# USD per 1M tokens (input, output). Claude prices are first-party list prices. Gemini prices come
# from third-party summaries as of Oct 2026 and Google may change them (Argon's $2/$10 is a promo
# through Nov 30; $4/$20 after), so treat the Gemini rows as estimates.
PRICES = {
    "anthropic_claude-haiku-5-5": (0.10, 0.50), "anthropic_claude-sonnet-5-5": (2.0, 10.0),
    "anthropic_claude-opus-5-5": (4.0, 20.0), "anthropic_claude-fable-5-1": (10.0, 50.0),
    "google-gla_gemini-3.1-flash-lite": (0.25, 1.50), "google-gla_gemini-3-flash-preview": (0.50, 3.0),
    "google-gla_gemini-4-argon": (2.0, 10.0),
}


def score_cost(runs, events_per_month: int = 3000) -> None:
    """Average real token use per image x list price. Thinking tokens bill as output."""
    print(f"\n{'model':42} {'in_tok':>6} {'out_tok':>7} {'$/1k imgs':>10} {'$/mo @' + str(events_per_month):>12}")
    for name, recs in runs.items():
        ok = [r for r in recs.values() if not r.get("error") and r.get("in_tok")]
        if not ok or name not in PRICES:
            continue
        pin, pout = PRICES[name]
        it = sum(r["in_tok"] for r in ok) / len(ok)
        ot = sum(r.get("out_tok") or 0 for r in ok) / len(ok)
        per = (it * pin + ot * pout) / 1e6
        print(f"{name:42} {it:6.0f} {ot:7.0f} {1000 * per:10.2f} {events_per_month * per:12.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="ground_truth")
    ap.add_argument("--prompt", default="default_v2")
    ap.add_argument("--gold")
    ap.add_argument("--prod-db", default=str(EVAL_DIR / "work" / "birds.db"))
    args = ap.parse_args()
    runs = load_runs(args.set, args.prompt)
    if args.set == "ground_truth":
        score_gold(runs, json.loads((EVAL_DIR / "ground_truth.json").read_text()))
    elif args.gold:
        score_gold(runs, json.loads(Path(args.gold).read_text()))
    else:
        import sqlite3
        c = sqlite3.connect(f"file:{args.prod_db}?mode=ro", uri=True)
        prod = {sid: klass({"is_bird": bool(b), "species": sp, "animal_type": at})
                for sid, b, sp, at in c.execute(
                    "SELECT sighting_id,is_bird,species,animal_type FROM vision_analyses")}
        score_consensus(runs, prod)
    score_cost(runs)


if __name__ == "__main__":
    main()
