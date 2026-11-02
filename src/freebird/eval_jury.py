"""Score models without expert labels, using a jury of the strongest models as the answer key.

  uv run python -m freebird.eval_jury                 # table: each model vs the jury (leave-one-out)
  uv run python -m freebird.eval_jury --make-check    # pick ~30 photos for an outside check (Merlin app)

For each image and each model being scored, the answer is what the OTHER jurors say, and only when at
least --agree (default 75%) of them say the same thing. A model never votes on its own score. Hard
views (tail only, cropped, blurry) are left out, because nobody can name the bird.

Human labels beat the jury where they exist:
  eval/sets/sample_gold.json        bird / empty / critter from eval_label --presence (species optional)
  eval/sets/sample_check_gold.json  species from an outside check of the photos --make-check picks

The jury shares blind spots (they can all call a house finch a house sparrow), so the outside check
on the photos they split on, plus a sample of the ones they agree on, is how we catch that.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
from pathlib import Path

from freebird.eval_compare import load_set
from freebird.eval_score import EVAL_DIR, cost_per_image, klass, load_runs, view_tier

JURY = [
    "anthropic_claude-fable-5-1", "anthropic_claude-opus-5-5", "anthropic_claude-sonnet-5-5",
    "google-gla_gemini-4-argon", "google-gla_gemini-3-flash-preview",
]
PROMPTS = ["default_v2", "default_v4"]  # v3 over-used "Unidentified bird", so it is not scored
CHECK_SET = "sample_check"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def usable(recs: dict, sid: str) -> dict | None:
    r = recs.get(sid)
    return r["pred"] if r and not r.get("error") else None


def load_all(set_name: str, prompts: list[str]):
    """({model@prompt: recs}, {model: recs of its newest prompt}, {sid: [all preds]})."""
    runs: dict[str, dict] = {}
    best: dict[str, dict] = {}
    for prompt in prompts:  # later prompts override earlier ones in `best`
        for model, recs in load_runs(set_name, prompt).items():
            runs[f"{model}@{prompt}"] = recs
            best[model] = recs
    preds: dict[str, list[dict]] = collections.defaultdict(list)
    for recs in runs.values():
        for sid in recs:
            if (p := usable(recs, sid)):
                preds[sid].append(p)
    return runs, best, preds


def jury_class(best: dict, jurors: list[str], sid: str, exclude: str | None, agree: float) -> str | None:
    """What the other jurors say about one image, or None if they are too split (or too few)."""
    votes = collections.Counter(
        klass(p) for m in jurors if m != exclude and m in best and (p := usable(best[m], sid)))
    total = sum(votes.values())
    if total < 3:
        return None
    top, n = votes.most_common(1)[0]
    return top if n / total >= agree else None


def human_truth(set_name: str) -> tuple[dict[str, bool], dict[str, str]]:
    """({sid: is_bird} from presence labels, {sid: class} from labels that name a species or animal)."""
    pres: dict[str, bool] = {}
    cls: dict[str, str] = {}
    for fname in (f"{set_name}_gold.json", f"{CHECK_SET}_gold.json"):
        for sid, lab in read_json(EVAL_DIR / "sets" / fname).items():
            if lab.get("unsure"):
                continue
            pres[sid] = bool(lab["is_bird"])
            if lab["label"] == "empty":
                cls[sid] = "empty"
            elif lab["label"]:
                from freebird.species import norm_species
                cls[sid] = ("bird:" if lab["is_bird"] else "animal:") + norm_species(lab["label"])
    return pres, cls


def score(set_name: str, jurors: list[str], prompts: list[str], agree: float, per_month: int) -> None:
    runs, best, preds = load_all(set_name, prompts)
    ids = [sid for sid, ps in preds.items() if view_tier(ps) != 2]
    h_pres, h_cls = human_truth(set_name)
    print(f"{len(preds)} images with runs, {len(preds) - len(ids)} hard views left out, "
          f"{len(h_pres)} with a human label\n")
    rows = []
    for name, recs in runs.items():
        model = name.split("@")[0]
        n_p = ok_p = n_s = ok_s = unk = 0
        for sid in ids:
            p = usable(recs, sid)
            if p is None:
                continue
            jc = jury_class(best, jurors, sid, model, agree)
            truth = h_cls.get(sid) or jc
            is_bird = h_pres.get(sid)
            if is_bird is None and truth is not None:
                is_bird = truth.startswith("bird:")
            if is_bird is not None:
                n_p += 1
                ok_p += p["is_bird"] == is_bird
            if truth and truth.startswith("bird:") and truth != "bird:unknown":
                n_s += 1
                ok_s += klass(p) == truth
                unk += klass(p) == "bird:unknown"
        c = cost_per_image(model, recs)
        rows.append((name, n_p, ok_p, n_s, ok_s, unk, c[2] * per_month if c else None))
    print(f"{'model@prompt':50} {'n':>4} {'bird/not':>8} {'n':>4} {'species':>8} {'unknown':>8} {'$/mo':>7}")
    for name, n_p, ok_p, n_s, ok_s, unk, cost in sorted(rows, key=lambda r: -(r[4] / r[3] if r[3] else 0)):
        mark = "*" if name.split("@")[0] in jurors else " "
        print(f"{name + mark:50} {n_p:4} {100 * ok_p / n_p if n_p else 0:7.1f}% {n_s:4} "
              f"{100 * ok_s / n_s if n_s else 0:7.1f}% {100 * unk / n_s if n_s else 0:7.1f}% "
              f"{cost if cost is not None else float('nan'):7.2f}")
    print(f"\n* = on the jury (scored against the other jurors only). $/mo is at {per_month} events a month.\n"
          "species = names the same species as the answer key; unknown = said 'unknown bird' on a photo that has an answer.")


def make_check(set_name: str, db: Path, jurors: list[str], n_random: int, seed: int) -> None:
    """Pick the photos to check outside: every clear bird the jury splits on, plus a random few they agree on."""
    _, best, preds = load_all(set_name, PROMPTS)
    split, agreed = [], []
    for sid, ps in sorted(preds.items()):
        if view_tier(ps) != 0:
            continue
        votes = collections.Counter(
            klass(p) for m in jurors if m in best and (p := usable(best[m], sid)))
        top, n = votes.most_common(1)[0]
        if n / sum(votes.values()) < 0.8:
            split.append(sid)
        elif top.startswith("bird:") and top != "bird:unknown":
            agreed.append(sid)  # agreed on a species; a sample of these tests for shared blind spots
    picked = sorted(set(split) | set(random.Random(seed).sample(agreed, min(n_random, len(agreed)))))
    path = EVAL_DIR / "sets" / f"{CHECK_SET}.json"
    path.write_text(json.dumps({"ids": picked, "seed": seed,
                                "note": f"outside check: {len(split)} the jury splits on + {len(picked) - len(split)} it agrees on"},
                               indent=1) + "\n")
    out = EVAL_DIR / "work" / "merlin_check"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    items = dict(load_set(CHECK_SET, db))
    for i, sid in enumerate(sorted(items), 1):  # same numbers eval_label shows as "photo NN"
        shutil.copy(items[sid], out / f"{i:02d}.jpg")
    print(f"{len(picked)} photos ({len(split)} split + {len(picked) - len(split)} agreed) -> {out}")
    print(f"Check them in Merlin, then type the species with:\n"
          f"  uv run python -m freebird.eval_label --set {CHECK_SET}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="sample")
    ap.add_argument("--jury", nargs="+", default=JURY, help="model slugs that vote")
    ap.add_argument("--prompts", nargs="+", default=PROMPTS)
    ap.add_argument("--agree", type=float, default=0.75, help="share of the other jurors that must agree")
    ap.add_argument("--events", type=int, default=3000, help="events per month for the cost column")
    ap.add_argument("--make-check", action="store_true")
    ap.add_argument("--random", type=int, default=15, help="with --make-check: agreed photos to add")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--db", default=str(EVAL_DIR / "work" / "birds.db"))
    args = ap.parse_args()
    if args.make_check:
        make_check(args.set, Path(args.db), args.jury, args.random, args.seed)
    else:
        score(args.set, args.jury, args.prompts, args.agree, args.events)


if __name__ == "__main__":
    main()
