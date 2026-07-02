#!/usr/bin/env python3
"""
human_reward.py
================================================================
A SYNTHETIC HUMAN reward — the human/community arm of the comparison, stood up
without real annotators. It scores the oracle's value dimensions FROM TEXT and
combines them with a PERSONA weighting (and optional noise), so we can run the
two-arm experiment (human-aligned vs linguistic-aligned) and study pluralism
("whose good detail?") today.

Dimensions (each ~[0,1], computed from the generation + its source record):
  coverage    – fraction of the record's salient specifics that appear
  specificity – surface concreteness (from linguistic_reward)
  grounding   – F, faithfulness (from composite_reward; exact in synthetic)
  source      – does it reference a source / attribute its claims?
  calibration – hedges vs over-claims (from linguistic_reward)
  restraint   – 1 − sensationalism
  (penalty)   – unsupported specifics (fabrications)

A PERSONA is a weighting over these — different historians value different
things. Swapping persona is the one-line pluralism knob.

Run `python human_reward.py` for the two-arm + pluralism report.
"""

import random
import re
import statistics

from linguistic_reward import features, linguistic_reward
from composite_reward import faithfulness, sensationalism

# Personas (the pluralism knob) + source-attribution words live in config.py
from config import SOURCE_WORDS, PERSONAS
POS_DIMS = ("coverage", "specificity", "grounding", "source", "calibration", "restraint")


_CASE_BY_ID = None


def _case_by_id():
    global _CASE_BY_ID
    if _CASE_BY_ID is None:
        from generate_synthetic_corpus import CASES
        _CASE_BY_ID = {c["case_id"]: c for c in CASES}
    return _CASE_BY_ID


def _salient_strings(case):
    out = set()
    for ent in case["entities"].values():
        if ent["type"] in ("person", "place", "org"):
            out.add(ent["name"])
    for ev in case["events"]:
        if ev["salience"] >= 2:
            out.add(ev["date"]["value"])
    return out


def component_scores(text, case):
    low = text.lower()
    sal = _salient_strings(case)
    coverage = sum(1 for s in sal
                   if re.search(rf"\b{re.escape(s)}\b", text, flags=re.I)) / max(1, len(sal))
    f = features(text)
    specificity = f["concreteness"]
    calibration = max(0.0, min(1.0, 0.5 + 2.0 * f["calibration"]))
    F, n_unsup = faithfulness(text, case)
    has_src = any(w in low for w in SOURCE_WORDS) or \
        any(s["archive_ref"].lower() in low for s in case["sources"].values())
    restraint = max(0.0, 1.0 - sensationalism(text, case=case))
    return dict(coverage=coverage, specificity=specificity, grounding=F,
                source=1.0 if has_src else 0.0, calibration=calibration,
                restraint=restraint, unsup=n_unsup)


def human_reward(text, case, persona="balanced", noise=0.0, rng=None):
    w = PERSONAS[persona]
    c = component_scores(text, case)
    score = sum(w[k] * c[k] for k in POS_DIMS) - w["fab"] * c["unsup"]
    if noise:
        score += (rng or random).gauss(0.0, noise)
    return round(score, 4)


def human_reward_by_id(text, case_id, persona="balanced", noise=0.0):
    return human_reward(text, _case_by_id()[case_id], persona, noise)


# ----------------------------------------------------------------------
# Two-arm + pluralism report
# ----------------------------------------------------------------------

def main():
    from generate_synthetic_corpus import CASES, REGISTERS, PROFILES, render
    cells = [(c, r) for c in CASES for r in REGISTERS]
    n = len(cells)

    # --- per-persona means + profile ordering -------------------------
    print("Synthetic-human reward by persona (mean over the grid):\n")
    print(f"{'persona':<18}{'good':>8}{'flattened':>11}{'fabricated':>12}   ordering")
    print("-" * 64)
    for p in PERSONAS:
        means = {prof: statistics.mean(human_reward(render(c, r, prof), c, p)
                                       for c, r in cells) for prof in PROFILES}
        order = " > ".join(sorted(PROFILES, key=lambda x: -means[x]))
        print(f"{p:<18}{means['good']:>8.2f}{means['flattened']:>11.2f}"
              f"{means['fabricated']:>12.2f}   {order}")

    # --- two-arm comparison: human(balanced) vs linguistic ------------
    print("\nTwo-arm comparison — human(balanced) vs linguistic (good > X):")
    for other in ("flattened", "fabricated"):
        h = sum(human_reward(render(c, r, "good"), c, "balanced") >
                human_reward(render(c, r, other), c, "balanced") for c, r in cells)
        l = sum(linguistic_reward(render(c, r, "good")) >
                linguistic_reward(render(c, r, other)) for c, r in cells)
        print(f"  good > {other:<11}: human {h}/{n}   linguistic {l}/{n}")

    # --- pluralism: do personas agree on flattened-vs-fabricated? -----
    print("\nPluralism probe — does each persona rank FLATTENED above FABRICATED?")
    for p in PERSONAS:
        flat_over_fab = sum(human_reward(render(c, r, "flattened"), c, p) >
                            human_reward(render(c, r, "fabricated"), c, p) for c, r in cells)
        print(f"  {p:<18}: flattened > fabricated in {flat_over_fab}/{n} cells")
    print("\nReading: all personas reject hallucinated fabrication, but they DISAGREE")
    print("on whether a bland-but-honest record (flattened) beats a vivid-but-fabricated")
    print("one — the ethics/grounding personas say yes emphatically, a specificity-first")
    print("persona is more tempted by the vivid fake. That disagreement is the synthetic")
    print("form of value pluralism: 'whose good detail?' becomes measurable.")

    # --- noise demo ---------------------------------------------------
    rng = random.Random(0)
    noisy = [human_reward(render(CASES[0], "testimony", "good"), CASES[0], "balanced",
                          noise=0.5, rng=rng) for _ in range(5)]
    print(f"\nNoise demo (balanced, good, sigma=0.5), 5 draws: {[round(x,2) for x in noisy]}")
    print("-> add noise / multiple personas to synthesise a realistic annotation set.")


if __name__ == "__main__":
    main()
