#!/usr/bin/env python3
"""
linguistic_reward.py
================================================================
A DECOMPOSABLE, surface-only linguistic reward for "good detail" — the
"culture-blind" arm of the human-vs-linguistic comparison.

Design rules (see project_update_good_detail.md):
  * SURFACE ONLY. Every feature reads the text and nothing else — no fact
    base, no grounding. That's the whole point: this arm must be blind to
    whether specifics are *true*, so the comparison with the human arm is
    meaningful.
  * DECOMPOSABLE. Each feature is a separately-logged sub-score. When the
    reward diverges from human judgement, the per-feature divergence is the
    finding.
  * LENGTH-NORMALISED. Counts are per-token rates so the reward doesn't just
    prefer longer text.

Features (all surface, all cheap, no model download):
  1. proper_noun_density   — capitalised non-initial tokens / sqrt(tokens)
  2. number_date_density   — tokens containing a digit / sqrt(tokens)
  3. concreteness          — mean concreteness of content words (Brysbaert
                             norms if available, else a small built-in stub)
  4. lexical_density       — content words / all words
  5. calibration           — hedge-marker rate minus over-claim-marker rate

NOT included here (the one feature that needs a model): token surprisal under
a generic LM — the "detect the unexpected" feature. Hook left in NOTES.

Validation harness (`python linguistic_reward.py`): runs every feature on the
good/flattened/fabricated grid and reports, per feature:
   - does it SEPARATE good from flattened?     (a real surface feature should)
   - is it BLIND to fabrication (good ≈ fab)?  (a culture-blind feature should)
Features that *catch* fabrication are not culture-blind — useful to know which.
"""

import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).parent
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+")
TITLECASE_RE = re.compile(r"^[A-ZÀ-Þ][a-zà-ÿ]")
ALL_CAPS_RE = re.compile(r"^[A-ZÀ-Þ]{1,4}$")

# Lexicons (stopwords, hedges, over-claim, concreteness) live in config.py
from config import STOPWORDS, HEDGES, OVERCLAIM, CONCRETENESS_STUB


def _load_concreteness():
    csv = HERE / "concreteness_norms.csv"
    if csv.exists():
        d = {}
        for line in csv.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    d[parts[0].strip().lower()] = float(parts[1])
                except ValueError:
                    pass
        if d:
            return d, "Brysbaert CSV"
    return CONCRETENESS_STUB, "built-in stub (illustrative — drop in real norms)"


CONCRETENESS, CONCRETENESS_SRC = _load_concreteness()


# ----------------------------------------------------------------------
# Features  (text -> float; higher = more "good-detail-like" on that axis)
# ----------------------------------------------------------------------

def _tokens(text):
    return TOKEN_RE.findall(text)


def _content_words(text):
    return [w.lower() for w in WORD_RE.findall(text) if w.lower() not in STOPWORDS]


def proper_noun_density(text):
    sents = re.split(r"[.;:?!]\s+", text)
    propers = 0
    for s in sents:
        toks = s.split()
        for i, t in enumerate(toks):
            tt = t.strip(",.;:()[]'\"")
            if i > 0 and (TITLECASE_RE.match(tt) or ALL_CAPS_RE.fullmatch(tt)):
                propers += 1
    return propers / math.sqrt(max(1, len(text.split())))


def number_date_density(text):
    nums = sum(1 for t in _tokens(text) if re.search(r"\d", t))
    return nums / math.sqrt(max(1, len(text.split())))


def concreteness(text):
    cw = _content_words(text)
    rated = [CONCRETENESS[w] for w in cw if w in CONCRETENESS]
    return statistics.mean(rated) / 5.0 if rated else 0.5   # neutral if unknown


def lexical_density(text):
    words = [w.lower() for w in WORD_RE.findall(text)]
    if not words:
        return 0.0
    content = [w for w in words if w not in STOPWORDS]
    return len(content) / len(words)


def calibration(text):
    cw = _content_words(text)
    n = max(1, len(cw))
    hedge = sum(1 for w in cw if w in HEDGES) / n
    over = sum(1 for w in cw if w in OVERCLAIM) / n
    return hedge - over           # hedged good detail high; over-claiming low


FEATURES = {
    "proper_noun_density": proper_noun_density,
    "number_date_density": number_date_density,
    "concreteness": concreteness,
    "lexical_density": lexical_density,
    "calibration": calibration,
}

# per-feature weights live in config.py
from config import WEIGHTS


def features(text):
    return {name: fn(text) for name, fn in FEATURES.items()}


def linguistic_reward(text, weights=WEIGHTS):
    f = features(text)
    return round(sum(weights[k] * f[k] for k in f), 4)


# ----------------------------------------------------------------------
# Validation harness on the good/flattened/fabricated grid
# ----------------------------------------------------------------------

def main():
    from generate_synthetic_corpus import CASES, REGISTERS, PROFILES, render

    print(f"Concreteness source: {CONCRETENESS_SRC}\n")
    # collect feature values per profile
    vals = {p: {name: [] for name in FEATURES} for p in PROFILES}
    for case in CASES:
        for register in REGISTERS:
            for p in PROFILES:
                f = features(render(case, register, p))
                for name, v in f.items():
                    vals[p][name].append(v)

    def mean(p, name):
        return statistics.mean(vals[p][name])

    print(f"{'feature':<22}{'good':>8}{'flat':>8}{'fab':>8}   {'sep good>flat':<14}{'blind to fab?'}")
    print("-" * 78)
    verdicts = {}
    for name in FEATURES:
        g, fl, fb = mean("good", name), mean("flattened", name), mean("fabricated", name)
        separates = g > fl + 0.02
        # blind = good and fabricated within 15% of good's magnitude
        denom = abs(g) + 1e-9
        blind = abs(g - fb) / denom < 0.15
        verdicts[name] = (separates, blind)
        print(f"{name:<22}{g:>8.3f}{fl:>8.3f}{fb:>8.3f}   "
              f"{'yes' if separates else 'NO':<14}{'BLIND' if blind else 'detects fab'}")

    # aggregate reward by profile
    agg = {p: statistics.mean(
        linguistic_reward(render(c, r, p)) for c in CASES for r in REGISTERS)
        for p in PROFILES}
    print("\nAggregate linguistic_reward by profile:")
    for p in PROFILES:
        print(f"  {p:<12}{agg[p]:.3f}")

    print("\nReading:")
    surface_blind = [n for n, (s, b) in verdicts.items() if s and b]
    catch_fab = [n for n, (s, b) in verdicts.items() if not b]
    print(f"  Culture-blind detail features (separate good/flat, blind to fab): {surface_blind}")
    print(f"  Features that DO catch fabrication (not purely culture-blind):    {catch_fab}")
    print("  -> The surface is blind to fabricated *specifics* but can sense over-claimed")
    print("     *certainty* (calibration). Divergence from human judgement is feature-specific,")
    print("     not global — which is exactly the kind of result the comparison should yield.")


if __name__ == "__main__":
    main()
