#!/usr/bin/env python3
"""
composite_reward.py
================================================================
The COMPOSITE, FAITHFULNESS-GATED reward — the deployable analogue of the
synthetic oracle, computable from a generated narrative + its source record.

    reward = F^gamma * (Q + wc*C)  -  wfab*(#unsupported_specifics)  -  wsens*S

  F     faithfulness/grounding in [0,1] (multiplicative GATE)
  Q     surface detail quality (from linguistic_reward features)
  C     calibration (hedge - over-claim)
  S     sensationalism penalty (restraint)
  gamma sharpness of the gate (gamma>1: faithfulness must be high to count)

In the SYNTHETIC phase, F is an EXACT, free lookup against the fact base
(we own the facts). On real corpora it becomes an NLI/LLM-judge estimate.

LIMITATION (honest): entity+year grounding catches fabricated *entities and
dates* (Gestapo, "March 1939"). It does NOT catch a fabricated *claim* that
introduces no new entity ("smuggled forty children") — that needs claim-level
NLI. Here the sensationalism penalty happens to catch many such cases, but
real claim-level faithfulness is the v2 upgrade.

Run `python composite_reward.py` to validate on the grid: the composite should
flip good > fabricated where the linguistic-only reward failed.
"""

import math
import re
import statistics

from linguistic_reward import features, STOPWORDS, WORD_RE

# Reward weights + sensational lexicon live in config.py
from config import SENSATIONAL, SENSATIONAL_EXEMPT_ATTESTED, GAMMA, W_C, W_FAB, W_SENS
# Faithfulness is now a PLUGGABLE estimator (lexical / nli / llm) in faithfulness.py.
# Re-exported here so `from composite_reward import faithfulness` keeps working and
# `composite_reward()` automatically uses whatever config.FAITHFULNESS_METHOD selects.
from faithfulness import case_premise_text, faithfulness, grounded_index  # noqa: F401
Q_KEYS = ("proper_noun_density", "number_date_density", "concreteness", "lexical_density")


_CASE_BY_ID = None


def _case_by_id():
    global _CASE_BY_ID
    if _CASE_BY_ID is None:
        from generate_synthetic_corpus import CASES
        _CASE_BY_ID = {c["case_id"]: c for c in CASES}
    return _CASE_BY_ID


def _simple_stem(word):
    w = word.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[:-len(suffix)]
    return w


def _attested_sensational_stems(case=None, source_text=None):
    if source_text is None and case is not None:
        source_text = case_premise_text(case)
    if not source_text:
        return set()
    return {_simple_stem(w) for w in WORD_RE.findall(source_text)}


def sensationalism(text, case=None, source_text=None):
    words = [w.lower() for w in WORD_RE.findall(text)]
    attested = (_attested_sensational_stems(case=case, source_text=source_text)
                if SENSATIONAL_EXEMPT_ATTESTED and (case is not None or source_text)
                else set())
    s = sum(1 for w in words if w in SENSATIONAL and _simple_stem(w) not in attested)
    return s / math.sqrt(max(1, len(words)))


# ----------------------------------------------------------------------
# Composite reward
# ----------------------------------------------------------------------

def composite_reward(text, case, gamma=GAMMA, w_c=W_C, w_fab=W_FAB, w_sens=W_SENS):
    f = features(text)
    Q = statistics.mean(f[k] for k in Q_KEYS)
    C = f["calibration"]
    F, n_unsup = faithfulness(text, case)
    S = sensationalism(text, case=case)
    return round((F ** gamma) * (Q + w_c * C) - w_fab * n_unsup - w_sens * S, 4)


def composite_reward_by_id(text, case_id, **kw):
    return composite_reward(text, _case_by_id()[case_id], **kw)


# ----------------------------------------------------------------------
# Validation on the grid
# ----------------------------------------------------------------------

def main():
    from generate_synthetic_corpus import CASES, REGISTERS, PROFILES, render
    from linguistic_reward import linguistic_reward
    rows = {p: {"F": [], "unsup": [], "S": [], "ling": [], "comp": []} for p in PROFILES}
    flip_comp = flip_ling = 0   # cells where good > fabricated

    for case in CASES:
        for reg in REGISTERS:
            per = {}
            for p in PROFILES:
                t = render(case, reg, p)
                F, nu = faithfulness(t, case)
                rows[p]["F"].append(F)
                rows[p]["unsup"].append(nu)
                rows[p]["S"].append(sensationalism(t))
                rows[p]["ling"].append(linguistic_reward(t))
                rows[p]["comp"].append(composite_reward(t, case))
                per[p] = (linguistic_reward(t), composite_reward(t, case))
            if per["good"][1] > per["fabricated"][1]:
                flip_comp += 1
            if per["good"][0] > per["fabricated"][0]:
                flip_ling += 1

    def m(p, k):
        return statistics.mean(rows[p][k])

    n = len(CASES) * len(REGISTERS)
    print(f"{'profile':<12}{'F':>7}{'unsup':>8}{'sens':>8}{'linguistic':>13}{'composite':>12}")
    print("-" * 60)
    for p in PROFILES:
        print(f"{p:<12}{m(p,'F'):>7.2f}{m(p,'unsup'):>8.2f}{m(p,'S'):>8.3f}"
              f"{m(p,'ling'):>13.3f}{m(p,'comp'):>12.3f}")

    print(f"\nCells where GOOD ranks above FABRICATED (out of {n}):")
    print(f"  linguistic-only reward : {flip_ling}/{n}")
    print(f"  composite reward       : {flip_comp}/{n}")
    print("\nReading: the faithfulness gate collapses fabricated text's reward")
    print("(low F scales quality toward 0, plus the unsupported-specifics and")
    print("sensationalism penalties), so the composite recovers the oracle's")
    print("ordering that the surface reward could not see.")


if __name__ == "__main__":
    main()
