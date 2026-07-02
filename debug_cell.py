#!/usr/bin/env python3
"""Find the composite grid cell(s) where good <= fabricated and dump why.

Usage:
  python debug_cell.py                          # full grid + verdicts for failing cells
  python debug_cell.py synth_case_005 testimony # per-claim verdicts for ONE cell
"""
import statistics
import sys
import faithfulness
from generate_synthetic_corpus import CASES, REGISTERS, render
from linguistic_reward import features
from composite_reward import composite_reward, sensationalism, Q_KEYS
from config import GAMMA, W_C, W_FAB, W_SENS


def components(text, case):
    f = features(text)
    Q = statistics.mean(f[k] for k in Q_KEYS)
    C = f["calibration"]
    F, nu = faithfulness.faithfulness(text, case, method="nli")
    S = sensationalism(text, case=case)
    comp = (F ** GAMMA) * (Q + W_C * C) - W_FAB * nu - W_SENS * S
    return dict(F=F, unsup=nu, Q=Q, C=C, S=S, comp=comp)


def dump_cell(case, reg):
    """Per-claim verdicts for one cell. Clears BOTH caches (the generic _CACHE and
    the fix(8) direct-NLI _NLI_CACHE) so the debug pass actually re-scores."""
    for prof in ("good", "fabricated"):
        t = render(case, reg, prof)
        print(f"\n--- {prof}:")
        print(t)
        print("\nPer-claim verdicts:")
        faithfulness._DEBUG = True
        faithfulness._CACHE.clear()
        getattr(faithfulness, "_NLI_CACHE", {}).clear()
        F, nu = faithfulness.faithfulness(t, case, method="nli")
        faithfulness._DEBUG = False
        print(f"  => F={F:.2f}  unsupported={nu}")


if len(sys.argv) == 3:                       # single-cell mode
    case = next(c for c in CASES if c["case_id"] == sys.argv[1])
    print("PREMISES:")
    for p in faithfulness.case_premise_facts(case):
        print("   ", p)
    dump_cell(case, sys.argv[2])
    raise SystemExit

fails = []
for case in CASES:
    for reg in REGISTERS:
        g = components(render(case, reg, "good"), case)
        b = components(render(case, reg, "fabricated"), case)
        marker = ""
        if g["comp"] <= b["comp"]:
            fails.append((case, reg, g, b))
            marker = "  <-- FAIL"
        print(f"{case['case_id']:<16}{reg:<13}good {g['comp']:+.3f} (F={g['F']:.2f} u={g['unsup']})   "
              f"fab {b['comp']:+.3f} (F={b['F']:.2f} u={b['unsup']}){marker}")

for case, reg, g, b in fails:
    print("\n" + "=" * 70)
    print(f"FAILING CELL: {case['case_id']} / {reg}")
    print("=" * 70)
    print("PREMISES:")
    for p in faithfulness.case_premise_facts(case):
        print("   ", p)
    dump_cell(case, reg)

if not fails:
    print("\nNo failing cell — composite is 16/16 in this environment.")
