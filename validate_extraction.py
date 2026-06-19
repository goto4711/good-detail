#!/usr/bin/env python3
"""
validate_extraction.py
================================================================
A trust check for the DSPy front-end before a full-scale run: run the extractor
on the GOLD documents' own text and compare predicted entities + relations to the
gold annotations (same text, pred vs gold) — precision / recall / F1.

Entity match is exact-text (case-insensitive) — strict, so treat it as a floor.
Relation match is type-AGNOSTIC on the (head, tail) pair: did the extractor find
the right *link*? (relation_type wording varies too much for exact match.)

  python validate_extraction.py --backend uva --limit 10
"""

import argparse
import json
import sys

from dspy_extract import build_extractor, make_lm


def _ent_set(items):
    return {e.get("entity_text", "").strip().lower() for e in items if e.get("entity_text")}


def _rel_set(items):
    out = set()
    for r in items:
        h = r.get("head_entity_text", "").strip().lower()
        t = r.get("tail_entity_text", "").strip().lower()
        if h and t:
            out.add((h, t))
    return out


def prf(gold, pred):
    if not gold and not pred:
        return (1.0, 1.0, 1.0)
    tp = len(gold & pred)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return (p, r, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/EHRI/iob/merged_intermediate_data.jsonl")
    ap.add_argument("--backend", default="uva")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    try:
        import dspy  # noqa: F401
    except ImportError:
        sys.exit("DSPy not installed:  pip install dspy-ai")

    make_lm(args.backend, args.model)
    ext = build_extractor()
    rows = [json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()]
    rows = [d for d in rows if d.get("document_text")][:args.limit]
    if not rows:
        sys.exit(f"No gold rows with document_text in {args.gold}")

    tot = {"ep": 0.0, "er": 0.0, "ef": 0.0, "rp": 0.0, "rr": 0.0, "rf": 0.0}
    n = 0
    for d in rows:
        gold_e = _ent_set(d.get("base_entities", []) + d.get("new_entities", []))
        gold_r = _rel_set(d.get("relationships", []))
        try:
            pred = ext(document_text=d["document_text"])
        except Exception as e:
            print(f"  fail {d.get('id')}: {e}")
            continue
        ep = prf(gold_e, _ent_set(pred["entities"]))
        rp = prf(gold_r, _rel_set(pred["relationships"]))
        tot["ep"] += ep[0]; tot["er"] += ep[1]; tot["ef"] += ep[2]
        tot["rp"] += rp[0]; tot["rr"] += rp[1]; tot["rf"] += rp[2]
        n += 1
        print(f"  {d.get('id')}: entities F1={ep[2]:.2f} (P{ep[0]:.2f}/R{ep[1]:.2f})  "
              f"relation-link F1={rp[2]:.2f} (P{rp[0]:.2f}/R{rp[1]:.2f})")

    if n:
        print(f"\nMEAN over {n}:  entities F1={tot['ef']/n:.2f} "
              f"(P{tot['ep']/n:.2f}/R{tot['er']/n:.2f})  "
              f"relation-link F1={tot['rf']/n:.2f} (P{tot['rp']/n:.2f}/R{tot['rr']/n:.2f})")
        print("Reading: entity recall ~ does it find the right names (exact-match floor); "
              "relation-link F1 ~ does it find the right entity pairs. Low scores -> the fact "
              "base is noisy and downstream grounding inherits it; consider --compile.")


if __name__ == "__main__":
    main()
