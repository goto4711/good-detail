#!/usr/bin/env python3
"""
llm_render.py
================================================================
v2 rendering: LLM-WRITTEN narratives with GROUNDING-VERIFICATION.

Until now the grid text was templated (deterministic, stilted). This generates
the narratives with an LLM (reusing the judge backends: uva / ollama / gemini)
and VERIFIES each against the fact base with the exact faithfulness check
(composite_reward.faithfulness): generate -> verify -> accept or regenerate.

Per style (the accept rule is the whole point):
  good       : accept only if 0 unsupported specifics (fully grounded);
               otherwise regenerate (up to --max-tries). This is the verified
               loop — it guarantees grounded SFT targets, fixing the templated-
               target caveat. Falls back to the templated 'good' if the model
               can't produce a clean one.
  flattened  : a vague, generic summary (negative).
  fabricated : a vivid narrative that DOES invent specifics (accept iff it
               hallucinates ≥1 ungrounded specific — synthetic adversarial data).

Writes synthetic_corpus/llm_renderings.jsonl  {case_id, register, style, text,
F, unsupported, accepted, tries, source}.

Usage:
  python llm_render.py --mock                              # offline (templated stand-in)
  python llm_render.py --backend ollama --model llama3.2 --limit 2
  python llm_render.py --backend gemini --limit 4
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import llm_judge_reward as J          # reuse its multi-backend chat client
    from composite_reward import faithfulness
    from generate_synthetic_corpus import CASES, REGISTERS, render
    from grpo_train import _record_block
except ImportError as e:
    sys.exit(f"Run from the project folder: {e}")

HERE = Path(__file__).parent
OUT = HERE / "synthetic_corpus" / "llm_renderings.jsonl"

# v2 rendering prompts live in config.py
from config import RENDER_SYS as SYS, STYLE_PROMPT


def _generate(case, register, style, model, mock):
    if mock:
        return render(case, register, style)   # offline stand-in (templated)
    record = _record_block(case)
    unit = case["entities"][case["focal"]]["name"]
    user = (STYLE_PROMPT[style].format(unit=unit, register=register.replace("_", "-"))
            + f"\n\nRECORD:\n{record}")
    return J.chat(model, [{"role": "system", "content": SYS},
                          {"role": "user", "content": user}], max_tokens=400).strip()


def _accept(style, n_unsup):
    if style == "good":
        return n_unsup == 0           # verified: fully grounded
    if style == "fabricated":
        return n_unsup >= 1           # must actually hallucinate
    return True                       # flattened: any


def render_verified(case, register, style, model, mock, max_tries):
    best = None
    for t in range(1, max_tries + 1):
        text = _generate(case, register, style, model, mock)
        F, nu = faithfulness(text, case)
        rec = {"text": text, "F": round(F, 3), "unsupported": nu, "tries": t,
               "accepted": _accept(style, nu), "source": "mock" if mock else "llm"}
        if rec["accepted"]:
            return rec
        best = rec
    # good couldn't be grounded -> fall back to the templated good (guaranteed clean)
    if style == "good":
        text = render(case, register, "good")
        F, nu = faithfulness(text, case)
        return {"text": text, "F": round(F, 3), "unsupported": nu, "tries": max_tries,
                "accepted": nu == 0, "source": "template_fallback"}
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=list(J.BACKENDS), default="ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=2, help="cases (bounds API calls)")
    ap.add_argument("--max-tries", type=int, default=3, dest="max_tries")
    ap.add_argument("--mock", action="store_true", help="offline templated stand-in")
    args = ap.parse_args()
    J._BACKEND = args.backend
    model = args.model or J.BACKENDS[args.backend]["default_model"]

    cells = [(c, r) for c in CASES[:args.limit] for r in REGISTERS]
    print(f"Backend: {'MOCK' if args.mock else f'{args.backend}:{model}'}  | "
          f"cases: {len(cells)}  (~{len(cells)*3*args.max_tries} max calls)\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    rows, stats = [], {s: {"acc": 0, "n": 0, "F": []} for s in STYLE_PROMPT}
    with OUT.open("w", encoding="utf-8") as fh:
        for case, reg in cells:
            for style in STYLE_PROMPT:
                rec = render_verified(case, reg, style, model, args.mock, args.max_tries)
                row = {"case_id": case["case_id"], "register": reg, "style": style, **rec}
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats[style]["n"] += 1
                stats[style]["acc"] += rec["accepted"]
                stats[style]["F"].append(rec["F"])

    print(f"{'style':<12}{'accepted':>10}{'mean F':>9}   example (truncated)")
    print("-" * 78)
    for s in STYLE_PROMPT:
        ex = next(r["text"] for r in rows if r["style"] == s)
        meanF = sum(stats[s]["F"]) / max(1, len(stats[s]["F"]))
        print(f"{s:<12}{stats[s]['acc']}/{stats[s]['n']:>8}{meanF:>9.2f}   {ex[:42].replace(chr(10),' ')}…")
    print(f"\nWrote {len(rows)} verified renderings -> {OUT}")
    print("Grounding-verification: 'good' is accepted ONLY with 0 unsupported specifics,")
    print("so these are clean SFT targets (use them in sft_narrative.py instead of templated).")
    if args.mock:
        print("\n[MOCK] templated stand-in — drop --mock with a backend to get LLM-written text.")


if __name__ == "__main__":
    main()
