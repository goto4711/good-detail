#!/usr/bin/env python3
"""
bestofn_demo.py
================================================================
The HISTORIAN-FACING demo: the rewards as a LENS over any current LLM, no
training. A strong, PLUGGABLE model writes K candidate micro-narratives for a
real record; each reward then PICKS its favourite. Where the picks DIVERGE is the
whole thesis, told with a model people recognise:

  - the grounded (composite/NLI) reward picks the faithful one,
  - the surface (linguistic) reward picks the vivid-but-maybe-fabricated one,
  - the LLM judge (RLAIF) picks the fluent one.

The generation model is fully pluggable (`--gen_backend` / `--gen_model`): point
it at whatever the historians want to see — the uva proxy's largest model, a
frontier API, or a local vLLM serving the newest open model. Reward functions are
model-agnostic, so the same lens works over any of them.

SAFEGUARDS: tiny N, focal name redacted, every narrative stamped UNVERIFIED.
A methods demonstration, never a historical source.

  python bestofn_demo.py --corpus EHRI --source extracted --gen_backend uva --k 5 --limit 2 --judge
  python bestofn_demo.py --corpus EHRI --source xml --gen_backend vllm --gen_model Qwen/Qwen2.5-7B-Instruct --k 6 --limit 2
"""

import argparse
import sys

try:
    from ingest import load_corpus, record_block
    from config import INSTRUCTION
    from faithfulness import nli_faithfulness
    from linguistic_reward import linguistic_reward
    from realdata_generate import retrieve, attested_index, _composite, _redact, _BANNER
    import llm_judge_reward as J
except ImportError as e:
    sys.exit(f"Run from the project folder: {e}")

REWARDS = ["composite", "linguistic"]   # + "judge" when --judge


def _gen(model, messages, k, temperature, max_tokens):
    cands = []
    for _ in range(k):
        try:
            cands.append(J.chat(model, messages, max_tokens=max_tokens, temperature=temperature).strip())
        except (Exception, SystemExit):   # a backend timeout/down must not kill the run
            cands.append("")
    # keep only real candidates (drop empty / reasoning-model truncations)
    return [c for c in cands if len(c.split()) >= 4]


def _judge_score(record, text, backend, model):
    old = J._BACKEND
    J._BACKEND = backend
    try:
        return float(J.judge(record, text, model=model)[0])
    except (Exception, SystemExit):   # a rate-limited/down judge must not kill the demo
        return float("nan")
    finally:
        J._BACKEND = old


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="EHRI")
    ap.add_argument("--source", default="extracted")
    ap.add_argument("--limit", type=int, default=2, help="records (keep small)")
    ap.add_argument("--k", type=int, default=5, help="candidate narratives per record")
    ap.add_argument("--gen_backend", default="uva", help="LLM that WRITES (uva/vllm/gemini/ollama)")
    ap.add_argument("--gen_model", default=None, help="generation model (default: backend default)")
    ap.add_argument("--temperature", type=float, default=0.9, help="diversity across candidates")
    ap.add_argument("--max_tokens", type=int, default=220)
    ap.add_argument("--premise_sents", type=int, default=20)
    ap.add_argument("--relations", action="store_true",
                    help="include extracted relations in the record block (off by default)")
    ap.add_argument("--no_redact", action="store_true")
    ap.add_argument("--gate", action="store_true",
                    help="deployable pick: keep only grounded candidates (unsup<=gate_unsup), "
                         "then the richest — always shows a grounded narrative")
    ap.add_argument("--gate_unsup", type=int, default=1,
                    help="max unsupported specifics allowed past the gate (0 is too strict on real data)")
    ap.add_argument("--judge", action="store_true", help="add the LLM-judge reward as a third selector")
    ap.add_argument("--judge_backend", default="gemini")
    ap.add_argument("--judge_model", default=None)
    ap.add_argument("--save", default=None,
                    help="dump the scored candidate sets to JSON (workshop input for make_annotation_sheet.py)")
    args = ap.parse_args()

    J._BACKEND = args.gen_backend
    gen_model = args.gen_model or J.BACKENDS[args.gen_backend]["default_model"]
    judge_model = args.judge_model or J.BACKENDS[args.judge_backend]["default_model"]
    rewards = REWARDS + (["judge"] if args.judge else [])
    wr = args.relations

    recs = load_corpus(args.corpus, args.source, limit=args.limit)
    if not recs:
        sys.exit(f"No records for {args.corpus}/{args.source}")
    print(f"Generation LLM: {args.gen_backend}/{gen_model}  |  K={args.k}  |  "
          f"rewards: {', '.join(rewards)}  |  relations: {'on' if wr else 'off'}")
    print(f"{_BANNER}\n")
    saved = []   # structured candidate sets for the workshop (--save)

    diverged = 0
    for ri, rec in enumerate(recs, 1):
        focal = rec.unit or rec.title
        print(f"  [{ri}/{len(recs)}] {rec.id}: generating {args.k} candidates …", flush=True)
        excerpt = " ".join(retrieve(rec.source_text, focal, 8))
        record = record_block(rec, with_relations=wr) + ("\n\nSOURCE EXCERPT:\n" + excerpt if excerpt else "")
        messages = [{"role": "user", "content": INSTRUCTION.format(
            register=rec.register.replace("_", "-"), unit=focal, record=record)}]
        cands = _gen(gen_model, messages, args.k, args.temperature, args.max_tokens)
        if not cands:
            print("      no usable candidates (backend empty/down) — skipping record", flush=True)
            continue
        print(f"      scoring {len(cands)} usable candidates with the rewards …", flush=True)

        att_t, att_y = attested_index(rec, with_relations=wr)
        scored = []
        for t in cands:
            premises = retrieve(rec.source_text, focal + " " + t, args.premise_sents)
            F, unsup = nli_faithfulness(t, premises, subject=focal, grounded_toks=att_t, grounded_years=att_y)
            row = {"text": t, "F": F, "unsup": unsup, "composite": _composite(t, F, unsup),
                   "linguistic": linguistic_reward(t)}
            if args.judge:
                row["judge"] = _judge_score(record, t, args.judge_backend, judge_model)
            scored.append(row)

        picks = {r: max(range(len(scored)), key=lambda i: scored[i][r]) for r in rewards}
        order = list(rewards)
        if args.gate:
            # the DEPLOYABLE pick: keep only grounded candidates (unsup <= gate_unsup),
            # then choose the richest (highest linguistic). Fall back to least-bad (max F).
            survivors = [i for i, s in enumerate(scored) if s["unsup"] <= args.gate_unsup]
            gkey = f"grounded-gate(unsup<={args.gate_unsup})"
            picks[gkey] = (max(survivors, key=lambda i: scored[i]["linguistic"]) if survivors
                           else max(range(len(scored)), key=lambda i: scored[i]["F"]))
            order.append(gkey)
            print(f"  ({len(survivors)}/{args.k} candidates passed the faithfulness gate)")
        rewards_iter = order
        unit_disp = focal if args.no_redact else "[unit]"
        print(f"=== {rec.id} — {unit_disp}  ({rec.title[:55]}…)  [{args.k} candidates] ===")
        seen = {}
        for r in rewards_iter:
            i = picks[r]
            row = scored[i]
            tag = f"#{i}" + (f" (= {seen[i]})" if i in seen else "")
            seen.setdefault(i, r)
            shown = row["text"] if args.no_redact else _redact(row["text"], focal)
            print(f"\n  ── {r.upper()} reward picks candidate {tag} ──")
            print(f"  {shown}")
            js = f"  judge={row['judge']:+.2f}" if args.judge else ""
            print(f"    [F={row['F']:.2f} unsup={row['unsup']}  composite={row['composite']:+.2f}  "
                  f"linguistic={row['linguistic']:.2f}{js}]")
        distinct = len(set(picks[r] for r in rewards))   # divergence among the REWARDS (not the gate)
        if distinct > 1:
            diverged += 1
        print(f"\n  -> the rewards chose {distinct} DIFFERENT candidate(s) "
              f"{'— they disagree on what good detail is' if distinct > 1 else '— they agree here'}.")
        print(f"  [{_BANNER}]\n")

        if args.save:
            saved.append({
                "id": rec.id, "title": rec.title,
                "unit": focal if args.no_redact else "[unit]",
                "register": rec.register,
                "record_block": record,
                "prompt_user": messages[0]["content"],   # exact instruction → lets the harness rebuild the training prompt
                "picks": {k: int(v) for k, v in picks.items()},
                "candidates": [
                    {"i": i,
                     "text": s["text"] if args.no_redact else _redact(s["text"], focal),
                     "F": s["F"], "unsup": s["unsup"], "composite": s["composite"],
                     "linguistic": s["linguistic"],
                     **({"judge": s["judge"]} if args.judge else {})}
                    for i, s in enumerate(scored)],
            })

    if args.save:
        import json
        json.dump({"corpus": args.corpus, "source": args.source, "gen_model": gen_model,
                   "k": args.k, "records": saved},
                  open(args.save, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"Saved {len(saved)} candidate sets -> {args.save}  "
              f"(feed to: python make_annotation_sheet.py {args.save})")

    print(f"Rewards diverged on {diverged}/{len(recs)} records "
          f"(higher = the definition of 'good detail' changes which narrative wins).")


if __name__ == "__main__":
    main()
