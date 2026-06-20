#!/usr/bin/env python3
"""
build_training_data.py
================================================================
The DATA-CENTRIC half of the thesis, made actionable.

"The Occasional Bitter Lesson" inverts for contested generative targets: you do
not win by scaling a culture-blind objective, you win with the RIGHT data. This
harness manufactures that data — turning a best-of-N candidate set
(bestofn_demo.py --save) into ready-to-train files:

  * SFT targets        {prompt, completion}            -> sft_narrative.py --source jsonl --data …
  * preference pairs   {prompt, chosen, rejected, …}   -> dpo_kto_train.py --pairs …

It labels which candidate is the GOOD detail two ways (`--label`):

  reward  — BOOTSTRAP, no humans needed. The grounded composite/faithfulness
            reward picks `chosen` (faithful AND rich, past the gate); a vivid-but-
            unfaithful candidate becomes `rejected` (tagged fabrication / flattened).
            Scales now; only as good as the reward.
  human   — GOLD. The historians' workshop ratings (analyze_annotations input)
            pick chosen = top-rated, rejected = bottom-rated, but ONLY when the
            rating gap is real (--min_gap) — so you never train on noise. Optionally
            still gate the chosen on faithfulness so a liked hallucination is dropped.

The prompt is rebuilt in the SAME chat-instruction format GRPO/SFT/generate feed,
so the data drops straight into the existing trainers.

SAFEGUARDS: every row carries provenance + an UNVERIFIED stamp; redaction is
inherited from the candidate file (generate with --no_redact only for local,
un-committed training and keep it out of git). A methods artifact, not a source.

  python build_training_data.py workshop_candidates.json --label reward
  python build_training_data.py workshop_candidates.json --label human --responses responses/ --min_gap 1.0
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter
from statistics import mean

try:
    from config import INSTRUCTION
except ImportError as e:
    sys.exit(f"Run from the project folder: {e}")

STAMP = "UNVERIFIED — machine-generated training target, not a historical source."


def _check_human_join(means, recs):
    """Loud, precise check for the silent failure mode: the ratings join by
    (record_id, candidate_i), so a candidates file that has been REGENERATED (different
    #candidates per record, or re-indexed) can share record IDs yet fail to line up at
    the candidate level — dropping the high/low-rated candidates and leaving only close
    survivors, so nothing clears --min_gap."""
    file_rids = {r["id"] for r in recs}
    file_keys = {(r["id"], c["i"]) for r in recs for c in r.get("candidates", [])}
    rated_keys = set(means)
    rec_overlap = file_rids & {k[0] for k in rated_keys}
    if not rec_overlap:
        print("  ⚠ ID MISMATCH: none of the rated records are in this candidates file — the ratings "
              "were collected against a DIFFERENT file. Use the workshop_candidates.json the "
              "annotation sheet was built from.")
        return
    shared_rated = {k for k in rated_keys if k[0] in file_rids}
    matched = shared_rated & file_keys
    counts = sorted({len(r.get("candidates", [])) for r in recs})
    rated_max_i = max((k[1] for k in rated_keys), default=-1)
    print(f"  ({len(rec_overlap)}/{len(file_rids)} records rated; "
          f"{len(matched)}/{len(shared_rated)} rated candidate-slots present in this file)")
    if len(matched) < len(shared_rated):
        print("  ⚠ CANDIDATE-SET MISMATCH: this candidates file has a different candidate set than the "
              "historians rated (it is a different bestofn run).")
        print(f"    This file has {counts[0]}–{counts[-1]} candidates/record; the ratings reference "
              f"candidate_i up to {rated_max_i}. Ratings join by (record_id, candidate_i): missing "
              f"slots are dropped and survivors may map to different texts, so pairs collapse.")
        print("    FIX: run against the SAME workshop_candidates.json that make_annotation_sheet.py "
              "used — the file whose candidate texts the historians actually rated.")


def _check_id_overlap(resp_rids, recs, verb):
    """Loud check: do the workshop record_ids actually match the records file? A silent
    mismatch (e.g. candidates regenerated after the workshop → new IDs) is the usual
    cause of 'every record skipped'."""
    cand_rids = {r.get("id") for r in recs}
    overlap = resp_rids & cand_rids
    if not overlap:
        print(f"  ⚠ ID MISMATCH: 0 of {len(cand_rids)} records in the records file are {verb} in "
              f"--responses/--authored.")
        print(f"    The ratings/summaries were collected against a DIFFERENT candidates file. "
              f"Use the SAME workshop_candidates.json that the sheet was built from — do not "
              f"regenerate candidates after the workshop (it mints new record IDs).")
        print(f"    e.g. records file has {sorted(cand_rids)[:1]}…, workshop has {sorted(resp_rids)[:1]}…")
    elif len(overlap) < len(cand_rids):
        print(f"  ({len(overlap)}/{len(cand_rids)} records {verb}; "
              f"{len(cand_rids) - len(overlap)} have no workshop data)")


def _user_text(rec):
    """The raw user instruction. Prefer the stored prompt; else rebuild from the
    record block (older candidate files)."""
    if rec.get("prompt_user"):
        return rec["prompt_user"]
    register = (rec.get("register") or "archival_micro_narrative").replace("_", "-")
    return INSTRUCTION.format(register=register, unit=rec.get("unit", "[unit]"),
                              record=rec.get("record_block", ""))


def _wrap(tok, user_text):
    """Wrap the instruction in the model's chat template (native EOS), exactly as
    grpo_train.chat_prompt / sft_narrative / generate do — so the TRAINING prompt is
    byte-identical to what the model sees at inference, and the prompt is a clean
    token-prefix of prompt+completion (no TRL tokenizer-mismatch warning). tok=None
    falls back to the raw instruction."""
    if tok is None:
        return user_text
    return tok.apply_chat_template([{"role": "user", "content": user_text}],
                                   tokenize=False, add_generation_prompt=True)


def _human_means(responses_paths, candidates_path):
    """(record_id, candidate_i) -> mean historian rating, from analyze_annotations input."""
    files = []
    for p in responses_paths:
        files += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
    cand_abs = os.path.abspath(candidates_path)
    ratings = {}
    n_annot = 0
    for path in files:
        if os.path.abspath(path) == cand_abs:
            continue
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if "responses" not in d:
            continue
        n_annot += 1
        for r in d["responses"]:
            ratings.setdefault((r["record_id"], r["candidate_i"]), []).append(r["overall"])
    return {k: mean(v) for k, v in ratings.items()}, n_annot


def _reward_label(cands, gate_unsup, fab_unsup, fab_F):
    """chosen = richest faithful candidate (past the gate); rejected = a vivid but
    unfaithful one. Returns (chosen, rejected, rejected_kind) or None if no clean pair."""
    survivors = [c for c in cands if c["unsup"] <= gate_unsup]
    if not survivors:
        return None
    chosen = max(survivors, key=lambda c: c["linguistic"])
    # reject the most surface-vivid candidate that the grounding flags as unfaithful
    bad = [c for c in cands if c["i"] != chosen["i"]
           and (c["unsup"] >= fab_unsup or c["F"] <= fab_F)]
    if not bad:
        return None
    rejected = max(bad, key=lambda c: c["linguistic"])
    kind = "fabrication" if rejected["unsup"] >= fab_unsup else "flattened"
    return chosen, rejected, kind


def _authored_by_rec(paths, records_path):
    """record_id -> list of {text, author, confidence} from authored_*.json (the
    historian authoring tool). Accepts files and/or a folder."""
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
    rec_abs = os.path.abspath(records_path)
    by_rec = {}
    n_auth = 0
    for path in files:
        if os.path.abspath(path) == rec_abs:
            continue
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        if "summaries" not in d:
            continue
        n_auth += 1
        author = d.get("author") or os.path.basename(path)
        for s in d["summaries"]:
            if s.get("text", "").strip():
                by_rec.setdefault(s["record_id"], []).append(
                    {"text": s["text"].strip(), "author": author,
                     "confidence": s.get("confidence")})
    return by_rec, n_auth


def _human_label(cands, means, rid, min_gap, gate_unsup, gate_chosen):
    rated = [(c, means[(rid, c["i"])]) for c in cands if (rid, c["i"]) in means]
    if len(rated) < 2:
        return None
    rated.sort(key=lambda t: t[1])
    (lo_c, lo_r), (hi_c, hi_r) = rated[0], rated[-1]
    if hi_r - lo_r < min_gap:
        return None                       # no real preference — don't train on noise
    if gate_chosen and hi_c["unsup"] > gate_unsup:
        return None                       # historians liked it, but it isn't grounded — drop
    kind = "fabrication" if lo_c["unsup"] >= gate_unsup + 1 else "low_rated"
    return hi_c, lo_c, kind, hi_r, lo_r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidates", help="a bestofn_demo.py --save JSON, or a records.json (authored mode)")
    ap.add_argument("--label", choices=["reward", "human", "authored"], default="reward")
    ap.add_argument("--responses", nargs="*", default=[],
                    help="(human mode) response JSON files and/or a FOLDER of them")
    ap.add_argument("--authored", nargs="*", default=[],
                    help="(authored mode) authored_*.json files and/or a FOLDER (historian-written summaries)")
    ap.add_argument("--gate_unsup", type=int, default=1,
                    help="max unsupported specifics allowed in a 'good' target")
    ap.add_argument("--fab_unsup", type=int, default=2,
                    help="(reward mode) unsupported specifics that mark a candidate fabricated")
    ap.add_argument("--fab_F", type=float, default=0.6,
                    help="(reward mode) faithfulness at/below which a candidate is 'unfaithful'")
    ap.add_argument("--min_gap", type=float, default=1.0,
                    help="(human mode) min top-vs-bottom mean-rating gap to emit a pair (1-5 scale)")
    ap.add_argument("--gate_human_chosen", action="store_true",
                    help="(human mode) also require the top-rated target to pass the faithfulness gate")
    ap.add_argument("--out_dir", default="data/EHRI/train")
    ap.add_argument("--model", default=None,
                    help="tokenizer for chat-wrapping prompts (default: config.DEFAULT_MODEL)")
    ap.add_argument("--no_chat_wrap", action="store_true",
                    help="emit RAW instruction prompts (NOT chat-templated). Only use if you train "
                         "with the same raw format — otherwise prompts won't match inference.")
    ap.add_argument("--dry_run", action="store_true", help="summarise, write nothing")
    args = ap.parse_args()

    blob = json.load(open(args.candidates, encoding="utf-8"))
    recs = blob.get("records", [])
    if not recs:
        sys.exit(f"No records in {args.candidates}")

    # Chat-template the prompts so train == inference (and TRL sees a clean token prefix).
    tok = None
    if not args.no_chat_wrap:
        try:
            from transformers import AutoTokenizer
            from config import DEFAULT_MODEL
            tok = AutoTokenizer.from_pretrained(args.model or DEFAULT_MODEL)
            print(f"Chat-wrapping prompts with {args.model or DEFAULT_MODEL}'s template "
                  f"(matches SFT/GRPO/generate).")
        except Exception as e:
            print(f"  ! could not load a tokenizer ({e}).")
            print(f"  ! emitting RAW prompts — these will MISMATCH inference. Re-run on a machine "
                  f"with transformers, or pass --no_chat_wrap to silence this intentionally.")

    means, n_annot = ({}, 0)
    authored, n_auth = ({}, 0)
    if args.label == "human":
        if not args.responses:
            sys.exit("--label human needs --responses <files or folder>")
        means, n_annot = _human_means(args.responses, args.candidates)
        if not means:
            sys.exit("No historian ratings found in --responses")
        print(f"Loaded ratings from {n_annot} annotator file(s); "
              f"{len(means)} rated candidate-items.")
        _check_human_join(means, recs)
    elif args.label == "authored":
        if not args.authored:
            sys.exit("--label authored needs --authored <files or folder>")
        authored, n_auth = _authored_by_rec(args.authored, args.candidates)
        if not authored:
            sys.exit("No historian-authored summaries found in --authored")
        print(f"Loaded {sum(len(v) for v in authored.values())} authored summaries "
              f"from {n_auth} historian file(s), covering {len(authored)} records.")
        _check_id_overlap(set(authored), recs, "summarised")

    sft, pairs = [], []
    skipped = 0
    kinds = Counter()
    for rec in recs:
        cands = rec.get("candidates", [])
        prompt = _wrap(tok, _user_text(rec))
        prov = {"corpus": blob.get("corpus"), "source": blob.get("source"),
                "gen_model": blob.get("gen_model"), "record_id": rec.get("id"),
                "label_source": args.label, "stamp": STAMP}

        # AUTHORED: each human summary is gold SFT; pair it against the least-faithful
        # machine candidate (if any) for a human-vs-machine preference signal.
        if args.label == "authored":
            mine = authored.get(rec.get("id"), [])
            if not mine:
                skipped += 1
                continue
            worst = min(cands, key=lambda c: c["F"]) if cands else None
            for a in mine:
                meta = {**prov, "author": a["author"], "confidence": a.get("confidence")}
                sft.append({"prompt": prompt, "completion": " " + a["text"], "meta": meta})
                if worst is not None and worst.get("text", "").strip() \
                        and worst["text"].strip() != a["text"]:
                    kinds["human_vs_machine"] += 1
                    pairs.append({"prompt": prompt, "chosen": a["text"],
                                  "rejected": worst["text"].strip(),
                                  "rejected_kind": "human_vs_machine",
                                  "meta": {**meta, "rejected_F": worst["F"],
                                           "rejected_unsup": worst["unsup"]}})
                else:
                    kinds["sft_only"] += 1
            continue

        if len(cands) < 2:
            skipped += 1
            continue

        if args.label == "reward":
            res = _reward_label(cands, args.gate_unsup, args.fab_unsup, args.fab_F)
            if not res:
                skipped += 1
                continue
            chosen, rejected, kind = res
            prov_extra = {"chosen_F": chosen["F"], "chosen_unsup": chosen["unsup"],
                          "rejected_F": rejected["F"], "rejected_unsup": rejected["unsup"]}
        else:
            res = _human_label(cands, means, rec.get("id"), args.min_gap,
                               args.gate_unsup, args.gate_human_chosen)
            if not res:
                skipped += 1
                continue
            chosen, rejected, kind, hi_r, lo_r = res
            prov_extra = {"chosen_rating": round(hi_r, 2), "rejected_rating": round(lo_r, 2),
                          "chosen_F": chosen["F"], "chosen_unsup": chosen["unsup"]}

        kinds[kind] += 1
        meta = {**prov, **prov_extra}
        sft.append({"prompt": prompt, "completion": " " + chosen["text"].strip(), "meta": meta})
        pairs.append({"prompt": prompt, "chosen": chosen["text"].strip(),
                      "rejected": rejected["text"].strip(), "rejected_kind": kind, "meta": meta})

    if not sft:
        sys.exit("No usable training rows (every record was skipped — loosen --min_gap / gate, "
                 "supply --authored summaries, or generate more candidates per record).")

    # --- summary ---
    print(f"\nLabel source: {args.label}   records: {len(recs)}   "
          f"SFT rows: {len(sft)}   pref pairs: {len(pairs)}   skipped: {skipped}")
    print(f"kind breakdown: {dict(kinds)}")
    sw = [len(r['completion'].split()) for r in sft]
    print(f"SFT target words: min {min(sw)}, median {int(sorted(sw)[len(sw)//2])}, max {max(sw)}")
    if pairs:
        bad = sum(1 for p in pairs if p["chosen"] == p["rejected"])
        print(f"pairs where chosen == rejected (must be 0): {bad}")
        ex = pairs[0]
        print("\n  e.g. CHOSEN  :", ex["chosen"][:140], "…")
        print("       REJECTED:", ex["rejected"][:140], "…", f"[{ex['rejected_kind']}]")
    else:
        print("  (SFT-only — no machine candidates to pair against; supply a bestofn --save "
              "file as the records arg to also get human-vs-machine preference pairs)")

    # --- faithfulness gate quality/quantity curve: how many records yield a 'good'
    #     target as the bar moves. On real archival data the grounded target is scarce
    #     — that scarcity IS the thesis (quality data beats scale, but costs effort). ---
    if any(rec.get("candidates") for rec in recs):
        print("\nGate sensitivity (records with ≥1 candidate at/below the unsupported-specifics bar):")
        for g in range(0, 5):
            survivors = sum(1 for rec in recs
                            if any(c["unsup"] <= g for c in rec.get("candidates", [])))
            mark = "  <- current" if g == args.gate_unsup else ""
            print(f"  unsup ≤ {g}:  {survivors}/{len(recs)} records have a groundable target{mark}")

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    sft_path = os.path.join(args.out_dir, f"sft_{args.label}.jsonl")
    pair_path = os.path.join(args.out_dir, f"preference_pairs_{args.label}.jsonl")
    with open(sft_path, "w", encoding="utf-8") as f:
        for r in sft:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(sft)} SFT rows   -> {sft_path}")
    print(f"\nTrain:\n  python sft_narrative.py --source jsonl --data {sft_path}")
    if pairs:
        with open(pair_path, "w", encoding="utf-8") as f:
            for r in pairs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(pairs)} pref pairs -> {pair_path}")
        print(f"  python dpo_kto_train.py --method dpo --pairs {pair_path}")
    print(f"\n[{STAMP}]")


if __name__ == "__main__":
    main()
